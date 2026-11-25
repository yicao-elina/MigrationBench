#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using Vec3 = std::array<double, 3>;

struct ModelSpec {
    std::string id;
    std::string family;
    long long min_step = 0;
    std::vector<std::string> paths;
};

struct Result {
    ModelSpec spec;
    long long raw_frames = 0;
    long long eligible_frames = 0;
    long long sampled_frames = 0;
    long long invalid_frames = 0;
    long long first_step = -1;
    long long last_step = -1;
    long double volume_sum = 0.0L;
    long double atom_count_sum = 0.0L;
    long double denominator_base = 0.0L;
    std::vector<long double> hist;
};

static double nanv() { return std::numeric_limits<double>::quiet_NaN(); }

static bool parse_key(const std::string& header, const std::string& key, double& value) {
    const std::string token = " " + key + "=";
    std::size_t p = header.find(token);
    if (p == std::string::npos && header.rfind(key + "=", 0) == 0) p = 0;
    if (p == std::string::npos) return false;
    const std::size_t start = p + (p == 0 ? key.size() + 1 : token.size());
    std::size_t end = start;
    while (end < header.size() && header[end] != ' ' && header[end] != '\t') ++end;
    try {
        value = std::stod(header.substr(start, end - start));
        return std::isfinite(value);
    } catch (...) {
        return false;
    }
}

static bool parse_lattice(const std::string& header, double c[3][3]) {
    const std::string key = "Lattice=\"";
    const std::size_t p = header.find(key);
    if (p == std::string::npos) return false;
    const std::size_t start = p + key.size();
    const std::size_t end = header.find('"', start);
    if (end == std::string::npos) return false;
    std::stringstream ss(header.substr(start, end - start));
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) {
        if (!(ss >> c[i][j]) || !std::isfinite(c[i][j])) return false;
    }
    return true;
}

static double determinant(const double c[3][3]) {
    return c[0][0] * (c[1][1] * c[2][2] - c[1][2] * c[2][1])
         - c[0][1] * (c[1][0] * c[2][2] - c[1][2] * c[2][0])
         + c[0][2] * (c[1][0] * c[2][1] - c[1][1] * c[2][0]);
}

static bool inverse(const double c[3][3], double inv[3][3]) {
    const double d = determinant(c);
    if (!std::isfinite(d) || std::abs(d) < 1e-12) return false;
    inv[0][0] = (c[1][1] * c[2][2] - c[1][2] * c[2][1]) / d;
    inv[0][1] = (c[0][2] * c[2][1] - c[0][1] * c[2][2]) / d;
    inv[0][2] = (c[0][1] * c[1][2] - c[0][2] * c[1][1]) / d;
    inv[1][0] = (c[1][2] * c[2][0] - c[1][0] * c[2][2]) / d;
    inv[1][1] = (c[0][0] * c[2][2] - c[0][2] * c[2][0]) / d;
    inv[1][2] = (c[0][2] * c[1][0] - c[0][0] * c[1][2]) / d;
    inv[2][0] = (c[1][0] * c[2][1] - c[1][1] * c[2][0]) / d;
    inv[2][1] = (c[0][1] * c[2][0] - c[0][0] * c[2][1]) / d;
    inv[2][2] = (c[0][0] * c[1][1] - c[0][1] * c[1][0]) / d;
    return true;
}

static Vec3 frac_from_cart(const Vec3& cart, const double inv[3][3]) {
    Vec3 f{0.0, 0.0, 0.0};
    for (int k = 0; k < 3; ++k) for (int l = 0; l < 3; ++l) f[k] += cart[l] * inv[l][k];
    return f;
}

static Vec3 cart_from_frac(const Vec3& frac, const double c[3][3]) {
    Vec3 cart{0.0, 0.0, 0.0};
    for (int k = 0; k < 3; ++k) for (int l = 0; l < 3; ++l) cart[k] += frac[l] * c[l][k];
    return cart;
}

static bool finite_vec(const Vec3& v) {
    return std::isfinite(v[0]) && std::isfinite(v[1]) && std::isfinite(v[2]);
}

static double wrap01(double x) {
    x -= std::floor(x);
    if (x >= 1.0) x = 0.0;
    if (x < 0.0) x += 1.0;
    return x;
}

static int bin_id(int i, int j, int k, const std::array<int, 3>& nb) {
    return (i * nb[1] + j) * nb[2] + k;
}

static std::vector<int> unique_bins(int center, int radius, int nbin) {
    std::vector<int> out;
    if (nbin <= 0) return out;
    if (2 * radius + 1 >= nbin) {
        for (int i = 0; i < nbin; ++i) out.push_back(i);
        return out;
    }
    for (int d = -radius; d <= radius; ++d) {
        int idx = (center + d) % nbin;
        if (idx < 0) idx += nbin;
        bool seen = false;
        for (int old : out) if (old == idx) { seen = true; break; }
        if (!seen) out.push_back(idx);
    }
    return out;
}

static void accumulate_frame(
    const std::vector<Vec3>& positions,
    const double cell[3][3],
    const double inv[3][3],
    double volume,
    double rmax,
    double dr,
    Result& result
) {
    const int natoms = static_cast<int>(positions.size());
    std::vector<Vec3> frac(static_cast<std::size_t>(natoms));
    for (int i = 0; i < natoms; ++i) {
        frac[i] = frac_from_cart(positions[i], inv);
        for (double& x : frac[i]) x = wrap01(x);
    }

    // A fractional-coordinate cell list limits candidate pairs while retaining
    // the exact triclinic minimum-image distance as the acceptance criterion.
    std::array<int, 3> nb{1, 1, 1};
    for (int k = 0; k < 3; ++k) {
        const double length = std::sqrt(cell[k][0] * cell[k][0] + cell[k][1] * cell[k][1] + cell[k][2] * cell[k][2]);
        nb[k] = std::max(1, static_cast<int>(std::floor(length / rmax)));
    }
    const int ncell = nb[0] * nb[1] * nb[2];
    std::vector<std::vector<int>> cells(static_cast<std::size_t>(ncell));
    std::vector<std::array<int, 3>> atom_bins(static_cast<std::size_t>(natoms));
    for (int i = 0; i < natoms; ++i) {
        std::array<int, 3> b{
            std::min(nb[0] - 1, static_cast<int>(frac[i][0] * nb[0])),
            std::min(nb[1] - 1, static_cast<int>(frac[i][1] * nb[1])),
            std::min(nb[2] - 1, static_cast<int>(frac[i][2] * nb[2]))
        };
        atom_bins[i] = b;
        cells[bin_id(b[0], b[1], b[2], nb)].push_back(i);
    }

    std::array<int, 3> search_radius{1, 1, 1};
    for (int k = 0; k < 3; ++k) {
        double reciprocal_norm = 0.0;
        for (int l = 0; l < 3; ++l) reciprocal_norm += inv[l][k] * inv[l][k];
        reciprocal_norm = std::sqrt(reciprocal_norm);
        search_radius[k] = std::max(1, static_cast<int>(std::ceil(rmax * reciprocal_norm * nb[k])) + 1);
    }

    for (int i = 0; i < natoms; ++i) {
        const auto xbins = unique_bins(atom_bins[i][0], search_radius[0], nb[0]);
        const auto ybins = unique_bins(atom_bins[i][1], search_radius[1], nb[1]);
        const auto zbins = unique_bins(atom_bins[i][2], search_radius[2], nb[2]);
        for (int bx : xbins) for (int by : ybins) for (int bz : zbins) {
            for (int j : cells[bin_id(bx, by, bz, nb)]) {
                if (j <= i) continue;
                Vec3 df{frac[j][0] - frac[i][0], frac[j][1] - frac[i][1], frac[j][2] - frac[i][2]};
                for (double& x : df) x -= std::nearbyint(x);
                const Vec3 dcart = cart_from_frac(df, cell);
                const double r2 = dcart[0] * dcart[0] + dcart[1] * dcart[1] + dcart[2] * dcart[2];
                if (!std::isfinite(r2) || r2 < 0.0 || r2 >= rmax * rmax) continue;
                const std::size_t bin = static_cast<std::size_t>(std::floor(std::sqrt(r2) / dr));
                if (bin < result.hist.size()) result.hist[bin] += 2.0L;
            }
        }
    }
    result.volume_sum += volume;
    result.atom_count_sum += natoms;
    result.denominator_base += static_cast<long double>(natoms) * (natoms - 1) / volume;
}

static Result process_model(const ModelSpec& spec, int frame_stride, double rmax, double dr) {
    Result result;
    result.spec = spec;
    result.hist.assign(static_cast<std::size_t>(std::ceil(rmax / dr)), 0.0L);

    long long last_step_seen = -1;
    for (const std::string& path : spec.paths) {
        std::ifstream in(path);
        if (!in) {
            std::cerr << "ERROR open " << path << "\n";
            ++result.invalid_frames;
            continue;
        }
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty()) continue;
            int natoms = 0;
            std::stringstream nss(line);
            if (!(nss >> natoms) || natoms <= 0 || natoms > 100000) continue;
            std::string header;
            if (!std::getline(in, header)) break;

            double step_value = 0.0;
            const bool step_ok = parse_key(header, "step", step_value);
            const long long step = step_ok ? static_cast<long long>(std::llround(step_value)) : result.raw_frames;
            ++result.raw_frames;

            double cell[3][3], inv[3][3];
            const bool cell_ok = parse_lattice(header, cell) && inverse(cell, inv);
            const double volume = cell_ok ? std::abs(determinant(cell)) : nanv();
            std::vector<Vec3> positions(static_cast<std::size_t>(natoms));
            bool finite = cell_ok && std::isfinite(volume) && volume > 0.0;
            for (int a = 0; a < natoms; ++a) {
                if (!std::getline(in, line)) { finite = false; break; }
                std::stringstream ass(line);
                std::string species;
                if (!(ass >> species >> positions[a][0] >> positions[a][1] >> positions[a][2])) {
                    finite = false;
                    continue;
                }
                if (!finite_vec(positions[a])) finite = false;
            }
            // Consume the full atom block before de-duplicating repeated segment
            // boundaries, so a repeated frame cannot desynchronise the parser.
            if (step_ok && step <= last_step_seen) continue;
            if (step_ok) last_step_seen = step;
            if (!finite || step < spec.min_step) {
                if (!finite) ++result.invalid_frames;
                continue;
            }
            ++result.eligible_frames;
            if (result.first_step < 0) result.first_step = step;
            result.last_step = step;
            if ((result.eligible_frames - 1) % frame_stride != 0) continue;
            ++result.sampled_frames;
            accumulate_frame(positions, cell, inv, volume, rmax, dr, result);
        }
    }
    return result;
}

static void write_outputs(const Result& result, const fs::path& out_dir, double rmax, double dr) {
    fs::create_directories(out_dir);
    std::ofstream out(out_dir / (result.spec.id + ".csv"));
    out << "r_A,g_global,sampled_frames,first_step,last_step,n_atoms,mean_volume_A3\n" << std::setprecision(12);
    for (std::size_t b = 0; b < result.hist.size(); ++b) {
        const double r0 = b * dr, r1 = (b + 1) * dr;
        const double shell = 4.0 * std::acos(-1.0) / 3.0 * (r1 * r1 * r1 - r0 * r0 * r0);
        const long double denom = result.denominator_base * shell;
        const double g = denom > 0.0L ? static_cast<double>(result.hist[b] / denom) : nanv();
        const double mean_volume = result.sampled_frames ? static_cast<double>(result.volume_sum / result.sampled_frames) : nanv();
        const double mean_atoms = result.sampled_frames ? static_cast<double>(result.atom_count_sum / result.sampled_frames) : nanv();
        out << (r0 + r1) / 2.0 << "," << g << "," << result.sampled_frames << ","
            << result.first_step << "," << result.last_step << "," << mean_atoms << "," << mean_volume << "\n";
    }
    out.close();

    std::ofstream meta(out_dir / (result.spec.id + ".meta.tsv"));
    meta << "model_id\tfamily\traw_frames\teligible_frames\tsampled_frames\tinvalid_frames\tfirst_step\tlast_step\n";
    meta << result.spec.id << "\t" << result.spec.family << "\t" << result.raw_frames << "\t"
         << result.eligible_frames << "\t" << result.sampled_frames << "\t" << result.invalid_frames
         << "\t" << result.first_step << "\t" << result.last_step << "\n";
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: compute_global_rdf_r06 manifest.tsv output_dir [frame_stride] [rmax] [dr]\n";
        return 2;
    }
    const std::string manifest_path = argv[1];
    const fs::path out_dir = argv[2];
    const int frame_stride = argc > 3 ? std::stoi(argv[3]) : 20;
    const double rmax = argc > 4 ? std::stod(argv[4]) : 10.0;
    const double dr = argc > 5 ? std::stod(argv[5]) : 0.1;

    std::ifstream mf(manifest_path);
    if (!mf) return 3;
    std::map<std::string, ModelSpec> specs;
    std::string line;
    while (std::getline(mf, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        std::string id, family, min_step, path;
        if (!std::getline(ss, id, '\t') || !std::getline(ss, family, '\t') ||
            !std::getline(ss, min_step, '\t') || !std::getline(ss, path)) continue;
        if (!specs.count(id)) specs[id] = ModelSpec{id, family, std::stoll(min_step), {}};
        specs[id].paths.push_back(path);
    }

    fs::create_directories(out_dir);
    std::ofstream summary(out_dir / "global_rdf_summary.tsv");
    summary << "model_id\tfamily\traw_frames\teligible_frames\tsampled_frames\tinvalid_frames\tfirst_step\tlast_step\n";
    for (const auto& kv : specs) {
        std::cerr << "GLOBAL_RDF " << kv.second.id << " (" << kv.second.paths.size() << " paths)\n";
        const Result result = process_model(kv.second, frame_stride, rmax, dr);
        write_outputs(result, out_dir, rmax, dr);
        summary << result.spec.id << "\t" << result.spec.family << "\t" << result.raw_frames << "\t"
                << result.eligible_frames << "\t" << result.sampled_frames << "\t" << result.invalid_frames
                << "\t" << result.first_step << "\t" << result.last_step << "\n";
    }
    return 0;
}
