#include <array>
#include <cmath>
#include <cstdlib>
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

struct PairSpec {
    std::string name;
    std::string center;
    std::string partner;
};

static const std::vector<PairSpec> PAIRS = {
    {"Cr-Sb", "Cr", "Sb"},
    {"Cr-Te", "Cr", "Te"},
    {"Sb-Te", "Sb", "Te"},
};

struct ModelSpec {
    std::string id;
    std::string family;
    long long min_step = 0;
    std::vector<std::string> paths;
};

struct PairAccum {
    std::vector<long double> hist;
    long double denominator_base = 0.0L;
    long double partner_density_sum = 0.0L;
    long double center_count_sum = 0.0L;
    long double partner_count_sum = 0.0L;
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
    std::vector<PairAccum> pairs;
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

static Result process_model(const ModelSpec& spec, int frame_stride, double rmax, double dr) {
    Result result;
    result.spec = spec;
    const std::size_t nbins = static_cast<std::size_t>(std::ceil(rmax / dr));
    result.pairs.resize(PAIRS.size());
    for (auto& p : result.pairs) p.hist.assign(nbins, 0.0L);

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
            std::vector<std::string> species(static_cast<std::size_t>(natoms));
            std::vector<Vec3> pos(static_cast<std::size_t>(natoms));
            bool finite = cell_ok && std::isfinite(volume) && volume > 0.0;
            for (int a = 0; a < natoms; ++a) {
                if (!std::getline(in, line)) { finite = false; break; }
                std::stringstream ass(line);
                if (!(ass >> species[a] >> pos[a][0] >> pos[a][1] >> pos[a][2])) {
                    finite = false;
                    continue;
                }
                if (!finite_vec(pos[a])) finite = false;
            }
            // Consume every atom block before applying the de-duplication
            // rule.  This mirrors the validated MD15 parser and prevents a
            // repeated boundary frame from disturbing the next extxyz frame.
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
            result.volume_sum += volume;

            std::vector<Vec3> frac(static_cast<std::size_t>(natoms));
            for (int a = 0; a < natoms; ++a) frac[a] = frac_from_cart(pos[a], inv);

            for (std::size_t q = 0; q < PAIRS.size(); ++q) {
                const auto& pair = PAIRS[q];
                long long ncenter = 0, npartner = 0;
                for (const auto& s : species) {
                    if (s == pair.center) ++ncenter;
                    if (s == pair.partner) ++npartner;
                }
                if (ncenter == 0 || npartner == 0) continue;
                auto& acc = result.pairs[q];
                acc.center_count_sum += ncenter;
                acc.partner_count_sum += npartner;
                acc.partner_density_sum += static_cast<long double>(npartner) / volume;
                acc.denominator_base += static_cast<long double>(ncenter) * npartner / volume;

                for (int i = 0; i < natoms; ++i) {
                    if (species[i] != pair.center) continue;
                    for (int j = 0; j < natoms; ++j) {
                        if (species[j] != pair.partner) continue;
                        Vec3 df{frac[j][0] - frac[i][0], frac[j][1] - frac[i][1], frac[j][2] - frac[i][2]};
                        for (double& x : df) x -= std::nearbyint(x);
                        const Vec3 dcart = cart_from_frac(df, cell);
                        const double r2 = dcart[0] * dcart[0] + dcart[1] * dcart[1] + dcart[2] * dcart[2];
                        if (!std::isfinite(r2) || r2 < 0.0 || r2 >= rmax * rmax) continue;
                        const std::size_t bin = static_cast<std::size_t>(std::floor(std::sqrt(r2) / dr));
                        if (bin < acc.hist.size()) acc.hist[bin] += 1.0L;
                    }
                }
            }
        }
    }
    return result;
}

static void write_outputs(const Result& r, const fs::path& out_dir, double rmax, double dr) {
    (void)rmax;
    fs::create_directories(out_dir);
    std::ofstream out(out_dir / (r.spec.id + ".csv"));
    out << "pair,r_A,g_pair,sampled_frames,first_step,last_step,mean_n_center,mean_n_partner,mean_volume_A3\n" << std::setprecision(12);
    for (std::size_t q = 0; q < PAIRS.size(); ++q) {
        const auto& acc = r.pairs[q];
        const double mean_center = r.sampled_frames ? static_cast<double>(acc.center_count_sum / r.sampled_frames) : 0.0;
        const double mean_partner = r.sampled_frames ? static_cast<double>(acc.partner_count_sum / r.sampled_frames) : 0.0;
        const double mean_volume = r.sampled_frames ? static_cast<double>(r.volume_sum / r.sampled_frames) : 0.0;
        for (std::size_t b = 0; b < acc.hist.size(); ++b) {
            const double r0 = b * dr, r1 = (b + 1) * dr;
            const double shell = 4.0 * std::acos(-1.0) / 3.0 * (r1 * r1 * r1 - r0 * r0 * r0);
            const long double denom = acc.denominator_base * shell;
            const double g = denom > 0.0L ? static_cast<double>(acc.hist[b] / denom) : nanv();
            out << PAIRS[q].name << "," << (r0 + r1) / 2.0 << "," << g << "," << r.sampled_frames << ","
                << r.first_step << "," << r.last_step << "," << mean_center << "," << mean_partner << "," << mean_volume << "\n";
        }
    }
    out.close();

    std::ofstream meta(out_dir / (r.spec.id + ".meta.tsv"));
    meta << "model_id\tfamily\traw_frames\teligible_frames\tsampled_frames\tinvalid_frames\tfirst_step\tlast_step\n";
    meta << r.spec.id << "\t" << r.spec.family << "\t" << r.raw_frames << "\t" << r.eligible_frames << "\t"
         << r.sampled_frames << "\t" << r.invalid_frames << "\t" << r.first_step << "\t" << r.last_step << "\n";
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: compute_partial_rdf_r05 manifest.tsv output_dir [frame_stride] [rmax] [dr]\n";
        return 2;
    }
    const std::string manifest_path = argv[1];
    const fs::path out_dir = argv[2];
    const int frame_stride = argc > 3 ? std::stoi(argv[3]) : 20;
    const double rmax = argc > 4 ? std::stod(argv[4]) : 8.0;
    const double dr = argc > 5 ? std::stod(argv[5]) : 0.05;

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
    std::ofstream summary(out_dir / "partial_rdf_summary.tsv");
    summary << "model_id\tfamily\traw_frames\teligible_frames\tsampled_frames\tinvalid_frames\tfirst_step\tlast_step\n";
    for (const auto& kv : specs) {
        std::cerr << "PARTIAL_RDF " << kv.second.id << " (" << kv.second.paths.size() << " paths)\n";
        const Result r = process_model(kv.second, frame_stride, rmax, dr);
        write_outputs(r, out_dir, rmax, dr);
        summary << r.spec.id << "\t" << r.spec.family << "\t" << r.raw_frames << "\t" << r.eligible_frames << "\t"
                << r.sampled_frames << "\t" << r.invalid_frames << "\t" << r.first_step << "\t" << r.last_step << "\n";
    }
    return 0;
}
