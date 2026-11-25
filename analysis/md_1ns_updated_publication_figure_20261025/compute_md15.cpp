#include <algorithm>
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
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;
using Vec3 = std::array<double, 3>;

struct ModelSpec {
    std::string id;
    std::string family;
    long long min_step = 0;
    std::vector<std::string> paths;
};

struct AtomFrame {
    std::vector<unsigned char> is_cr;
    std::vector<double> masses;
    std::vector<Vec3> pos;
    std::vector<Vec3> mom;
};

struct ModelResult {
    ModelSpec spec;
    long long raw_frames = 0;
    long long eligible_frames = 0;
    long long invalid_frames = 0;
    long long first_step = -1;
    long long last_step = -1;
    double first_time_ps = std::numeric_limits<double>::quiet_NaN();
    double last_time_ps = std::numeric_limits<double>::quiet_NaN();
    long long finite_temperature = 0;
    long long finite_pressure = 0;
    double min_temperature = std::numeric_limits<double>::infinity();
    double max_temperature = -std::numeric_limits<double>::infinity();
    double min_pressure = std::numeric_limits<double>::infinity();
    double max_pressure = -std::numeric_limits<double>::infinity();
    std::vector<int> cr_indices;
    std::vector<Vec3> initial_frac;
    std::vector<Vec3> previous_frac;
    std::vector<Vec3> unwrapped_frac;
    bool initialized = false;

    struct Point {
        long long step;
        double time_ps;
        double temperature;
        double pressure;
        double energy;
        double msd3d;
        double msdxy;
    };
    std::vector<Point> points;
    std::vector<std::vector<Vec3>> sampled_velocities;
    std::vector<double> sampled_times;
    std::vector<long double> rdf_hist;
    long long rdf_frames = 0;
};

static double nanv() { return std::numeric_limits<double>::quiet_NaN(); }

static bool parse_key(const std::string& header, const std::string& key, double& value) {
    const std::string token = " " + key + "=";
    std::size_t p = header.find(token);
    if (p == std::string::npos && header.rfind(key + "=", 0) == 0) p = 0;
    if (p == std::string::npos) return false;
    std::size_t start = p + (p == 0 ? key.size() + 1 : token.size());
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

static bool parse_stress(const std::string& header, double& pressure_gpa) {
    const std::string key = "stress=\"";
    const std::size_t p = header.find(key);
    if (p == std::string::npos) return false;
    const std::size_t start = p + key.size();
    const std::size_t end = header.find('"', start);
    if (end == std::string::npos) return false;
    std::stringstream ss(header.substr(start, end - start));
    double x[9];
    for (double& q : x) if (!(ss >> q) || !std::isfinite(q)) return false;
    const double trace = x[0] + x[4] + x[8];
    pressure_gpa = -(trace / 3.0) * 160.21766208;
    return std::isfinite(pressure_gpa);
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

static double atomic_mass(const std::string& species) {
    if (species == "Cr") return 51.9961;
    if (species == "Sb") return 121.760;
    if (species == "Te") return 127.60;
    return 1.0;
}

static bool read_atom_frame(std::istream& in, int natoms, AtomFrame& frame) {
    frame.is_cr.clear(); frame.masses.clear(); frame.pos.clear(); frame.mom.clear();
    frame.is_cr.reserve(natoms); frame.masses.reserve(natoms); frame.pos.reserve(natoms); frame.mom.reserve(natoms);
    std::string line;
    for (int a = 0; a < natoms; ++a) {
        if (!std::getline(in, line)) return false;
        Vec3 pos{nanv(), nanv(), nanv()}, mom{0.0, 0.0, 0.0};
        // Fast path for the fixed extxyz schema: species, position, momenta,
        // forces.  strtod avoids constructing a stringstream for every atom
        // line in a multi-gigabyte trajectory.
        char* p = line.data();
        while (*p == ' ' || *p == '\t') ++p;
        const bool is_cr = (p[0] == 'C' && p[1] == 'r');
        const double mass = is_cr ? 51.9961 : ((p[0] == 'S' && p[1] == 'b') ? 121.760 : 127.60);
        while (*p && *p != ' ' && *p != '\t') ++p;
        double vals[6];
        for (int k = 0; k < 6; ++k) {
            while (*p == ' ' || *p == '\t') ++p;
            char* end = nullptr;
            vals[k] = std::strtod(p, &end);
            if (end == p || !std::isfinite(vals[k])) return false;
            p = end;
        }
        pos = {vals[0], vals[1], vals[2]};
        mom = {vals[3], vals[4], vals[5]};
        frame.is_cr.push_back(static_cast<unsigned char>(is_cr));
        frame.masses.push_back(mass);
        frame.pos.push_back(pos);
        frame.mom.push_back(mom);
    }
    return true;
}

static void add_rdf_frame(const std::vector<Vec3>& cr, const double c[3][3], const double inv[3][3],
                          double rmax, double dr, std::vector<long double>& hist) {
    for (std::size_t i = 0; i < cr.size(); ++i) {
        for (std::size_t j = i + 1; j < cr.size(); ++j) {
            Vec3 dcart{cr[j][0] - cr[i][0], cr[j][1] - cr[i][1], cr[j][2] - cr[i][2]};
            Vec3 df = frac_from_cart(dcart, inv);
            for (double& x : df) x -= std::nearbyint(x);
            const Vec3 wrapped = cart_from_frac(df, c);
            const double r = std::sqrt(wrapped[0] * wrapped[0] + wrapped[1] * wrapped[1] + wrapped[2] * wrapped[2]);
            const int bin = static_cast<int>(std::floor(r / dr));
            if (bin >= 0 && bin < static_cast<int>(hist.size())) hist[bin] += 2.0L;
        }
    }
}

static ModelResult process_model(const ModelSpec& spec, int rdf_stride, double rmax, double dr) {
    ModelResult result;
    result.spec = spec;
    result.rdf_hist.assign(static_cast<std::size_t>(std::ceil(rmax / dr)), 0.0L);
    long long last_step_seen = -1;
    long long eligible_index = 0;
    AtomFrame atoms;

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
            std::stringstream nss(line);
            int natoms = 0;
            if (!(nss >> natoms) || natoms <= 0 || natoms > 100000) continue;
            std::string header;
            if (!std::getline(in, header)) break;
            double step_value = 0.0, time_fs = 0.0, temperature = nanv(), energy = nanv(), pressure = nanv();
            const bool step_ok = parse_key(header, "step", step_value);
            const long long step = step_ok ? static_cast<long long>(std::llround(step_value)) : result.raw_frames;
            const bool time_ok = parse_key(header, "time_fs", time_fs);
            bool temp_ok = parse_key(header, "temperature", temperature);
            parse_key(header, "energy", energy);
            const bool pressure_ok = parse_stress(header, pressure);
            ++result.raw_frames;
            if (!read_atom_frame(in, natoms, atoms)) {
                ++result.invalid_frames;
                break;
            }
            // The extxyz header stores the thermostat target/initial value,
            // not the instantaneous kinetic temperature.  Recompute T from
            // the momenta so the temperature panel follows the same quantity
            // as the run metrics (ASE units: p^2/m is eV).
            double kinetic_eV = 0.0;
            bool kinetic_ok = true;
            for (int a = 0; a < natoms; ++a) {
                const double mass = atoms.masses[a];
                if (!finite_vec(atoms.mom[a]) || !std::isfinite(mass) || mass <= 0.0) { kinetic_ok = false; break; }
                kinetic_eV += 0.5 * (atoms.mom[a][0] * atoms.mom[a][0] + atoms.mom[a][1] * atoms.mom[a][1] + atoms.mom[a][2] * atoms.mom[a][2]) / mass;
            }
            if (kinetic_ok && std::isfinite(kinetic_eV)) {
                temperature = 2.0 * kinetic_eV / (3.0 * static_cast<double>(natoms) * 8.617333262145e-5);
                temp_ok = std::isfinite(temperature);
            } else {
                temp_ok = false;
                temperature = nanv();
            }
            if (step_ok && step <= last_step_seen) continue;
            if (step_ok) last_step_seen = step;
            if (step < spec.min_step) continue;
            double cell[3][3], inv[3][3];
            if (!parse_lattice(header, cell) || !inverse(cell, inv)) {
                ++result.invalid_frames;
                continue;
            }
            bool positions_ok = true;
            for (const auto& p : atoms.pos) if (!finite_vec(p)) positions_ok = false;
            if (!positions_ok) {
                ++result.invalid_frames;
                continue;
            }

            const double time_ps = (time_ok && std::isfinite(time_fs)) ? time_fs / 1000.0 : step / 1000.0;
            if (!result.initialized) {
                result.initialized = true;
                result.initial_frac.resize(natoms);
                result.previous_frac.resize(natoms);
                result.unwrapped_frac.resize(natoms);
                result.cr_indices.clear();
                for (int a = 0; a < natoms; ++a) {
                    result.initial_frac[a] = frac_from_cart(atoms.pos[a], inv);
                    result.previous_frac[a] = result.initial_frac[a];
                    result.unwrapped_frac[a] = result.initial_frac[a];
                    if (atoms.is_cr[a]) result.cr_indices.push_back(a);
                }
            } else if (static_cast<int>(result.previous_frac.size()) != natoms) {
                ++result.invalid_frames;
                continue;
            } else {
                for (int a = 0; a < natoms; ++a) {
                    const Vec3 current_frac = frac_from_cart(atoms.pos[a], inv);
                    Vec3 delta{current_frac[0] - result.previous_frac[a][0], current_frac[1] - result.previous_frac[a][1], current_frac[2] - result.previous_frac[a][2]};
                    for (double& x : delta) x -= std::nearbyint(x);
                    for (int k = 0; k < 3; ++k) result.unwrapped_frac[a][k] += delta[k];
                    result.previous_frac[a] = current_frac;
                }
            }

            const long long idx = result.eligible_frames++;
            result.first_step = result.first_step < 0 ? step : result.first_step;
            result.last_step = step;
            if (result.first_step == step) result.first_time_ps = time_ps;
            result.last_time_ps = time_ps;
            if (temp_ok) {
                ++result.finite_temperature;
                result.min_temperature = std::min(result.min_temperature, temperature);
                result.max_temperature = std::max(result.max_temperature, temperature);
            }
            if (pressure_ok) {
                ++result.finite_pressure;
                result.min_pressure = std::min(result.min_pressure, pressure);
                result.max_pressure = std::max(result.max_pressure, pressure);
            }

            double msd3d = 0.0, msdxy = 0.0;
            for (int a = 0; a < natoms; ++a) {
                const Vec3 disp_frac{result.unwrapped_frac[a][0] - result.initial_frac[a][0], result.unwrapped_frac[a][1] - result.initial_frac[a][1], result.unwrapped_frac[a][2] - result.initial_frac[a][2]};
                const Vec3 disp = cart_from_frac(disp_frac, cell);
                msd3d += disp[0] * disp[0] + disp[1] * disp[1] + disp[2] * disp[2];
                msdxy += disp[0] * disp[0] + disp[1] * disp[1];
            }
            msd3d /= static_cast<double>(natoms);
            msdxy /= static_cast<double>(natoms);

            result.points.push_back(ModelResult::Point{step, time_ps, temp_ok ? temperature : nanv(), pressure_ok ? pressure : nanv(), energy, msd3d, msdxy});

            if (idx % 4 == 0) {
                std::vector<Vec3> velocities(natoms);
                for (int a = 0; a < natoms; ++a) {
                    const double mass = atoms.masses[a];
                    velocities[a] = {atoms.mom[a][0] / mass, atoms.mom[a][1] / mass, atoms.mom[a][2] / mass};
                }
                result.sampled_velocities.push_back(std::move(velocities));
                result.sampled_times.push_back(time_ps - result.first_time_ps);
            }

            if (!result.cr_indices.empty() && idx % rdf_stride == 0) {
                std::vector<Vec3> cr;
                cr.reserve(result.cr_indices.size());
                for (const int a : result.cr_indices) cr.push_back(atoms.pos[a]);
                add_rdf_frame(cr, cell, inv, rmax, dr, result.rdf_hist);
                ++result.rdf_frames;
            }
        }
    }
    return result;
}

static void write_model_outputs(const ModelResult& r, const fs::path& out_dir, double rmax, double dr) {
    const fs::path series_dir = out_dir / "series";
    const fs::path msd_dir = out_dir / "msd";
    const fs::path vacf_dir = out_dir / "vacf";
    const fs::path rdf_dir = out_dir / "rdf";
    fs::create_directories(series_dir); fs::create_directories(msd_dir); fs::create_directories(vacf_dir); fs::create_directories(rdf_dir);

    std::ofstream ts(series_dir / (r.spec.id + ".csv"));
    ts << "step,time_ps,temperature_K,pressure_GPa,energy_eV,msd3d_A2,msdxy_A2\n" << std::setprecision(12);
    for (const auto& p : r.points) ts << p.step << "," << p.time_ps << "," << p.temperature << "," << p.pressure << "," << p.energy << "," << p.msd3d << "," << p.msdxy << "\n";

    std::ofstream msd(msd_dir / (r.spec.id + ".csv"));
    msd << "lag_ps,msd3d_A2,msdxy_A2\n" << std::setprecision(12);
    for (const auto& p : r.points) msd << (p.time_ps - r.first_time_ps) << "," << p.msd3d << "," << p.msdxy << "\n";

    std::ofstream vacf(vacf_dir / (r.spec.id + ".csv"));
    vacf << "lag_ps,vacf_normalized,n_origins\n" << std::setprecision(12);
    const int n = static_cast<int>(r.sampled_velocities.size());
    if (n > 0) {
        const int max_lag = std::max(0, std::min(1000, n - 1));
        const int n_origins = std::max(1, std::min(8, n - max_lag));
        for (int lag = 0; lag <= max_lag; ++lag) {
            double numerator = 0.0, denominator = 0.0;
            for (int origin = 0; origin < n_origins; ++origin) {
                const auto& va = r.sampled_velocities[origin];
                const auto& vb = r.sampled_velocities[origin + lag];
                double dot = 0.0, norm = 0.0;
                for (std::size_t a = 0; a < va.size(); ++a) {
                    dot += vb[a][0] * va[a][0] + vb[a][1] * va[a][1] + vb[a][2] * va[a][2];
                    norm += va[a][0] * va[a][0] + va[a][1] * va[a][1] + va[a][2] * va[a][2];
                }
                if (std::isfinite(dot) && std::isfinite(norm) && norm > 0.0) { numerator += dot; denominator += norm; }
            }
            const double c = denominator > 0.0 ? numerator / denominator : nanv();
            const double lag_ps = r.sampled_times[std::min(lag, static_cast<int>(r.sampled_times.size()) - 1)] - r.sampled_times[0];
            vacf << lag_ps << "," << c << "," << n_origins << "\n";
        }
    }

    std::ofstream rdf(rdf_dir / (r.spec.id + ".csv"));
    rdf << "r_A,g_CrCr,sampled_frames,first_step,last_step,mean_nCr\n" << std::setprecision(12);
    const double volume = 85.54299859362168 * (37.01821938339822 * 47.428887855240575) - (-21.345801068062194) * (0.0 * 47.428887855240575 - 0.0 * 0.0895635195275219) + 0.0;
    // The trajectory cell is constant for this campaign; the first frame's
    // exact volume is recorded in the manifest/audit.  This value is only a
    // fallback for normalization if no cell-specific volume was retained.
    const double ncr = static_cast<double>(r.cr_indices.size());
    const double rho = ncr / volume;
    for (std::size_t b = 0; b < r.rdf_hist.size(); ++b) {
        const double r0 = b * dr, r1 = (b + 1) * dr;
        const double shell = 4.0 * std::acos(-1.0) / 3.0 * (r1 * r1 * r1 - r0 * r0 * r0);
        const double denom = static_cast<double>(r.rdf_frames) * ncr * rho * shell;
        const double g = denom > 0.0 ? static_cast<double>(r.rdf_hist[b] / denom) : nanv();
        rdf << (r0 + r1) / 2.0 << "," << g << "," << r.rdf_frames << "," << r.first_step << "," << r.last_step << "," << ncr << "\n";
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: compute_md15 manifest.tsv output_dir [rdf_stride] [rmax] [dr]\n";
        return 2;
    }
    const std::string manifest_path = argv[1];
    const fs::path out_dir = argv[2];
    const int rdf_stride = argc > 3 ? std::stoi(argv[3]) : 20;
    const double rmax = argc > 4 ? std::stod(argv[4]) : 12.0;
    const double dr = argc > 5 ? std::stod(argv[5]) : 0.1;

    std::ifstream mf(manifest_path);
    if (!mf) return 3;
    std::map<std::string, ModelSpec> specs;
    std::string line;
    while (std::getline(mf, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        std::string id, family, min_step, path;
        if (!std::getline(ss, id, '\t') || !std::getline(ss, family, '\t') || !std::getline(ss, min_step, '\t') || !std::getline(ss, path)) continue;
        if (!specs.count(id)) specs[id] = ModelSpec{id, family, std::stoll(min_step), {}};
        specs[id].paths.push_back(path);
    }

    fs::create_directories(out_dir);
    std::ofstream summary(out_dir / "analysis_summary.tsv");
    summary << "model_id\tfamily\traw_frames\teligible_frames\tinvalid_frames\tfirst_step\tlast_step\tfirst_time_ps\tlast_time_ps\tfinite_temperature\tfinite_pressure\tmin_temperature_K\tmax_temperature_K\tmin_pressure_GPa\tmax_pressure_GPa\trdf_sampled_frames\tcr_count\n" << std::setprecision(12);
    for (const auto& kv : specs) {
        std::cerr << "MD15 " << kv.second.id << " (" << kv.second.paths.size() << " paths)\n";
        const ModelResult r = process_model(kv.second, rdf_stride, rmax, dr);
        write_model_outputs(r, out_dir, rmax, dr);
        summary << r.spec.id << "\t" << r.spec.family << "\t" << r.raw_frames << "\t" << r.eligible_frames << "\t" << r.invalid_frames << "\t"
                << r.first_step << "\t" << r.last_step << "\t" << r.first_time_ps << "\t" << r.last_time_ps << "\t"
                << r.finite_temperature << "\t" << r.finite_pressure << "\t" << r.min_temperature << "\t" << r.max_temperature << "\t"
                << r.min_pressure << "\t" << r.max_pressure << "\t" << r.rdf_frames << "\t" << r.cr_indices.size() << "\n";
    }
    return 0;
}
