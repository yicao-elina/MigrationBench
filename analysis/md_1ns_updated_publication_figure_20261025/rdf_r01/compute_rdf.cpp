#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

struct RunSpec {
    std::string family;
    std::string run;
    long long min_step = 0;
    std::vector<std::string> paths;
};

struct RunResult {
    RunSpec spec;
    std::vector<long double> hist;
    long long frames = 0;
    long long eligible_frames = 0;
    long long sampled_frames = 0;
    long long invalid_frames = 0;
    long long first_step = -1;
    long long last_step = -1;
    long double ncr_sum = 0.0L;
    long double volume_sum = 0.0L;
};

static bool parse_key(const std::string& header, const std::string& key, double& value) {
    const std::string token = " " + key;
    const std::size_t p = header.find(token);
    if (p == std::string::npos) return false;
    std::size_t start = p + token.size();
    if (start < header.size() && header[start] == '=') ++start;
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

static void add_rdf_frame(const std::vector<std::array<double, 3>>& cr,
                          const double c[3][3], const double inv[3][3],
                          double rmax, double dr, std::vector<long double>& hist) {
    const std::size_t n = cr.size();
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            double dcart[3] = {cr[j][0] - cr[i][0], cr[j][1] - cr[i][1], cr[j][2] - cr[i][2]};
            double df[3] = {0.0, 0.0, 0.0};
            for (int k = 0; k < 3; ++k) for (int l = 0; l < 3; ++l) df[k] += dcart[l] * inv[l][k];
            for (int k = 0; k < 3; ++k) df[k] -= std::nearbyint(df[k]);
            double wrapped[3] = {0.0, 0.0, 0.0};
            for (int k = 0; k < 3; ++k) for (int l = 0; l < 3; ++l) wrapped[k] += df[l] * c[l][k];
            const double r = std::sqrt(wrapped[0] * wrapped[0] + wrapped[1] * wrapped[1] + wrapped[2] * wrapped[2]);
            const int bin = static_cast<int>(std::floor(r / dr));
            if (bin >= 0 && bin < static_cast<int>(hist.size())) hist[bin] += 2.0L;
        }
    }
}

static RunResult process_run(const RunSpec& spec, int frame_stride, double rmax, double dr) {
    RunResult result;
    result.spec = spec;
    const int nbins = static_cast<int>(std::ceil(rmax / dr));
    result.hist.assign(nbins, 0.0L);
    long long sequence = 0;
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
            std::stringstream nss(line);
            int natoms = 0;
            if (!(nss >> natoms) || natoms <= 0 || natoms > 100000) continue;
            std::string header;
            if (!std::getline(in, header)) break;

            double cell[3][3];
            double step_value = 0.0;
            const bool cell_ok = parse_lattice(header, cell);
            const bool step_ok = parse_key(header, "step", step_value);
            const long long step = step_ok ? static_cast<long long>(std::llround(step_value)) : sequence;

            std::vector<std::array<double, 3>> cr;
            cr.reserve(64);
            bool finite = cell_ok;
            for (int a = 0; a < natoms; ++a) {
                if (!std::getline(in, line)) { finite = false; break; }
                std::stringstream ass(line);
                std::string species;
                std::array<double, 3> xyz{};
                if (!(ass >> species >> xyz[0] >> xyz[1] >> xyz[2])) { finite = false; continue; }
                if (!std::isfinite(xyz[0]) || !std::isfinite(xyz[1]) || !std::isfinite(xyz[2])) finite = false;
                if (species == "Cr") cr.push_back(xyz);
            }
            ++result.frames;
            ++sequence;
            if (step_ok && step <= last_step_seen) continue;
            if (step_ok) last_step_seen = step;
            if (!finite || cr.empty()) { ++result.invalid_frames; continue; }
            if (step < spec.min_step) continue;
            double inv[3][3];
            if (!inverse(cell, inv)) { ++result.invalid_frames; continue; }
            const double volume = std::abs(determinant(cell));
            if (!std::isfinite(volume) || volume <= 0.0) { ++result.invalid_frames; continue; }
            ++result.eligible_frames;
            if (result.first_step < 0) result.first_step = step;
            result.last_step = step;
            if ((result.eligible_frames - 1) % frame_stride != 0) continue;
            ++result.sampled_frames;
            result.ncr_sum += static_cast<long double>(cr.size());
            result.volume_sum += static_cast<long double>(volume);
            add_rdf_frame(cr, cell, inv, rmax, dr, result.hist);
        }
    }
    return result;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: rdf_compute manifest.tsv output_prefix [frame_stride] [rmax] [dr]\n";
        return 2;
    }
    const std::string manifest_path = argv[1];
    const std::string out_prefix = argv[2];
    const int frame_stride = argc > 3 ? std::stoi(argv[3]) : 20;
    const double rmax = argc > 4 ? std::stod(argv[4]) : 12.0;
    const double dr = argc > 5 ? std::stod(argv[5]) : 0.1;

    std::ifstream manifest(manifest_path);
    if (!manifest) return 3;
    std::map<std::pair<std::string, std::string>, RunSpec> specs;
    std::string line;
    while (std::getline(manifest, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        std::string family, run, min_step_s, path;
        if (!std::getline(ss, family, '\t') || !std::getline(ss, run, '\t') ||
            !std::getline(ss, min_step_s, '\t') || !std::getline(ss, path)) continue;
        if (family.empty() || run.empty() || path.empty()) continue;
        const auto key = std::make_pair(family, run);
        if (!specs.count(key)) specs[key] = RunSpec{family, run, std::stoll(min_step_s), {}};
        specs[key].paths.push_back(path);
    }

    std::vector<RunResult> results;
    for (const auto& kv : specs) {
        std::cerr << "RDF " << kv.second.family << " " << kv.second.run << "\n";
        results.push_back(process_run(kv.second, frame_stride, rmax, dr));
    }

    const std::string runs_path = out_prefix + "_runs.csv";
    std::ofstream runs(runs_path);
    runs << "family,run,r_A,g_CrCr,sampled_frames,eligible_frames,raw_frames,invalid_frames,first_step,last_step,mean_nCr,mean_volume_A3\n";
    runs << std::setprecision(10);
    for (const auto& rr : results) {
        const long double mean_n = rr.sampled_frames ? rr.ncr_sum / rr.sampled_frames : 0.0L;
        const long double mean_v = rr.sampled_frames ? rr.volume_sum / rr.sampled_frames : 0.0L;
        const long double rho = mean_v > 0.0L ? mean_n / mean_v : 0.0L;
        for (int b = 0; b < static_cast<int>(rr.hist.size()); ++b) {
            const double r0 = b * dr;
            const double r1 = (b + 1) * dr;
            const long double shell = 4.0L * acosl(-1.0L) / 3.0L * (r1 * r1 * r1 - r0 * r0 * r0);
            const long double denom = rr.sampled_frames * mean_n * rho * shell;
            const long double g = denom > 0.0L ? rr.hist[b] / denom : std::numeric_limits<long double>::quiet_NaN();
            runs << rr.spec.family << "," << rr.spec.run << "," << (r0 + r1) / 2.0 << "," << static_cast<double>(g)
                 << "," << rr.sampled_frames << "," << rr.eligible_frames << "," << rr.frames << "," << rr.invalid_frames
                 << "," << rr.first_step << "," << rr.last_step << "," << static_cast<double>(mean_n) << "," << static_cast<double>(mean_v) << "\n";
        }
    }
    runs.close();

    std::map<std::string, std::vector<const RunResult*>> groups;
    for (const auto& rr : results) groups[rr.spec.family].push_back(&rr);
    const std::string groups_path = out_prefix + "_groups.csv";
    std::ofstream groups_out(groups_path);
    groups_out << "family,r_A,g_mean,g_sd,n_runs,n_sampled_frames\n";
    groups_out << std::setprecision(10);
    for (const auto& kv : groups) {
        const auto& group = kv.second;
        for (int b = 0; b < static_cast<int>(std::ceil(rmax / dr)); ++b) {
            std::vector<long double> vals;
            for (const auto* rr : group) {
                const long double mean_n = rr->sampled_frames ? rr->ncr_sum / rr->sampled_frames : 0.0L;
                const long double mean_v = rr->sampled_frames ? rr->volume_sum / rr->sampled_frames : 0.0L;
                const long double rho = mean_v > 0.0L ? mean_n / mean_v : 0.0L;
                const double r0 = b * dr, r1 = (b + 1) * dr;
                const long double shell = 4.0L * acosl(-1.0L) / 3.0L * (r1 * r1 * r1 - r0 * r0 * r0);
                const long double denom = rr->sampled_frames * mean_n * rho * shell;
                if (denom > 0.0L) vals.push_back(rr->hist[b] / denom);
            }
            long double mean = 0.0L, sd = 0.0L;
            for (const auto v : vals) mean += v;
            if (!vals.empty()) mean /= vals.size();
            if (vals.size() > 1) {
                for (const auto v : vals) sd += (v - mean) * (v - mean);
                sd = std::sqrt(sd / (vals.size() - 1));
            }
            groups_out << kv.first << "," << ((b + 0.5) * dr) << "," << static_cast<double>(mean) << "," << static_cast<double>(sd)
                       << "," << vals.size() << ",";
            long long nframes = 0;
            for (const auto* rr : group) nframes += rr->sampled_frames;
            groups_out << nframes << "\n";
        }
    }
    groups_out.close();

    std::ofstream meta(out_prefix + "_meta.tsv");
    meta << "family\trun\tmin_step\tpaths\traw_frames\teligible_frames\tsampled_frames\tinvalid_frames\tfirst_step\tlast_step\n";
    for (const auto& rr : results) {
        meta << rr.spec.family << "\t" << rr.spec.run << "\t" << rr.spec.min_step << "\t" << rr.spec.paths.size() << "\t"
             << rr.frames << "\t" << rr.eligible_frames << "\t" << rr.sampled_frames << "\t" << rr.invalid_frames << "\t"
             << rr.first_step << "\t" << rr.last_step << "\n";
    }
    return 0;
}
