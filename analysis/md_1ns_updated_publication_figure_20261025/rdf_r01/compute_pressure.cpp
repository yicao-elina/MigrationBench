#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>

struct RunSpec { std::string family, run; long long min_step; std::vector<std::string> paths; };

static bool key_value(const std::string& h, const std::string& key, double& v) {
    const std::string token = " " + key + "=";
    const auto p = h.find(token);
    if (p == std::string::npos) return false;
    auto s = p + token.size();
    auto e = s;
    while (e < h.size() && h[e] != ' ' && h[e] != '\t') ++e;
    try { v = std::stod(h.substr(s, e - s)); return std::isfinite(v); }
    catch (...) { return false; }
}

static bool stress_trace(const std::string& h, double& trace) {
    const std::string key = "stress=\"";
    const auto p = h.find(key);
    if (p == std::string::npos) return false;
    const auto s = p + key.size();
    const auto e = h.find('"', s);
    if (e == std::string::npos) return false;
    std::stringstream ss(h.substr(s, e - s));
    double x[9];
    for (double& q : x) if (!(ss >> q) || !std::isfinite(q)) return false;
    trace = x[0] + x[4] + x[8];
    return std::isfinite(trace);
}

int main(int argc, char** argv) {
    if (argc < 3) return 2;
    const std::string manifest_path = argv[1], out_path = argv[2];
    const int stride = argc > 3 ? std::stoi(argv[3]) : 20;
    std::ifstream mf(manifest_path);
    if (!mf) return 3;
    std::map<std::pair<std::string, std::string>, RunSpec> specs;
    std::string line;
    while (std::getline(mf, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        std::string fam, run, minstep, path;
        if (!std::getline(ss, fam, '\t') || !std::getline(ss, run, '\t') || !std::getline(ss, minstep, '\t') || !std::getline(ss, path)) continue;
        auto k = std::make_pair(fam, run);
        if (!specs.count(k)) specs[k] = RunSpec{fam, run, std::stoll(minstep), {}};
        specs[k].paths.push_back(path);
    }

    std::ofstream out(out_path);
    out << "family,run,time_ps,pressure_GPa\n" << std::setprecision(10);
    for (const auto& kv : specs) {
        const auto& spec = kv.second;
        std::cerr << "PRESSURE " << spec.family << " " << spec.run << "\n";
        long long last_step = -1, eligible = 0, sampled = 0;
        for (const auto& path : spec.paths) {
            std::ifstream in(path);
            if (!in) { std::cerr << "ERROR open " << path << "\n"; continue; }
            std::string h;
            while (std::getline(in, line)) {
                if (line.empty()) continue;
                std::stringstream ns(line); int natoms = 0;
                if (!(ns >> natoms) || natoms <= 0 || natoms > 100000) continue;
                if (!std::getline(in, h)) break;
                double step_v = 0.0, time_fs = 0.0, trace = 0.0;
                const bool step_ok = key_value(h, "step", step_v);
                const long long step = step_ok ? static_cast<long long>(std::llround(step_v)) : 0;
                const bool time_ok = key_value(h, "time_fs", time_fs);
                const bool stress_ok = stress_trace(h, trace);
                for (int a = 0; a < natoms; ++a) if (!std::getline(in, line)) break;
                if (step_ok && step <= last_step) continue;
                if (step_ok) last_step = step;
                if (!stress_ok || !step_ok || step < spec.min_step) continue;
                ++eligible;
                if ((eligible - 1) % stride != 0) continue;
                ++sampled;
                const double t_ps = time_ok ? time_fs / 1000.0 : step / 1000.0;
                const double p_gpa = -(trace / 3.0) * 160.21766208;
                if (std::isfinite(t_ps) && std::isfinite(p_gpa)) out << spec.family << "," << spec.run << "," << t_ps << "," << p_gpa << "\n";
            }
        }
        std::cerr << "  eligible=" << eligible << " sampled=" << sampled << "\n";
    }
    return 0;
}
