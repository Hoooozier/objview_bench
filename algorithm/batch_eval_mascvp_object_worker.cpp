#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <json/json.h>

namespace fs = std::filesystem;

namespace {

struct Config {
    std::string exe_path = "./EvalMascvpObjectWorker";
    std::string analysis_json;
    std::string rollout_root;
    std::string view_cache_root;
    std::string raw_results_dir;
    std::string balanced_results_dir;
    std::string output_root;

    int expected_num_views = 128;
    double resolution = 0.02;
    double gamma = 0.5;
    int step_id = 0;
    int max_step_id = -1;
    int start_idx = 0;
    int end_idx = -1;
    int skip_existing = 1;
};

struct Job {
    int object_idx = -1;
    int object_total = 0;
    std::string uid;
    fs::path rollout_json;
    fs::path view_cache_dir;
    fs::path output_json;
    fs::path log_path;
};

std::string quote(const std::string& s) {
    std::string out = "\"";
    for (char c : s) {
        if (c == '"') out += "\\\"";
        else out += c;
    }
    out += "\"";
    return out;
}

void printUsage(const char* argv0) {
    std::cout
        << "Usage: " << argv0 << " [options]\n"
        << "Required:\n"
        << "  --analysis_json PATH\n"
        << "  --rollout_root DIR\n"
        << "  --view_cache_root DIR\n"
        << "  --raw_results_dir DIR\n"
        << "  --balanced_results_dir DIR\n"
        << "  --output_root DIR\n"
        << "Optional:\n"
        << "  --exe ./EvalMascvpObjectWorker\n"
        << "  --expected_num_views 128\n"
        << "  --resolution 0.02\n"
        << "  --gamma 0.5\n"
        << "  --step_id 0\n"
        << "  --max_step_id -1\n"
        << "  --start_idx 0\n"
        << "  --end_idx -1\n"
        << "  --skip_existing 1\n";
}

Config parseArgs(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto needValue = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("Missing value for " + name);
            return argv[++i];
        };

        if (arg == "--exe") cfg.exe_path = needValue(arg);
        else if (arg == "--analysis_json") cfg.analysis_json = needValue(arg);
        else if (arg == "--rollout_root") cfg.rollout_root = needValue(arg);
        else if (arg == "--view_cache_root") cfg.view_cache_root = needValue(arg);
        else if (arg == "--raw_results_dir") cfg.raw_results_dir = needValue(arg);
        else if (arg == "--balanced_results_dir") cfg.balanced_results_dir = needValue(arg);
        else if (arg == "--output_root") cfg.output_root = needValue(arg);
        else if (arg == "--expected_num_views") cfg.expected_num_views = std::stoi(needValue(arg));
        else if (arg == "--resolution") cfg.resolution = std::stod(needValue(arg));
        else if (arg == "--gamma") cfg.gamma = std::stod(needValue(arg));
        else if (arg == "--step_id") cfg.step_id = std::stoi(needValue(arg));
        else if (arg == "--max_step_id") cfg.max_step_id = std::stoi(needValue(arg));
        else if (arg == "--start_idx") cfg.start_idx = std::stoi(needValue(arg));
        else if (arg == "--end_idx") cfg.end_idx = std::stoi(needValue(arg));
        else if (arg == "--skip_existing") cfg.skip_existing = std::stoi(needValue(arg));
        else if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (cfg.analysis_json.empty() || cfg.rollout_root.empty() || cfg.view_cache_root.empty() ||
        cfg.raw_results_dir.empty() || cfg.balanced_results_dir.empty() || cfg.output_root.empty()) {
        throw std::runtime_error(
            "--analysis_json, --rollout_root, --view_cache_root, --raw_results_dir, "
            "--balanced_results_dir, and --output_root are required.");
    }
    if (cfg.start_idx < 0 || (cfg.end_idx != -1 && cfg.end_idx < 0)) {
        throw std::runtime_error("Invalid start/end range.");
    }
    if (cfg.skip_existing != 0 && cfg.skip_existing != 1) {
        throw std::runtime_error("--skip_existing must be 0 or 1.");
    }
    if (cfg.max_step_id != -1 && cfg.max_step_id < cfg.step_id) {
        throw std::runtime_error("--max_step_id must be -1 or >= --step_id.");
    }
    return cfg;
}

std::vector<std::string> loadUidsFromJson(const std::string& json_path) {
    std::ifstream fin(json_path, std::ios::binary);
    if (!fin) throw std::runtime_error("Failed to open json: " + json_path);

    Json::CharReaderBuilder builder;
    builder["collectComments"] = false;
    Json::Value root;
    std::string errs;
    if (!Json::parseFromStream(builder, fin, &root, &errs)) {
        throw std::runtime_error("Failed to parse json: " + errs);
    }
    if (!root.isArray()) throw std::runtime_error("Expected top-level JSON array: " + json_path);

    std::vector<std::string> uids;
    std::set<std::string> seen;
    for (const auto& x : root) {
        if (!x.isObject() || !x.isMember("uid") || !x["uid"].isString()) continue;
        const std::string uid = x["uid"].asString();
        if (!uid.empty() && seen.insert(uid).second) uids.push_back(uid);
    }
    return uids;
}

std::string buildCommand(const Config& cfg, const Job& job) {
    std::ostringstream cmd;
    cmd << quote(cfg.exe_path)
        << " --uid " << quote(job.uid)
        << " --analysis_json " << quote(cfg.analysis_json)
        << " --rollout_json " << quote(job.rollout_json.string())
        << " --view_cache_dir " << quote(job.view_cache_dir.string())
        << " --raw_results_dir " << quote(cfg.raw_results_dir)
        << " --balanced_results_dir " << quote(cfg.balanced_results_dir)
        << " --output_json " << quote(job.output_json.string())
        << " --expected_num_views " << cfg.expected_num_views
        << " --resolution " << cfg.resolution
        << " --gamma " << cfg.gamma
        << " --step_id " << cfg.step_id
        << " --max_step_id " << cfg.max_step_id
        << " --skip_existing " << cfg.skip_existing
        << " > " << quote(job.log_path.string()) << " 2>&1";
    return cmd.str();
}

int runJob(const Config& cfg, const Job& job) {
    fs::create_directories(job.output_json.parent_path());
    fs::create_directories(job.log_path.parent_path());
    const std::string cmd = buildCommand(cfg, job);
    return std::system(cmd.c_str());
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Config cfg = parseArgs(argc, argv);

        if (!fs::exists(cfg.exe_path)) throw std::runtime_error("Worker executable does not exist: " + cfg.exe_path);
        if (!fs::exists(cfg.analysis_json)) throw std::runtime_error("analysis_json does not exist: " + cfg.analysis_json);
        if (!fs::exists(cfg.rollout_root) || !fs::is_directory(cfg.rollout_root)) {
            throw std::runtime_error("Invalid rollout_root: " + cfg.rollout_root);
        }
        if (!fs::exists(cfg.view_cache_root) || !fs::is_directory(cfg.view_cache_root)) {
            throw std::runtime_error("Invalid view_cache_root: " + cfg.view_cache_root);
        }
        if (!fs::exists(cfg.raw_results_dir) || !fs::is_directory(cfg.raw_results_dir)) {
            throw std::runtime_error("Invalid raw_results_dir: " + cfg.raw_results_dir);
        }
        if (!fs::exists(cfg.balanced_results_dir) || !fs::is_directory(cfg.balanced_results_dir)) {
            throw std::runtime_error("Invalid balanced_results_dir: " + cfg.balanced_results_dir);
        }
        fs::create_directories(cfg.output_root);

        const std::vector<std::string> all_uids = loadUidsFromJson(cfg.analysis_json);
        const int total = static_cast<int>(all_uids.size());
        const int begin = std::min(cfg.start_idx, total);
        const int end = (cfg.end_idx == -1) ? total : std::min(cfg.end_idx, total);
        if (end < begin) throw std::runtime_error("--end_idx must be >= --start_idx");

        std::vector<Job> jobs;
        jobs.reserve(end - begin);
        int missing = 0;
        const fs::path log_root = fs::path(cfg.output_root) / "_logs";

        for (int idx = begin; idx < end; ++idx) {
            const std::string& uid = all_uids[idx];
            const fs::path rollout_json = fs::path(cfg.rollout_root) / (uid + ".json");
            const fs::path view_cache_dir = fs::path(cfg.view_cache_root) / uid;
            const fs::path output_json = fs::path(cfg.output_root) / (uid + ".json");

            if (!fs::exists(rollout_json)) {
                ++missing;
                std::cerr << "[missing] rollout " << uid << ": " << rollout_json.string() << "\n";
                continue;
            }
            if (!fs::exists(view_cache_dir) || !fs::is_directory(view_cache_dir)) {
                ++missing;
                std::cerr << "[missing] view cache " << uid << ": " << view_cache_dir.string() << "\n";
                continue;
            }

            Job job;
            job.object_idx = idx;
            job.object_total = total;
            job.uid = uid;
            job.rollout_json = rollout_json;
            job.view_cache_dir = view_cache_dir;
            job.output_json = output_json;
            job.log_path = log_root / (uid + ".log");
            jobs.push_back(std::move(job));
        }

        std::cout << "Total uid in split: " << total << "\n";
        std::cout << "Processing sorted range [" << begin << ", " << end << ")"
                  << " -> " << (end - begin) << " uid(s).\n";
        std::cout << "Runnable jobs: " << jobs.size()
                  << ", missing: " << missing
                  << ", skip_existing: " << cfg.skip_existing << "\n";

        int ok = 0;
        int fail = 0;
        for (const Job& job : jobs) {
            std::cout << "[run] (" << (job.object_idx + 1) << "/" << job.object_total << ") "
                      << job.uid << "\n";
            const int rc = runJob(cfg, job);
            if (rc == 0) {
                ++ok;
            } else {
                ++fail;
                std::cerr << "[fail] uid=" << job.uid
                          << " rc=" << rc
                          << " log=" << job.log_path.string() << "\n";
            }
        }

        std::cout << "Done. ok=" << ok << " fail=" << fail << " missing=" << missing << "\n";
        return fail == 0 ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}

