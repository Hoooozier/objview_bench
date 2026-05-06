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
    std::string exe_path = "./ExportMascvpOfflineCase";
    std::string analysis_split_json;
    std::string rollout_root;
    std::string view_cache_root;
    std::string views_path;
    std::string output_root;
    std::string constraint_name;
    std::string debug_dir = "mascvp_offline_debug";

    int expected_num_views = 128;
    int grid_size = 64;
    int start_view_id = -1;
    int step_id = 0;
    int max_step_id = -1;
    int start_idx = 0;
    int end_idx = -1;
    int skip_existing = 1;
    int debug_save_ot = 0;

    double bbox_min = -1.0;
    double bbox_max = 1.0;
    double unknown_occ = 0.5;
    double max_range = -1.0;
    double view_radius = 3.0;
};

struct Job {
    int object_idx = -1;
    int object_total = 0;
    std::string uid;
    fs::path output_dir;
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
        << "  --analysis_split_json PATH\n"
        << "  --rollout_root DIR\n"
        << "  --view_cache_root DIR\n"
        << "  --views PATH\n"
        << "  --output_root DIR\n"
        << "Optional:\n"
        << "  --exe ./ExportMascvpOfflineCase\n"
        << "  --constraint_name NAME\n"
        << "  --debug_dir DIR\n"
        << "  --debug_save_ot 0\n"
        << "  --expected_num_views 128\n"
        << "  --grid_size 64\n"
        << "  --step_id 0\n"
        << "  --max_step_id -1\n"
        << "  --start_view_id -1\n"
        << "  --start_idx 0\n"
        << "  --end_idx -1\n"
        << "  --skip_existing 1\n"
        << "  --bbox_min -1.0\n"
        << "  --bbox_max 1.0\n"
        << "  --unknown_occ 0.5\n"
        << "  --max_range -1\n"
        << "  --view_radius 3.0\n";
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
        else if (arg == "--analysis_split_json") cfg.analysis_split_json = needValue(arg);
        else if (arg == "--rollout_root") cfg.rollout_root = needValue(arg);
        else if (arg == "--view_cache_root") cfg.view_cache_root = needValue(arg);
        else if (arg == "--views") cfg.views_path = needValue(arg);
        else if (arg == "--output_root") cfg.output_root = needValue(arg);
        else if (arg == "--constraint_name") cfg.constraint_name = needValue(arg);
        else if (arg == "--debug_dir") cfg.debug_dir = needValue(arg);
        else if (arg == "--debug_save_ot") cfg.debug_save_ot = std::stoi(needValue(arg));
        else if (arg == "--expected_num_views") cfg.expected_num_views = std::stoi(needValue(arg));
        else if (arg == "--grid_size") cfg.grid_size = std::stoi(needValue(arg));
        else if (arg == "--step_id") cfg.step_id = std::stoi(needValue(arg));
        else if (arg == "--max_step_id") cfg.max_step_id = std::stoi(needValue(arg));
        else if (arg == "--start_view_id") cfg.start_view_id = std::stoi(needValue(arg));
        else if (arg == "--start_idx") cfg.start_idx = std::stoi(needValue(arg));
        else if (arg == "--end_idx") cfg.end_idx = std::stoi(needValue(arg));
        else if (arg == "--skip_existing") cfg.skip_existing = std::stoi(needValue(arg));
        else if (arg == "--bbox_min") cfg.bbox_min = std::stod(needValue(arg));
        else if (arg == "--bbox_max") cfg.bbox_max = std::stod(needValue(arg));
        else if (arg == "--unknown_occ") cfg.unknown_occ = std::stod(needValue(arg));
        else if (arg == "--max_range") cfg.max_range = std::stod(needValue(arg));
        else if (arg == "--view_radius") cfg.view_radius = std::stod(needValue(arg));
        else if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (cfg.analysis_split_json.empty() || cfg.rollout_root.empty() || cfg.view_cache_root.empty() ||
        cfg.views_path.empty() || cfg.output_root.empty()) {
        throw std::runtime_error(
            "--analysis_split_json, --rollout_root, --view_cache_root, --views, and --output_root are required.");
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
        << " --analysis_split_json " << quote(cfg.analysis_split_json)
        << " --rollout_root " << quote(cfg.rollout_root)
        << " --uid " << quote(job.uid)
        << " --view_cache_root " << quote(cfg.view_cache_root)
        << " --views " << quote(cfg.views_path)
        << " --output_root " << quote(cfg.output_root)
        << " --debug_dir " << quote(cfg.debug_dir)
        << " --debug_save_ot " << cfg.debug_save_ot
        << " --expected_num_views " << cfg.expected_num_views
        << " --grid_size " << cfg.grid_size
        << " --step_id " << cfg.step_id
        << " --max_step_id " << cfg.max_step_id
        << " --start_view_id " << cfg.start_view_id
        << " --skip_existing " << cfg.skip_existing
        << " --bbox_min " << cfg.bbox_min
        << " --bbox_max " << cfg.bbox_max
        << " --unknown_occ " << cfg.unknown_occ
        << " --max_range " << cfg.max_range
        << " --view_radius " << cfg.view_radius;
    if (!cfg.constraint_name.empty()) {
        cmd << " --constraint_name " << quote(cfg.constraint_name);
    }
    cmd << " > " << quote(job.log_path.string()) << " 2>&1";
    return cmd.str();
}

int runJob(const Config& cfg, const Job& job) {
    fs::create_directories(job.output_dir);
    fs::create_directories(job.log_path.parent_path());
    const std::string cmd = buildCommand(cfg, job);
    return std::system(cmd.c_str());
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Config cfg = parseArgs(argc, argv);

        if (!fs::exists(cfg.exe_path)) throw std::runtime_error("Worker executable does not exist: " + cfg.exe_path);
        if (!fs::exists(cfg.analysis_split_json)) throw std::runtime_error("analysis_split_json does not exist: " + cfg.analysis_split_json);
        if (!fs::exists(cfg.rollout_root) || !fs::is_directory(cfg.rollout_root)) {
            throw std::runtime_error("Invalid rollout_root: " + cfg.rollout_root);
        }
        if (!fs::exists(cfg.view_cache_root) || !fs::is_directory(cfg.view_cache_root)) {
            throw std::runtime_error("Invalid view_cache_root: " + cfg.view_cache_root);
        }
        if (!fs::exists(cfg.views_path)) throw std::runtime_error("Views file does not exist: " + cfg.views_path);
        fs::create_directories(cfg.output_root);

        const std::vector<std::string> all_uids = loadUidsFromJson(cfg.analysis_split_json);
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
            const fs::path view_cache_dir = fs::path(cfg.view_cache_root) / uid;
            if (!fs::exists(view_cache_dir) || !fs::is_directory(view_cache_dir)) {
                ++missing;
                std::cerr << "[missing] view cache " << uid << ": " << view_cache_dir.string() << "\n";
                continue;
            }

            Job job;
            job.object_idx = idx;
            job.object_total = total;
            job.uid = uid;
            job.output_dir = fs::path(cfg.output_root) / uid;
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

