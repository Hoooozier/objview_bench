#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <json/json.h>

namespace fs = std::filesystem;

struct Config {
    std::string exe_path = "./BenbvSingleWorker";
    std::string public_train_json;
    std::string pcd_root;
    std::string view_cache_root;
    std::string views_path;
    std::string output_root;

    int start_idx = 0;
    int end_idx = -1;
    int start_count = 4;
    int skip_existing = 1;

    int random_seed = 42;
    double resolution = 0.02;
    double camera_distance = 2.0;
    int width = 512;
    int height = 512;
    double fov_deg = 45.0;
    double max_range = 6.0;
    int ignore_unknown = 1;
    int max_steps = 128;
    double coverage_threshold = 0.99;
    int min_gain = 10;
    int candidate_count = 20;
    int point_sample_count = 4096;
    int knn = 30;
    int debug = 0;
};

struct Job {
    int object_idx = -1;
    int object_total = 0;
    std::string uid;
    int start_id = 0;
    fs::path pcd_path;
    fs::path view_cache_dir;
    fs::path output_dir;
    fs::path log_path;
};

static std::string quote(const std::string& s) {
    std::string out = "\"";
    for (char c : s) {
        if (c == '"') out += "\\\"";
        else out += c;
    }
    out += "\"";
    return out;
}

static void printUsage(const char* argv0) {
    std::cout
        << "Usage: " << argv0 << " [options]\n"
        << "Required:\n"
        << "  --public_train_json PATH\n"
        << "  --pcd_root DIR\n"
        << "  --view_cache_root DIR\n"
        << "  --views PATH\n"
        << "  --output_root DIR\n"
        << "Optional:\n"
        << "  --exe ./BenbvSingleWorker\n"
        << "  --start_idx 0 --end_idx -1\n"
        << "  --start-count 4\n"
        << "  --skip_existing 1\n"
        << "  --random-seed 42\n"
        << "  --resolution 0.02 --camera-distance 2.0\n"
        << "  --width 512 --height 512 --fov 45\n"
        << "  --max-range 6 --ignore-unknown 1\n"
        << "  --max-steps 128 --coverage-threshold 0.99 --min-gain 10\n"
        << "  --candidate-count 20 --point-sample-count 4096 --knn 30\n"
        << "  --debug 0\n";
}

static Config parseArgs(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto needValue = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("Missing value for " + name);
            return argv[++i];
        };

        if (arg == "--exe") cfg.exe_path = needValue(arg);
        else if (arg == "--public_train_json") cfg.public_train_json = needValue(arg);
        else if (arg == "--pcd_root") cfg.pcd_root = needValue(arg);
        else if (arg == "--view_cache_root") cfg.view_cache_root = needValue(arg);
        else if (arg == "--views") cfg.views_path = needValue(arg);
        else if (arg == "--output_root") cfg.output_root = needValue(arg);
        else if (arg == "--start_idx") cfg.start_idx = std::stoi(needValue(arg));
        else if (arg == "--end_idx") cfg.end_idx = std::stoi(needValue(arg));
        else if (arg == "--start-count") cfg.start_count = std::stoi(needValue(arg));
        else if (arg == "--skip_existing") cfg.skip_existing = std::stoi(needValue(arg));
        else if (arg == "--random-seed") cfg.random_seed = std::stoi(needValue(arg));
        else if (arg == "--resolution") cfg.resolution = std::stod(needValue(arg));
        else if (arg == "--camera-distance") cfg.camera_distance = std::stod(needValue(arg));
        else if (arg == "--width") cfg.width = std::stoi(needValue(arg));
        else if (arg == "--height") cfg.height = std::stoi(needValue(arg));
        else if (arg == "--fov") cfg.fov_deg = std::stod(needValue(arg));
        else if (arg == "--max-range") cfg.max_range = std::stod(needValue(arg));
        else if (arg == "--ignore-unknown") cfg.ignore_unknown = std::stoi(needValue(arg));
        else if (arg == "--max-steps") cfg.max_steps = std::stoi(needValue(arg));
        else if (arg == "--coverage-threshold") cfg.coverage_threshold = std::stod(needValue(arg));
        else if (arg == "--min-gain") cfg.min_gain = std::stoi(needValue(arg));
        else if (arg == "--candidate-count") cfg.candidate_count = std::stoi(needValue(arg));
        else if (arg == "--point-sample-count") cfg.point_sample_count = std::stoi(needValue(arg));
        else if (arg == "--knn") cfg.knn = std::stoi(needValue(arg));
        else if (arg == "--debug") cfg.debug = std::stoi(needValue(arg));
        else if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (cfg.public_train_json.empty() || cfg.pcd_root.empty() || cfg.view_cache_root.empty() ||
        cfg.views_path.empty() || cfg.output_root.empty()) {
        throw std::runtime_error("--public_train_json, --pcd_root, --view_cache_root, --views, and --output_root are required.");
    }
    if (cfg.start_idx < 0 || (cfg.end_idx != -1 && cfg.end_idx < 0)) {
        throw std::runtime_error("Invalid start/end range.");
    }
    if (cfg.start_count <= 0) throw std::runtime_error("--start-count must be > 0.");
    if (cfg.skip_existing != 0 && cfg.skip_existing != 1) throw std::runtime_error("--skip_existing must be 0 or 1.");
    return cfg;
}

static std::vector<std::string> loadUidsFromJson(const std::string& json_path) {
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

static std::string buildCommand(const Config& cfg, const Job& job) {
    std::ostringstream cmd;
    cmd << quote(cfg.exe_path)
        << " --uid " << quote(job.uid)
        << " --pcd " << quote(job.pcd_path.string())
        << " --view_cache_dir " << quote(job.view_cache_dir.string())
        << " --views " << quote(cfg.views_path)
        << " --output_dir " << quote(job.output_dir.string())
        << " --start_id " << job.start_id
        << " --start_view_id -1"
        << " --random_seed " << cfg.random_seed
        << " --resolution " << cfg.resolution
        << " --camera-distance " << cfg.camera_distance
        << " --width " << cfg.width
        << " --height " << cfg.height
        << " --fov " << cfg.fov_deg
        << " --max-range " << cfg.max_range
        << " --ignore-unknown " << cfg.ignore_unknown
        << " --max-steps " << cfg.max_steps
        << " --coverage-threshold " << cfg.coverage_threshold
        << " --min-gain " << cfg.min_gain
        << " --candidate-count " << cfg.candidate_count
        << " --point-sample-count " << cfg.point_sample_count
        << " --knn " << cfg.knn
        << " --debug " << cfg.debug
        << " > " << quote(job.log_path.string()) << " 2>&1";
    return cmd.str();
}

static int runJob(const Config& cfg, const Job& job) {
    fs::create_directories(job.output_dir);
    fs::create_directories(job.log_path.parent_path());
    const std::string cmd = buildCommand(cfg, job);
    return std::system(cmd.c_str());
}

int main(int argc, char** argv) {
    try {
        const Config cfg = parseArgs(argc, argv);

        if (!fs::exists(cfg.exe_path)) throw std::runtime_error("Worker executable does not exist: " + cfg.exe_path);
        if (!fs::exists(cfg.public_train_json)) throw std::runtime_error("public_train_json does not exist: " + cfg.public_train_json);
        if (!fs::exists(cfg.pcd_root) || !fs::is_directory(cfg.pcd_root)) throw std::runtime_error("Invalid pcd_root: " + cfg.pcd_root);
        if (!fs::exists(cfg.view_cache_root) || !fs::is_directory(cfg.view_cache_root)) throw std::runtime_error("Invalid view_cache_root: " + cfg.view_cache_root);
        if (!fs::exists(cfg.views_path)) throw std::runtime_error("Views file does not exist: " + cfg.views_path);
        fs::create_directories(cfg.output_root);

        const std::vector<std::string> all_uids = loadUidsFromJson(cfg.public_train_json);
        const int total = static_cast<int>(all_uids.size());
        const int begin = std::min(cfg.start_idx, total);
        const int end = (cfg.end_idx == -1) ? total : std::min(cfg.end_idx, total);
        if (end < begin) throw std::runtime_error("--end_idx must be >= --start_idx");

        std::vector<Job> jobs;
        int missing = 0;
        int skipped = 0;
        const fs::path log_root = fs::path(cfg.output_root) / "_logs";

        for (int idx = begin; idx < end; ++idx) {
            const std::string& uid = all_uids[idx];
            const fs::path pcd_path = fs::path(cfg.pcd_root) / (uid + ".pcd");
            const fs::path view_cache_dir = fs::path(cfg.view_cache_root) / uid;
            const fs::path output_dir = fs::path(cfg.output_root) / uid;

            if (!fs::exists(pcd_path)) {
                ++missing;
                std::cerr << "[missing] pcd " << uid << ": " << pcd_path.string() << "\n";
                continue;
            }
            if (!fs::exists(view_cache_dir) || !fs::is_directory(view_cache_dir)) {
                ++missing;
                std::cerr << "[missing] view cache " << uid << ": " << view_cache_dir.string() << "\n";
                continue;
            }

            for (int start_id = 0; start_id < cfg.start_count; ++start_id) {
                std::ostringstream stem;
                stem << "start" << std::setw(3) << std::setfill('0') << start_id;
                const fs::path sentinel = output_dir / (stem.str() + ".npz");
                if (cfg.skip_existing == 1 && fs::exists(sentinel)) {
                    ++skipped;
                    continue;
                }

                Job job;
                job.object_idx = idx;
                job.object_total = end - begin;
                job.uid = uid;
                job.start_id = start_id;
                job.pcd_path = pcd_path;
                job.view_cache_dir = view_cache_dir;
                job.output_dir = output_dir;
                job.log_path = log_root / (uid + "_" + stem.str() + ".log");
                jobs.push_back(job);
            }
        }

        std::cout << "Public uid count: " << total << "\n";
        std::cout << "Object range: [" << begin << ", " << end << ") -> " << (end - begin) << " uid(s)\n";
        std::cout << "Start count per uid: " << cfg.start_count << "\n";
        std::cout << "Jobs queued: " << jobs.size() << "\n";
        std::cout << "Skipped existing: " << skipped << "\n";
        std::cout << "Missing inputs: " << missing << "\n";

        int success = 0;
        int failed = 0;

        for (int i = 0; i < static_cast<int>(jobs.size()); ++i) {
            const Job& job = jobs[i];
            std::cout << "[" << (i + 1) << "/" << jobs.size() << "] "
                      << "global=" << job.object_idx << " uid=" << job.uid
                      << " start=" << job.start_id
                      << " log=" << job.log_path.string() << "\n";

            const int ret = runJob(cfg, job);
            if (ret == 0) {
                ++success;
                std::cout << "  -> ok\n";
            } else {
                ++failed;
                std::cout << "  -> failed code=" << ret
                          << " log=" << job.log_path.string() << "\n";
            }
        }

        std::cout << "\nDone.\n";
        std::cout << "Success: " << success << "\n";
        std::cout << "Skipped: " << skipped << "\n";
        std::cout << "Missing: " << missing << "\n";
        std::cout << "Failed: " << failed << "\n";
        return failed == 0 && missing == 0 ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}

