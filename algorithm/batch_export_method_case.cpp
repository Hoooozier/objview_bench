#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <set>
#include <algorithm>
#include <stdexcept>

#include <json/json.h>

namespace fs = std::filesystem;

struct Config {
    std::string exe_path = "./ExportMethodCase";

    std::string raw_train_json;
    std::string balanced_train_json;
    std::string public_train_json;

    std::string rollout_root;
    std::string view_cache_root;
    std::string views_path;
    std::string output_root;

    int expected_num_views = 128;
    double view_radius = 3.0;
    double cache_resolution = 0.02;
    int grid_size = 64;
    double bbox_min = -1.0;
    double bbox_max = 1.0;
    double unknown_occ = 0.5;

    int scvp_num_start_views = 64;
    int longtail_need_case_1 = 64;
    int random_seed = 42;

    int start_idx = 0;      // inclusive
    int end_idx = -1;       // exclusive, -1 means all
    int skip_existing = 1;  // 0/1
};

static void printUsage(const char* argv0) {
    std::cout
        << "Usage: " << argv0 << " [options]\n"
        << "Required:\n"
        << "  --raw_train_json PATH\n"
        << "  --balanced_train_json PATH\n"
        << "  --public_train_json PATH\n"
        << "  --rollout_root DIR\n"
        << "  --view_cache_root DIR\n"
        << "  --views PATH\n"
        << "  --output_root DIR\n"
        << "Optional:\n"
        << "  --exe PATH                  Path to ExportMethodCase executable (default: ./ExportMethodCase)\n"
        << "  --expected_num_views 128\n"
        << "  --view-radius 3.0\n"
        << "  --cache-resolution 0.02\n"
        << "  --grid-size 64\n"
        << "  --bbox-min -1.0\n"
        << "  --bbox-max 1.0\n"
        << "  --unknown-occ 0.5\n"
        << "  --scvp-num-start-views 64\n"
        << "  --longtail-need-case-1 64\n"
        << "  --random-seed 42\n"
        << "  --start_idx 0\n"
        << "  --end_idx -1\n"
        << "  --skip_existing 1\n";
}

static Config parseArgs(int argc, char** argv) {
    Config cfg;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        auto needValue = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for " + name);
            }
            return argv[++i];
        };

        if (arg == "--exe") cfg.exe_path = needValue(arg);
        else if (arg == "--raw_train_json") cfg.raw_train_json = needValue(arg);
        else if (arg == "--balanced_train_json") cfg.balanced_train_json = needValue(arg);
        else if (arg == "--public_train_json") cfg.public_train_json = needValue(arg);
        else if (arg == "--rollout_root") cfg.rollout_root = needValue(arg);
        else if (arg == "--view_cache_root") cfg.view_cache_root = needValue(arg);
        else if (arg == "--views") cfg.views_path = needValue(arg);
        else if (arg == "--output_root") cfg.output_root = needValue(arg);
        else if (arg == "--expected_num_views") cfg.expected_num_views = std::stoi(needValue(arg));
        else if (arg == "--view-radius") cfg.view_radius = std::stod(needValue(arg));
        else if (arg == "--cache-resolution") cfg.cache_resolution = std::stod(needValue(arg));
        else if (arg == "--grid-size") cfg.grid_size = std::stoi(needValue(arg));
        else if (arg == "--bbox-min") cfg.bbox_min = std::stod(needValue(arg));
        else if (arg == "--bbox-max") cfg.bbox_max = std::stod(needValue(arg));
        else if (arg == "--unknown-occ") cfg.unknown_occ = std::stod(needValue(arg));
        else if (arg == "--scvp-num-start-views") cfg.scvp_num_start_views = std::stoi(needValue(arg));
        else if (arg == "--longtail-need-case-1") cfg.longtail_need_case_1 = std::stoi(needValue(arg));
        else if (arg == "--random-seed") cfg.random_seed = std::stoi(needValue(arg));
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

    if (cfg.raw_train_json.empty() || cfg.balanced_train_json.empty() || cfg.public_train_json.empty() ||
        cfg.rollout_root.empty() || cfg.view_cache_root.empty() ||
        cfg.views_path.empty() || cfg.output_root.empty()) {
        throw std::runtime_error(
            "--raw_train_json, --balanced_train_json, --public_train_json, "
            "--rollout_root, --view_cache_root, --views, --output_root are required.");
    }

    if (cfg.start_idx < 0) {
        throw std::runtime_error("--start_idx must be >= 0");
    }
    if (cfg.end_idx != -1 && cfg.end_idx < 0) {
        throw std::runtime_error("--end_idx must be -1 or >= 0");
    }
    if (cfg.skip_existing != 0 && cfg.skip_existing != 1) {
        throw std::runtime_error("--skip_existing must be 0 or 1");
    }

    return cfg;
}

static std::string quote(const std::string& s) {
    return "\"" + s + "\"";
}

static std::set<std::string> loadUidSetFromJson(const std::string& json_path) {
    std::ifstream fin(json_path, std::ios::binary);
    if (!fin) {
        throw std::runtime_error("Failed to open json: " + json_path);
    }

    Json::CharReaderBuilder reader_builder;
    reader_builder["collectComments"] = false;

    Json::Value root;
    std::string errs;
    if (!Json::parseFromStream(reader_builder, fin, &root, &errs)) {
        throw std::runtime_error("Failed to parse json: " + errs);
    }

    if (!root.isArray()) {
        throw std::runtime_error("Expected top-level JSON array in: " + json_path);
    }

    std::set<std::string> uid_set;
    for (const auto& x : root) {
        if (!x.isObject()) continue;
        if (!x.isMember("uid")) continue;
        if (!x["uid"].isString()) continue;

        const std::string uid = x["uid"].asString();
        if (!uid.empty()) uid_set.insert(uid);
    }

    return uid_set;
}

int main(int argc, char** argv) {
    try {
        Config cfg = parseArgs(argc, argv);

        if (!fs::exists(cfg.exe_path)) {
            throw std::runtime_error("Worker executable does not exist: " + cfg.exe_path);
        }
        if (!fs::exists(cfg.raw_train_json)) {
            throw std::runtime_error("raw_train_json does not exist: " + cfg.raw_train_json);
        }
        if (!fs::exists(cfg.balanced_train_json)) {
            throw std::runtime_error("balanced_train_json does not exist: " + cfg.balanced_train_json);
        }
        if (!fs::exists(cfg.public_train_json)) {
            throw std::runtime_error("public_train_json does not exist: " + cfg.public_train_json);
        }
        if (!fs::exists(cfg.rollout_root) || !fs::is_directory(cfg.rollout_root)) {
            throw std::runtime_error("Invalid rollout_root: " + cfg.rollout_root);
        }
        if (!fs::exists(cfg.view_cache_root) || !fs::is_directory(cfg.view_cache_root)) {
            throw std::runtime_error("Invalid view_cache_root: " + cfg.view_cache_root);
        }
        if (!fs::exists(cfg.views_path)) {
            throw std::runtime_error("Views file does not exist: " + cfg.views_path);
        }

        fs::create_directories(cfg.output_root);

        std::set<std::string> uid_union;
        {
            auto s = loadUidSetFromJson(cfg.raw_train_json);
            uid_union.insert(s.begin(), s.end());
        }
        {
            auto s = loadUidSetFromJson(cfg.balanced_train_json);
            uid_union.insert(s.begin(), s.end());
        }
        {
            auto s = loadUidSetFromJson(cfg.public_train_json);
            uid_union.insert(s.begin(), s.end());
        }

        if (uid_union.empty()) {
            std::cout << "No uid found from the three training json files.\n";
            return 0;
        }

        std::vector<std::string> uids(uid_union.begin(), uid_union.end());
        std::sort(uids.begin(), uids.end());

        const int total = static_cast<int>(uids.size());
        int begin = std::min(cfg.start_idx, total);
        int end = (cfg.end_idx == -1) ? total : std::min(cfg.end_idx, total);

        if (end < begin) {
            throw std::runtime_error("--end_idx must be >= --start_idx");
        }

        if (begin == end) {
            std::cout << "Empty range after clipping: [" << begin << ", " << end << ")\n";
            std::cout << "Total uid in union: " << total << std::endl;
            return 0;
        }

        std::cout << "Union uid count = " << total << "\n";
        std::cout << "Processing sorted range [" << begin << ", " << end << ")"
                  << " -> " << (end - begin) << " uid(s).\n";
        std::cout << "skip_existing = " << cfg.skip_existing << "\n";

        int success = 0;
        int failed = 0;
        int skipped = 0;

        for (int idx = begin; idx < end; ++idx) {
            const std::string& uid = uids[idx];

            const fs::path rollout_json = fs::path(cfg.rollout_root) / (uid + ".json");
            const fs::path view_cache_dir = fs::path(cfg.view_cache_root) / uid;

            const fs::path sentinel =
                fs::path(cfg.output_root) / "cases" / uid / "longtail_sampling.json";

            std::cout << "[" << (idx - begin + 1) << "/" << (end - begin) << "] "
                      << "(global " << idx << ") "
                      << uid;

            if (!fs::exists(rollout_json)) {
                ++failed;
                std::cout << "  -> missing rollout json: " << rollout_json.string() << "\n";
                continue;
            }

            if (!fs::exists(view_cache_dir) || !fs::is_directory(view_cache_dir)) {
                ++failed;
                std::cout << "  -> missing view cache dir: " << view_cache_dir.string() << "\n";
                continue;
            }

            if (cfg.skip_existing == 1 && fs::exists(sentinel)) {
                ++skipped;
                std::cout << "  -> skipped (exists)\n";
                continue;
            }

            std::cout << std::endl;

            std::ostringstream cmd;
            cmd << quote(cfg.exe_path)
                << " --rollout_json " << quote(rollout_json.string())
                << " --view_cache_dir " << quote(view_cache_dir.string())
                << " --views " << quote(cfg.views_path)
                << " --output_root " << quote(cfg.output_root)
                << " --expected_num_views " << cfg.expected_num_views
                << " --view-radius " << cfg.view_radius
                << " --cache-resolution " << cfg.cache_resolution
                << " --grid-size " << cfg.grid_size
                << " --bbox-min " << cfg.bbox_min
                << " --bbox-max " << cfg.bbox_max
                << " --unknown-occ " << cfg.unknown_occ
                << " --scvp-num-start-views " << cfg.scvp_num_start_views
                << " --longtail-need-case-1 " << cfg.longtail_need_case_1
                << " --random-seed " << cfg.random_seed
                << " --skip_existing " << cfg.skip_existing;

            const int ret = std::system(cmd.str().c_str());
            if (ret == 0) {
                ++success;
            } else {
                ++failed;
                std::cerr << "  Failed with code: " << ret << std::endl;
            }
        }

        std::cout << "\nDone.\n";
        std::cout << "Success: " << success << "\n";
        std::cout << "Skipped: " << skipped << "\n";
        std::cout << "Failed: " << failed << "\n";

        return failed == 0 ? 0 : 1;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
