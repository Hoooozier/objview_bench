#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <stdexcept>

#include <json/json.h>

namespace fs = std::filesystem;

struct Config {
    std::string exe_path = "./PcdViewCache";
    std::string clean_pool_json;
    std::string pcd_dir;
    std::string views_path;
    std::string output_root;

    double resolution = 0.02;
    double view_radius = 3.0;
    double look_at_x = 0.0;
    double look_at_y = 0.0;
    double look_at_z = 0.0;
    int width = 512;
    int height = 512;
    double fov_deg = 45.0;
    double max_range = 6.0;
    int ignore_unknown = 1;
    std::string visibility_mode = "render_cuda";
    int save_meta = 1;

    int start_idx = 0;      // inclusive
    int end_idx = -1;       // exclusive; -1 means all
    int skip_existing = 1;  // 0/1
};

struct PoolItem {
    std::string uid;
};

void printUsage(const char* argv0) {
    std::cout
        << "Usage: " << argv0
        << " --clean_pool_json pool.json --pcd_dir DIR --views PATH --output_root DIR [options]\n"
        << "Options:\n"
        << "  --exe PATH                  Path to worker executable (default: ./PcdViewCache)\n"
        << "  --clean_pool_json PATH      Path to final_clean_pool.json / final_clean_main_pool.json\n"
        << "  --pcd_dir DIR               Directory containing input .pcd files\n"
        << "  --views PATH                Path to 128_xyz.txt / other view file\n"
        << "  --output_root DIR           Root directory to save per-object outputs\n"
        << "  --resolution 0.02\n"
        << "  --view-radius 3.0\n"
        << "  --look-at x y z             Default: 0 0 0\n"
        << "  --width 512\n"
        << "  --height 512\n"
        << "  --fov 45\n"
        << "  --max-range 6\n"
        << "  --ignore-unknown 1\n"
        << "  --visibility-mode render_cuda\n"
        << "  --save-meta 1\n"
        << "  --start_idx 0               Start index after sorting uids (inclusive)\n"
        << "  --end_idx -1                End index after sorting uids (exclusive, -1 means all)\n"
        << "  --skip_existing 1           Skip if output sentinel already exists (0/1)\n";
}

Config parseArgs(int argc, char** argv) {
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
        else if (arg == "--clean_pool_json") cfg.clean_pool_json = needValue(arg);
        else if (arg == "--pcd_dir") cfg.pcd_dir = needValue(arg);
        else if (arg == "--views") cfg.views_path = needValue(arg);
        else if (arg == "--output_root") cfg.output_root = needValue(arg);
        else if (arg == "--resolution") cfg.resolution = std::stod(needValue(arg));
        else if (arg == "--view-radius") cfg.view_radius = std::stod(needValue(arg));
        else if (arg == "--width") cfg.width = std::stoi(needValue(arg));
        else if (arg == "--height") cfg.height = std::stoi(needValue(arg));
        else if (arg == "--fov") cfg.fov_deg = std::stod(needValue(arg));
        else if (arg == "--max-range") cfg.max_range = std::stod(needValue(arg));
        else if (arg == "--ignore-unknown") cfg.ignore_unknown = std::stoi(needValue(arg));
        else if (arg == "--visibility-mode") cfg.visibility_mode = needValue(arg);
        else if (arg == "--save-meta") cfg.save_meta = std::stoi(needValue(arg));
        else if (arg == "--start_idx") cfg.start_idx = std::stoi(needValue(arg));
        else if (arg == "--end_idx") cfg.end_idx = std::stoi(needValue(arg));
        else if (arg == "--skip_existing") cfg.skip_existing = std::stoi(needValue(arg));
        else if (arg == "--look-at") {
            if (i + 3 >= argc) {
                throw std::runtime_error("Missing 3 values for --look-at");
            }
            cfg.look_at_x = std::stod(argv[++i]);
            cfg.look_at_y = std::stod(argv[++i]);
            cfg.look_at_z = std::stod(argv[++i]);
        }
        else if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            std::exit(0);
        }
        else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (cfg.clean_pool_json.empty() || cfg.pcd_dir.empty() ||
        cfg.views_path.empty() || cfg.output_root.empty()) {
        throw std::runtime_error("--clean_pool_json, --pcd_dir, --views, and --output_root are required.");
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

std::string quote(const std::string& s) {
    return "\"" + s + "\"";
}

std::vector<PoolItem> loadPoolItems(const std::string& json_path) {
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

    std::vector<PoolItem> items;
    items.reserve(root.size());

    for (const auto& x : root) {
        if (!x.isObject()) continue;
        if (!x.isMember("uid")) continue;
        if (!x["uid"].isString()) continue;

        PoolItem item;
        item.uid = x["uid"].asString();

        if (!item.uid.empty()) {
            items.push_back(std::move(item));
        }
    }

    if (items.empty()) {
        throw std::runtime_error("No valid uid found in json: " + json_path);
    }

    return items;
}

int main(int argc, char** argv) {
    try {
        Config cfg = parseArgs(argc, argv);

        if (!fs::exists(cfg.clean_pool_json)) {
            throw std::runtime_error("clean_pool_json does not exist: " + cfg.clean_pool_json);
        }
        if (!fs::exists(cfg.pcd_dir) || !fs::is_directory(cfg.pcd_dir)) {
            throw std::runtime_error("Invalid pcd_dir: " + cfg.pcd_dir);
        }
        if (!fs::exists(cfg.views_path)) {
            throw std::runtime_error("Views file does not exist: " + cfg.views_path);
        }
        if (!fs::exists(cfg.exe_path)) {
            throw std::runtime_error("Worker executable does not exist: " + cfg.exe_path);
        }

        fs::create_directories(cfg.output_root);

        auto items = loadPoolItems(cfg.clean_pool_json);
        std::sort(items.begin(), items.end(),
                  [](const PoolItem& a, const PoolItem& b) {
                      return a.uid < b.uid;
                  });

        const int total = static_cast<int>(items.size());
        int begin = std::min(cfg.start_idx, total);
        int end = (cfg.end_idx == -1) ? total : std::min(cfg.end_idx, total);

        if (end < begin) {
            throw std::runtime_error("--end_idx must be >= --start_idx");
        }

        if (begin == end) {
            std::cout << "Empty range after clipping: [" << begin << ", " << end << ")\n";
            std::cout << "Total items loaded: " << total << std::endl;
            return 0;
        }

        std::cout << "Loaded " << total << " items from pool json.\n";
        std::cout << "Processing sorted range [" << begin << ", " << end << ")"
                  << " -> " << (end - begin) << " items.\n";
        std::cout << "skip_existing = " << cfg.skip_existing << "\n";

        int success = 0;
        int failed = 0;
        int skipped = 0;

        for (int idx = begin; idx < end; ++idx) {
            const std::string& uid = items[idx].uid;

            const fs::path pcd_path = fs::path(cfg.pcd_dir) / (uid + ".pcd");
            const fs::path out_dir = fs::path(cfg.output_root) / uid;

            // Sentinel rule: if the last-view cache file exists, treat this UID as done.
            const fs::path sentinel = out_dir / "view_127.pcd";

            std::cout << "[" << (idx - begin + 1) << "/" << (end - begin) << "] "
                      << "(global " << idx << ") "
                      << uid;

            if (!fs::exists(pcd_path)) {
                ++failed;
                std::cout << "  -> missing pcd: " << pcd_path.string() << "\n";
                continue;
            }

            if (cfg.skip_existing == 1 && fs::exists(sentinel)) {
                ++skipped;
                std::cout << "  -> skipped (exists)\n";
                continue;
            }

            std::cout << std::endl;

            fs::create_directories(out_dir);

            std::ostringstream cmd;
            cmd << quote(cfg.exe_path)
                << " --pcd " << quote(pcd_path.string())
                << " --views " << quote(cfg.views_path)
                << " --output_dir " << quote(out_dir.string())
                << " --resolution " << cfg.resolution
                << " --view-radius " << cfg.view_radius
                << " --look-at " << cfg.look_at_x << " " << cfg.look_at_y << " " << cfg.look_at_z
                << " --width " << cfg.width
                << " --height " << cfg.height
                << " --fov " << cfg.fov_deg
                << " --max-range " << cfg.max_range
                << " --ignore-unknown " << cfg.ignore_unknown
                << " --visibility-mode " << cfg.visibility_mode
                << " --save-meta " << cfg.save_meta;

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

