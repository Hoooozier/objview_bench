#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <stdexcept>

namespace fs = std::filesystem;

struct Config {
    std::string exe_path = "./PcdSetCover";
    std::string pcd_dir;
    std::string views_path;
    std::string output_dir;

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
    double time_limit_sec = -1.0;
    std::string visibility_mode = "inverse_cuda";
    int save_vis_pcd = 0;
    int min_visible_views = 1;

    // batch control
    int start_idx = 0;      // inclusive
    int end_idx = -1;       // exclusive; -1 means all
    int skip_existing = 0;  // 0/1
};

void printUsage(const char* argv0) {
    std::cout
        << "Usage: " << argv0 << " --pcd_dir DIR --views PATH --output_dir DIR [options]\n"
        << "Options:\n"
        << "  --exe PATH                  Path to set_cover executable (default: ./PcdSetCover)\n"
        << "  --pcd_dir DIR               Directory containing input .pcd files\n"
        << "  --views PATH                Path to 64_xyz.txt / 128_xyz.txt / other view file\n"
        << "  --output_dir DIR            Directory to save output .txt files\n"
        << "  --resolution 0.02\n"
        << "  --view-radius 3.0\n"
        << "  --look-at x y z             Default: 0 0 0\n"
        << "  --width 512\n"
        << "  --height 512\n"
        << "  --fov 45\n"
        << "  --max-range 6.0\n"
        << "  --ignore-unknown 1\n"
        << "  --time-limit -1\n"
        << "  --visibility-mode MODE      render_cpu | render_cuda | inverse_cpu | inverse_cuda | membership_cpu | membership_cuda\n"
        << "  --min-visible-views 1\n"
        << "  --save-vis-pcd 0\n"
        << "  --start_idx 0               Start index after sorting .pcd files (inclusive)\n"
        << "  --end_idx -1                End index after sorting .pcd files (exclusive, -1 means all)\n"
        << "  --skip_existing 0           Skip if output txt already exists (0/1)\n";
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
        else if (arg == "--pcd_dir") cfg.pcd_dir = needValue(arg);
        else if (arg == "--views") cfg.views_path = needValue(arg);
        else if (arg == "--output_dir") cfg.output_dir = needValue(arg);
        else if (arg == "--resolution") cfg.resolution = std::stod(needValue(arg));
        else if (arg == "--view-radius") cfg.view_radius = std::stod(needValue(arg));
        else if (arg == "--width") cfg.width = std::stoi(needValue(arg));
        else if (arg == "--height") cfg.height = std::stoi(needValue(arg));
        else if (arg == "--fov") cfg.fov_deg = std::stod(needValue(arg));
        else if (arg == "--max-range") cfg.max_range = std::stod(needValue(arg));
        else if (arg == "--ignore-unknown") cfg.ignore_unknown = std::stoi(needValue(arg));
        else if (arg == "--time-limit") cfg.time_limit_sec = std::stod(needValue(arg));
        else if (arg == "--visibility-mode") cfg.visibility_mode = needValue(arg);
        else if (arg == "--min-visible-views") cfg.min_visible_views = std::stoi(needValue(arg));
        else if (arg == "--save-vis-pcd") cfg.save_vis_pcd = std::stoi(needValue(arg));
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

    if (cfg.pcd_dir.empty() || cfg.views_path.empty() || cfg.output_dir.empty()) {
        throw std::runtime_error("--pcd_dir, --views, and --output_dir are required.");
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

int main(int argc, char** argv) {
    try {
        Config cfg = parseArgs(argc, argv);

        if (!fs::exists(cfg.pcd_dir) || !fs::is_directory(cfg.pcd_dir)) {
            throw std::runtime_error("Invalid pcd_dir: " + cfg.pcd_dir);
        }
        if (!fs::exists(cfg.views_path)) {
            throw std::runtime_error("Views file does not exist: " + cfg.views_path);
        }
        if (!fs::exists(cfg.exe_path)) {
            throw std::runtime_error("Executable does not exist: " + cfg.exe_path);
        }

        fs::create_directories(cfg.output_dir);

        std::vector<fs::path> pcd_files;
        for (const auto& entry : fs::directory_iterator(cfg.pcd_dir)) {
            if (!entry.is_regular_file()) continue;
            if (entry.path().extension() == ".pcd") {
                pcd_files.push_back(entry.path());
            }
        }

        std::sort(pcd_files.begin(), pcd_files.end());

        if (pcd_files.empty()) {
            std::cout << "No .pcd files found in: " << cfg.pcd_dir << std::endl;
            return 0;
        }

        const int total = static_cast<int>(pcd_files.size());
        int begin = std::min(cfg.start_idx, total);
        int end = (cfg.end_idx == -1) ? total : std::min(cfg.end_idx, total);

        if (end < begin) {
            throw std::runtime_error("--end_idx must be >= --start_idx");
        }

        if (begin == end) {
            std::cout << "Empty range after clipping: [" << begin << ", " << end << ")\n";
            std::cout << "Total .pcd files found: " << total << std::endl;
            return 0;
        }

        std::cout << "Found " << total << " pcd files.\n";
        std::cout << "Processing sorted range [" << begin << ", " << end << ")"
                  << " -> " << (end - begin) << " files.\n";
        std::cout << "skip_existing = " << cfg.skip_existing << "\n";

        int success = 0;
        int failed = 0;
        int skipped = 0;

        for (int idx = begin; idx < end; ++idx) {
            const fs::path& pcd_path = pcd_files[idx];
            const std::string stem = pcd_path.stem().string();
            const fs::path output_txt = fs::path(cfg.output_dir) / (stem + ".txt");

            std::cout << "[" << (idx - begin + 1) << "/" << (end - begin) << "] "
                      << "(global " << idx << ") "
                      << pcd_path.filename().string();

            if (cfg.skip_existing == 1 && fs::exists(output_txt)) {
                ++skipped;
                std::cout << "  -> skipped (exists)\n";
                continue;
            }

            std::cout << std::endl;

            std::ostringstream cmd;
            cmd << quote(cfg.exe_path)
                << " --pcd " << quote(pcd_path.string())
                << " --views " << quote(cfg.views_path)
                << " --output " << quote(output_txt.string())
                << " --resolution " << cfg.resolution
                << " --view-radius " << cfg.view_radius
                << " --look-at " << cfg.look_at_x << " " << cfg.look_at_y << " " << cfg.look_at_z
                << " --width " << cfg.width
                << " --height " << cfg.height
                << " --fov " << cfg.fov_deg
                << " --max-range " << cfg.max_range
                << " --ignore-unknown " << cfg.ignore_unknown
                << " --time-limit " << cfg.time_limit_sec
                << " --visibility-mode " << cfg.visibility_mode
                << " --min-visible-views " << cfg.min_visible_views
                << " --save-vis-pcd " << cfg.save_vis_pcd;

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
