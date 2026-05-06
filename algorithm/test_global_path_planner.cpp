#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "global_path_planner.h"
#include "objview_view_io.h"

namespace {

struct Config {
    std::string views_path;
    std::vector<int> view_ids;
    int start_view_id = -1;
    int end_view_id = -1;
    double view_radius = 3.0;
    double obstacle_radius = 0.0;
    double time_limit_sec = -1.0;
    bool silent = true;
};

void printUsage(const char* argv0) {
    std::cout
        << "Usage: " << argv0 << " --views PATH --view_ids LIST --start ID [options]\n"
        << "Options:\n"
        << "  --end ID                  Optional fixed endpoint (default: -1)\n"
        << "  --view-radius 3.0\n"
        << "  --obstacle-radius 0.0     Collision sphere radius centered at origin\n"
        << "  --time-limit -1\n"
        << "  --silent 1\n"
        << "\n"
        << "Example:\n"
        << "  " << argv0
        << " --views ../Tammes_sphere/8_xyz.txt --view_ids 0,1,2,3,4,5,6,7"
        << " --start 0 --end 3 --view-radius 3.0 --obstacle-radius 1.414\n";
}

std::vector<int> parseIdList(const std::string& s) {
    std::vector<int> ids;
    std::stringstream ss(s);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (token.empty()) continue;
        ids.push_back(std::stoi(token));
    }
    if (ids.empty()) {
        throw std::runtime_error("--view_ids must contain at least one id.");
    }
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    return ids;
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

        if (arg == "--views") cfg.views_path = needValue(arg);
        else if (arg == "--view_ids") cfg.view_ids = parseIdList(needValue(arg));
        else if (arg == "--start") cfg.start_view_id = std::stoi(needValue(arg));
        else if (arg == "--end") cfg.end_view_id = std::stoi(needValue(arg));
        else if (arg == "--view-radius") cfg.view_radius = std::stod(needValue(arg));
        else if (arg == "--obstacle-radius") cfg.obstacle_radius = std::stod(needValue(arg));
        else if (arg == "--time-limit") cfg.time_limit_sec = std::stod(needValue(arg));
        else if (arg == "--silent") cfg.silent = (std::stoi(needValue(arg)) != 0);
        else if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            std::exit(0);
        }
        else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (cfg.views_path.empty() || cfg.view_ids.empty() || cfg.start_view_id < 0) {
        throw std::runtime_error("--views, --view_ids, and --start are required.");
    }
    if (cfg.view_radius <= 0.0) {
        throw std::runtime_error("--view-radius must be positive.");
    }
    if (cfg.obstacle_radius < 0.0) {
        throw std::runtime_error("--obstacle-radius must be non-negative.");
    }
    return cfg;
}

}  // namespace

void runRuntimeAccountingTest() {
    objview::warmupGurobi();

    objview::HamiltonianPathConfig cfg;
    cfg.view_positions = {
        objview::Vec3(3.0, 0.0, 0.0),
        objview::Vec3(0.0, 3.0, 0.0),
        objview::Vec3(-3.0, 0.0, 0.0),
        objview::Vec3(0.0, -3.0, 0.0),
    };
    cfg.view_ids = {0, 1, 2, 3};
    cfg.start_view_id = 0;
    cfg.end_view_id = -1;
    cfg.obstacle_center = objview::Vec3(0.0, 0.0, 0.0);
    cfg.obstacle_radius = 1.0;
    cfg.time_limit_sec = -1.0;
    cfg.silent = true;

    const auto result = objview::solveHamiltonianPath(cfg);
    if (!result.solved) {
        throw std::runtime_error("Runtime accounting test failed to solve.");
    }
    if (result.path_view_ids.size() != cfg.view_ids.size()) {
        throw std::runtime_error("Runtime accounting test returned wrong path length.");
    }
    if (result.model_build_runtime_sec < 0.0 ||
        result.optimize_runtime_sec < 0.0 ||
        result.solution_runtime_sec < 0.0 ||
        result.billable_runtime_sec < 0.0) {
        throw std::runtime_error("Runtime accounting contains negative value.");
    }

    const double component_sum =
        result.model_build_runtime_sec + result.optimize_runtime_sec + result.solution_runtime_sec;
    if (std::abs(component_sum - result.billable_runtime_sec) > 1e-9) {
        throw std::runtime_error("Runtime accounting total does not match component sum.");
    }
    if (result.billable_runtime_sec <= 0.0) {
        throw std::runtime_error("Runtime accounting billable time should be positive.");
    }

    std::cout << "[ok] Gurobi runtime accounting"
              << " build=" << result.model_build_runtime_sec
              << " optimize=" << result.optimize_runtime_sec
              << " solution=" << result.solution_runtime_sec
              << " billable=" << result.billable_runtime_sec
              << "\n";
}

int main(int argc, char** argv) {
    try {
        if (argc == 1) {
            runRuntimeAccountingTest();
            return 0;
        }

        const Config cfg = parseArgs(argc, argv);

        objview::HamiltonianPathConfig planner_cfg;
        planner_cfg.view_positions = objview::loadViewPositions(cfg.views_path, cfg.view_radius);
        planner_cfg.view_ids = cfg.view_ids;
        planner_cfg.start_view_id = cfg.start_view_id;
        planner_cfg.end_view_id = cfg.end_view_id;
        planner_cfg.obstacle_center = objview::Vec3(0.0, 0.0, 0.0);
        planner_cfg.obstacle_radius = cfg.obstacle_radius;
        planner_cfg.time_limit_sec = cfg.time_limit_sec;
        planner_cfg.silent = cfg.silent;

        const auto result = objview::solveHamiltonianPath(planner_cfg);
        if (!result.solved) {
            std::cout << "No solution found.\n";
            return 1;
        }

        std::cout << "Total length: " << result.total_length << "\n";
        std::cout << "Path:";
        for (int vid : result.path_view_ids) {
            std::cout << " " << vid;
        }
        std::cout << "\n";
        return 0;
    }
    catch (const GRBException& e) {
        std::cerr << "Gurobi error " << e.getErrorCode() << ": " << e.getMessage() << "\n";
        return 2;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}

