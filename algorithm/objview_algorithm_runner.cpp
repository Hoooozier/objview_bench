#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include <gurobi_c++.h>

#include "benbv_planning_network_algorithm.h"
#include "nbvnet_planning_network_algorithm.h"
#include "objview_algorithm.h"
#include "objview_benchmark_submitter.h"
#include "mascvp_planning_network_algorithm.h"
#include "pointr_c_mcp_algorithm.h"
#include "pointr_c_nbv_algorithm.h"
#include "pointr_c_scp_algorithm.h"
#include "random_tsp_order_algorithm.h"
#include "voxel_ig_algorithm.h"

namespace fs = std::filesystem;

namespace {

struct RunnerConfig {
    objview::SubmitterConfig submitter;
    std::string algorithm = "random_tsp_order";
    int budget = 5;
    int seed = 42;
    std::string views_path = "../Tammes_sphere/360_xyz.txt";
    double view_radius = 3.0;
    double obstacle_radius = 1.0;
    double tsp_time_limit_sec = -1.0;
    int voxel_ig_grid_dim = 64;
    double voxel_ig_resolution = -1.0;
    double voxel_ig_map_bbox_min = -1.0;
    double voxel_ig_map_bbox_max = 1.0;
    std::string voxel_ig_backend = "cuda";
    std::string voxel_ig_method = "rse";
    int voxel_ig_ray_stride = 0;
    int voxel_ig_use_movement_cost = 0;
    double voxel_ig_movement_cost_weight = 0.7;
    int voxel_ig_filter_rays_to_bbox = 1;
    int voxel_ig_debug_save_ot = 0;
    std::string voxel_ig_debug_ot_dir;
    double pointr_partial_voxel_leaf_size = 0.015625;
    double pointr_planning_voxel_size = 0.03125;
    double pointr_pointcloud_bbox_min = -1.0;
    double pointr_pointcloud_bbox_max = 1.0;
    double pointr_hpr_radius_scale = 100.0;
    double pointr_tau = 0.95;
    int pointr_sor_mean_k = 16;
    double pointr_sor_stddev_mul = 1.5;
    double pointr_visibility_max_range = 6.0;
    std::string pointr_visibility_mode = "inverse_cuda";
    int pointr_min_visible_views = 1;
    double pointr_scp_time_limit_sec = 10.0;
    int pointr_scp_debug_save = 0;
    std::string pointr_scp_debug_dir = "pointr_c_scp_debug";
    double pointr_mcp_time_limit_sec = 10.0;
    int pointr_mcp_debug_save = 0;
    std::string pointr_mcp_debug_dir = "pointr_c_mcp_debug";
    std::string pointr_completion_backend = "PoinTr-C";
    int mascvp_grid_dim = 64;
    double mascvp_map_bbox_min = -1.0;
    double mascvp_map_bbox_max = 1.0;
    double mascvp_unknown_occ = 0.5;
    std::string mascvp_service_name = "mascvp";
    std::string mascvp_decode_key = "gamma_0.5";
    int mascvp_topk = 5;
    int mascvp_debug_save = 0;
    int mascvp_debug_save_ot = 0;
    std::string mascvp_debug_dir = "mascvp_planning_network_debug";
    std::string benbv_service_name = "benbv";
    double benbv_resolution = 0.02;
    double benbv_camera_distance = 2.0;
    int benbv_point_sample_count = 4096;
    int benbv_candidate_count = 20;
    int benbv_knn = 30;
    double benbv_boundary_angle_deg = 120.0;
    double benbv_partial_voxel_leaf = 0.015625;
    double benbv_pointcloud_bbox_min = -1.0;
    double benbv_pointcloud_bbox_max = 1.0;
    int benbv_topk = 5;
    int benbv_debug_save = 0;
    std::string benbv_debug_dir = "benbv_planning_network_debug";
    int nbvnet_grid_dim = 64;
    double nbvnet_map_bbox_min = -1.0;
    double nbvnet_map_bbox_max = 1.0;
    double nbvnet_unknown_occ = 0.5;
    std::string nbvnet_service_name = "nbvnet";
    int nbvnet_topk = 5;
    int nbvnet_debug_save = 0;
    int nbvnet_debug_save_ot = 0;
    std::string nbvnet_debug_dir = "nbvnet_planning_network_debug";
    int silent = 1;
};

void printUsage(const char* argv0) {
    std::cout
        << "Usage: " << argv0 << " --session-dir DIR [options]\n"
        << "Submitter options:\n"
        << "  --cache-index-json PATH       default: render_cache/cache_index.json\n"
        << "  --query-feasibility 1\n"
        << "  --algorithm-runtime-sec 0.01   accepted for compatibility; runtime is measured around decideNext\n"
        << "  --wait-timeout-sec 120\n"
        << "  --poll-interval-sec 0.01\n"
        << "Algorithm options:\n"
        << "  --algorithm random_tsp_order|voxel_ig|voxel_ig_iterative|pointr_c_nbv|pointr_c_scp|pointr_c_mcp|mascvp_planning_network|benbv_planning_network|nbvnet_planning_network\n"
        << "  --budget 5|10|30|50        Used by finite-plan algorithms such as random_tsp_order; online Voxel-IG runs until benchmark stops it\n"
        << "  --seed 42                  Base seed; per-episode seed hashes uid/constraint/start/method/budget\n"
        << "  --views ../Tammes_sphere/360_xyz.txt\n"
        << "  --view-radius 3.0\n"
        << "  --obstacle-radius 1.0\n"
        << "  --tsp-time-limit -1\n"
        << "  --voxel-ig-grid-dim 64        Default dense grid per axis for bbox initialization; "
           "default bbox [-1,1] gives 0.03125 voxels\n"
        << "  --voxel-ig-resolution 0.03125 Optional override; must evenly divide bbox extent\n"
        << "  --voxel-ig-map-bbox-min -1.0  Default cubic bbox min for all axes\n"
        << "  --voxel-ig-map-bbox-max 1.0  Default cubic bbox max for all axes\n"
        << "  --voxel-ig-backend cpu|cuda    default: cuda\n"
        << "  --voxel-ig-method oa|uv|rse|apora|kr|pcv   default: rse\n"
        << "  --voxel-ig-ray-stride 0        0 means backend default: cpu=16, cuda=4\n"
        << "  --voxel-ig-use-movement-cost 0  1 enables paper-style utility with normalized move cost\n"
        << "  --voxel-ig-movement-cost-weight 0.7  gamma in [0,1] for utility=(1-gamma)*IG-gamma*cost\n"
        << "  --voxel-ig-filter-rays-to-bbox 1\n"
        << "  --voxel-ig-debug-save-ot 0\n"
        << "  --voxel-ig-debug-ot-dir voxel_ig_debug\n"
        << "  --pointr-partial-voxel-leaf-size 0.015625  Default partial-cloud voxel filter leaf size\n"
        << "  --pointr-planning-voxel-size 0.03125  Default planning voxel size\n"
        << "  --pointr-pointcloud-bbox-min -1.0\n"
        << "  --pointr-pointcloud-bbox-max 1.0\n"
        << "  --pointr-hpr-radius-scale 100.0\n"
        << "  --pointr-tau 0.95\n"
        << "  --pointr-sor-mean-k 16\n"
        << "  --pointr-sor-stddev-mul 1.5\n"
        << "  --pointr-visibility-max-range 6.0\n"
        << "  --pointr-visibility-mode inverse_cpu|inverse_cuda  (default: inverse_cuda)\n"
        << "  --pointr-min-visible-views 1\n"
        << "  --pointr-scp-time-limit 10.0\n"
        << "  --pointr-scp-debug-save 0\n"
        << "  --pointr-scp-debug-dir pointr_c_scp_debug\n"
        << "  --pointr-mcp-time-limit 10.0\n"
        << "  --pointr-mcp-debug-save 0\n"
        << "  --pointr-mcp-debug-dir pointr_c_mcp_debug\n"
        << "  --pointr-completion-backend PoinTr-C\n"
        << "  --mascvp-grid-dim 64\n"
        << "  --mascvp-map-bbox-min -1.0\n"
        << "  --mascvp-map-bbox-max 1.0\n"
        << "  --mascvp-unknown-occ 0.5\n"
        << "  --mascvp-service-name mascvp\n"
        << "  --mascvp-decode-key gamma_0.5\n"
        << "  --mascvp-topk 5\n"
        << "  --mascvp-debug-save 0\n"
        << "  --mascvp-debug-save-ot 0\n"
        << "  --mascvp-debug-dir mascvp_planning_network_debug\n"
        << "  --benbv-service-name benbv\n"
        << "  --benbv-resolution 0.02\n"
        << "  --benbv-camera-distance 2.0\n"
        << "  --benbv-point-sample-count 4096\n"
        << "  --benbv-candidate-count 20\n"
        << "  --benbv-knn 30\n"
        << "  --benbv-boundary-angle-deg 120\n"
        << "  --benbv-partial-voxel-leaf 0.015625\n"
        << "  --benbv-pointcloud-bbox-min -1.0\n"
        << "  --benbv-pointcloud-bbox-max 1.0\n"
        << "  --benbv-topk 5\n"
        << "  --benbv-debug-save 0\n"
        << "  --benbv-debug-dir benbv_planning_network_debug\n"
        << "  --nbvnet-grid-dim 64\n"
        << "  --nbvnet-map-bbox-min -1.0\n"
        << "  --nbvnet-map-bbox-max 1.0\n"
        << "  --nbvnet-unknown-occ 0.5\n"
        << "  --nbvnet-service-name nbvnet\n"
        << "  --nbvnet-topk 5\n"
        << "  --nbvnet-debug-save 0\n"
        << "  --nbvnet-debug-save-ot 0\n"
        << "  --nbvnet-debug-dir nbvnet_planning_network_debug\n"
        << "  --silent 1\n";
}

RunnerConfig parseArgs(int argc, char** argv) {
    RunnerConfig cfg;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto needValue = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("Missing value for " + name);
            return argv[++i];
        };

        if (arg == "--session-dir") cfg.submitter.session_dir = needValue(arg);
        else if (arg == "--cache-index-json") cfg.submitter.cache_index_json = needValue(arg);
        else if (arg == "--view-set") {
            (void)needValue(arg);  // Backward-compatible no-op; algorithms define their own view space.
        }
        else if (arg == "--query-feasibility") cfg.submitter.query_feasibility = (std::stoi(needValue(arg)) != 0);
        else if (arg == "--algorithm-runtime-sec") cfg.submitter.algorithm_runtime_sec = std::stod(needValue(arg));
        else if (arg == "--wait-timeout-sec") cfg.submitter.wait_timeout_sec = std::stod(needValue(arg));
        else if (arg == "--poll-interval-sec") cfg.submitter.poll_interval_sec = std::stod(needValue(arg));
        else if (arg == "--algorithm") cfg.algorithm = needValue(arg);
        else if (arg == "--budget") cfg.budget = std::stoi(needValue(arg));
        else if (arg == "--seed") cfg.seed = std::stoi(needValue(arg));
        else if (arg == "--views") cfg.views_path = needValue(arg);
        else if (arg == "--view-radius") cfg.view_radius = std::stod(needValue(arg));
        else if (arg == "--obstacle-radius") cfg.obstacle_radius = std::stod(needValue(arg));
        else if (arg == "--tsp-time-limit") cfg.tsp_time_limit_sec = std::stod(needValue(arg));
        else if (arg == "--voxel-ig-grid-dim") cfg.voxel_ig_grid_dim = std::stoi(needValue(arg));
        else if (arg == "--voxel-ig-resolution") cfg.voxel_ig_resolution = std::stod(needValue(arg));
        else if (arg == "--voxel-ig-map-bbox-min") cfg.voxel_ig_map_bbox_min = std::stod(needValue(arg));
        else if (arg == "--voxel-ig-map-bbox-max") cfg.voxel_ig_map_bbox_max = std::stod(needValue(arg));
        else if (arg == "--voxel-ig-backend") cfg.voxel_ig_backend = needValue(arg);
        else if (arg == "--voxel-ig-method") cfg.voxel_ig_method = needValue(arg);
        else if (arg == "--voxel-ig-ray-stride") cfg.voxel_ig_ray_stride = std::stoi(needValue(arg));
        else if (arg == "--voxel-ig-use-movement-cost") {
            cfg.voxel_ig_use_movement_cost = std::stoi(needValue(arg));
        }
        else if (arg == "--voxel-ig-movement-cost-weight") {
            cfg.voxel_ig_movement_cost_weight = std::stod(needValue(arg));
        }
        else if (arg == "--voxel-ig-filter-rays-to-bbox") {
            cfg.voxel_ig_filter_rays_to_bbox = std::stoi(needValue(arg));
        }
        else if (arg == "--voxel-ig-debug-save-ot") cfg.voxel_ig_debug_save_ot = std::stoi(needValue(arg));
        else if (arg == "--voxel-ig-debug-ot-dir") cfg.voxel_ig_debug_ot_dir = needValue(arg);
        else if (arg == "--pointr-partial-voxel-leaf-size") {
            cfg.pointr_partial_voxel_leaf_size = std::stod(needValue(arg));
        }
        else if (arg == "--pointr-planning-voxel-size") {
            cfg.pointr_planning_voxel_size = std::stod(needValue(arg));
        }
        else if (arg == "--pointr-pointcloud-bbox-min") {
            cfg.pointr_pointcloud_bbox_min = std::stod(needValue(arg));
        }
        else if (arg == "--pointr-pointcloud-bbox-max") {
            cfg.pointr_pointcloud_bbox_max = std::stod(needValue(arg));
        }
        else if (arg == "--pointr-hpr-radius-scale") {
            cfg.pointr_hpr_radius_scale = std::stod(needValue(arg));
        }
        else if (arg == "--pointr-tau") {
            cfg.pointr_tau = std::stod(needValue(arg));
        }
        else if (arg == "--pointr-sor-mean-k") {
            cfg.pointr_sor_mean_k = std::stoi(needValue(arg));
        }
        else if (arg == "--pointr-sor-stddev-mul") {
            cfg.pointr_sor_stddev_mul = std::stod(needValue(arg));
        }
        else if (arg == "--pointr-visibility-max-range") {
            cfg.pointr_visibility_max_range = std::stod(needValue(arg));
        }
        else if (arg == "--pointr-visibility-mode") {
            cfg.pointr_visibility_mode = needValue(arg);
        }
        else if (arg == "--pointr-min-visible-views") {
            cfg.pointr_min_visible_views = std::stoi(needValue(arg));
        }
        else if (arg == "--pointr-scp-time-limit") {
            cfg.pointr_scp_time_limit_sec = std::stod(needValue(arg));
        }
        else if (arg == "--pointr-scp-debug-save") {
            cfg.pointr_scp_debug_save = std::stoi(needValue(arg));
        }
        else if (arg == "--pointr-scp-debug-dir") {
            cfg.pointr_scp_debug_dir = needValue(arg);
        }
        else if (arg == "--pointr-mcp-time-limit") {
            cfg.pointr_mcp_time_limit_sec = std::stod(needValue(arg));
        }
        else if (arg == "--pointr-mcp-debug-save") {
            cfg.pointr_mcp_debug_save = std::stoi(needValue(arg));
        }
        else if (arg == "--pointr-mcp-debug-dir") {
            cfg.pointr_mcp_debug_dir = needValue(arg);
        }
        else if (arg == "--pointr-completion-backend") {
            cfg.pointr_completion_backend = needValue(arg);
        }
        else if (arg == "--mascvp-grid-dim") {
            cfg.mascvp_grid_dim = std::stoi(needValue(arg));
        }
        else if (arg == "--mascvp-map-bbox-min") {
            cfg.mascvp_map_bbox_min = std::stod(needValue(arg));
        }
        else if (arg == "--mascvp-map-bbox-max") {
            cfg.mascvp_map_bbox_max = std::stod(needValue(arg));
        }
        else if (arg == "--mascvp-unknown-occ") {
            cfg.mascvp_unknown_occ = std::stod(needValue(arg));
        }
        else if (arg == "--mascvp-service-name") {
            cfg.mascvp_service_name = needValue(arg);
        }
        else if (arg == "--mascvp-decode-key") {
            cfg.mascvp_decode_key = needValue(arg);
        }
        else if (arg == "--mascvp-topk") {
            cfg.mascvp_topk = std::stoi(needValue(arg));
        }
        else if (arg == "--mascvp-debug-save") {
            cfg.mascvp_debug_save = std::stoi(needValue(arg));
        }
        else if (arg == "--mascvp-debug-save-ot") {
            cfg.mascvp_debug_save_ot = std::stoi(needValue(arg));
        }
        else if (arg == "--mascvp-debug-dir") {
            cfg.mascvp_debug_dir = needValue(arg);
        }
        else if (arg == "--benbv-service-name") {
            cfg.benbv_service_name = needValue(arg);
        }
        else if (arg == "--benbv-resolution") {
            cfg.benbv_resolution = std::stod(needValue(arg));
        }
        else if (arg == "--benbv-camera-distance") {
            cfg.benbv_camera_distance = std::stod(needValue(arg));
        }
        else if (arg == "--benbv-point-sample-count") {
            cfg.benbv_point_sample_count = std::stoi(needValue(arg));
        }
        else if (arg == "--benbv-candidate-count") {
            cfg.benbv_candidate_count = std::stoi(needValue(arg));
        }
        else if (arg == "--benbv-knn") {
            cfg.benbv_knn = std::stoi(needValue(arg));
        }
        else if (arg == "--benbv-boundary-angle-deg") {
            cfg.benbv_boundary_angle_deg = std::stod(needValue(arg));
        }
        else if (arg == "--benbv-partial-voxel-leaf") {
            cfg.benbv_partial_voxel_leaf = std::stod(needValue(arg));
        }
        else if (arg == "--benbv-pointcloud-bbox-min") {
            cfg.benbv_pointcloud_bbox_min = std::stod(needValue(arg));
        }
        else if (arg == "--benbv-pointcloud-bbox-max") {
            cfg.benbv_pointcloud_bbox_max = std::stod(needValue(arg));
        }
        else if (arg == "--benbv-topk") {
            cfg.benbv_topk = std::stoi(needValue(arg));
        }
        else if (arg == "--benbv-debug-save") {
            cfg.benbv_debug_save = std::stoi(needValue(arg));
        }
        else if (arg == "--benbv-debug-dir") {
            cfg.benbv_debug_dir = needValue(arg);
        }
        else if (arg == "--nbvnet-grid-dim") {
            cfg.nbvnet_grid_dim = std::stoi(needValue(arg));
        }
        else if (arg == "--nbvnet-map-bbox-min") {
            cfg.nbvnet_map_bbox_min = std::stod(needValue(arg));
        }
        else if (arg == "--nbvnet-map-bbox-max") {
            cfg.nbvnet_map_bbox_max = std::stod(needValue(arg));
        }
        else if (arg == "--nbvnet-unknown-occ") {
            cfg.nbvnet_unknown_occ = std::stod(needValue(arg));
        }
        else if (arg == "--nbvnet-service-name") {
            cfg.nbvnet_service_name = needValue(arg);
        }
        else if (arg == "--nbvnet-topk") {
            cfg.nbvnet_topk = std::stoi(needValue(arg));
        }
        else if (arg == "--nbvnet-debug-save") {
            cfg.nbvnet_debug_save = std::stoi(needValue(arg));
        }
        else if (arg == "--nbvnet-debug-save-ot") {
            cfg.nbvnet_debug_save_ot = std::stoi(needValue(arg));
        }
        else if (arg == "--nbvnet-debug-dir") {
            cfg.nbvnet_debug_dir = needValue(arg);
        }
        else if (arg == "--silent") cfg.silent = std::stoi(needValue(arg));
        else if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            std::exit(0);
        }
        else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (cfg.submitter.session_dir.empty()) {
        throw std::runtime_error("--session-dir is required.");
    }
    return cfg;
}

std::unique_ptr<objview::Algorithm> makeAlgorithm(const RunnerConfig& cfg) {
    if (cfg.algorithm == "random_tsp_order") {
        objview::warmupGurobi();
        objview::RandomTspOrderConfig algo_cfg;
        algo_cfg.budget = cfg.budget;
        algo_cfg.seed = cfg.seed;
        algo_cfg.views_path = cfg.views_path;
        algo_cfg.view_radius = cfg.view_radius;
        algo_cfg.obstacle_radius = cfg.obstacle_radius;
        algo_cfg.tsp_time_limit_sec = cfg.tsp_time_limit_sec;
        algo_cfg.silent = (cfg.silent != 0);
        return std::make_unique<objview::RandomTspOrderAlgorithm>(algo_cfg);
    }
    if (cfg.algorithm == "voxel_ig" || cfg.algorithm == "voxel_ig_sequential" ||
        cfg.algorithm == "voxel_ig_iterative") {
        objview::VoxelIgAlgorithmConfig algo_cfg;
        algo_cfg.views_path = cfg.views_path;
        algo_cfg.view_radius = cfg.view_radius;
        algo_cfg.grid_dim = cfg.voxel_ig_grid_dim;
        algo_cfg.octomap_resolution = cfg.voxel_ig_resolution;
        algo_cfg.map_bbox_min = cfg.voxel_ig_map_bbox_min;
        algo_cfg.map_bbox_max = cfg.voxel_ig_map_bbox_max;
        algo_cfg.backend = objview::parseVoxelIgBackend(cfg.voxel_ig_backend);
        algo_cfg.method = objview::parseVoxelIgMethod(cfg.voxel_ig_method);
        algo_cfg.ray_stride = cfg.voxel_ig_ray_stride;
        algo_cfg.use_movement_cost = (cfg.voxel_ig_use_movement_cost != 0);
        algo_cfg.movement_cost_weight = cfg.voxel_ig_movement_cost_weight;
        algo_cfg.filter_rays_to_bbox = (cfg.voxel_ig_filter_rays_to_bbox != 0);
        algo_cfg.debug_save_ot = (cfg.voxel_ig_debug_save_ot != 0);
        algo_cfg.debug_ot_dir = cfg.voxel_ig_debug_ot_dir;
        algo_cfg.silent = (cfg.silent != 0);
        return std::make_unique<objview::VoxelIgAlgorithm>(algo_cfg);
    }
    if (cfg.algorithm == "pointr_c_nbv") {
        objview::PointrCNbvAlgorithmConfig algo_cfg;
        algo_cfg.views_path = cfg.views_path;
        algo_cfg.view_radius = cfg.view_radius;
        algo_cfg.obstacle_radius = cfg.obstacle_radius;
        algo_cfg.partial_voxel_leaf_size = cfg.pointr_partial_voxel_leaf_size;
        algo_cfg.planning_voxel_size = cfg.pointr_planning_voxel_size;
        algo_cfg.pointcloud_bbox_min = cfg.pointr_pointcloud_bbox_min;
        algo_cfg.pointcloud_bbox_max = cfg.pointr_pointcloud_bbox_max;
        algo_cfg.hpr_radius_scale = cfg.pointr_hpr_radius_scale;
        algo_cfg.tau = cfg.pointr_tau;
        algo_cfg.completion_backend_name = cfg.pointr_completion_backend;
        algo_cfg.silent = (cfg.silent != 0);
        return std::make_unique<objview::PointrCNbvAlgorithm>(algo_cfg);
    }
    if (cfg.algorithm == "pointr_c_scp") {
        objview::warmupGurobi();
        objview::PointrCScpAlgorithmConfig algo_cfg;
        algo_cfg.views_path = cfg.views_path;
        algo_cfg.view_radius = cfg.view_radius;
        algo_cfg.obstacle_radius = cfg.obstacle_radius;
        algo_cfg.partial_voxel_leaf_size = cfg.pointr_partial_voxel_leaf_size;
        algo_cfg.planning_voxel_size = cfg.pointr_planning_voxel_size;
        algo_cfg.pointcloud_bbox_min = cfg.pointr_pointcloud_bbox_min;
        algo_cfg.pointcloud_bbox_max = cfg.pointr_pointcloud_bbox_max;
        algo_cfg.sor_mean_k = cfg.pointr_sor_mean_k;
        algo_cfg.sor_stddev_mul = cfg.pointr_sor_stddev_mul;
        algo_cfg.visibility_max_range = cfg.pointr_visibility_max_range;
        algo_cfg.visibility_mode = cfg.pointr_visibility_mode;
        algo_cfg.min_visible_views = cfg.pointr_min_visible_views;
        algo_cfg.scp_time_limit_sec = cfg.pointr_scp_time_limit_sec;
        algo_cfg.debug_save_intermediate = (cfg.pointr_scp_debug_save != 0);
        algo_cfg.debug_dir = cfg.pointr_scp_debug_dir;
        algo_cfg.completion_backend_name = cfg.pointr_completion_backend;
        algo_cfg.silent = (cfg.silent != 0);
        return std::make_unique<objview::PointrCScpAlgorithm>(algo_cfg);
    }
    if (cfg.algorithm == "pointr_c_mcp") {
        objview::warmupGurobi();
        objview::PointrCMcpAlgorithmConfig algo_cfg;
        algo_cfg.budget = cfg.budget;
        algo_cfg.views_path = cfg.views_path;
        algo_cfg.view_radius = cfg.view_radius;
        algo_cfg.obstacle_radius = cfg.obstacle_radius;
        algo_cfg.partial_voxel_leaf_size = cfg.pointr_partial_voxel_leaf_size;
        algo_cfg.planning_voxel_size = cfg.pointr_planning_voxel_size;
        algo_cfg.pointcloud_bbox_min = cfg.pointr_pointcloud_bbox_min;
        algo_cfg.pointcloud_bbox_max = cfg.pointr_pointcloud_bbox_max;
        algo_cfg.sor_mean_k = cfg.pointr_sor_mean_k;
        algo_cfg.sor_stddev_mul = cfg.pointr_sor_stddev_mul;
        algo_cfg.visibility_max_range = cfg.pointr_visibility_max_range;
        algo_cfg.visibility_mode = cfg.pointr_visibility_mode;
        algo_cfg.min_visible_views = cfg.pointr_min_visible_views;
        algo_cfg.mcp_time_limit_sec = cfg.pointr_mcp_time_limit_sec;
        algo_cfg.debug_save_intermediate = (cfg.pointr_mcp_debug_save != 0);
        algo_cfg.debug_dir = cfg.pointr_mcp_debug_dir;
        algo_cfg.completion_backend_name = cfg.pointr_completion_backend;
        algo_cfg.silent = (cfg.silent != 0);
        return std::make_unique<objview::PointrCMcpAlgorithm>(algo_cfg);
    }
    if (cfg.algorithm == "mascvp_planning_network" || cfg.algorithm == "mascvp_scp") {
        objview::warmupGurobi();
        objview::MascvpPlanningNetworkConfig algo_cfg;
        algo_cfg.views_path = cfg.views_path;
        algo_cfg.view_radius = cfg.view_radius;
        algo_cfg.obstacle_radius = cfg.obstacle_radius;
        algo_cfg.grid_dim = cfg.mascvp_grid_dim;
        algo_cfg.map_bbox_min = cfg.mascvp_map_bbox_min;
        algo_cfg.map_bbox_max = cfg.mascvp_map_bbox_max;
        algo_cfg.unknown_occ = cfg.mascvp_unknown_occ;
        algo_cfg.service_name = cfg.mascvp_service_name;
        algo_cfg.decode_key = cfg.mascvp_decode_key;
        algo_cfg.topk = cfg.mascvp_topk;
        algo_cfg.tsp_time_limit_sec = cfg.tsp_time_limit_sec;
        algo_cfg.debug_save = (cfg.mascvp_debug_save != 0);
        algo_cfg.debug_save_ot = (cfg.mascvp_debug_save_ot != 0);
        algo_cfg.debug_dir = cfg.mascvp_debug_dir;
        algo_cfg.silent = (cfg.silent != 0);
        return std::make_unique<objview::MascvpPlanningNetworkAlgorithm>(algo_cfg);
    }
    if (cfg.algorithm == "benbv_planning_network" || cfg.algorithm == "benbv_nbv") {
        objview::BenbvPlanningNetworkConfig algo_cfg;
        algo_cfg.service_name = cfg.benbv_service_name;
        algo_cfg.resolution = cfg.benbv_resolution;
        algo_cfg.camera_distance = cfg.benbv_camera_distance;
        algo_cfg.point_sample_count = cfg.benbv_point_sample_count;
        algo_cfg.candidate_count = cfg.benbv_candidate_count;
        algo_cfg.knn = cfg.benbv_knn;
        algo_cfg.boundary_angle_deg = cfg.benbv_boundary_angle_deg;
        algo_cfg.partial_voxel_leaf = cfg.benbv_partial_voxel_leaf;
        algo_cfg.pointcloud_bbox_min = cfg.benbv_pointcloud_bbox_min;
        algo_cfg.pointcloud_bbox_max = cfg.benbv_pointcloud_bbox_max;
        algo_cfg.seed = cfg.seed;
        algo_cfg.topk = cfg.benbv_topk;
        algo_cfg.debug_save = (cfg.benbv_debug_save != 0);
        algo_cfg.debug_dir = cfg.benbv_debug_dir;
        algo_cfg.silent = (cfg.silent != 0);
        return std::make_unique<objview::BenbvPlanningNetworkAlgorithm>(algo_cfg);
    }
    if (cfg.algorithm == "nbvnet_planning_network" || cfg.algorithm == "nbvnet_nbv") {
        objview::NbvnetPlanningNetworkConfig algo_cfg;
        algo_cfg.views_path = cfg.views_path;
        algo_cfg.view_radius = cfg.view_radius;
        algo_cfg.grid_dim = cfg.nbvnet_grid_dim;
        algo_cfg.map_bbox_min = cfg.nbvnet_map_bbox_min;
        algo_cfg.map_bbox_max = cfg.nbvnet_map_bbox_max;
        algo_cfg.unknown_occ = cfg.nbvnet_unknown_occ;
        algo_cfg.service_name = cfg.nbvnet_service_name;
        algo_cfg.topk = cfg.nbvnet_topk;
        algo_cfg.debug_save = (cfg.nbvnet_debug_save != 0);
        algo_cfg.debug_save_ot = (cfg.nbvnet_debug_save_ot != 0);
        algo_cfg.debug_dir = cfg.nbvnet_debug_dir;
        algo_cfg.silent = (cfg.silent != 0);
        return std::make_unique<objview::NbvnetPlanningNetworkAlgorithm>(algo_cfg);
    }

    throw std::runtime_error("Unknown algorithm: " + cfg.algorithm);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const RunnerConfig cfg = parseArgs(argc, argv);
        std::unique_ptr<objview::Algorithm> algorithm = makeAlgorithm(cfg);
        objview::BenchmarkSubmitter submitter(cfg.submitter);
        return submitter.run(*algorithm);
    }
    catch (const GRBException& e) {
        std::cerr << "Gurobi error " << e.getErrorCode() << ": " << e.getMessage() << std::endl;
        return 2;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
