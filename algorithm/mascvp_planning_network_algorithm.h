#ifndef OBJVIEWBENCH_MASCVP_PLANNING_NETWORK_ALGORITHM_H_
#define OBJVIEWBENCH_MASCVP_PLANNING_NETWORK_ALGORITHM_H_

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <Eigen/Dense>
#include <cnpy.h>
#include <json/json.h>
#include <octomap/ColorOcTree.h>

#include "objview_algorithm.h"
#include "objview_observation_io.h"
#include "objview_planning_network_client.h"
#include "objview_view_io.h"

namespace objview {

struct MascvpPlanningNetworkConfig {
    std::string views_path = "../Tammes_sphere/128_xyz.txt";
    double view_radius = 3.0;
    double obstacle_radius = 1.0;
    int grid_dim = 64;
    double map_bbox_min = -1.0;
    double map_bbox_max = 1.0;
    double unknown_occ = 0.5;
    std::string service_name = "mascvp";
    std::string decode_key = "gamma_0.5";
    int topk = 5;
    double tsp_time_limit_sec = -1.0;
    bool debug_save = false;
    bool debug_save_ot = false;
    std::string debug_dir = "mascvp_planning_network_debug";
    bool silent = true;
};

class MascvpPlanningNetworkAlgorithm : public Algorithm {
public:
    explicit MascvpPlanningNetworkAlgorithm(MascvpPlanningNetworkConfig cfg)
        : cfg_(std::move(cfg)), map_(computeResolution(cfg_)) {
        if (cfg_.grid_dim <= 0) {
            throw std::runtime_error("MASCVP grid_dim must be positive.");
        }
        if (cfg_.map_bbox_min >= cfg_.map_bbox_max) {
            throw std::runtime_error("MASCVP map bbox min must be smaller than max.");
        }
        if (cfg_.view_radius <= 0.0) {
            throw std::runtime_error("MASCVP view_radius must be positive.");
        }
        initializeUnknownMap();
    }

    void prepareEpisode(
        const Json::Value& episode_config,
        const std::filesystem::path& session_dir) override {
        episode_config_ = episode_config;
        session_dir_ = session_dir;
        service_root_ = session_dir_ / "planning_network" / cfg_.service_name;
        if (cfg_.debug_save || cfg_.debug_save_ot) {
            std::filesystem::path p(cfg_.debug_dir);
            debug_dir_abs_ = p.is_absolute() ? p : (session_dir_ / p);
            std::filesystem::create_directories(debug_dir_abs_);
        }
    }

    std::vector<ViewEntry> candidateViewSpace() const override {
        const auto positions = loadViewPositions(cfg_.views_path, cfg_.view_radius);
        std::vector<ViewEntry> views;
        views.reserve(positions.size());
        for (size_t i = 0; i < positions.size(); ++i) {
            ViewEntry entry;
            entry.view_idx = static_cast<int>(i);
            entry.pose.v = {
                positions[i].x(),
                positions[i].y(),
                positions[i].z(),
                0.0,
                0.0,
                0.0,
                0.0,
            };
            views.push_back(entry);
        }
        return views;
    }

    AlgorithmDecision decideNext(const AlgorithmContext& ctx) override {
        ensureStartInitialized(ctx);

        if (planned_) {
            if (cursor_ >= planned_path_.size()) {
                return AlgorithmDecision::stop(terminal_stop_reason_).withRuntime(0.0);
            }
            const int next_view_id = planned_path_[cursor_++];
            visited_128_.insert(next_view_id);
            return AlgorithmDecision::move(next_view_id).withRuntime(0.0);
        }

        const auto step_start = std::chrono::steady_clock::now();
        const ObservationFrame observation = readObservationFrame(ctx);
        const auto update_start = std::chrono::steady_clock::now();
        updateMapFromObservation(ctx, observation);
        const double map_update_sec = secondsSince(update_start);

        if (!bootstrap_selected_) {
            const int fp_id = selectBootstrapFp(ctx);
            if (fp_id < 0) {
                const double runtime_sec = map_update_sec;
                return AlgorithmDecision::stop("candidate_exhausted").withRuntime(runtime_sec);
            }
            bootstrap_view_id_ = fp_id;
            bootstrap_selected_ = true;
            visited_128_.insert(fp_id);
            if (!cfg_.silent) {
                std::cout << "[mascvp] bootstrap fp=" << fp_id
                          << " init128=" << init_nearest128_id_ << std::endl;
            }
            return AlgorithmDecision::move(fp_id).withRuntime(map_update_sec);
        }

        const PlanResult plan = makePlan(ctx, map_update_sec, step_start);
        planned_path_ = plan.path;
        terminal_stop_reason_ = plan.terminal_stop_reason;
        planned_ = true;
        cursor_ = 0;

        if (!cfg_.silent) {
            std::cout << "[mascvp] path:";
            for (int vid : planned_path_) std::cout << " " << vid;
            std::cout << " terminal=" << terminal_stop_reason_
                      << " detail=" << plan.stop_detail
                      << " runtime_sec=" << plan.billable_runtime_sec << std::endl;
        }

        if (cursor_ >= planned_path_.size()) {
            return AlgorithmDecision::stop(terminal_stop_reason_)
                .withRuntime(plan.billable_runtime_sec);
        }

        const int next_view_id = planned_path_[cursor_++];
        visited_128_.insert(next_view_id);
        return AlgorithmDecision::move(next_view_id).withRuntime(plan.billable_runtime_sec);
    }

private:
    struct ObservationFrame {
        Json::Value frame_meta;
        Pose7d pose;
        CameraIntrinsics intrinsics;
        std::filesystem::path depth_path;
        std::filesystem::path mask_path;
        bool has_depth = false;
    };

    struct PlanResult {
        std::vector<int> path;
        double billable_runtime_sec = 0.0;
        std::string terminal_stop_reason = "plan_end";
        std::string stop_detail = "normal_plan_end";
        Json::Value debug;
    };

    MascvpPlanningNetworkConfig cfg_;
    octomap::ColorOcTree map_;
    Json::Value episode_config_;
    std::filesystem::path session_dir_;
    std::filesystem::path service_root_;
    std::filesystem::path debug_dir_abs_;
    mutable std::optional<std::vector<ViewEntry>> all_views_cache_;
    std::unordered_set<int> observed_step_indices_;
    std::set<int> visited_128_;
    int init_nearest128_id_ = -1;
    int bootstrap_view_id_ = -1;
    bool bootstrap_selected_ = false;
    bool planned_ = false;
    std::vector<int> planned_path_;
    size_t cursor_ = 0;
    std::string terminal_stop_reason_ = "plan_end";

    static double computeResolution(const MascvpPlanningNetworkConfig& cfg) {
        return (cfg.map_bbox_max - cfg.map_bbox_min) / static_cast<double>(cfg.grid_dim);
    }

    double voxelResolution() const {
        return map_.getResolution();
    }

    void initializeUnknownMap() {
        const double resolution = voxelResolution();
        for (int ix = 0; ix < cfg_.grid_dim; ++ix) {
            const double x = cfg_.map_bbox_min + (static_cast<double>(ix) + 0.5) * resolution;
            for (int iy = 0; iy < cfg_.grid_dim; ++iy) {
                const double y = cfg_.map_bbox_min + (static_cast<double>(iy) + 0.5) * resolution;
                for (int iz = 0; iz < cfg_.grid_dim; ++iz) {
                    const double z = cfg_.map_bbox_min + (static_cast<double>(iz) + 0.5) * resolution;
                    map_.updateNode(
                        octomap::point3d(
                            static_cast<float>(x),
                            static_cast<float>(y),
                            static_cast<float>(z)),
                        0.0f,
                        true);
                }
            }
        }
        map_.updateInnerOccupancy();
    }

    void ensureStartInitialized(const AlgorithmContext& ctx) {
        if (init_nearest128_id_ >= 0) return;
        init_nearest128_id_ = nearestViewId(cameraPosition(ctx.current_pose));
        visited_128_.insert(init_nearest128_id_);
    }

    ObservationFrame readObservationFrame(const AlgorithmContext& ctx) const {
        ObservationFrame observation;
        observation.pose = ctx.current_pose;

        const std::string frame_meta_rel =
            ctx.observation_manifest.get("frame_meta_path", "").asString();
        if (!frame_meta_rel.empty()) {
            observation.frame_meta = readJsonFile(resolveSessionPath(ctx, frame_meta_rel));
            observation.pose = poseFromFrameMeta(observation.frame_meta);
            observation.intrinsics = parseCameraIntrinsics(observation.frame_meta);
        } else {
            throw std::runtime_error("MASCVP requires frame_meta_path in observation_manifest.");
        }

        const std::string depth_rel =
            ctx.observation_manifest.get("depth_path", "").asString();
        const std::string mask_rel =
            ctx.observation_manifest.get("mask_path", "").asString();
        observation.has_depth = !depth_rel.empty() && !mask_rel.empty();
        if (!observation.has_depth) {
            throw std::runtime_error("MASCVP requires depth_path and mask_path.");
        }
        observation.depth_path = resolveSessionPath(ctx, depth_rel);
        observation.mask_path = resolveSessionPath(ctx, mask_rel);
        return observation;
    }

    void updateMapFromObservation(const AlgorithmContext& ctx, const ObservationFrame& observation) {
        if (ctx.step_index < 0) return;
        if (!observed_step_indices_.insert(ctx.step_index).second) return;

        const DepthImage depth = loadDepthNpz(observation.depth_path);
        const MaskImage mask = loadMaskImage(observation.mask_path);
        const Eigen::Matrix4d camera_to_world =
            parseMatrix4d(observation.frame_meta["camera_to_world"], "camera_to_world");
        const octomap::Pointcloud cloud = backprojectDepthToWorldPointcloud(
            depth,
            mask,
            observation.intrinsics,
            camera_to_world,
            cfg_.map_bbox_min,
            cfg_.map_bbox_max);
        const Vec3 camera = cameraPosition(observation.pose);
        map_.insertPointCloud(
            cloud,
            octomap::point3d(camera.x(), camera.y(), camera.z()),
            -1.0,
            false,
            false);
        map_.updateInnerOccupancy();
    }

    PlanResult makePlan(
        const AlgorithmContext& ctx,
        double map_update_sec,
        const std::chrono::steady_clock::time_point& step_start) const {
        PlanResult plan;
        plan.debug["uid"] = ctx.uid;
        plan.debug["episode_id"] = ctx.episode_id;
        plan.debug["step_index"] = ctx.step_index;
        plan.debug["init_nearest128_id"] = init_nearest128_id_;
        plan.debug["fp_view_id_128"] = bootstrap_view_id_;

        const auto grid_start = std::chrono::steady_clock::now();
        double grid_min = 0.0;
        double grid_max = 0.0;
        const std::vector<float> grid = buildDenseOccupancyGrid(&grid_min, &grid_max);
        const std::vector<uint8_t> vs = makeViewState();
        const double grid_compute_sec = secondsSince(grid_start);

        std::filesystem::path input_npz;
        double npz_write_elapsed_sec = 0.0;
        {
            const auto write_start = std::chrono::steady_clock::now();
            input_npz = saveInputNpz(ctx, grid, vs);
            npz_write_elapsed_sec = secondsSince(write_start);
        }

        PlanningNetworkRpcClient client(
            service_root_,
            ctx.interaction_wait_timeout_sec,
            ctx.interaction_poll_interval_sec);
        const std::string request_id = makeRequestId(ctx);
        const PlanningNetworkInferResult infer =
            client.infer(request_id, input_npz, cfg_.topk);
        const Json::Value result = infer.result;
        const double service_runtime_sec =
            result["runtime"].get("total_sec", 0.0).asDouble();

        const auto filter_start = std::chrono::steady_clock::now();
        const std::vector<int> network_ids = readGammaSelectedIds(result);
        const std::vector<int> dedup_ids = deduplicatePreserveOrder(network_ids);
        std::vector<int> visited_filtered_ids;
        visited_filtered_ids.reserve(dedup_ids.size());
        for (int vid : dedup_ids) {
            if (visited_128_.count(vid)) continue;
            visited_filtered_ids.push_back(vid);
        }

        std::unordered_set<int> feasible_ids;
        feasible_ids.reserve(ctx.candidate_views.size());
        for (const auto& v : ctx.candidate_views) feasible_ids.insert(v.view_idx);

        std::vector<int> feasible_filtered_ids;
        feasible_filtered_ids.reserve(visited_filtered_ids.size());
        for (int vid : visited_filtered_ids) {
            if (feasible_ids.count(vid)) feasible_filtered_ids.push_back(vid);
        }
        const double filter_sec = secondsSince(filter_start);

        const auto tsp_start = std::chrono::steady_clock::now();
        plan.path = orderPlannedViews(ctx.current_pose, feasible_filtered_ids);
        const double tsp_sec = secondsSince(tsp_start);

        if (network_ids.empty()) {
            plan.terminal_stop_reason = "plan_end";
            plan.stop_detail = "empty_network_set";
        } else if (visited_filtered_ids.empty()) {
            plan.terminal_stop_reason = "plan_end";
            plan.stop_detail = "network_set_only_visited";
        } else if (feasible_filtered_ids.empty()) {
            plan.terminal_stop_reason = "candidate_exhausted";
            plan.stop_detail = "network_set_all_infeasible";
        } else {
            plan.terminal_stop_reason = "plan_end";
            plan.stop_detail = "normal_plan_end";
        }

        plan.billable_runtime_sec =
            map_update_sec + grid_compute_sec + service_runtime_sec + filter_sec + tsp_sec;

        fillDebugPlan(
            plan.debug,
            input_npz,
            infer,
            result,
            network_ids,
            dedup_ids,
            visited_filtered_ids,
            feasible_filtered_ids,
            plan.path,
            grid_min,
            grid_max,
            map_update_sec,
            grid_compute_sec,
            npz_write_elapsed_sec,
            service_runtime_sec,
            infer.rpc_elapsed_sec,
            filter_sec,
            tsp_sec,
            plan);
        saveDebugArtifacts(ctx, plan.debug);
        return plan;
    }

    std::vector<float> buildDenseOccupancyGrid(double* out_min, double* out_max) const {
        const double resolution = voxelResolution();
        std::vector<float> grid(
            static_cast<size_t>(cfg_.grid_dim) *
            static_cast<size_t>(cfg_.grid_dim) *
            static_cast<size_t>(cfg_.grid_dim),
            static_cast<float>(cfg_.unknown_occ));

        float min_val = std::numeric_limits<float>::infinity();
        float max_val = -std::numeric_limits<float>::infinity();
        size_t idx = 0;
        for (int ix = 0; ix < cfg_.grid_dim; ++ix) {
            const double x = cfg_.map_bbox_min + (static_cast<double>(ix) + 0.5) * resolution;
            for (int iy = 0; iy < cfg_.grid_dim; ++iy) {
                const double y = cfg_.map_bbox_min + (static_cast<double>(iy) + 0.5) * resolution;
                for (int iz = 0; iz < cfg_.grid_dim; ++iz, ++idx) {
                    const double z = cfg_.map_bbox_min + (static_cast<double>(iz) + 0.5) * resolution;
                    const octomap::ColorOcTreeNode* node = map_.search(
                        octomap::point3d(
                            static_cast<float>(x),
                            static_cast<float>(y),
                            static_cast<float>(z)));
                    const float occ = node == nullptr
                        ? static_cast<float>(cfg_.unknown_occ)
                        : static_cast<float>(node->getOccupancy());
                    grid[idx] = occ;
                    min_val = std::min(min_val, occ);
                    max_val = std::max(max_val, occ);
                }
            }
        }
        if (out_min) *out_min = min_val;
        if (out_max) *out_max = max_val;
        return grid;
    }

    std::vector<uint8_t> makeViewState() const {
        const size_t n = allViews().size();
        std::vector<uint8_t> vs(n, static_cast<uint8_t>(0));
        for (int vid : visited_128_) {
            if (vid >= 0 && vid < static_cast<int>(n)) vs[static_cast<size_t>(vid)] = 1;
        }
        return vs;
    }

    std::filesystem::path saveInputNpz(
        const AlgorithmContext& ctx,
        const std::vector<float>& grid,
        const std::vector<uint8_t>& vs) const {
        std::filesystem::path dir = debug_dir_abs_.empty()
            ? (service_root_ / "inputs")
            : debug_dir_abs_;
        std::filesystem::create_directories(dir);
        const std::filesystem::path path = dir / (makeStepStem(ctx) + ".npz");

        const std::vector<std::size_t> grid_shape{
            static_cast<size_t>(cfg_.grid_dim),
            static_cast<size_t>(cfg_.grid_dim),
            static_cast<size_t>(cfg_.grid_dim)};
        const std::vector<std::size_t> vs_shape{vs.size()};
        const std::vector<std::size_t> scalar_shape{1};
        const int32_t step_i32 = static_cast<int32_t>(ctx.step_index);
        const int32_t init_i32 = static_cast<int32_t>(init_nearest128_id_);
        const int32_t fp_i32 = static_cast<int32_t>(bootstrap_view_id_);
        cnpy::npz_save(path.string(), "grid", grid.data(), grid_shape, "w");
        cnpy::npz_save(path.string(), "vs", vs.data(), vs_shape, "a");
        cnpy::npz_save(path.string(), "step_id", &step_i32, scalar_shape, "a");
        cnpy::npz_save(path.string(), "init_nearest128_id", &init_i32, scalar_shape, "a");
        cnpy::npz_save(path.string(), "fp_view_id_128", &fp_i32, scalar_shape, "a");
        return path;
    }

    std::vector<int> readGammaSelectedIds(const Json::Value& result) const {
        const Json::Value selected =
            result["decodes"][cfg_.decode_key]["selected_indices"];
        if (!selected.isArray() || selected.empty() || !selected[0].isArray()) {
            return {};
        }
        std::vector<int> ids;
        ids.reserve(selected[0].size());
        for (const Json::Value& x : selected[0]) {
            if (x.isInt()) ids.push_back(x.asInt());
        }
        return ids;
    }

    static std::vector<int> deduplicatePreserveOrder(const std::vector<int>& ids) {
        std::unordered_set<int> seen;
        std::vector<int> out;
        out.reserve(ids.size());
        for (int id : ids) {
            if (id < 0) continue;
            if (!seen.insert(id).second) continue;
            out.push_back(id);
        }
        return out;
    }

    int selectBootstrapFp(const AlgorithmContext& ctx) const {
        int best_view_id = -1;
        double best_cost = -std::numeric_limits<double>::infinity();
        for (const auto& candidate : ctx.candidate_views) {
            if (candidate.view_idx == init_nearest128_id_) continue;
            if (visited_128_.count(candidate.view_idx)) continue;
            const double cost = collisionAvoidSphereDistance(
                cameraPosition(ctx.current_pose),
                cameraPosition(candidate.pose),
                Vec3::Zero(),
                cfg_.obstacle_radius);
            if (cost > best_cost) {
                best_cost = cost;
                best_view_id = candidate.view_idx;
            }
        }
        return best_view_id;
    }

    std::vector<int> orderPlannedViews(
        const Pose7d& current_pose,
        const std::vector<int>& selected_view_ids) const {
        if (selected_view_ids.empty()) return {};
        const auto& all_views = allViews();
        std::unordered_map<int, Vec3> pos_map;
        pos_map.reserve(all_views.size());
        for (const auto& v : all_views) pos_map[v.view_idx] = cameraPosition(v.pose);

        std::vector<Vec3> tsp_positions;
        std::vector<int> tsp_ids;
        std::vector<int> local_to_global;
        tsp_positions.reserve(selected_view_ids.size() + 1);
        tsp_ids.reserve(selected_view_ids.size() + 1);
        local_to_global.reserve(selected_view_ids.size() + 1);
        int local_id = 0;
        for (int vid : selected_view_ids) {
            const auto it = pos_map.find(vid);
            if (it == pos_map.end()) continue;
            tsp_positions.push_back(it->second);
            tsp_ids.push_back(local_id++);
            local_to_global.push_back(vid);
        }
        if (tsp_positions.empty()) return {};
        const int virtual_start_id = local_id;
        tsp_positions.push_back(cameraPosition(current_pose));
        tsp_ids.push_back(virtual_start_id);
        local_to_global.push_back(-1);

        HamiltonianPathConfig tsp_cfg;
        tsp_cfg.view_positions = tsp_positions;
        tsp_cfg.view_ids = tsp_ids;
        tsp_cfg.start_view_id = virtual_start_id;
        tsp_cfg.end_view_id = -1;
        tsp_cfg.obstacle_center = Vec3(0.0, 0.0, 0.0);
        tsp_cfg.obstacle_radius = cfg_.obstacle_radius;
        tsp_cfg.time_limit_sec = cfg_.tsp_time_limit_sec;
        tsp_cfg.silent = cfg_.silent;
        const auto tsp = solveHamiltonianPath(tsp_cfg);
        if (!tsp.solved) return selected_view_ids;

        std::vector<int> ordered;
        ordered.reserve(selected_view_ids.size());
        for (int local : tsp.path_view_ids) {
            if (local == virtual_start_id) continue;
            if (local < 0 || local >= static_cast<int>(local_to_global.size())) continue;
            const int global = local_to_global[static_cast<size_t>(local)];
            if (global >= 0) ordered.push_back(global);
        }
        return ordered;
    }

    void fillDebugPlan(
        Json::Value& debug,
        const std::filesystem::path& input_npz,
        const PlanningNetworkInferResult& infer,
        const Json::Value& result,
        const std::vector<int>& network_ids,
        const std::vector<int>& dedup_ids,
        const std::vector<int>& visited_filtered_ids,
        const std::vector<int>& feasible_filtered_ids,
        const std::vector<int>& tsp_ordered_ids,
        double grid_min,
        double grid_max,
        double map_update_sec,
        double grid_compute_sec,
        double npz_write_elapsed_sec,
        double service_runtime_sec,
        double rpc_elapsed_sec,
        double filter_sec,
        double tsp_sec,
        const PlanResult& plan) const {
        debug["input_npz"] = input_npz.string();
        debug["request_id"] = infer.request_id;
        debug["request_path"] = infer.request_path.string();
        debug["response_path"] = infer.response_path.string();
        debug["backend"] = result.get("backend", "").asString();
        debug["decode_key"] = cfg_.decode_key;
        debug["grid_min"] = grid_min;
        debug["grid_max"] = grid_max;
        debug["map_update_sec"] = map_update_sec;
        debug["grid_compute_sec"] = grid_compute_sec;
        debug["npz_write_elapsed_sec"] = npz_write_elapsed_sec;
        debug["service_runtime_total_sec"] = service_runtime_sec;
        debug["rpc_elapsed_sec"] = rpc_elapsed_sec;
        debug["filter_sec"] = filter_sec;
        debug["tsp_sec"] = tsp_sec;
        debug["billable_runtime_sec"] = plan.billable_runtime_sec;
        debug["terminal_stop_reason"] = plan.terminal_stop_reason;
        debug["stop_detail"] = plan.stop_detail;
        debug["network_selected_ids"] = toJsonArray(network_ids);
        debug["dedup_ids"] = toJsonArray(dedup_ids);
        debug["visited_filtered_ids"] = toJsonArray(visited_filtered_ids);
        debug["feasible_filtered_ids"] = toJsonArray(feasible_filtered_ids);
        debug["tsp_ordered_ids"] = toJsonArray(tsp_ordered_ids);
        debug["network_selected_count"] = static_cast<Json::UInt64>(network_ids.size());
        debug["dedup_count"] = static_cast<Json::UInt64>(dedup_ids.size());
        debug["visited_filtered_count"] = static_cast<Json::UInt64>(visited_filtered_ids.size());
        debug["feasible_filtered_count"] = static_cast<Json::UInt64>(feasible_filtered_ids.size());
        debug["tsp_ordered_count"] = static_cast<Json::UInt64>(tsp_ordered_ids.size());
    }

    void saveDebugArtifacts(const AlgorithmContext& ctx, const Json::Value& debug) const {
        if (debug_dir_abs_.empty()) return;
        const std::string stem = makeStepStem(ctx);
        if (cfg_.debug_save_ot) {
            saveDebugOctomaps(stem);
        }
        if (cfg_.debug_save) {
            Json::StreamWriterBuilder builder;
            builder["indentation"] = "  ";
            std::ofstream fout(debug_dir_abs_ / (stem + "_plan.json"), std::ios::binary);
            if (fout) fout << Json::writeString(builder, debug) << "\n";
        }
    }

    void saveDebugOctomaps(const std::string& stem) const {
        std::filesystem::create_directories(debug_dir_abs_);
        const std::filesystem::path observed_path = debug_dir_abs_ / (stem + "__observed_only.ot");
        octomap::ColorOcTree observed_only(voxelResolution());
        for (auto it = map_.begin_leafs(), end = map_.end_leafs(); it != end; ++it) {
            const double occupancy = (*it).getOccupancy();
            if (std::abs(occupancy - cfg_.unknown_occ) < 1e-6) continue;
            observed_only.updateNode(it.getCoordinate(), occupancy >= cfg_.unknown_occ);
            if (auto* node = observed_only.search(it.getKey())) {
                node->setLogOdds((*it).getLogOdds());
            }
        }
        observed_only.updateInnerOccupancy();
        if (!observed_only.write(observed_path.string())) {
            throw std::runtime_error("Failed to write MASCVP observed-only octomap: " +
                                     observed_path.string());
        }

        const std::filesystem::path with_unknown_path = debug_dir_abs_ / (stem + "__with_unknown.ot");
        if (!map_.write(with_unknown_path.string())) {
            throw std::runtime_error("Failed to write MASCVP with-unknown octomap: " +
                                     with_unknown_path.string());
        }
    }

    int nearestViewId(const Vec3& position) const {
        const auto& all = allViews();
        int best_id = -1;
        double best_dist = std::numeric_limits<double>::infinity();
        for (const auto& v : all) {
            const double d = (cameraPosition(v.pose) - position).squaredNorm();
            if (d < best_dist) {
                best_dist = d;
                best_id = v.view_idx;
            }
        }
        if (best_id < 0) {
            throw std::runtime_error("MASCVP failed to map init pose to nearest Tammes-128 view.");
        }
        return best_id;
    }

    const std::vector<ViewEntry>& allViews() const {
        if (!all_views_cache_.has_value()) {
            all_views_cache_ = candidateViewSpace();
        }
        return *all_views_cache_;
    }

    static Json::Value readJsonFile(const std::filesystem::path& path) {
        std::ifstream fin(path, std::ios::binary);
        if (!fin) throw std::runtime_error("Failed to open json: " + path.string());
        Json::CharReaderBuilder builder;
        builder["collectComments"] = false;
        Json::Value root;
        std::string errs;
        if (!Json::parseFromStream(builder, fin, &root, &errs)) {
            throw std::runtime_error("Failed to parse json " + path.string() + ": " + errs);
        }
        return root;
    }

    static std::filesystem::path resolveSessionPath(
        const AlgorithmContext& ctx,
        const std::string& path) {
        std::filesystem::path p(path);
        if (p.empty() || p.is_absolute()) return p;
        return ctx.session_dir / p;
    }

    static Pose7d poseFromFrameMeta(const Json::Value& frame_meta) {
        Pose7d pose;
        const Json::Value pose_json = frame_meta["pose"];
        const Json::Value camera_xyz = pose_json["camera_xyz"];
        const Json::Value lookat_xyz = pose_json["lookat_xyz"];
        if (!camera_xyz.isArray() || camera_xyz.size() != 3 ||
            !lookat_xyz.isArray() || lookat_xyz.size() != 3) {
            throw std::runtime_error(
                "frame_meta pose must contain camera_xyz and lookat_xyz arrays.");
        }
        pose.v = {
            camera_xyz[0].asDouble(),
            camera_xyz[1].asDouble(),
            camera_xyz[2].asDouble(),
            lookat_xyz[0].asDouble(),
            lookat_xyz[1].asDouble(),
            lookat_xyz[2].asDouble(),
            pose_json.get("roll_rad", 0.0).asDouble(),
        };
        return pose;
    }

    static Json::Value toJsonArray(const std::vector<int>& xs) {
        Json::Value arr(Json::arrayValue);
        for (int x : xs) arr.append(x);
        return arr;
    }

    static std::string makeStepStem(const AlgorithmContext& ctx) {
        std::ostringstream oss;
        oss << "mascvp_step_" << std::setw(3) << std::setfill('0') << ctx.step_index;
        return oss.str();
    }

    static std::string makeRequestId(const AlgorithmContext& ctx) {
        std::ostringstream oss;
        oss << "mascvp_" << sanitizeRequestIdPart(ctx.uid)
            << "_step" << ctx.step_index;
        return oss.str();
    }

    static std::string sanitizeRequestIdPart(const std::string& value) {
        std::string out;
        out.reserve(value.size());
        for (unsigned char c : value) {
            if ((c >= 'a' && c <= 'z') ||
                (c >= 'A' && c <= 'Z') ||
                (c >= '0' && c <= '9') ||
                c == '_' || c == '-') {
                out.push_back(static_cast<char>(c));
            } else {
                out.push_back('_');
            }
        }
        if (out.empty()) return "unknown";
        return out;
    }

    static double secondsSince(const std::chrono::steady_clock::time_point& start) {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    }
};

}  // namespace objview

#endif  // OBJVIEWBENCH_MASCVP_PLANNING_NETWORK_ALGORITHM_H_
