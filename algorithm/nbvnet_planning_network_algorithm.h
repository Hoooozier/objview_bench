#ifndef OBJVIEWBENCH_NBVNET_PLANNING_NETWORK_ALGORITHM_H_
#define OBJVIEWBENCH_NBVNET_PLANNING_NETWORK_ALGORITHM_H_

#include <algorithm>
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

struct NbvnetPlanningNetworkConfig {
    std::string views_path = "../Tammes_sphere/128_xyz.txt";
    double view_radius = 3.0;
    int grid_dim = 64;
    double map_bbox_min = -1.0;
    double map_bbox_max = 1.0;
    double unknown_occ = 0.5;
    std::string service_name = "nbvnet";
    int topk = 5;
    bool debug_save = false;
    bool debug_save_ot = false;
    std::string debug_dir = "nbvnet_planning_network_debug";
    bool silent = true;
};

class NbvnetPlanningNetworkAlgorithm : public Algorithm {
public:
    explicit NbvnetPlanningNetworkAlgorithm(NbvnetPlanningNetworkConfig cfg)
        : cfg_(std::move(cfg)), map_(computeResolution(cfg_)) {
        if (cfg_.grid_dim <= 0) {
            throw std::runtime_error("NBVNet grid_dim must be positive.");
        }
        if (cfg_.map_bbox_min >= cfg_.map_bbox_max) {
            throw std::runtime_error("NBVNet map bbox min must be smaller than max.");
        }
        if (cfg_.view_radius <= 0.0) {
            throw std::runtime_error("NBVNet view_radius must be positive.");
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
        const ObservationFrame observation = readObservationFrame(ctx);
        const auto update_start = std::chrono::steady_clock::now();
        updateMapFromObservation(ctx, observation);
        const double map_update_sec = secondsSince(update_start);

        const auto grid_start = std::chrono::steady_clock::now();
        double grid_min = 0.0;
        double grid_max = 0.0;
        const std::vector<float> grid = buildDenseOccupancyGrid(&grid_min, &grid_max);
        const double grid_compute_sec = secondsSince(grid_start);

        std::filesystem::path input_npz;
        double npz_write_elapsed_sec = 0.0;
        {
            const auto write_start = std::chrono::steady_clock::now();
            input_npz = saveInputNpz(ctx, grid);
            npz_write_elapsed_sec = secondsSince(write_start);
        }

        PlanningNetworkRpcClient client(
            service_root_,
            ctx.interaction_wait_timeout_sec,
            ctx.interaction_poll_interval_sec);
        const PlanningNetworkInferResult infer =
            client.infer(makeRequestId(ctx), input_npz, cfg_.topk);
        const Json::Value result = infer.result;
        const double service_runtime_sec =
            result["runtime"].get("total_sec", 0.0).asDouble();

        const auto check_start = std::chrono::steady_clock::now();
        const int pred = readBestIndex(result);
        DecisionCheck check = checkPrediction(ctx, pred);
        const double check_sec = secondsSince(check_start);

        const double billable_runtime_sec =
            map_update_sec + grid_compute_sec + service_runtime_sec + check_sec;

        Json::Value debug = makeDebugJson(
            ctx,
            input_npz,
            infer,
            result,
            pred,
            check,
            grid_min,
            grid_max,
            map_update_sec,
            grid_compute_sec,
            npz_write_elapsed_sec,
            service_runtime_sec,
            infer.rpc_elapsed_sec,
            check_sec,
            billable_runtime_sec);
        saveDebugArtifacts(ctx, debug);

        if (!check.ok) {
            if (!cfg_.silent) {
                std::cout << "[nbvnet] stop=" << check.stop_reason
                          << " detail=" << check.stop_detail
                          << " pred=" << pred << std::endl;
            }
            return AlgorithmDecision::stop(check.stop_reason)
                .withRuntime(billable_runtime_sec);
        }

        visited_128_.insert(pred);
        if (!cfg_.silent) {
            std::cout << "[nbvnet] step=" << ctx.step_index
                      << " pred=" << pred
                      << " runtime_sec=" << billable_runtime_sec << std::endl;
        }
        return AlgorithmDecision::move(pred).withRuntime(billable_runtime_sec);
    }

private:
    struct ObservationFrame {
        Json::Value frame_meta;
        Pose7d pose;
        CameraIntrinsics intrinsics;
        std::filesystem::path depth_path;
        std::filesystem::path mask_path;
    };

    struct DecisionCheck {
        bool ok = false;
        std::string stop_reason = "next_view_infeasible";
        std::string stop_detail = "unknown";
        std::vector<int> feasible_remaining_ids;
    };

    NbvnetPlanningNetworkConfig cfg_;
    octomap::ColorOcTree map_;
    Json::Value episode_config_;
    std::filesystem::path session_dir_;
    std::filesystem::path service_root_;
    std::filesystem::path debug_dir_abs_;
    mutable std::optional<std::vector<ViewEntry>> all_views_cache_;
    std::unordered_set<int> observed_step_indices_;
    std::set<int> visited_128_;

    static double computeResolution(const NbvnetPlanningNetworkConfig& cfg) {
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

    ObservationFrame readObservationFrame(const AlgorithmContext& ctx) const {
        ObservationFrame observation;
        observation.pose = ctx.current_pose;

        const std::string frame_meta_rel =
            ctx.observation_manifest.get("frame_meta_path", "").asString();
        if (frame_meta_rel.empty()) {
            throw std::runtime_error("NBVNet requires frame_meta_path in observation_manifest.");
        }
        observation.frame_meta = readJsonFile(resolveSessionPath(ctx, frame_meta_rel));
        observation.pose = poseFromFrameMeta(observation.frame_meta);
        observation.intrinsics = parseCameraIntrinsics(observation.frame_meta);

        const std::string depth_rel =
            ctx.observation_manifest.get("depth_path", "").asString();
        const std::string mask_rel =
            ctx.observation_manifest.get("mask_path", "").asString();
        if (depth_rel.empty() || mask_rel.empty()) {
            throw std::runtime_error("NBVNet requires depth_path and mask_path.");
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

    std::filesystem::path saveInputNpz(
        const AlgorithmContext& ctx,
        const std::vector<float>& grid) const {
        std::filesystem::path dir = debug_dir_abs_.empty()
            ? (service_root_ / "inputs")
            : debug_dir_abs_;
        std::filesystem::create_directories(dir);
        const std::filesystem::path path = dir / (makeStepStem(ctx) + ".npz");

        const std::vector<std::size_t> grid_shape{
            static_cast<size_t>(cfg_.grid_dim),
            static_cast<size_t>(cfg_.grid_dim),
            static_cast<size_t>(cfg_.grid_dim)};
        const std::vector<std::size_t> scalar_shape{1};
        const int32_t step_i32 = static_cast<int32_t>(ctx.step_index);
        cnpy::npz_save(path.string(), "grid", grid.data(), grid_shape, "w");
        cnpy::npz_save(path.string(), "step_id", &step_i32, scalar_shape, "a");
        return path;
    }

    static int readBestIndex(const Json::Value& result) {
        const Json::Value best = result["best_index"];
        if (!best.isArray() || best.empty() || !best[0].isInt()) {
            return -1;
        }
        return best[0].asInt();
    }

    DecisionCheck checkPrediction(const AlgorithmContext& ctx, int pred) const {
        DecisionCheck check;
        check.feasible_remaining_ids.reserve(ctx.candidate_views.size());
        std::unordered_set<int> feasible_ids;
        feasible_ids.reserve(ctx.candidate_views.size());
        for (const auto& v : ctx.candidate_views) {
            feasible_ids.insert(v.view_idx);
            check.feasible_remaining_ids.push_back(v.view_idx);
        }

        const int n = static_cast<int>(allViews().size());
        if (pred < 0) {
            check.stop_reason = "next_view_infeasible";
            check.stop_detail = "missing_best_index";
            return check;
        }
        if (pred >= n) {
            check.stop_reason = "next_view_infeasible";
            check.stop_detail = "pred_out_of_range";
            return check;
        }
        if (visited_128_.count(pred)) {
            check.stop_reason = "next_view_visited";
            check.stop_detail = "pred_already_visited";
            return check;
        }
        if (!feasible_ids.count(pred)) {
            check.stop_reason = "next_view_infeasible";
            check.stop_detail = "pred_not_feasible";
            return check;
        }
        check.ok = true;
        check.stop_reason = "";
        check.stop_detail = "move";
        return check;
    }

    Json::Value makeDebugJson(
        const AlgorithmContext& ctx,
        const std::filesystem::path& input_npz,
        const PlanningNetworkInferResult& infer,
        const Json::Value& result,
        int pred,
        const DecisionCheck& check,
        double grid_min,
        double grid_max,
        double map_update_sec,
        double grid_compute_sec,
        double npz_write_elapsed_sec,
        double service_runtime_sec,
        double rpc_elapsed_sec,
        double check_sec,
        double billable_runtime_sec) const {
        Json::Value debug(Json::objectValue);
        debug["uid"] = ctx.uid;
        debug["episode_id"] = ctx.episode_id;
        debug["step_index"] = ctx.step_index;
        debug["input_npz"] = input_npz.string();
        debug["request_id"] = infer.request_id;
        debug["request_path"] = infer.request_path.string();
        debug["response_path"] = infer.response_path.string();
        debug["backend"] = result.get("backend", "").asString();
        debug["pred_best_index"] = pred;
        debug["decision"] = check.ok ? "move" : "stop";
        debug["stop_reason"] = check.stop_reason;
        debug["stop_detail"] = check.stop_detail;
        debug["visited_128_ids"] = toJsonArray(std::vector<int>(visited_128_.begin(), visited_128_.end()));
        debug["feasible_remaining_ids"] = toJsonArray(check.feasible_remaining_ids);
        debug["topk_indices"] = result["topk_indices"];
        debug["grid_min"] = grid_min;
        debug["grid_max"] = grid_max;
        debug["map_update_sec"] = map_update_sec;
        debug["grid_compute_sec"] = grid_compute_sec;
        debug["npz_write_elapsed_sec"] = npz_write_elapsed_sec;
        debug["service_runtime_total_sec"] = service_runtime_sec;
        debug["rpc_elapsed_sec"] = rpc_elapsed_sec;
        debug["check_sec"] = check_sec;
        debug["billable_runtime_sec"] = billable_runtime_sec;
        return debug;
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
            std::ofstream fout(debug_dir_abs_ / (stem + "_decision.json"), std::ios::binary);
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
            throw std::runtime_error("Failed to write NBVNet observed-only octomap: " +
                                     observed_path.string());
        }

        const std::filesystem::path with_unknown_path = debug_dir_abs_ / (stem + "__with_unknown.ot");
        if (!map_.write(with_unknown_path.string())) {
            throw std::runtime_error("Failed to write NBVNet with-unknown octomap: " +
                                     with_unknown_path.string());
        }
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
        oss << "nbvnet_step_" << std::setw(3) << std::setfill('0') << ctx.step_index;
        return oss.str();
    }

    static std::string makeRequestId(const AlgorithmContext& ctx) {
        std::ostringstream oss;
        oss << "nbvnet_" << sanitizeRequestIdPart(ctx.uid)
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

#endif  // OBJVIEWBENCH_NBVNET_PLANNING_NETWORK_ALGORITHM_H_
