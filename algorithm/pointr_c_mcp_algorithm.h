#ifndef OBJVIEWBENCH_POINTR_C_MCP_ALGORITHM_H_
#define OBJVIEWBENCH_POINTR_C_MCP_ALGORITHM_H_

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
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <Eigen/Dense>
#include <json/json.h>
#include <octomap/ColorOcTree.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <cuda_runtime.h>

#include <gurobi_c++.h>

#include "objview_algorithm.h"
#include "objview_observation_io.h"
#include "objview_pointcloud_io.h"
#include "objview_shape_completion_client.h"
#include "objview_view_io.h"
#include "cuda_raycaster.h"

namespace objview {

struct PointrCMcpAlgorithmConfig {
    int budget = 30;
    std::string views_path = "../Tammes_sphere/360_xyz.txt";
    double view_radius = 3.0;
    double obstacle_radius = 1.0;
    double partial_voxel_leaf_size = 0.015625;
    double planning_voxel_size = 0.03125;
    double pointcloud_bbox_min = -1.0;
    double pointcloud_bbox_max = 1.0;
    int sor_mean_k = 16;
    double sor_stddev_mul = 1.5;
    double visibility_max_range = 6.0;
    std::string visibility_mode = "inverse_cuda";
    int min_visible_views = 1;
    double mcp_time_limit_sec = 10.0;
    std::string completion_backend_name = "PoinTr-C";
    bool debug_save_intermediate = false;
    std::string debug_dir = "pointr_c_mcp_debug";
    bool silent = true;
};

class PointrCMcpAlgorithm : public Algorithm {
public:
    explicit PointrCMcpAlgorithm(PointrCMcpAlgorithmConfig cfg)
        : cfg_(std::move(cfg)) {
        if (cfg_.budget < 0) {
            throw std::runtime_error("PoinTr-C MCP budget must be non-negative.");
        }
        if (cfg_.partial_voxel_leaf_size <= 0.0) {
            throw std::runtime_error("PoinTr-C MCP partial_voxel_leaf_size must be positive.");
        }
        if (cfg_.planning_voxel_size <= 0.0) {
            throw std::runtime_error("PoinTr-C MCP planning_voxel_size must be positive.");
        }
        if (cfg_.pointcloud_bbox_min >= cfg_.pointcloud_bbox_max) {
            throw std::runtime_error("PoinTr-C MCP pointcloud bbox min must be smaller than max.");
        }
        if (cfg_.sor_mean_k <= 0) {
            throw std::runtime_error("PoinTr-C MCP sor_mean_k must be positive.");
        }
        if (cfg_.sor_stddev_mul <= 0.0) {
            throw std::runtime_error("PoinTr-C MCP sor_stddev_mul must be positive.");
        }
        if (cfg_.visibility_max_range <= 0.0) {
            throw std::runtime_error("PoinTr-C MCP visibility_max_range must be positive.");
        }
        if (cfg_.visibility_mode != "inverse_cpu" && cfg_.visibility_mode != "inverse_cuda") {
            throw std::runtime_error(
                "PoinTr-C MCP visibility_mode must be inverse_cpu or inverse_cuda.");
        }
        if (cfg_.min_visible_views <= 0) {
            throw std::runtime_error("PoinTr-C MCP min_visible_views must be positive.");
        }
        if (cfg_.mcp_time_limit_sec <= 0.0) {
            throw std::runtime_error("PoinTr-C MCP mcp_time_limit_sec must be positive.");
        }
        if (cfg_.visibility_mode == "inverse_cuda") {
            const cudaError_t err = cudaFree(nullptr);
            if (err != cudaSuccess) {
                throw std::runtime_error(
                    std::string("PoinTr-C MCP failed to warm up CUDA runtime: ") +
                    cudaGetErrorString(err));
            }
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
        ensureStartViewInitialized(ctx);

        if (planned_) {
            if (cursor_ >= planned_path_.size()) {
                return AlgorithmDecision::stop(terminal_stop_reason_).withRuntime(0.0);
            }
            const int next_view_id = planned_path_[cursor_++];
            chosen_view_ids_.insert(next_view_id);
            return AlgorithmDecision::move(next_view_id).withRuntime(0.0);
        }

        const auto step_start = std::chrono::steady_clock::now();
        const ObservationFrame observation = readObservationFrame(ctx);
        mergeObservationIntoPartial(observation);

        if (!bootstrap_selected_) {
            if (ctx.candidate_views.empty()) {
                const double runtime_sec = secondsSince(step_start);
                return AlgorithmDecision::stop("candidate_exhausted").withRuntime(runtime_sec);
            }
            bootstrap_view_id_ = selectBootstrapFp(ctx);
            if (bootstrap_view_id_ < 0) {
                const double runtime_sec = secondsSince(step_start);
                return AlgorithmDecision::stop("candidate_exhausted").withRuntime(runtime_sec);
            }
            bootstrap_selected_ = true;
            chosen_view_ids_.insert(bootstrap_view_id_);
            const double runtime_sec = secondsSince(step_start);
            return AlgorithmDecision::move(bootstrap_view_id_).withRuntime(runtime_sec);
        }

        if (remainingBudgetAfterBootstrap() <= 0) {
            planned_ = true;
            terminal_stop_reason_ = "plan_end";
            return AlgorithmDecision::stop(terminal_stop_reason_).withRuntime(0.0);
        }

        double reported_runtime_sec = 0.0;
        if (!planned_) {
            const PlanResult plan = makePlan(ctx, step_start);
            planned_path_ = plan.path;
            terminal_stop_reason_ = plan.terminal_stop_reason;
            planned_ = true;
            cursor_ = 0;
            reported_runtime_sec = plan.billable_runtime_sec;
            if (!cfg_.silent) {
                std::cout << "PoinTr-C MCP path:";
                for (int vid : planned_path_) std::cout << " " << vid;
                std::cout << " terminal=" << terminal_stop_reason_
                          << " runtime_sec=" << reported_runtime_sec << std::endl;
            }
        }

        if (cursor_ >= planned_path_.size()) {
            return AlgorithmDecision::stop(terminal_stop_reason_).withRuntime(reported_runtime_sec);
        }

        const int next_view_id = planned_path_[cursor_++];
        chosen_view_ids_.insert(next_view_id);
        return AlgorithmDecision::move(next_view_id).withRuntime(reported_runtime_sec);
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

    struct CameraBasis {
        Eigen::Vector3d X;
        Eigen::Vector3d Y;
        Eigen::Vector3d Z;
        double fx = 0.0;
        double fy = 0.0;
        double cx = 0.0;
        double cy = 0.0;
        int width = 0;
        int height = 0;
    };

    struct PlanResult {
        std::vector<int> path;
        double billable_runtime_sec = 0.0;
        std::string terminal_stop_reason = "plan_end";
        std::vector<int> mcp_selected_view_ids;
        std::vector<int> fill_selected_view_ids;
        int budget_total = 0;
        int budget_remaining = 0;
        std::size_t mcp_covered_target_count = 0;
        std::string fill_stop_reason = "budget_filled";
    };

    using KeySet = std::unordered_set<octomap::OcTreeKey, octomap::OcTreeKey::KeyHash>;

    PointrCMcpAlgorithmConfig cfg_;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr partial_cloud_{
        new pcl::PointCloud<pcl::PointXYZRGB>()};
    mutable std::optional<std::vector<ViewEntry>> all_views_cache_;
    int start_view_id_ = -1;
    int bootstrap_view_id_ = -1;
    bool bootstrap_selected_ = false;
    bool planned_ = false;
    std::vector<int> planned_path_;
    size_t cursor_ = 0;
    std::string terminal_stop_reason_ = "plan_end";
    std::set<int> chosen_view_ids_;
    std::filesystem::path debug_dir_abs_;

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

    static std::filesystem::path resolveSessionPath(const AlgorithmContext& ctx,
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
            throw std::runtime_error("PoinTr-C MCP requires frame_meta_path in observation_manifest.");
        }

        const std::string depth_rel =
            ctx.observation_manifest.get("depth_path", "").asString();
        const std::string mask_rel =
            ctx.observation_manifest.get("mask_path", "").asString();
        observation.has_depth = !depth_rel.empty() && !mask_rel.empty();
        if (!observation.has_depth) {
            throw std::runtime_error("PoinTr-C MCP requires depth_path and mask_path.");
        }
        observation.depth_path = resolveSessionPath(ctx, depth_rel);
        observation.mask_path = resolveSessionPath(ctx, mask_rel);
        return observation;
    }

    void ensureStartViewInitialized(const AlgorithmContext& ctx) {
        if (start_view_id_ >= 0) return;
        start_view_id_ = ctx.episode_config["start_state"].get("start_view_id", -1).asInt();
        if (start_view_id_ >= 0) chosen_view_ids_.insert(start_view_id_);
        if (cfg_.debug_save_intermediate && debug_dir_abs_.empty()) {
            std::filesystem::path p(cfg_.debug_dir);
            debug_dir_abs_ = p.is_absolute() ? p : (ctx.session_dir / p);
            std::filesystem::create_directories(debug_dir_abs_);
        }
    }

    void mergeObservationIntoPartial(const ObservationFrame& observation) {
        const DepthImage depth = loadDepthNpz(observation.depth_path);
        const MaskImage mask = loadMaskImage(observation.mask_path);
        const Eigen::Matrix4d camera_to_world =
            parseMatrix4d(observation.frame_meta["camera_to_world"], "camera_to_world");
        const octomap::Pointcloud new_points = backprojectDepthToWorldPointcloud(
            depth,
            mask,
            observation.intrinsics,
            camera_to_world,
            cfg_.pointcloud_bbox_min,
            cfg_.pointcloud_bbox_max);

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr merged(new pcl::PointCloud<pcl::PointXYZRGB>());
        merged->reserve(partial_cloud_->size() + new_points.size());
        *merged = *partial_cloud_;
        for (size_t i = 0; i < new_points.size(); ++i) {
            const auto& p = new_points.getPoint(static_cast<unsigned int>(i));
            pcl::PointXYZRGB q;
            q.x = p.x();
            q.y = p.y();
            q.z = p.z();
            q.r = 255;
            q.g = 255;
            q.b = 255;
            merged->push_back(q);
        }

        pcl::VoxelGrid<pcl::PointXYZRGB> voxel_grid;
        voxel_grid.setInputCloud(merged);
        const float leaf = static_cast<float>(cfg_.partial_voxel_leaf_size);
        voxel_grid.setLeafSize(leaf, leaf, leaf);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZRGB>());
        voxel_grid.filter(*filtered);
        filtered->width = static_cast<uint32_t>(filtered->size());
        filtered->height = 1;
        filtered->is_dense = false;
        partial_cloud_ = filtered;
    }

    pcl::PointCloud<pcl::PointXYZRGB> loadCompletionPointCloud(
        const std::filesystem::path& path) const {
        pcl::PointCloud<pcl::PointXYZRGB> cloud;
        if (pcl::io::loadPCDFile<pcl::PointXYZRGB>(path.string(), cloud) == 0) {
            return cloud;
        }

        pcl::PointCloud<pcl::PointXYZ> xyz_cloud;
        if (pcl::io::loadPCDFile<pcl::PointXYZ>(path.string(), xyz_cloud) != 0) {
            throw std::runtime_error("Failed to read completion point cloud: " + path.string());
        }
        cloud.reserve(xyz_cloud.size());
        for (const auto& p : xyz_cloud.points) {
            pcl::PointXYZRGB q;
            q.x = p.x;
            q.y = p.y;
            q.z = p.z;
            q.r = 0;
            q.g = 255;
            q.b = 0;
            cloud.push_back(q);
        }
        cloud.width = static_cast<uint32_t>(cloud.size());
        cloud.height = 1;
        cloud.is_dense = false;
        return cloud;
    }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr filterCompletionLightSor(
        const pcl::PointCloud<pcl::PointXYZRGB>& cloud) const {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr input(new pcl::PointCloud<pcl::PointXYZRGB>(cloud));
        if (input->size() < static_cast<size_t>(cfg_.sor_mean_k)) return input;
        pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
        sor.setInputCloud(input);
        sor.setMeanK(cfg_.sor_mean_k);
        sor.setStddevMulThresh(cfg_.sor_stddev_mul);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZRGB>());
        sor.filter(*filtered);
        if (filtered->empty()) return input;
        filtered->width = static_cast<uint32_t>(filtered->size());
        filtered->height = 1;
        filtered->is_dense = false;
        return filtered;
    }

    bool pointInsideBbox(const Eigen::Vector3d& p) const {
        const double minv = cfg_.pointcloud_bbox_min;
        const double maxv = cfg_.pointcloud_bbox_max;
        return p.x() >= minv && p.x() <= maxv &&
               p.y() >= minv && p.y() <= maxv &&
               p.z() >= minv && p.z() <= maxv;
    }

    KeySet voxelizePointCloudToKeys(const pcl::PointCloud<pcl::PointXYZRGB>& cloud,
                                    double voxel_size) const {
        octomap::ColorOcTree key_tree(voxel_size);
        KeySet keys;
        keys.reserve(cloud.size());
        for (const auto& p : cloud.points) {
            if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) continue;
            const Eigen::Vector3d v(p.x, p.y, p.z);
            if (!pointInsideBbox(v)) continue;
            octomap::OcTreeKey key;
            if (!key_tree.coordToKeyChecked(octomap::point3d(p.x, p.y, p.z), key)) continue;
            keys.insert(key);
        }
        return keys;
    }

    KeySet buildCoverageReferenceVoxelSet(const KeySet& current_keys,
                                          const KeySet& predicted_keys) const {
        KeySet reference;
        reference.reserve(current_keys.size() + predicted_keys.size());
        reference.insert(current_keys.begin(), current_keys.end());
        reference.insert(predicted_keys.begin(), predicted_keys.end());
        return reference;
    }

    KeySet buildCoverageTargetVoxelSet(const KeySet& current_keys,
                                       const KeySet& predicted_keys) const {
        KeySet target;
        target.reserve(predicted_keys.size());
        for (const auto& key : predicted_keys) {
            if (current_keys.find(key) == current_keys.end()) {
                target.insert(key);
            }
        }
        return target;
    }

    octomap::ColorOcTree buildReferenceTree(const KeySet& reference_keys) const {
        octomap::ColorOcTree tree(cfg_.planning_voxel_size);
        for (const auto& key : reference_keys) {
            tree.setNodeValue(key, tree.getProbHitLog(), true);
        }
        tree.updateInnerOccupancy();
        return tree;
    }

    static std::vector<octomap::OcTreeKey> collectAllOccupiedKeys(const octomap::ColorOcTree& tree) {
        std::vector<octomap::OcTreeKey> keys;
        for (auto it = tree.begin_leafs(), end = tree.end_leafs(); it != end; ++it) {
            if (!tree.isNodeOccupied(*it)) continue;
            keys.push_back(it.getKey());
        }
        return keys;
    }

    CameraBasis makeCameraBasis(const Eigen::Vector3d& camera_pos,
                                const Eigen::Vector3d& look_at,
                                const CameraIntrinsics& intrinsics) const {
        CameraBasis cam;
        cam.fx =
            0.5 * static_cast<double>(intrinsics.width) / std::tan(0.5 * intrinsics.fov_x_rad);
        cam.fy =
            0.5 * static_cast<double>(intrinsics.height) / std::tan(0.5 * intrinsics.fov_y_rad);
        cam.cx = intrinsics.principal_x;
        cam.cy = intrinsics.principal_y;
        cam.width = intrinsics.width;
        cam.height = intrinsics.height;
        cam.Z = (look_at - camera_pos).normalized();
        if ((cam.Z - Eigen::Vector3d(0.0, 0.0, -1.0)).norm() < 1e-6) {
            cam.Z = Eigen::Vector3d(1e-8, 1e-8, -1.0).normalized();
        }
        if ((cam.Z - Eigen::Vector3d(0.0, 0.0, 1.0)).norm() < 1e-6) {
            cam.Z = Eigen::Vector3d(1e-8, 1e-8, 1.0).normalized();
        }
        cam.X = ((-cam.Z).cross(Eigen::Vector3d(0.0, 0.0, 1.0))).normalized();
        cam.Y = (cam.X.cross(-cam.Z)).normalized();
        return cam;
    }

    bool projectWorldPointToPixel(const Eigen::Vector3d& world_pt,
                                  const Eigen::Vector3d& camera_pos,
                                  const CameraBasis& cam,
                                  int& u,
                                  int& v,
                                  double& z_cam) const {
        const Eigen::Vector3d d = world_pt - camera_pos;
        const double x_cam = d.dot(cam.X);
        const double y_cam = d.dot(cam.Y);
        z_cam = d.dot(cam.Z);
        if (z_cam <= 1e-9) return false;

        const double u_f = cam.fx * (x_cam / z_cam) + cam.cx;
        const double v_f = cam.fy * (y_cam / z_cam) + cam.cy;
        if (u_f < 0.0 || u_f >= static_cast<double>(cam.width) ||
            v_f < 0.0 || v_f >= static_cast<double>(cam.height)) {
            return false;
        }
        u = static_cast<int>(std::floor(u_f));
        v = static_cast<int>(std::floor(v_f));
        return true;
    }

    std::vector<octomap::OcTreeKey> computeVisibleKeysInverseCpu(
        const octomap::ColorOcTree& tree,
        const std::vector<octomap::OcTreeKey>& all_keys,
        const Pose7d& pose,
        const CameraIntrinsics& intrinsics) const {
        const Eigen::Vector3d camera_pos(pose.v[0], pose.v[1], pose.v[2]);
        const Eigen::Vector3d look_at(0.0, 0.0, 0.0);
        const CameraBasis cam = makeCameraBasis(camera_pos, look_at, intrinsics);
        KeySet visible_keys;
        visible_keys.reserve(all_keys.size() / 8 + 1);

        for (const auto& candidate_key : all_keys) {
            const octomap::point3d coord = tree.keyToCoord(candidate_key);
            const Eigen::Vector3d world_pt(coord.x(), coord.y(), coord.z());

            int u = -1;
            int v = -1;
            double z_cam = 0.0;
            if (!projectWorldPointToPixel(world_pt, camera_pos, cam, u, v, z_cam)) {
                continue;
            }
            if (z_cam > cfg_.visibility_max_range) continue;

            const Eigen::Vector3d dir = pixelToWorldRay(u, v, cam);
            octomap::point3d end_pt;
            const bool hit = tree.castRay(
                octomap::point3d(camera_pos.x(), camera_pos.y(), camera_pos.z()),
                octomap::point3d(dir.x(), dir.y(), dir.z()),
                end_pt,
                true,
                cfg_.visibility_max_range);
            if (!hit) continue;

            octomap::OcTreeKey hit_key;
            if (!tree.coordToKeyChecked(end_pt, hit_key)) continue;
            auto* node = tree.search(hit_key);
            if (node == nullptr || !tree.isNodeOccupied(node)) continue;
            visible_keys.insert(hit_key);
        }

        return std::vector<octomap::OcTreeKey>(visible_keys.begin(), visible_keys.end());
    }

    std::vector<octomap::OcTreeKey> computeVisibleKeysInverseCuda(
        const octomap::ColorOcTree& tree,
        octomap::CudaRayCaster& raycaster,
        const std::vector<octomap::OcTreeKey>& all_keys,
        const Pose7d& pose,
        const CameraIntrinsics& intrinsics) const {
        const Eigen::Vector3d camera_pos(pose.v[0], pose.v[1], pose.v[2]);
        const Eigen::Vector3d look_at(0.0, 0.0, 0.0);
        const CameraBasis cam = makeCameraBasis(camera_pos, look_at, intrinsics);

        std::vector<octomap::point3d> origins;
        std::vector<octomap::point3d> dirs;
        std::vector<double> max_ranges;
        origins.reserve(all_keys.size());
        dirs.reserve(all_keys.size());
        max_ranges.reserve(all_keys.size());

        for (const auto& key : all_keys) {
            const octomap::point3d coord = tree.keyToCoord(key);
            int u = 0;
            int v = 0;
            double z_cam = 0.0;
            if (!projectWorldPointToPixel(
                    Eigen::Vector3d(coord.x(), coord.y(), coord.z()),
                    camera_pos,
                    cam,
                    u,
                    v,
                    z_cam)) {
                continue;
            }
            if (z_cam > cfg_.visibility_max_range) continue;
            const Eigen::Vector3d dir = pixelToWorldRay(u, v, cam);
            origins.emplace_back(camera_pos.x(), camera_pos.y(), camera_pos.z());
            dirs.emplace_back(dir.x(), dir.y(), dir.z());
            max_ranges.push_back(cfg_.visibility_max_range);
        }

        if (origins.empty()) return {};

        std::vector<octomap::point3d> end_pts;
        bool* hits = raycaster.castRay(origins, dirs, &end_pts, true, max_ranges);
        if (hits == nullptr) {
            throw std::runtime_error("CudaRayCaster::castRay returned null hits pointer.");
        }

        std::unordered_set<octomap::OcTreeKey, octomap::OcTreeKey::KeyHash> visible_set;
        visible_set.reserve(end_pts.size());
        for (std::size_t i = 0; i < end_pts.size(); ++i) {
            if (!hits[i]) continue;
            octomap::OcTreeKey hit_key;
            if (!tree.coordToKeyChecked(end_pts[i], hit_key)) continue;
            auto* node = tree.search(hit_key);
            if (node == nullptr || !tree.isNodeOccupied(node)) continue;
            visible_set.insert(hit_key);
        }
        delete[] hits;
        return std::vector<octomap::OcTreeKey>(visible_set.begin(), visible_set.end());
    }

    std::vector<octomap::OcTreeKey> computeVisibleKeysForCoverage(
        const octomap::ColorOcTree& tree,
        octomap::CudaRayCaster* cuda_raycaster,
        const std::vector<octomap::OcTreeKey>& all_keys,
        const Pose7d& pose,
        const CameraIntrinsics& intrinsics) const {
        if (cfg_.visibility_mode == "inverse_cuda") {
            if (cuda_raycaster == nullptr) {
                throw std::runtime_error("inverse_cuda requested but CUDA raycaster is null.");
            }
            return computeVisibleKeysInverseCuda(tree, *cuda_raycaster, all_keys, pose, intrinsics);
        }
        return computeVisibleKeysInverseCpu(tree, all_keys, pose, intrinsics);
    }

    std::vector<std::vector<octomap::OcTreeKey>> filterCoveredKeysByMinVisibleViews(
        const std::vector<std::vector<octomap::OcTreeKey>>& covered_keys_per_view,
        std::vector<octomap::OcTreeKey>& kept_universe) const {
        std::unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash> voxel_view_count;
        voxel_view_count.reserve(covered_keys_per_view.size() * 1024);
        for (const auto& view_keys : covered_keys_per_view) {
            for (const auto& key : view_keys) voxel_view_count[key] += 1;
        }

        std::unordered_set<octomap::OcTreeKey, octomap::OcTreeKey::KeyHash> kept_voxels;
        kept_voxels.reserve(voxel_view_count.size());
        for (const auto& kv : voxel_view_count) {
            if (kv.second >= cfg_.min_visible_views) {
                kept_voxels.insert(kv.first);
            }
        }

        kept_universe.assign(kept_voxels.begin(), kept_voxels.end());
        std::vector<std::vector<octomap::OcTreeKey>> filtered;
        filtered.reserve(covered_keys_per_view.size());
        for (const auto& view_keys : covered_keys_per_view) {
            std::vector<octomap::OcTreeKey> kept_keys;
            kept_keys.reserve(view_keys.size());
            for (const auto& key : view_keys) {
                if (kept_voxels.find(key) != kept_voxels.end()) {
                    kept_keys.push_back(key);
                }
            }
            filtered.push_back(std::move(kept_keys));
        }
        return filtered;
    }

    std::vector<octomap::OcTreeKey> filterKeysToTarget(
        const std::vector<octomap::OcTreeKey>& visible_keys,
        const KeySet& target_keys) const {
        std::vector<octomap::OcTreeKey> filtered;
        filtered.reserve(visible_keys.size());
        for (const auto& key : visible_keys) {
            if (target_keys.find(key) != target_keys.end()) {
                filtered.push_back(key);
            }
        }
        return filtered;
    }

    std::vector<int> fillRemainingBudgetFps(
        const std::vector<int>& candidate_view_ids,
        const std::vector<int>& already_selected_view_ids,
        int fill_count) const {
        if (fill_count <= 0) return {};

        std::unordered_map<int, Vec3> pos_map;
        for (const auto& view : allViews()) {
            pos_map.emplace(view.view_idx, cameraPosition(view.pose));
        }

        std::unordered_set<int> selected(already_selected_view_ids.begin(), already_selected_view_ids.end());
        std::vector<int> filled;
        filled.reserve(fill_count);

        auto current_anchor_points = [&]() {
            std::vector<Vec3> anchors;
            anchors.reserve(selected.size());
            for (int vid : selected) {
                const auto it = pos_map.find(vid);
                if (it != pos_map.end()) anchors.push_back(it->second);
            }
            return anchors;
        };

        while (static_cast<int>(filled.size()) < fill_count) {
            const auto anchors = current_anchor_points();
            int best_view_id = -1;
            double best_min_dist_sq = -1.0;
            for (int candidate_view_id : candidate_view_ids) {
                if (selected.find(candidate_view_id) != selected.end()) continue;
                const auto pos_it = pos_map.find(candidate_view_id);
                if (pos_it == pos_map.end()) continue;
                double min_dist_sq = std::numeric_limits<double>::infinity();
                if (anchors.empty()) {
                    min_dist_sq = pos_it->second.squaredNorm();
                } else {
                    for (const auto& anchor : anchors) {
                        min_dist_sq = std::min(min_dist_sq, (pos_it->second - anchor).squaredNorm());
                    }
                }
                if (min_dist_sq > best_min_dist_sq) {
                    best_min_dist_sq = min_dist_sq;
                    best_view_id = candidate_view_id;
                }
            }
            if (best_view_id < 0) break;
            selected.insert(best_view_id);
            filled.push_back(best_view_id);
        }
        return filled;
    }

    PlanResult solveMcpAndOrder(
        const AlgorithmContext& ctx,
        const std::vector<std::vector<octomap::OcTreeKey>>& chosen_cover_sets,
        const std::vector<std::vector<octomap::OcTreeKey>>& candidate_cover_sets,
        const std::vector<int>& candidate_view_ids) const {
        PlanResult result;
        result.budget_total = cfg_.budget;
        result.budget_remaining = std::max(0, remainingBudgetAfterBootstrap());

        std::vector<std::vector<octomap::OcTreeKey>> all_sets = chosen_cover_sets;
        all_sets.insert(all_sets.end(), candidate_cover_sets.begin(), candidate_cover_sets.end());
        std::vector<octomap::OcTreeKey> kept_universe;
        const auto filtered_all = filterCoveredKeysByMinVisibleViews(all_sets, kept_universe);

        const size_t chosen_count = chosen_cover_sets.size();
        std::vector<std::vector<octomap::OcTreeKey>> filtered_chosen(
            filtered_all.begin(),
            filtered_all.begin() + static_cast<long long>(chosen_count));
        std::vector<std::vector<octomap::OcTreeKey>> filtered_candidates(
            filtered_all.begin() + static_cast<long long>(chosen_count),
            filtered_all.end());

        KeySet covered_by_chosen;
        for (const auto& keys : filtered_chosen) {
            covered_by_chosen.insert(keys.begin(), keys.end());
        }

        std::unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash> needed_voxel_id_map;
        needed_voxel_id_map.reserve(kept_universe.size());
        int next_id = 0;
        for (const auto& key : kept_universe) {
            if (covered_by_chosen.find(key) != covered_by_chosen.end()) continue;
            bool has_candidate_cover = false;
            for (const auto& keys : filtered_candidates) {
                for (const auto& candidate_key : keys) {
                    if (sameKey(candidate_key, key)) {
                        has_candidate_cover = true;
                        break;
                    }
                }
                if (has_candidate_cover) break;
            }
            if (!has_candidate_cover) continue;
            needed_voxel_id_map.emplace(key, next_id++);
        }

        const int num_views = static_cast<int>(filtered_candidates.size());
        const int num_voxels = next_id;
        std::vector<std::vector<int>> views_per_voxel(num_voxels);
        for (int local_view_id = 0; local_view_id < num_views; ++local_view_id) {
            for (const auto& key : filtered_candidates[local_view_id]) {
                const auto it = needed_voxel_id_map.find(key);
                if (it == needed_voxel_id_map.end()) continue;
                views_per_voxel[it->second].push_back(local_view_id);
            }
        }

        std::vector<int> mcp_selected_local_ids;
        if (result.budget_remaining > 0 && num_views > 0 && num_voxels > 0) {
            GRBEnv& env = sharedGurobiEnv();
            GRBModel model(env);
            model.set(GRB_IntParam_OutputFlag, cfg_.silent ? 0 : 1);
            model.set(GRB_DoubleParam_TimeLimit, cfg_.mcp_time_limit_sec);

            std::vector<GRBVar> x(num_views);
            for (int i = 0; i < num_views; ++i) {
                x[i] = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, "x_" + std::to_string(i));
            }
            std::vector<GRBVar> y(num_voxels);
            for (int j = 0; j < num_voxels; ++j) {
                y[j] = model.addVar(0.0, 1.0, 1.0, GRB_BINARY, "y_" + std::to_string(j));
            }

            GRBLinExpr objective = 0;
            for (int j = 0; j < num_voxels; ++j) objective += y[j];
            model.setObjective(objective, GRB_MAXIMIZE);

            GRBLinExpr budget_expr = 0;
            for (int i = 0; i < num_views; ++i) budget_expr += x[i];
            model.addConstr(
                budget_expr <= std::min(result.budget_remaining, num_views),
                "budget");

            for (int j = 0; j < num_voxels; ++j) {
                GRBLinExpr cover = 0;
                for (int local_view_id : views_per_voxel[j]) cover += x[local_view_id];
                model.addConstr(y[j] <= cover, "cover_" + std::to_string(j));
            }
            model.optimize();

            const int status = model.get(GRB_IntAttr_Status);
            if (status != GRB_OPTIMAL && status != GRB_SUBOPTIMAL && status != GRB_TIME_LIMIT) {
                throw std::runtime_error("Gurobi failed with status: " + std::to_string(status));
            }

            for (int i = 0; i < num_views; ++i) {
                if (x[i].get(GRB_DoubleAttr_X) > 0.5) {
                    mcp_selected_local_ids.push_back(i);
                    result.mcp_selected_view_ids.push_back(
                        candidate_view_ids[static_cast<size_t>(i)]);
                }
            }
        }

        KeySet mcp_covered_target;
        for (int local_view_id : mcp_selected_local_ids) {
            for (const auto& key : filtered_candidates[local_view_id]) {
                const auto it = needed_voxel_id_map.find(key);
                if (it != needed_voxel_id_map.end()) mcp_covered_target.insert(key);
            }
        }
        result.mcp_covered_target_count = mcp_covered_target.size();

        std::vector<int> fps_anchor_ids;
        fps_anchor_ids.reserve(chosen_view_ids_.size() + result.mcp_selected_view_ids.size());
        for (int vid : chosen_view_ids_) fps_anchor_ids.push_back(vid);
        fps_anchor_ids.insert(
            fps_anchor_ids.end(),
            result.mcp_selected_view_ids.begin(),
            result.mcp_selected_view_ids.end());

        const int fill_needed =
            std::max(0, result.budget_remaining - static_cast<int>(result.mcp_selected_view_ids.size()));
        result.fill_selected_view_ids =
            fillRemainingBudgetFps(candidate_view_ids, fps_anchor_ids, fill_needed);

        std::vector<int> selected_view_ids = result.mcp_selected_view_ids;
        selected_view_ids.insert(
            selected_view_ids.end(),
            result.fill_selected_view_ids.begin(),
            result.fill_selected_view_ids.end());

        if (static_cast<int>(selected_view_ids.size()) < result.budget_remaining) {
            result.terminal_stop_reason = "candidate_exhausted";
            result.fill_stop_reason = "candidate_exhausted";
        } else {
            result.terminal_stop_reason = "plan_end";
            result.fill_stop_reason =
                (fill_needed > 0 ? "budget_filled" : "no_fill_needed");
        }

        result.path = orderPlannedViews(ctx.current_pose, selected_view_ids);
        return result;
    }

    std::vector<int> orderPlannedViews(const Pose7d& current_pose,
                                       const std::vector<int>& selected_view_ids) const {
        if (selected_view_ids.empty()) return {};
        const auto& all_views = allViews();
        std::unordered_map<int, Vec3> pos_map;
        pos_map.reserve(all_views.size());
        for (const auto& v : all_views) {
            pos_map[v.view_idx] = cameraPosition(v.pose);
        }
        std::vector<Vec3> tsp_positions;
        tsp_positions.reserve(selected_view_ids.size() + 1);
        std::vector<int> tsp_ids;
        tsp_ids.reserve(selected_view_ids.size() + 1);
        std::vector<int> local_to_global;
        local_to_global.reserve(selected_view_ids.size() + 1);
        int local_id = 0;
        for (int vid : selected_view_ids) {
            const auto it = pos_map.find(vid);
            if (it == pos_map.end()) {
                throw std::runtime_error("Missing selected view id in candidateViewSpace cache: " +
                                         std::to_string(vid));
            }
            tsp_positions.push_back(it->second);
            tsp_ids.push_back(local_id++);
            local_to_global.push_back(vid);
        }
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
        tsp_cfg.time_limit_sec = -1.0;
        tsp_cfg.silent = cfg_.silent;
        const auto result = solveHamiltonianPath(tsp_cfg);
        if (!result.solved) {
            return selected_view_ids;
        }
        std::vector<int> ordered;
        ordered.reserve(selected_view_ids.size());
        for (int vid : result.path_view_ids) {
            if (vid == virtual_start_id) continue;
            if (vid < 0 || vid >= static_cast<int>(local_to_global.size())) {
                throw std::runtime_error("Local TSP path id is out of range.");
            }
            const int global_view_id = local_to_global[static_cast<size_t>(vid)];
            if (global_view_id < 0) continue;
            ordered.push_back(global_view_id);
        }
        return ordered;
    }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr keySetToCloud(const KeySet& keys) const {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
        cloud->reserve(keys.size());
        octomap::ColorOcTree key_tree(cfg_.planning_voxel_size);
        for (const auto& key : keys) {
            const octomap::point3d coord = key_tree.keyToCoord(key);
            pcl::PointXYZRGB p;
            p.x = coord.x();
            p.y = coord.y();
            p.z = coord.z();
            p.r = 255;
            p.g = 255;
            p.b = 255;
            cloud->push_back(p);
        }
        cloud->width = static_cast<uint32_t>(cloud->size());
        cloud->height = 1;
        cloud->is_dense = false;
        return cloud;
    }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr keyVectorToCloud(
        const std::vector<octomap::OcTreeKey>& keys) const {
        KeySet unique_keys;
        unique_keys.reserve(keys.size());
        for (const auto& key : keys) unique_keys.insert(key);
        return keySetToCloud(unique_keys);
    }

    void saveDebugArtifacts(
        int step_index,
        const std::filesystem::path& partial_path,
        const std::filesystem::path& completed_path,
        const pcl::PointCloud<pcl::PointXYZRGB>& filtered_predicted,
        const KeySet& current_keys,
        const KeySet& predicted_keys,
        const KeySet& target_keys,
        const KeySet& reference_keys,
        const std::vector<ViewEntry>& chosen_views,
        const std::vector<std::vector<octomap::OcTreeKey>>& chosen_reference_cover_sets,
        const std::vector<std::vector<octomap::OcTreeKey>>& chosen_cover_sets,
        const std::vector<int>& candidate_view_ids,
        const std::vector<std::vector<octomap::OcTreeKey>>& candidate_reference_cover_sets,
        const std::vector<std::vector<octomap::OcTreeKey>>& candidate_cover_sets,
        int budget_total,
        int budget_remaining,
        std::size_t mcp_covered_target_count,
        const std::vector<int>& mcp_selected_view_ids,
        const std::vector<int>& fill_selected_view_ids,
        const std::vector<int>& selected_view_ids,
        const std::string& fill_stop_reason,
        const std::string& terminal_stop_reason) const {
        if (!cfg_.debug_save_intermediate || debug_dir_abs_.empty()) return;

        const auto make_name = [&](const std::string& stem) {
            std::ostringstream oss;
            oss << stem << "_step_" << std::setw(3) << std::setfill('0') << step_index;
            return oss.str();
        };

        const auto save_if_nonempty =
            [&](const std::filesystem::path& path,
                const pcl::PointCloud<pcl::PointXYZRGB>& cloud) {
                if (cloud.empty()) return;
                savePointCloudXYZRGBBinary(path, cloud, !cfg_.silent);
            };

        const std::filesystem::path predicted_path =
            debug_dir_abs_ / (make_name("completed_filtered") + ".pcd");
        save_if_nonempty(predicted_path, filtered_predicted);
        save_if_nonempty(
            debug_dir_abs_ / (make_name("current_keys") + ".pcd"),
            *keySetToCloud(current_keys));
        save_if_nonempty(
            debug_dir_abs_ / (make_name("predicted_keys") + ".pcd"),
            *keySetToCloud(predicted_keys));
        save_if_nonempty(
            debug_dir_abs_ / (make_name("target_keys") + ".pcd"),
            *keySetToCloud(target_keys));
        save_if_nonempty(
            debug_dir_abs_ / (make_name("reference") + ".pcd"),
            *keySetToCloud(reference_keys));

        const std::filesystem::path coverage_dir = debug_dir_abs_ / make_name("coverage");
        const std::filesystem::path reference_coverage_dir =
            debug_dir_abs_ / make_name("reference_coverage");
        std::filesystem::create_directories(coverage_dir);
        std::filesystem::create_directories(reference_coverage_dir);
        for (size_t i = 0; i < chosen_views.size() && i < chosen_cover_sets.size(); ++i) {
            std::ostringstream oss;
            oss << "chosen_view_" << std::setw(3) << std::setfill('0') << chosen_views[i].view_idx
                << ".pcd";
            save_if_nonempty(coverage_dir / oss.str(), *keyVectorToCloud(chosen_cover_sets[i]));
        }
        for (size_t i = 0; i < chosen_views.size() && i < chosen_reference_cover_sets.size(); ++i) {
            std::ostringstream oss;
            oss << "chosen_view_" << std::setw(3) << std::setfill('0') << chosen_views[i].view_idx
                << ".pcd";
            save_if_nonempty(
                reference_coverage_dir / oss.str(),
                *keyVectorToCloud(chosen_reference_cover_sets[i]));
        }
        for (size_t i = 0; i < candidate_view_ids.size() && i < candidate_cover_sets.size(); ++i) {
            std::ostringstream oss;
            oss << "candidate_view_" << std::setw(3) << std::setfill('0')
                << candidate_view_ids[i] << ".pcd";
            save_if_nonempty(coverage_dir / oss.str(), *keyVectorToCloud(candidate_cover_sets[i]));
        }
        for (size_t i = 0; i < candidate_view_ids.size() && i < candidate_reference_cover_sets.size();
             ++i) {
            std::ostringstream oss;
            oss << "candidate_view_" << std::setw(3) << std::setfill('0')
                << candidate_view_ids[i] << ".pcd";
            save_if_nonempty(
                reference_coverage_dir / oss.str(),
                *keyVectorToCloud(candidate_reference_cover_sets[i]));
        }

        Json::Value meta;
        meta["step_index"] = step_index;
        meta["partial_path"] = partial_path.string();
        meta["completed_raw_path"] = completed_path.string();
        meta["completed_filtered_path"] = predicted_path.string();
        meta["current_key_count"] = static_cast<Json::UInt64>(current_keys.size());
        meta["predicted_key_count"] = static_cast<Json::UInt64>(predicted_keys.size());
        meta["target_key_count"] = static_cast<Json::UInt64>(target_keys.size());
        meta["reference_key_count"] = static_cast<Json::UInt64>(reference_keys.size());
        meta["coverage_dir"] = coverage_dir.string();
        meta["reference_coverage_dir"] = reference_coverage_dir.string();
        meta["visibility_mode"] = cfg_.visibility_mode;
        meta["budget_total"] = budget_total;
        meta["budget_remaining"] = budget_remaining;
        meta["chosen_view_count"] = static_cast<Json::UInt64>(chosen_views.size());
        meta["mcp_selected_view_count"] =
            static_cast<Json::UInt64>(mcp_selected_view_ids.size());
        meta["fill_selected_view_count"] =
            static_cast<Json::UInt64>(fill_selected_view_ids.size());
        meta["selected_view_count"] = static_cast<Json::UInt64>(selected_view_ids.size());
        meta["mcp_covered_target_count"] =
            static_cast<Json::UInt64>(mcp_covered_target_count);
        meta["fill_stop_reason"] = fill_stop_reason;
        meta["terminal_stop_reason"] = terminal_stop_reason;
        Json::Value candidate_json(Json::arrayValue);
        for (int vid : candidate_view_ids) candidate_json.append(vid);
        meta["candidate_view_ids"] = candidate_json;
        Json::Value mcp_selected_json(Json::arrayValue);
        for (int vid : mcp_selected_view_ids) mcp_selected_json.append(vid);
        meta["mcp_selected_view_ids"] = mcp_selected_json;
        Json::Value fill_selected_json(Json::arrayValue);
        for (int vid : fill_selected_view_ids) fill_selected_json.append(vid);
        meta["fill_selected_view_ids"] = fill_selected_json;
        Json::Value selected_json(Json::arrayValue);
        for (int vid : selected_view_ids) selected_json.append(vid);
        meta["selected_view_ids"] = selected_json;

        Json::StreamWriterBuilder builder;
        builder["indentation"] = "  ";
        std::ofstream fout(debug_dir_abs_ / (make_name("plan") + ".json"));
        if (fout) {
            fout << Json::writeString(builder, meta);
        }
    }

    PlanResult makePlan(const AlgorithmContext& ctx,
                        const std::chrono::steady_clock::time_point& step_start) const {
        if (!ctx.shape_completion.has_value()) {
            throw std::runtime_error("PoinTr-C MCP requires shape_completion capability.");
        }
        const ShapeCompletionCapability& capability = *ctx.shape_completion;
        const ShapeCompletionBackendInfo* backend =
            capability.findBackend(cfg_.completion_backend_name);
        if (backend == nullptr) {
            throw std::runtime_error(
                "PoinTr-C MCP missing completion backend metadata for: " +
                cfg_.completion_backend_name);
        }
        (void)backend;

        const double t_observation_update_sec = secondsSince(step_start);
        const std::filesystem::path partial_path =
            capability.outputs_dir / makeStepFilename("partial", ctx.step_index);
        savePointCloudXYZRGBBinary(partial_path, *partial_cloud_, !cfg_.silent);

        ShapeCompletionClient client(
            capability,
            ctx.interaction_wait_timeout_sec,
            ctx.interaction_poll_interval_sec,
            !cfg_.silent);
        const std::filesystem::path completed_path =
            capability.outputs_dir / makeStepFilename("completed", ctx.step_index);
        const ShapeCompletionResult completion =
            client.completeShape(partial_path, completed_path, makeRequestId(ctx));
        const auto post_completion_start = std::chrono::steady_clock::now();

        const pcl::PointCloud<pcl::PointXYZRGB> predicted_raw =
            loadCompletionPointCloud(completion.completed_pointcloud_path);
        const auto filtered_predicted = filterCompletionLightSor(predicted_raw);
        const KeySet current_keys =
            voxelizePointCloudToKeys(*partial_cloud_, cfg_.planning_voxel_size);
        const KeySet predicted_keys =
            voxelizePointCloudToKeys(*filtered_predicted, cfg_.planning_voxel_size);
        const KeySet target_keys =
            buildCoverageTargetVoxelSet(current_keys, predicted_keys);
        const KeySet reference_keys =
            buildCoverageReferenceVoxelSet(current_keys, predicted_keys);
        const octomap::ColorOcTree tree = buildReferenceTree(reference_keys);
        const std::vector<octomap::OcTreeKey> all_keys = collectAllOccupiedKeys(tree);
        std::unique_ptr<octomap::CudaRayCaster> cuda_raycaster;
        if (cfg_.visibility_mode == "inverse_cuda") {
            cuda_raycaster = std::make_unique<octomap::CudaRayCaster>(tree, !cfg_.silent);
        }

        std::vector<ViewEntry> chosen_views;
        chosen_views.reserve(chosen_view_ids_.size());
        std::vector<ViewEntry> candidate_views = ctx.candidate_views;
        for (int vid : chosen_view_ids_) {
            const auto* entry = findViewById(vid);
            if (entry != nullptr) chosen_views.push_back(*entry);
        }

        CameraIntrinsics intrinsics;
        if (!ctx.observation_manifest.isNull()) {
            const ObservationFrame current_observation = readObservationFrame(ctx);
            intrinsics = current_observation.intrinsics;
        } else {
            throw std::runtime_error("PoinTr-C MCP requires current observation intrinsics.");
        }

        std::vector<std::vector<octomap::OcTreeKey>> chosen_reference_cover_sets;
        chosen_reference_cover_sets.reserve(chosen_views.size());
        std::vector<std::vector<octomap::OcTreeKey>> chosen_cover_sets;
        chosen_cover_sets.reserve(chosen_views.size());
        for (const auto& view : chosen_views) {
            auto visible_reference = computeVisibleKeysForCoverage(
                tree, cuda_raycaster.get(), all_keys, view.pose, intrinsics);
            chosen_reference_cover_sets.push_back(visible_reference);
            chosen_cover_sets.push_back(filterKeysToTarget(visible_reference, target_keys));
        }

        std::vector<std::vector<octomap::OcTreeKey>> candidate_reference_cover_sets;
        candidate_reference_cover_sets.reserve(candidate_views.size());
        std::vector<std::vector<octomap::OcTreeKey>> candidate_cover_sets;
        candidate_cover_sets.reserve(candidate_views.size());
        std::vector<int> candidate_view_ids;
        candidate_view_ids.reserve(candidate_views.size());
        for (const auto& view : candidate_views) {
            candidate_view_ids.push_back(view.view_idx);
            auto visible_reference = computeVisibleKeysForCoverage(
                tree, cuda_raycaster.get(), all_keys, view.pose, intrinsics);
            candidate_reference_cover_sets.push_back(visible_reference);
            candidate_cover_sets.push_back(filterKeysToTarget(visible_reference, target_keys));
        }

        PlanResult result = solveMcpAndOrder(
            ctx,
            chosen_cover_sets,
            candidate_cover_sets,
            candidate_view_ids);

        saveDebugArtifacts(
            ctx.step_index,
            partial_path,
            completion.completed_pointcloud_path,
            *filtered_predicted,
            current_keys,
            predicted_keys,
            target_keys,
            reference_keys,
            chosen_views,
            chosen_reference_cover_sets,
            chosen_cover_sets,
            candidate_view_ids,
            candidate_reference_cover_sets,
            candidate_cover_sets,
            result.budget_total,
            result.budget_remaining,
            result.mcp_covered_target_count,
            result.mcp_selected_view_ids,
            result.fill_selected_view_ids,
            result.path,
            result.fill_stop_reason,
            result.terminal_stop_reason);

        const double t_post_completion_compute_sec =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - post_completion_start)
                .count();
        result.billable_runtime_sec =
            t_observation_update_sec + t_post_completion_compute_sec + completion.runtime.total_sec;
        return result;
    }

    int selectBootstrapFp(const AlgorithmContext& ctx) const {
        int best_view_id = -1;
        double best_cost = -std::numeric_limits<double>::infinity();
        for (const auto& candidate : ctx.candidate_views) {
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

    const std::vector<ViewEntry>& allViews() const {
        if (!all_views_cache_.has_value()) {
            all_views_cache_ = candidateViewSpace();
        }
        return *all_views_cache_;
    }

    const ViewEntry* findViewById(int view_id) const {
        const auto& all = allViews();
        for (const auto& v : all) {
            if (v.view_idx == view_id) return &v;
        }
        return nullptr;
    }

    static std::string makeStepFilename(const std::string& prefix, int step_index) {
        std::ostringstream oss;
        oss << prefix << "_step_" << std::setw(3) << std::setfill('0') << step_index << ".pcd";
        return oss.str();
    }

    static std::string makeRequestId(const AlgorithmContext& ctx) {
        std::ostringstream oss;
        oss << "pointr_c_mcp_" << ctx.uid << "_step" << ctx.step_index;
        return oss.str();
    }

    static double secondsSince(const std::chrono::steady_clock::time_point& start) {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    }

    int remainingBudgetAfterBootstrap() const {
        return cfg_.budget - 1;
    }

    static bool sameKey(const octomap::OcTreeKey& a, const octomap::OcTreeKey& b) {
        return a.k[0] == b.k[0] && a.k[1] == b.k[1] && a.k[2] == b.k[2];
    }

    static Eigen::Vector3d pixelToWorldRay(int u, int v, const CameraBasis& cam) {
        const double x_cam = (static_cast<double>(u) + 0.5 - cam.cx) / cam.fx;
        const double y_cam = (static_cast<double>(v) + 0.5 - cam.cy) / cam.fy;
        return (cam.X * x_cam + cam.Y * y_cam + cam.Z).normalized();
    }
};

}  // namespace objview

#endif  // OBJVIEWBENCH_POINTR_C_MCP_ALGORITHM_H_
