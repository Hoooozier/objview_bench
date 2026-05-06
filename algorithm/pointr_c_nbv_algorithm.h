#ifndef OBJVIEWBENCH_POINTR_C_NBV_ALGORITHM_H_
#define OBJVIEWBENCH_POINTR_C_NBV_ALGORITHM_H_

#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <Eigen/Dense>
#include <json/json.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/surface/convex_hull.h>

#include "objview_algorithm.h"
#include "objview_observation_io.h"
#include "objview_pointcloud_io.h"
#include "objview_shape_completion_client.h"
#include "objview_view_io.h"

namespace objview {

struct PointrCNbvAlgorithmConfig {
    std::string views_path = "../Tammes_sphere/360_xyz.txt";
    double view_radius = 3.0;
    double obstacle_radius = 1.0;
    double partial_voxel_leaf_size = 1.0 / 64.0;
    double planning_voxel_size = 2.0 / 64.0;
    double pointcloud_bbox_min = -1.0;
    double pointcloud_bbox_max = 1.0;
    double hpr_radius_scale = 100.0;
    double tau = 0.95;
    std::string completion_backend_name = "PoinTr-C";
    bool silent = true;
};

class PointrCNbvAlgorithm : public Algorithm {
public:
    explicit PointrCNbvAlgorithm(PointrCNbvAlgorithmConfig cfg)
        : cfg_(std::move(cfg)) {
        if (cfg_.partial_voxel_leaf_size <= 0.0) {
            throw std::runtime_error("PoinTr-C NBV partial_voxel_leaf_size must be positive.");
        }
        if (cfg_.planning_voxel_size <= 0.0) {
            throw std::runtime_error("PoinTr-C NBV planning_voxel_size must be positive.");
        }
        if (cfg_.pointcloud_bbox_min >= cfg_.pointcloud_bbox_max) {
            throw std::runtime_error("PoinTr-C NBV pointcloud bbox min must be smaller than max.");
        }
        if (cfg_.hpr_radius_scale <= 1.0) {
            throw std::runtime_error("PoinTr-C NBV hpr_radius_scale must be greater than 1.");
        }
        if (cfg_.tau <= 0.0 || cfg_.tau > 1.0) {
            throw std::runtime_error("PoinTr-C NBV tau must be in (0, 1].");
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
        const auto step_start = std::chrono::steady_clock::now();
        if (!ctx.shape_completion.has_value()) {
            throw std::runtime_error("PoinTr-C NBV requires shape_completion capability.");
        }

        const ShapeCompletionCapability& capability = *ctx.shape_completion;
        const ShapeCompletionBackendInfo* backend =
            capability.findBackend(cfg_.completion_backend_name);
        if (backend == nullptr) {
            throw std::runtime_error(
                "PoinTr-C NBV missing completion backend metadata for: " +
                cfg_.completion_backend_name);
        }
        (void)backend;

        const ObservationFrame observation = readObservationFrame(ctx);
        mergeObservationIntoPartial(observation);
        const double t_observation_update_sec = secondsSince(step_start);

        if (partial_cloud_->empty()) {
            return AlgorithmDecision::stop("candidate_exhausted")
                .withRuntime(t_observation_update_sec);
        }

        const std::filesystem::path partial_path =
            capability.outputs_dir / makeStepFilename("partial", ctx.step_index);
        if (!cfg_.silent) {
            logResolvedPath("shape_completion.service_root", capability.service_root);
            logResolvedPath("shape_completion.requests_dir", capability.requests_dir);
            logResolvedPath("shape_completion.responses_dir", capability.responses_dir);
            logResolvedPath("shape_completion.outputs_dir", capability.outputs_dir);
            logResolvedPath("shape_completion.ready_path", capability.ready_path);
            logResolvedPath("partial_pointcloud_path", partial_path);
            logParentPathForCreate("partial_pointcloud_path", partial_path);
        }

        // The file-based completion RPC is a benchmark-side service plumbing
        // detail, so its request/response serialization overhead is excluded
        // from the reported algorithm runtime.
        savePointCloudXYZRGBBinary(partial_path, *partial_cloud_, !cfg_.silent);

        ShapeCompletionClient client(
            capability,
            ctx.interaction_wait_timeout_sec,
            ctx.interaction_poll_interval_sec,
            !cfg_.silent);
        const std::string request_id = makeRequestId(ctx);
        const std::filesystem::path completed_path =
            capability.outputs_dir / makeStepFilename("completed", ctx.step_index);
        if (!cfg_.silent) {
            logResolvedPath("completion_output_pointcloud_path", completed_path);
            logParentPathForCreate("completion_output_pointcloud_path", completed_path);
        }
        const ShapeCompletionResult completion =
            client.completeShape(partial_path, completed_path, request_id);
        last_completed_pointcloud_path_ = completion.completed_pointcloud_path;
        const auto post_completion_start = std::chrono::steady_clock::now();

        const VoxelSet observed_voxels = voxelizePointCloud(*partial_cloud_, cfg_.planning_voxel_size);
        const VoxelSet predicted_voxels =
            voxelizePointCloud(loadCompletionPointCloud(completion.completed_pointcloud_path),
                              cfg_.planning_voxel_size);
        const VoxelSet unobserved_predicted_voxels =
            subtractVoxelSets(predicted_voxels, observed_voxels);

        const double t_post_completion_compute_sec =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - post_completion_start)
                .count();
        const double local_planning_compute_sec =
            t_observation_update_sec + t_post_completion_compute_sec;
        const double runtime_sec =
            local_planning_compute_sec + completion.runtime.total_sec;

        if (ctx.candidate_views.empty()) {
            return AlgorithmDecision::stop("candidate_exhausted").withRuntime(runtime_sec);
        }

        const int next_view_id =
            selectNextView(ctx, predicted_voxels, unobserved_predicted_voxels);
        if (next_view_id < 0) {
            return AlgorithmDecision::stop("candidate_exhausted").withRuntime(runtime_sec);
        }
        return AlgorithmDecision::move(next_view_id).withRuntime(runtime_sec);
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

    struct VoxelKey {
        int x = 0;
        int y = 0;
        int z = 0;

        bool operator==(const VoxelKey& other) const {
            return x == other.x && y == other.y && z == other.z;
        }
    };

    struct VoxelKeyHash {
        size_t operator()(const VoxelKey& key) const {
            const size_t hx = std::hash<int>{}(key.x);
            const size_t hy = std::hash<int>{}(key.y);
            const size_t hz = std::hash<int>{}(key.z);
            return hx ^ (hy << 1) ^ (hz << 7);
        }
    };

    struct QuantizedPointKey {
        int64_t x = 0;
        int64_t y = 0;
        int64_t z = 0;

        bool operator==(const QuantizedPointKey& other) const {
            return x == other.x && y == other.y && z == other.z;
        }
    };

    struct QuantizedPointKeyHash {
        size_t operator()(const QuantizedPointKey& key) const {
            const size_t hx = std::hash<int64_t>{}(key.x);
            const size_t hy = std::hash<int64_t>{}(key.y);
            const size_t hz = std::hash<int64_t>{}(key.z);
            return hx ^ (hy << 1) ^ (hz << 7);
        }
    };

    struct CandidateScore {
        int view_id = -1;
        double information_gain = 0.0;
        double motion_cost = std::numeric_limits<double>::infinity();
    };

    using VoxelSet = std::unordered_set<VoxelKey, VoxelKeyHash>;

    PointrCNbvAlgorithmConfig cfg_;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr partial_cloud_{
        new pcl::PointCloud<pcl::PointXYZRGB>()};
    std::filesystem::path last_completed_pointcloud_path_;

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
            throw std::runtime_error("PoinTr-C NBV requires frame_meta_path in observation_manifest.");
        }

        const std::string depth_rel =
            ctx.observation_manifest.get("depth_path", "").asString();
        const std::string mask_rel =
            ctx.observation_manifest.get("mask_path", "").asString();
        observation.has_depth = !depth_rel.empty() && !mask_rel.empty();
        if (!observation.has_depth) {
            throw std::runtime_error("PoinTr-C NBV requires depth_path and mask_path.");
        }
        observation.depth_path = resolveSessionPath(ctx, depth_rel);
        observation.mask_path = resolveSessionPath(ctx, mask_rel);
        return observation;
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

    bool pointInsideBbox(const Eigen::Vector3d& p) const {
        const double minv = cfg_.pointcloud_bbox_min;
        const double maxv = cfg_.pointcloud_bbox_max;
        return p.x() >= minv && p.x() <= maxv &&
               p.y() >= minv && p.y() <= maxv &&
               p.z() >= minv && p.z() <= maxv;
    }

    std::optional<VoxelKey> pointToVoxelKey(const Eigen::Vector3d& p,
                                            double voxel_size) const {
        if (!pointInsideBbox(p)) return std::nullopt;
        const double minv = cfg_.pointcloud_bbox_min;
        const double extent = cfg_.pointcloud_bbox_max - cfg_.pointcloud_bbox_min;
        const int max_index = static_cast<int>(std::floor(extent / voxel_size)) - 1;
        VoxelKey key;
        key.x = static_cast<int>(std::floor((p.x() - minv) / voxel_size));
        key.y = static_cast<int>(std::floor((p.y() - minv) / voxel_size));
        key.z = static_cast<int>(std::floor((p.z() - minv) / voxel_size));
        if (key.x < 0 || key.y < 0 || key.z < 0) return std::nullopt;
        if (key.x > max_index || key.y > max_index || key.z > max_index) return std::nullopt;
        return key;
    }

    Eigen::Vector3d voxelKeyToCenter(const VoxelKey& key, double voxel_size) const {
        const double minv = cfg_.pointcloud_bbox_min;
        return Eigen::Vector3d(
            minv + (static_cast<double>(key.x) + 0.5) * voxel_size,
            minv + (static_cast<double>(key.y) + 0.5) * voxel_size,
            minv + (static_cast<double>(key.z) + 0.5) * voxel_size);
    }

    VoxelSet voxelizePointCloud(const pcl::PointCloud<pcl::PointXYZRGB>& cloud,
                                double voxel_size) const {
        VoxelSet voxels;
        voxels.reserve(cloud.size());
        for (const auto& p : cloud.points) {
            const auto key = pointToVoxelKey(Eigen::Vector3d(p.x, p.y, p.z), voxel_size);
            if (key.has_value()) voxels.insert(*key);
        }
        return voxels;
    }

    VoxelSet subtractVoxelSets(const VoxelSet& predicted,
                               const VoxelSet& observed) const {
        VoxelSet diff;
        diff.reserve(predicted.size());
        for (const auto& key : predicted) {
            if (observed.find(key) == observed.end()) diff.insert(key);
        }
        return diff;
    }

    std::vector<Eigen::Vector3d> voxelSetToCenters(const VoxelSet& voxels,
                                                   double voxel_size) const {
        std::vector<Eigen::Vector3d> centers;
        centers.reserve(voxels.size());
        for (const auto& key : voxels) {
            centers.push_back(voxelKeyToCenter(key, voxel_size));
        }
        return centers;
    }

    static QuantizedPointKey quantizePoint(const Eigen::Vector3d& p) {
        constexpr double scale = 1e6;
        return QuantizedPointKey{
            static_cast<int64_t>(std::llround(p.x() * scale)),
            static_cast<int64_t>(std::llround(p.y() * scale)),
            static_cast<int64_t>(std::llround(p.z() * scale)),
        };
    }

    double computeHprRadius(const std::vector<Eigen::Vector3d>& points,
                            const Eigen::Vector3d& camera_position) const {
        if (points.empty()) return cfg_.hpr_radius_scale * cfg_.planning_voxel_size;
        Eigen::Vector3d min_pt = points.front();
        Eigen::Vector3d max_pt = points.front();
        double max_distance = 0.0;
        for (const auto& p : points) {
            min_pt = min_pt.cwiseMin(p);
            max_pt = max_pt.cwiseMax(p);
            max_distance = std::max(max_distance, (p - camera_position).norm());
        }
        const double diameter = (max_pt - min_pt).norm();
        const double base = std::max(diameter, cfg_.planning_voxel_size);
        return std::max(max_distance + cfg_.planning_voxel_size,
                        cfg_.hpr_radius_scale * base);
    }

    std::vector<int> extractVisibleVoxelIndicesHpr(
        const std::vector<Eigen::Vector3d>& points,
        const Eigen::Vector3d& camera_position,
        double radius) const {
        if (points.empty()) return {};
        if (points.size() <= 4) {
            std::vector<int> all(points.size());
            for (size_t i = 0; i < points.size(); ++i) all[i] = static_cast<int>(i);
            return all;
        }

        pcl::PointCloud<pcl::PointXYZ>::Ptr transformed(new pcl::PointCloud<pcl::PointXYZ>());
        transformed->reserve(points.size() + 1);
        std::unordered_map<QuantizedPointKey, std::vector<int>, QuantizedPointKeyHash> index_map;
        index_map.reserve(points.size());
        for (size_t i = 0; i < points.size(); ++i) {
            const Eigen::Vector3d rel = points[i] - camera_position;
            const double r = rel.norm();
            if (r <= 1e-12) continue;
            const Eigen::Vector3d flipped = rel + 2.0 * (radius - r) * (rel / r);
            pcl::PointXYZ q;
            q.x = static_cast<float>(flipped.x());
            q.y = static_cast<float>(flipped.y());
            q.z = static_cast<float>(flipped.z());
            transformed->push_back(q);
            index_map[quantizePoint(flipped)].push_back(static_cast<int>(i));
        }
        pcl::PointXYZ origin;
        origin.x = 0.0f;
        origin.y = 0.0f;
        origin.z = 0.0f;
        transformed->push_back(origin);
        transformed->width = static_cast<uint32_t>(transformed->size());
        transformed->height = 1;
        transformed->is_dense = true;

        pcl::ConvexHull<pcl::PointXYZ> hull;
        hull.setInputCloud(transformed);
        hull.setDimension(3);
        pcl::PointCloud<pcl::PointXYZ> hull_points;
        std::vector<pcl::Vertices> polygons;
        try {
            hull.reconstruct(hull_points, polygons);
        } catch (const std::exception&) {
            std::vector<int> all(points.size());
            for (size_t i = 0; i < points.size(); ++i) all[i] = static_cast<int>(i);
            return all;
        }

        std::unordered_set<int> visible;
        for (const auto& p : hull_points.points) {
            const Eigen::Vector3d v(p.x, p.y, p.z);
            if (v.norm() <= 1e-9) continue;
            const auto it = index_map.find(quantizePoint(v));
            if (it == index_map.end()) continue;
            for (int idx : it->second) visible.insert(idx);
        }

        return std::vector<int>(visible.begin(), visible.end());
    }

    int countVisibleVoxelsFromCandidate(const std::vector<Eigen::Vector3d>& points,
                                        const Pose7d& candidate_pose) const {
        if (points.empty()) return 0;
        const Eigen::Vector3d camera_position(
            candidate_pose.v[0],
            candidate_pose.v[1],
            candidate_pose.v[2]);
        const double radius = computeHprRadius(points, camera_position);
        return static_cast<int>(
            extractVisibleVoxelIndicesHpr(points, camera_position, radius).size());
    }

    double motionCostToCandidate(const Pose7d& current_pose,
                                 const Pose7d& candidate_pose) const {
        return collisionAvoidSphereDistance(
            cameraPosition(current_pose),
            cameraPosition(candidate_pose),
            Vec3::Zero(),
            cfg_.obstacle_radius);
    }

    int selectNextView(const AlgorithmContext& ctx,
                       const VoxelSet& predicted_voxels,
                       const VoxelSet& unobserved_predicted_voxels) const {
        const std::vector<Eigen::Vector3d> hpr_points =
            voxelSetToCenters(unobserved_predicted_voxels, cfg_.planning_voxel_size);
        const std::vector<Eigen::Vector3d> predicted_points =
            voxelSetToCenters(predicted_voxels, cfg_.planning_voxel_size);

        std::vector<CandidateScore> scores;
        scores.reserve(ctx.candidate_views.size());
        double max_gain = 0.0;
        for (const auto& candidate : ctx.candidate_views) {
            CandidateScore score;
            score.view_id = candidate.view_idx;
            score.motion_cost = motionCostToCandidate(ctx.current_pose, candidate.pose);
            const std::vector<Eigen::Vector3d>& score_points =
                hpr_points.empty() ? predicted_points : hpr_points;
            score.information_gain =
                static_cast<double>(countVisibleVoxelsFromCandidate(score_points, candidate.pose));
            max_gain = std::max(max_gain, score.information_gain);
            scores.push_back(score);
        }

        if (scores.empty()) return -1;

        const double gain_threshold = max_gain * cfg_.tau;
        bool found = false;
        CandidateScore best;
        for (const auto& score : scores) {
            if (score.information_gain + 1e-9 < gain_threshold) continue;
            if (!found ||
                score.motion_cost < best.motion_cost - 1e-9 ||
                (std::fabs(score.motion_cost - best.motion_cost) <= 1e-9 &&
                 score.information_gain > best.information_gain + 1e-9)) {
                best = score;
                found = true;
            }
        }
        if (found) return best.view_id;

        for (const auto& score : scores) {
            if (!found || score.motion_cost < best.motion_cost) {
                best = score;
                found = true;
            }
        }
        return found ? best.view_id : -1;
    }

    static std::string makeStepFilename(const std::string& prefix, int step_index) {
        std::ostringstream oss;
        oss << prefix << "_step_" << std::setw(3) << std::setfill('0') << step_index << ".pcd";
        return oss.str();
    }

    static std::string makeRequestId(const AlgorithmContext& ctx) {
        std::ostringstream oss;
        oss << "pointr_c_nbv_" << ctx.uid << "_step" << ctx.step_index;
        return oss.str();
    }

    static double secondsSince(const std::chrono::steady_clock::time_point& start) {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    }

    static void logResolvedPath(const std::string& label, const std::filesystem::path& path) {
        std::cerr << "[pointr_c_nbv] " << label << "=" << path.string() << std::endl;
    }

    static void logParentPathForCreate(const std::string& label, const std::filesystem::path& path) {
        std::cerr << "[pointr_c_nbv] create_directories parent for " << label
                  << "=" << path.parent_path().string() << std::endl;
    }
};

}  // namespace objview

#endif  // OBJVIEWBENCH_POINTR_C_NBV_ALGORITHM_H_
