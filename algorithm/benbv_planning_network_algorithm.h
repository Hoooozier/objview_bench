#ifndef OBJVIEWBENCH_BENBV_PLANNING_NETWORK_ALGORITHM_H_
#define OBJVIEWBENCH_BENBV_PLANNING_NETWORK_ALGORITHM_H_

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
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include <Eigen/Dense>
#include <cnpy.h>
#include <json/json.h>
#include <octomap/Pointcloud.h>
#include <pcl/features/boundary.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>

#include "objview_algorithm.h"
#include "objview_interaction_rpc_client.h"
#include "objview_observation_io.h"
#include "objview_planning_network_client.h"

namespace objview {

struct BenbvPlanningNetworkConfig {
    std::string service_name = "benbv";
    double resolution = 0.02;
    double camera_distance = 2.0;
    int point_sample_count = 4096;
    int candidate_count = 20;
    int knn = 30;
    double boundary_angle_deg = 120.0;
    double partial_voxel_leaf = 0.015625;
    double pointcloud_bbox_min = -1.0;
    double pointcloud_bbox_max = 1.0;
    int seed = 42;
    int topk = 5;
    bool debug_save = false;
    std::string debug_dir = "benbv_planning_network_debug";
    bool silent = true;
};

class BenbvPlanningNetworkAlgorithm : public Algorithm {
public:
    explicit BenbvPlanningNetworkAlgorithm(BenbvPlanningNetworkConfig cfg)
        : cfg_(std::move(cfg)), rng_(static_cast<uint32_t>(cfg_.seed)) {
        if (cfg_.candidate_count != 20) {
            throw std::runtime_error("BENBV currently expects candidate_count=20.");
        }
        if (cfg_.point_sample_count <= 0 || cfg_.knn <= 0 ||
            cfg_.camera_distance <= 0.0 || cfg_.partial_voxel_leaf <= 0.0) {
            throw std::runtime_error("Invalid BENBV numeric configuration.");
        }
        partial_cloud_.reset(new CloudXYZ);
    }

    void prepareEpisode(
        const Json::Value& episode_config,
        const std::filesystem::path& session_dir) override {
        episode_config_ = episode_config;
        session_dir_ = session_dir;
        service_root_ = session_dir_ / "planning_network" / cfg_.service_name;
        if (cfg_.debug_save) {
            std::filesystem::path p(cfg_.debug_dir);
            debug_dir_abs_ = p.is_absolute() ? p : (session_dir_ / p);
            std::filesystem::create_directories(debug_dir_abs_);
        }
    }

    std::vector<ViewEntry> candidateViewSpace() const override {
        return {};
    }

    AlgorithmDecision decideNext(const AlgorithmContext& ctx) override {
        const auto total_start = std::chrono::steady_clock::now();
        const ObservationFrame observation = readObservationFrame(ctx);

        const auto update_start = std::chrono::steady_clock::now();
        updatePartialCloud(ctx, observation);
        const double cloud_update_sec = secondsSince(update_start);

        const auto normal_start = std::chrono::steady_clock::now();
        CloudPN::Ptr cloud_pn = estimateNormalsOutward(partialCloudPoints(), cfg_.knn);
        const double normal_sec = secondsSince(normal_start);
        if (cloud_pn->size() < 8) {
            const double runtime_sec = cloud_update_sec + normal_sec;
            saveStopDebug(ctx, "too_few_points", runtime_sec, cloud_pn, {}, {});
            return AlgorithmDecision::stop("candidate_exhausted").withRuntime(runtime_sec);
        }

        const auto boundary_start = std::chrono::steady_clock::now();
        const std::vector<int> boundary =
            computeBoundaryIndices(cloud_pn, cfg_.knn, cfg_.boundary_angle_deg);
        const double boundary_sec = secondsSince(boundary_start);
        if (boundary.empty()) {
            const double runtime_sec = cloud_update_sec + normal_sec + boundary_sec;
            saveStopDebug(ctx, "boundary_empty", runtime_sec, cloud_pn, boundary, {});
            return AlgorithmDecision::stop("candidate_exhausted").withRuntime(runtime_sec);
        }

        const auto candidate_start = std::chrono::steady_clock::now();
        const std::vector<int> selected = runKMeansSelect(
            cloud_pn,
            boundary,
            cfg_.candidate_count,
            cfg_.seed + ctx.step_index * 4099);
        std::vector<Candidate> candidates = generateCandidates(
            cloud_pn,
            selected,
            cfg_.candidate_count,
            cfg_.camera_distance,
            cfg_.knn);
        const double candidate_sec = secondsSince(candidate_start);
        if (!hasValidCandidate(candidates)) {
            const double runtime_sec =
                cloud_update_sec + normal_sec + boundary_sec + candidate_sec;
            saveStopDebug(ctx, "candidate_empty", runtime_sec, cloud_pn, boundary, candidates);
            return AlgorithmDecision::stop("candidate_exhausted").withRuntime(runtime_sec);
        }

        const auto input_start = std::chrono::steady_clock::now();
        const std::vector<float> P = buildSampleP(cloud_pn, cfg_.point_sample_count, rng_);
        const std::vector<float> S = buildS(candidates);
        const std::vector<float> C = computeDensityC(cloud_pn, candidates, ctx.step_index, cfg_.knn);
        const double input_build_sec = secondsSince(input_start);

        std::filesystem::path input_npz;
        double npz_write_elapsed_sec = 0.0;
        {
            const auto write_start = std::chrono::steady_clock::now();
            input_npz = saveInputNpz(ctx, P, S, C);
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

        const auto select_start = std::chrono::steady_clock::now();
        const std::vector<double> scores = readScores(result);
        const std::vector<int> ranked = rankCandidates(scores);
        std::vector<Pose7d> poses = candidatePoses(candidates);
        const double score_decode_sec = secondsSince(select_start);

        InteractionRpcClient interaction(
            ctx.interaction_requests_dir,
            ctx.interaction_responses_dir,
            ctx.interaction_wait_timeout_sec,
            ctx.interaction_poll_interval_sec);
        const InteractionFeasibilityResult feasible = interaction.isFeasible(
            InteractionRpcClient::makeRequestId(
                "benbv_is_feasible",
                ctx.episode_id,
                ctx.step_index),
            poses);

        const auto choose_start = std::chrono::steady_clock::now();
        int selected_candidate = -1;
        for (int idx : ranked) {
            if (idx < 0 || idx >= static_cast<int>(candidates.size())) continue;
            if (!candidates[static_cast<size_t>(idx)].valid) continue;
            if (idx >= static_cast<int>(feasible.feasible.size())) continue;
            if (!feasible.feasible[static_cast<size_t>(idx)]) continue;
            selected_candidate = idx;
            break;
        }
        const double choose_sec = secondsSince(choose_start);

        const double billable_runtime_sec =
            cloud_update_sec + normal_sec + boundary_sec + candidate_sec +
            input_build_sec + service_runtime_sec + score_decode_sec + choose_sec;

        Json::Value debug = makeDebugJson(
            ctx,
            input_npz,
            infer,
            result,
            feasible,
            scores,
            ranked,
            candidates,
            selected_candidate,
            cloud_update_sec,
            normal_sec,
            boundary_sec,
            candidate_sec,
            input_build_sec,
            npz_write_elapsed_sec,
            service_runtime_sec,
            score_decode_sec,
            choose_sec,
            billable_runtime_sec);

        if (selected_candidate < 0) {
            debug["stop_detail"] = "all_candidates_infeasible_or_invalid";
            saveStepDebug(ctx, cloud_pn, boundary, candidates, selected_candidate, debug);
            if (!cfg_.silent) {
                std::cout << "[benbv] candidate_exhausted at step " << ctx.step_index << "\n";
            }
            return AlgorithmDecision::stop("candidate_exhausted")
                .withRuntime(billable_runtime_sec);
        }

        saveStepDebug(ctx, cloud_pn, boundary, candidates, selected_candidate, debug);
        if (!cfg_.silent) {
            std::cout << "[benbv] step=" << ctx.step_index
                      << " selected=" << selected_candidate
                      << " score=" << scores[static_cast<size_t>(selected_candidate)]
                      << " runtime_sec=" << billable_runtime_sec
                      << " wall_sec=" << secondsSince(total_start) << "\n";
        }
        return AlgorithmDecision::movePose(poses[static_cast<size_t>(selected_candidate)])
            .withRuntime(billable_runtime_sec);
    }

private:
    using PointXYZ = pcl::PointXYZ;
    using PointXYZRGB = pcl::PointXYZRGB;
    using PointNormal = pcl::PointNormal;
    using CloudXYZ = pcl::PointCloud<PointXYZ>;
    using CloudXYZRGB = pcl::PointCloud<PointXYZRGB>;
    using CloudNormal = pcl::PointCloud<pcl::Normal>;
    using CloudPN = pcl::PointCloud<PointNormal>;

    struct ObservationFrame {
        Json::Value frame_meta;
        Pose7d pose;
        CameraIntrinsics intrinsics;
        std::filesystem::path depth_path;
        std::filesystem::path mask_path;
    };

    struct Candidate {
        Vec3 target = Vec3::Zero();
        Vec3 direction = Vec3::Zero();
        Vec3 camera = Vec3::Zero();
        bool valid = false;
    };

    BenbvPlanningNetworkConfig cfg_;
    Json::Value episode_config_;
    std::filesystem::path session_dir_;
    std::filesystem::path service_root_;
    std::filesystem::path debug_dir_abs_;
    std::unordered_set<int> observed_step_indices_;
    CloudXYZ::Ptr partial_cloud_;
    std::mt19937 rng_;

    ObservationFrame readObservationFrame(const AlgorithmContext& ctx) const {
        ObservationFrame observation;
        observation.pose = ctx.current_pose;
        const std::string frame_meta_rel =
            ctx.observation_manifest.get("frame_meta_path", "").asString();
        if (frame_meta_rel.empty()) {
            throw std::runtime_error("BENBV requires frame_meta_path in observation_manifest.");
        }
        observation.frame_meta = readJsonFile(resolveSessionPath(ctx, frame_meta_rel));
        observation.pose = poseFromFrameMeta(observation.frame_meta);
        observation.intrinsics = parseCameraIntrinsics(observation.frame_meta);

        const std::string depth_rel = ctx.observation_manifest.get("depth_path", "").asString();
        const std::string mask_rel = ctx.observation_manifest.get("mask_path", "").asString();
        if (depth_rel.empty() || mask_rel.empty()) {
            throw std::runtime_error("BENBV requires depth_path and mask_path.");
        }
        observation.depth_path = resolveSessionPath(ctx, depth_rel);
        observation.mask_path = resolveSessionPath(ctx, mask_rel);
        return observation;
    }

    void updatePartialCloud(const AlgorithmContext& ctx, const ObservationFrame& observation) {
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
            cfg_.pointcloud_bbox_min,
            cfg_.pointcloud_bbox_max);

        partial_cloud_->reserve(partial_cloud_->size() + cloud.size());
        for (size_t i = 0; i < cloud.size(); ++i) {
            const octomap::point3d p = cloud.getPoint(i);
            partial_cloud_->push_back(makePointXYZ(p.x(), p.y(), p.z()));
        }
        partial_cloud_->width = static_cast<uint32_t>(partial_cloud_->size());
        partial_cloud_->height = 1;
        partial_cloud_->is_dense = false;

        if (partial_cloud_->empty()) return;
        pcl::VoxelGrid<PointXYZ> voxel;
        voxel.setInputCloud(partial_cloud_);
        voxel.setLeafSize(
            static_cast<float>(cfg_.partial_voxel_leaf),
            static_cast<float>(cfg_.partial_voxel_leaf),
            static_cast<float>(cfg_.partial_voxel_leaf));
        CloudXYZ::Ptr filtered(new CloudXYZ);
        voxel.filter(*filtered);
        partial_cloud_ = filtered;
    }

    std::vector<Vec3> partialCloudPoints() const {
        std::vector<Vec3> pts;
        pts.reserve(partial_cloud_->size());
        for (const auto& p : partial_cloud_->points) {
            if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) continue;
            pts.emplace_back(p.x, p.y, p.z);
        }
        return pts;
    }

    static PointXYZ makePointXYZ(float x, float y, float z) {
        PointXYZ p;
        p.x = x;
        p.y = y;
        p.z = z;
        return p;
    }

    static Vec3 unit(const Vec3& v, const Vec3& fallback = Vec3(0.0, 0.0, 1.0)) {
        const double n = v.norm();
        if (n < 1e-12) return fallback;
        return v / n;
    }

    static CloudXYZ::Ptr makeCloud(const std::vector<Vec3>& pts) {
        CloudXYZ::Ptr cloud(new CloudXYZ);
        cloud->reserve(pts.size());
        for (const auto& p : pts) {
            cloud->push_back(makePointXYZ(
                static_cast<float>(p.x()),
                static_cast<float>(p.y()),
                static_cast<float>(p.z())));
        }
        cloud->width = static_cast<uint32_t>(cloud->size());
        cloud->height = 1;
        cloud->is_dense = false;
        return cloud;
    }

    static CloudPN::Ptr estimateNormalsOutward(const std::vector<Vec3>& pts, int knn) {
        CloudXYZ::Ptr cloud = makeCloud(pts);
        CloudNormal::Ptr normals(new CloudNormal);
        CloudPN::Ptr out(new CloudPN);
        if (cloud->empty()) return out;

        pcl::NormalEstimationOMP<PointXYZ, pcl::Normal> ne;
        ne.setInputCloud(cloud);
        ne.setSearchMethod(pcl::search::KdTree<PointXYZ>::Ptr(
            new pcl::search::KdTree<PointXYZ>));
        ne.setKSearch(std::max(3, std::min(knn, static_cast<int>(cloud->size()))));
        ne.compute(*normals);

        out->reserve(cloud->size());
        for (std::size_t i = 0; i < cloud->size(); ++i) {
            const auto& p = (*cloud)[i];
            const auto& n = (*normals)[i];
            Vec3 normal(n.normal_x, n.normal_y, n.normal_z);
            if (!std::isfinite(normal.x()) || !std::isfinite(normal.y()) ||
                !std::isfinite(normal.z()) || normal.norm() < 1e-12) {
                normal = unit(Vec3(p.x, p.y, p.z));
            } else {
                normal = unit(normal);
            }
            const Vec3 pos(p.x, p.y, p.z);
            if (normal.dot(pos) < 0.0) normal = -normal;

            PointNormal pn;
            pn.x = p.x;
            pn.y = p.y;
            pn.z = p.z;
            pn.normal_x = static_cast<float>(normal.x());
            pn.normal_y = static_cast<float>(normal.y());
            pn.normal_z = static_cast<float>(normal.z());
            out->push_back(pn);
        }
        out->width = static_cast<uint32_t>(out->size());
        out->height = 1;
        out->is_dense = false;
        return out;
    }

    static std::vector<int> computeBoundaryIndices(
        const CloudPN::Ptr& cloud_pn,
        int knn,
        double angle_deg) {
        std::vector<int> indices;
        if (cloud_pn->size() < 8) return indices;

        CloudXYZ::Ptr points(new CloudXYZ);
        CloudNormal::Ptr normals(new CloudNormal);
        points->reserve(cloud_pn->size());
        normals->reserve(cloud_pn->size());
        for (const auto& p : cloud_pn->points) {
            points->push_back(makePointXYZ(p.x, p.y, p.z));
            pcl::Normal n;
            n.normal_x = p.normal_x;
            n.normal_y = p.normal_y;
            n.normal_z = p.normal_z;
            normals->push_back(n);
        }

        pcl::BoundaryEstimation<PointXYZ, pcl::Normal, pcl::Boundary> be;
        be.setInputCloud(points);
        be.setInputNormals(normals);
        be.setSearchMethod(pcl::search::KdTree<PointXYZ>::Ptr(
            new pcl::search::KdTree<PointXYZ>));
        be.setKSearch(std::max(3, std::min(knn, static_cast<int>(points->size()))));
        be.setAngleThreshold(angle_deg * M_PI / 180.0);

        pcl::PointCloud<pcl::Boundary> boundaries;
        be.compute(boundaries);
        for (std::size_t i = 0; i < boundaries.size(); ++i) {
            if (boundaries[i].boundary_point != 0) indices.push_back(static_cast<int>(i));
        }
        return indices;
    }

    static std::vector<int> runKMeansSelect(
        const CloudPN::Ptr& cloud,
        const std::vector<int>& boundary_indices,
        int k,
        int seed) {
        if (boundary_indices.empty()) return {};
        k = std::min(k, static_cast<int>(boundary_indices.size()));
        if (k <= 0) return {};

        std::mt19937 rng(static_cast<uint32_t>(seed));
        std::vector<int> shuffled = boundary_indices;
        std::shuffle(shuffled.begin(), shuffled.end(), rng);

        std::vector<Vec3> centroids;
        centroids.reserve(k);
        for (int i = 0; i < k; ++i) {
            const auto& p = (*cloud)[shuffled[i]];
            centroids.emplace_back(p.x, p.y, p.z);
        }

        std::vector<int> labels(boundary_indices.size(), 0);
        for (int iter = 0; iter < 15; ++iter) {
            for (std::size_t i = 0; i < boundary_indices.size(); ++i) {
                const auto& p = (*cloud)[boundary_indices[i]];
                const Vec3 x(p.x, p.y, p.z);
                double best_d = std::numeric_limits<double>::max();
                int best = 0;
                for (int c = 0; c < k; ++c) {
                    const double d = (x - centroids[static_cast<size_t>(c)]).squaredNorm();
                    if (d < best_d) {
                        best_d = d;
                        best = c;
                    }
                }
                labels[i] = best;
            }

            std::vector<Vec3> sums(static_cast<size_t>(k), Vec3::Zero());
            std::vector<int> counts(static_cast<size_t>(k), 0);
            for (std::size_t i = 0; i < boundary_indices.size(); ++i) {
                const auto& p = (*cloud)[boundary_indices[i]];
                sums[static_cast<size_t>(labels[i])] += Vec3(p.x, p.y, p.z);
                counts[static_cast<size_t>(labels[i])] += 1;
            }
            for (int c = 0; c < k; ++c) {
                if (counts[static_cast<size_t>(c)] > 0) {
                    centroids[static_cast<size_t>(c)] =
                        sums[static_cast<size_t>(c)] /
                        static_cast<double>(counts[static_cast<size_t>(c)]);
                }
            }
        }

        std::vector<int> selected;
        selected.reserve(k);
        for (int c = 0; c < k; ++c) {
            double best_d = std::numeric_limits<double>::max();
            int best_idx = -1;
            for (std::size_t i = 0; i < boundary_indices.size(); ++i) {
                if (labels[i] != c) continue;
                const auto& p = (*cloud)[boundary_indices[i]];
                const double d =
                    (Vec3(p.x, p.y, p.z) - centroids[static_cast<size_t>(c)]).squaredNorm();
                if (d < best_d) {
                    best_d = d;
                    best_idx = boundary_indices[i];
                }
            }
            if (best_idx >= 0) selected.push_back(best_idx);
        }
        return selected;
    }

    static Vec3 rotateAroundAxis(const Vec3& v, const Vec3& axis, double rad) {
        const Vec3 a = unit(axis, Vec3(0.0, 0.0, 1.0));
        return v * std::cos(rad) + a.cross(v) * std::sin(rad) +
               a * (a.dot(v)) * (1.0 - std::cos(rad));
    }

    static std::vector<Candidate> generateCandidates(
        const CloudPN::Ptr& cloud_pn,
        const std::vector<int>& selected_indices,
        int candidate_count,
        double camera_distance,
        int knn) {
        std::vector<Candidate> candidates(static_cast<size_t>(candidate_count));
        if (cloud_pn->empty()) return candidates;

        CloudXYZ::Ptr xyz(new CloudXYZ);
        xyz->reserve(cloud_pn->size());
        for (const auto& p : cloud_pn->points) xyz->push_back(makePointXYZ(p.x, p.y, p.z));

        pcl::KdTreeFLANN<PointXYZ> tree;
        tree.setInputCloud(xyz);

        for (std::size_t ci = 0;
             ci < selected_indices.size() && ci < static_cast<std::size_t>(candidate_count);
             ++ci) {
            const int idx = selected_indices[ci];
            const auto& p = (*cloud_pn)[idx];
            const Vec3 target(p.x, p.y, p.z);
            Vec3 normal(p.normal_x, p.normal_y, p.normal_z);
            normal = unit(normal, unit(target));

            PointXYZ query = makePointXYZ(p.x, p.y, p.z);
            std::vector<int> nn_idx;
            std::vector<float> nn_dist;
            const int k = std::max(3, std::min(knn, static_cast<int>(xyz->size())));
            tree.nearestKSearch(query, k, nn_idx, nn_dist);

            Vec3 center = Vec3::Zero();
            int count = 0;
            for (int ni : nn_idx) {
                if (ni == idx) continue;
                const auto& q = (*xyz)[ni];
                center += Vec3(q.x, q.y, q.z);
                ++count;
            }
            if (count > 0) center /= static_cast<double>(count);
            Vec3 outer = unit(target - center, unit(target));
            if (outer.dot(unit(target)) < -0.2) outer = -outer;

            Vec3 axis = outer.cross(normal);
            Vec3 direction = normal;
            if (axis.norm() > 1e-8) {
                const double angles[] = {-45.0, 0.0, 45.0};
                const double rad = angles[ci % 3] * M_PI / 180.0;
                direction = unit(rotateAroundAxis(normal, axis, rad), normal);
            }
            if (direction.dot(unit(target)) < -0.1) direction = -direction;

            candidates[ci].target = target;
            candidates[ci].direction = unit(direction, unit(target));
            candidates[ci].camera = target + camera_distance * candidates[ci].direction;
            candidates[ci].valid = true;
        }
        return candidates;
    }

    std::vector<float> buildSampleP(
        const CloudPN::Ptr& cloud_pn,
        int sample_count,
        std::mt19937& rng) const {
        std::vector<float> out(static_cast<std::size_t>(sample_count) * 6, 0.0f);
        if (cloud_pn->empty()) return out;

        std::vector<int> ids(cloud_pn->size());
        std::iota(ids.begin(), ids.end(), 0);
        if (static_cast<int>(ids.size()) > sample_count) {
            std::shuffle(ids.begin(), ids.end(), rng);
            ids.resize(static_cast<std::size_t>(sample_count));
        }

        for (std::size_t i = 0; i < ids.size(); ++i) {
            const auto& p = (*cloud_pn)[ids[i]];
            const std::size_t o = i * 6;
            out[o + 0] = p.x;
            out[o + 1] = p.y;
            out[o + 2] = p.z;
            out[o + 3] = p.normal_x;
            out[o + 4] = p.normal_y;
            out[o + 5] = p.normal_z;
        }
        return out;
    }

    static std::vector<float> buildS(const std::vector<Candidate>& candidates) {
        std::vector<float> out(20 * 6, 0.0f);
        for (std::size_t i = 0; i < candidates.size() && i < 20; ++i) {
            if (!candidates[i].valid) continue;
            const std::size_t o = i * 6;
            out[o + 0] = static_cast<float>(candidates[i].target.x());
            out[o + 1] = static_cast<float>(candidates[i].target.y());
            out[o + 2] = static_cast<float>(candidates[i].target.z());
            out[o + 3] = static_cast<float>(candidates[i].direction.x());
            out[o + 4] = static_cast<float>(candidates[i].direction.y());
            out[o + 5] = static_cast<float>(candidates[i].direction.z());
        }
        return out;
    }

    static std::vector<float> computeDensityC(
        const CloudPN::Ptr& cloud_pn,
        const std::vector<Candidate>& candidates,
        int step_id,
        int knn) {
        std::vector<float> C(21, 0.0f);
        if (cloud_pn->empty()) {
            C[20] = static_cast<float>(step_id);
            return C;
        }

        CloudXYZ::Ptr xyz(new CloudXYZ);
        xyz->reserve(cloud_pn->size());
        for (const auto& p : cloud_pn->points) xyz->push_back(makePointXYZ(p.x, p.y, p.z));
        pcl::KdTreeFLANN<PointXYZ> tree;
        tree.setInputCloud(xyz);

        float max_density = 0.0f;
        for (std::size_t i = 0; i < candidates.size() && i < 20; ++i) {
            if (!candidates[i].valid) continue;
            PointXYZ q = makePointXYZ(
                static_cast<float>(candidates[i].target.x()),
                static_cast<float>(candidates[i].target.y()),
                static_cast<float>(candidates[i].target.z()));
            std::vector<int> idx;
            std::vector<float> dist2;
            const int k = std::max(1, std::min(knn, static_cast<int>(xyz->size())));
            if (tree.nearestKSearch(q, k, idx, dist2) <= 0) continue;
            double mean = 0.0;
            for (float d2 : dist2) mean += std::sqrt(std::max(0.0f, d2));
            mean /= static_cast<double>(dist2.size());
            C[i] = static_cast<float>(1.0 / (mean + 1e-6));
            max_density = std::max(max_density, C[i]);
        }

        if (max_density > 0.0f) {
            for (int i = 0; i < 20; ++i) C[i] /= max_density;
        }
        C[20] = static_cast<float>(step_id);
        return C;
    }

    std::filesystem::path saveInputNpz(
        const AlgorithmContext& ctx,
        const std::vector<float>& P,
        const std::vector<float>& S,
        const std::vector<float>& C) const {
        std::filesystem::path dir = debug_dir_abs_.empty()
            ? (service_root_ / "inputs")
            : debug_dir_abs_;
        std::filesystem::create_directories(dir);
        const std::filesystem::path path = dir / (makeStepStem(ctx) + ".npz");

        cnpy::npz_save(
            path.string(),
            "P",
            P.data(),
            {static_cast<size_t>(cfg_.point_sample_count), static_cast<size_t>(6)},
            "w");
        cnpy::npz_save(path.string(), "S", S.data(), {static_cast<size_t>(20), static_cast<size_t>(6)}, "a");
        cnpy::npz_save(path.string(), "C", C.data(), {static_cast<size_t>(21), static_cast<size_t>(1)}, "a");
        return path;
    }

    static bool hasValidCandidate(const std::vector<Candidate>& candidates) {
        for (const auto& c : candidates) {
            if (c.valid) return true;
        }
        return false;
    }

    static std::vector<double> readScores(const Json::Value& result) {
        const Json::Value scores_json = result["scores"];
        if (!scores_json.isArray() || scores_json.empty() || !scores_json[0].isArray()) {
            return std::vector<double>(20, -std::numeric_limits<double>::infinity());
        }
        std::vector<double> scores(20, -std::numeric_limits<double>::infinity());
        const Json::Value row = scores_json[0];
        for (Json::ArrayIndex i = 0; i < row.size() && i < 20; ++i) {
            scores[static_cast<size_t>(i)] = row[i].asDouble();
        }
        return scores;
    }

    static std::vector<int> rankCandidates(const std::vector<double>& scores) {
        std::vector<int> ids(scores.size());
        std::iota(ids.begin(), ids.end(), 0);
        std::stable_sort(ids.begin(), ids.end(), [&](int a, int b) {
            return scores[static_cast<size_t>(a)] > scores[static_cast<size_t>(b)];
        });
        return ids;
    }

    static std::vector<Pose7d> candidatePoses(const std::vector<Candidate>& candidates) {
        std::vector<Pose7d> poses;
        poses.reserve(candidates.size());
        for (const auto& c : candidates) {
            Pose7d p;
            if (c.valid) {
                p.v = {
                    c.camera.x(),
                    c.camera.y(),
                    c.camera.z(),
                    c.target.x(),
                    c.target.y(),
                    c.target.z(),
                    0.0,
                };
            }
            poses.push_back(p);
        }
        return poses;
    }

    Json::Value makeDebugJson(
        const AlgorithmContext& ctx,
        const std::filesystem::path& input_npz,
        const PlanningNetworkInferResult& infer,
        const Json::Value& result,
        const InteractionFeasibilityResult& feasible,
        const std::vector<double>& scores,
        const std::vector<int>& ranked,
        const std::vector<Candidate>& candidates,
        int selected_candidate,
        double cloud_update_sec,
        double normal_sec,
        double boundary_sec,
        double candidate_sec,
        double input_build_sec,
        double npz_write_elapsed_sec,
        double service_runtime_sec,
        double score_decode_sec,
        double choose_sec,
        double billable_runtime_sec) const {
        Json::Value debug(Json::objectValue);
        debug["uid"] = ctx.uid;
        debug["episode_id"] = ctx.episode_id;
        debug["step_index"] = ctx.step_index;
        debug["backend"] = result.get("backend", "").asString();
        debug["input_npz"] = input_npz.string();
        debug["request_id"] = infer.request_id;
        debug["planning_rpc_elapsed_sec"] = infer.rpc_elapsed_sec;
        debug["feasibility_rpc_elapsed_sec"] = feasible.rpc_elapsed_sec;
        debug["cloud_update_sec"] = cloud_update_sec;
        debug["normal_sec"] = normal_sec;
        debug["boundary_sec"] = boundary_sec;
        debug["candidate_sec"] = candidate_sec;
        debug["input_build_sec"] = input_build_sec;
        debug["npz_write_elapsed_sec"] = npz_write_elapsed_sec;
        debug["service_runtime_total_sec"] = service_runtime_sec;
        debug["score_decode_sec"] = score_decode_sec;
        debug["choose_sec"] = choose_sec;
        debug["billable_runtime_sec"] = billable_runtime_sec;
        debug["selected_candidate_index"] = selected_candidate;
        debug["scores"] = toJsonArray(scores);
        debug["ranked_candidate_indices"] = toJsonArray(ranked);
        debug["valid_mask"] = validMaskJson(candidates);
        debug["feasible_mask"] = boolArrayJson(feasible.feasible);
        debug["candidate_poses"] = candidatePosesJson(candidates);
        if (selected_candidate >= 0 &&
            selected_candidate < static_cast<int>(candidates.size())) {
            debug["selected_score"] = scores[static_cast<size_t>(selected_candidate)];
            debug["selected_pose"] =
                poseToJson(candidatePoses(candidates)[static_cast<size_t>(selected_candidate)]);
        }
        return debug;
    }

    void saveStopDebug(
        const AlgorithmContext& ctx,
        const std::string& detail,
        double runtime_sec,
        const CloudPN::Ptr& cloud_pn,
        const std::vector<int>& boundary,
        const std::vector<Candidate>& candidates) const {
        if (!cfg_.debug_save) return;
        Json::Value debug(Json::objectValue);
        debug["uid"] = ctx.uid;
        debug["episode_id"] = ctx.episode_id;
        debug["step_index"] = ctx.step_index;
        debug["terminal_stop_reason"] = "candidate_exhausted";
        debug["stop_detail"] = detail;
        debug["num_partial_points"] = static_cast<Json::UInt64>(partial_cloud_->size());
        debug["num_normal_points"] = static_cast<Json::UInt64>(cloud_pn ? cloud_pn->size() : 0);
        debug["num_boundary_points"] = static_cast<Json::UInt64>(boundary.size());
        debug["billable_runtime_sec"] = runtime_sec;
        saveStepDebug(ctx, cloud_pn, boundary, candidates, -1, debug);
    }

    void saveStepDebug(
        const AlgorithmContext& ctx,
        const CloudPN::Ptr& cloud_pn,
        const std::vector<int>& boundary,
        const std::vector<Candidate>& candidates,
        int selected_candidate,
        const Json::Value& debug) const {
        if (!cfg_.debug_save) return;
        std::filesystem::create_directories(debug_dir_abs_);
        const std::string stem = makeStepStem(ctx);
        saveCloud(partialCloudPoints(), debug_dir_abs_ / (stem + "_partial.pcd"));
        if (cloud_pn) {
            saveCloud(boundaryPointsForDebug(cloud_pn, boundary),
                      debug_dir_abs_ / (stem + "_boundary.pcd"));
        }
        saveCloud(candidateTargetsForDebug(candidates),
                  debug_dir_abs_ / (stem + "_candidate_targets.pcd"));
        saveCloud(candidateCamerasForDebug(candidates),
                  debug_dir_abs_ / (stem + "_candidate_cameras.pcd"));
        saveFrameCloud(
            candidateFramesForDebug(candidates),
            debug_dir_abs_ / (stem + "_candidate_frames_xyz_rgb.pcd"));
        if (selected_candidate >= 0 &&
            selected_candidate < static_cast<int>(candidates.size()) &&
            candidates[static_cast<size_t>(selected_candidate)].valid) {
            saveCloud(selectedCandidateForDebug(candidates[static_cast<size_t>(selected_candidate)]),
                      debug_dir_abs_ / (stem + "_selected_candidate.pcd"));
            saveFrameCloud(
                singleCandidateFrameForDebug(candidates[static_cast<size_t>(selected_candidate)]),
                debug_dir_abs_ / (stem + "_selected_frame_xyz_rgb.pcd"));
        }

        Json::StreamWriterBuilder builder;
        builder["indentation"] = "  ";
        std::ofstream fout(debug_dir_abs_ / (stem + "_debug.json"), std::ios::binary);
        if (fout) fout << Json::writeString(builder, debug) << "\n";
    }

    static std::vector<Vec3> boundaryPointsForDebug(
        const CloudPN::Ptr& cloud_pn,
        const std::vector<int>& idxs) {
        std::vector<Vec3> pts;
        pts.reserve(idxs.size());
        for (int idx : idxs) {
            if (idx < 0 || idx >= static_cast<int>(cloud_pn->size())) continue;
            const auto& p = (*cloud_pn)[idx];
            pts.emplace_back(p.x, p.y, p.z);
        }
        return pts;
    }

    static std::vector<Vec3> candidateTargetsForDebug(const std::vector<Candidate>& candidates) {
        std::vector<Vec3> pts;
        for (const auto& c : candidates) {
            if (c.valid) pts.push_back(c.target);
        }
        return pts;
    }

    static std::vector<Vec3> candidateCamerasForDebug(const std::vector<Candidate>& candidates) {
        std::vector<Vec3> pts;
        for (const auto& c : candidates) {
            if (c.valid) pts.push_back(c.camera);
        }
        return pts;
    }

    static std::vector<Vec3> selectedCandidateForDebug(const Candidate& c) {
        std::vector<Vec3> pts;
        pts.push_back(c.target);
        pts.push_back(c.camera);
        const Vec3 dir = unit(c.target - c.camera);
        for (int i = 1; i <= 8; ++i) {
            pts.push_back(c.camera + dir * (0.05 * static_cast<double>(i)));
        }
        return pts;
    }

    struct DebugPointRGB {
        Vec3 p = Vec3::Zero();
        uint8_t r = 255;
        uint8_t g = 255;
        uint8_t b = 255;
    };

    static std::vector<DebugPointRGB> candidateFramesForDebug(
        const std::vector<Candidate>& candidates) {
        std::vector<DebugPointRGB> pts;
        for (const auto& c : candidates) {
            if (!c.valid) continue;
            appendCameraFrameForDebug(pts, c);
        }
        return pts;
    }

    static std::vector<DebugPointRGB> singleCandidateFrameForDebug(const Candidate& c) {
        std::vector<DebugPointRGB> pts;
        if (c.valid) appendCameraFrameForDebug(pts, c);
        return pts;
    }

    static void appendCameraFrameForDebug(std::vector<DebugPointRGB>& pts, const Candidate& c) {
        const Vec3 origin = c.camera;
        const Vec3 z_axis = unit(c.target - c.camera, Vec3(0.0, 0.0, -1.0));
        Vec3 up(0.0, 0.0, 1.0);
        if (std::abs(z_axis.dot(up)) > 0.98) up = Vec3(0.0, 1.0, 0.0);
        const Vec3 x_axis = unit(z_axis.cross(up), Vec3(1.0, 0.0, 0.0));
        const Vec3 y_axis = unit(x_axis.cross(z_axis), Vec3(0.0, 1.0, 0.0));

        appendAxisSamples(pts, origin, x_axis, 0.25, 255, 0, 0);
        appendAxisSamples(pts, origin, y_axis, 0.25, 0, 255, 0);
        appendAxisSamples(pts, origin, z_axis, 0.35, 0, 0, 255);
        pts.push_back({c.target, 255, 255, 0});
    }

    static void appendAxisSamples(
        std::vector<DebugPointRGB>& pts,
        const Vec3& origin,
        const Vec3& axis,
        double length,
        uint8_t r,
        uint8_t g,
        uint8_t b) {
        constexpr int samples = 12;
        for (int i = 0; i <= samples; ++i) {
            const double t = length * static_cast<double>(i) / static_cast<double>(samples);
            pts.push_back({origin + axis * t, r, g, b});
        }
    }

    static void saveCloud(const std::vector<Vec3>& pts, const std::filesystem::path& path) {
        std::filesystem::create_directories(path.parent_path());
        CloudXYZ cloud;
        cloud.reserve(pts.size());
        for (const auto& p : pts) {
            cloud.push_back(makePointXYZ(
                static_cast<float>(p.x()),
                static_cast<float>(p.y()),
                static_cast<float>(p.z())));
        }
        cloud.width = static_cast<uint32_t>(cloud.size());
        cloud.height = 1;
        cloud.is_dense = false;
        pcl::io::savePCDFileBinary(path.string(), cloud);
    }

    static void saveFrameCloud(
        const std::vector<DebugPointRGB>& pts,
        const std::filesystem::path& path) {
        std::filesystem::create_directories(path.parent_path());
        CloudXYZRGB cloud;
        cloud.reserve(pts.size());
        for (const auto& q : pts) {
            PointXYZRGB p;
            p.x = static_cast<float>(q.p.x());
            p.y = static_cast<float>(q.p.y());
            p.z = static_cast<float>(q.p.z());
            p.r = q.r;
            p.g = q.g;
            p.b = q.b;
            cloud.push_back(p);
        }
        cloud.width = static_cast<uint32_t>(cloud.size());
        cloud.height = 1;
        cloud.is_dense = false;
        pcl::io::savePCDFileBinary(path.string(), cloud);
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

    static Json::Value toJsonArray(const std::vector<double>& xs) {
        Json::Value arr(Json::arrayValue);
        for (double x : xs) arr.append(x);
        return arr;
    }

    static Json::Value boolArrayJson(const std::vector<bool>& xs) {
        Json::Value arr(Json::arrayValue);
        for (bool x : xs) arr.append(x);
        return arr;
    }

    static Json::Value validMaskJson(const std::vector<Candidate>& candidates) {
        Json::Value arr(Json::arrayValue);
        for (const auto& c : candidates) arr.append(c.valid);
        return arr;
    }

    static Json::Value poseToJson(const Pose7d& pose) {
        Json::Value arr(Json::arrayValue);
        for (double x : pose.v) arr.append(x);
        return arr;
    }

    static Json::Value candidatePosesJson(const std::vector<Candidate>& candidates) {
        Json::Value arr(Json::arrayValue);
        const std::vector<Pose7d> poses = candidatePoses(candidates);
        for (const Pose7d& p : poses) arr.append(poseToJson(p));
        return arr;
    }

    static std::string makeStepStem(const AlgorithmContext& ctx) {
        std::ostringstream oss;
        oss << "benbv_step_" << std::setw(3) << std::setfill('0') << ctx.step_index;
        return oss.str();
    }

    static std::string makeRequestId(const AlgorithmContext& ctx) {
        std::ostringstream oss;
        oss << "benbv_" << sanitizeRequestIdPart(ctx.uid)
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
        return out.empty() ? "unknown" : out;
    }

    static double secondsSince(const std::chrono::steady_clock::time_point& start) {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    }
};

}  // namespace objview

#endif  // OBJVIEWBENCH_BENBV_PLANNING_NETWORK_ALGORITHM_H_
