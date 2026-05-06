#ifndef OBJVIEWBENCH_VOXEL_IG_ALGORITHM_H_
#define OBJVIEWBENCH_VOXEL_IG_ALGORITHM_H_

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <json/json.h>
#include <octomap/ColorOcTree.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "objview_algorithm.h"
#include "objview_observation_io.h"
#include "objview_view_io.h"
#include "voxel_ig_cuda.h"

namespace objview {

enum class VoxelIgMethod : int {
    OA = 1,
    UV = 2,
    RSE = 3,
    PCV = 4,
    APORA = 5,
    Kr = 6,
    MCMF = 10,
    GMC = 11,
};

enum class VoxelIgBackend : int {
    CPU = 0,
    CUDA = 1,
};

inline std::string voxelIgBackendName(VoxelIgBackend backend) {
    switch (backend) {
        case VoxelIgBackend::CPU: return "cpu";
        case VoxelIgBackend::CUDA: return "cuda";
    }
    return "unknown";
}

inline VoxelIgBackend parseVoxelIgBackend(const std::string& value) {
    std::string s;
    s.reserve(value.size());
    for (char c : value) s.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));

    if (s == "cpu") return VoxelIgBackend::CPU;
    if (s == "cuda") return VoxelIgBackend::CUDA;
    throw std::runtime_error("Unknown voxel IG backend: " + value);
}

inline std::string voxelIgMethodName(VoxelIgMethod method) {
    switch (method) {
        case VoxelIgMethod::MCMF: return "mcmf";
        case VoxelIgMethod::OA: return "oa";
        case VoxelIgMethod::UV: return "uv";
        case VoxelIgMethod::RSE: return "rse";
        case VoxelIgMethod::PCV: return "pcv";
        case VoxelIgMethod::APORA: return "apora";
        case VoxelIgMethod::Kr: return "kr";
        case VoxelIgMethod::GMC: return "gmc";
    }
    return "unknown";
}

inline VoxelIgMethod parseVoxelIgMethod(const std::string& value) {
    std::string s;
    s.reserve(value.size());
    for (char c : value) s.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));

    if (s == "1" || s == "oa") return VoxelIgMethod::OA;
    if (s == "2" || s == "uv") return VoxelIgMethod::UV;
    if (s == "3" || s == "rse") return VoxelIgMethod::RSE;
    if (s == "4" || s == "pcv" || s == "proximity" || s == "proximity_count") {
        return VoxelIgMethod::PCV;
    }
    if (s == "5" || s == "apora") return VoxelIgMethod::APORA;
    if (s == "6" || s == "kr") return VoxelIgMethod::Kr;
    if (s == "10" || s == "mcmf") return VoxelIgMethod::MCMF;
    if (s == "11" || s == "gmc") return VoxelIgMethod::GMC;

    throw std::runtime_error("Unknown voxel IG method: " + value);
}

struct VoxelIgAlgorithmConfig {
    std::string views_path = "../Tammes_sphere/360_xyz.txt";
    double view_radius = 3.0;
    int grid_dim = 64;
    double octomap_resolution = -1.0;  // optional override; default is bbox extent / grid_dim
    double map_bbox_min = -1.0;
    double map_bbox_max = 1.0;
    double p_unknown_lower_bound = 0.45;
    double p_unknown_upper_bound = 0.65;
    VoxelIgMethod method = VoxelIgMethod::RSE;
    VoxelIgBackend backend = VoxelIgBackend::CUDA;
    int ray_stride = 0;  // 0 means backend default: CPU=16, CUDA=4
    bool use_movement_cost = false;
  double movement_cost_weight = 0.7;
    bool filter_rays_to_bbox = true;
    bool debug_save_ot = false;
    std::string debug_ot_dir;
    bool silent = true;
};

class VoxelIgAlgorithm : public Algorithm {
public:
    explicit VoxelIgAlgorithm(VoxelIgAlgorithmConfig cfg)
        : cfg_(std::move(cfg)), map_(computeInitialResolution(cfg_)) {
        if (cfg_.map_bbox_min >= cfg_.map_bbox_max) {
            throw std::runtime_error("Voxel-IG map bbox min must be smaller than bbox max.");
        }
        if (cfg_.grid_dim <= 0) {
            throw std::runtime_error("Voxel-IG grid_dim must be positive.");
        }
        if (cfg_.octomap_resolution > 0.0) {
            const double implied_grid =
                (cfg_.map_bbox_max - cfg_.map_bbox_min) / cfg_.octomap_resolution;
            if (std::fabs(implied_grid - std::round(implied_grid)) > 1e-6) {
                throw std::runtime_error(
                    "Voxel-IG resolution override must evenly divide the bbox extent.");
            }
        }
        if (cfg_.p_unknown_lower_bound < 0.0 ||
            cfg_.p_unknown_upper_bound > 1.0 ||
            cfg_.p_unknown_lower_bound >= cfg_.p_unknown_upper_bound) {
            throw std::runtime_error("Voxel-IG unknown occupancy bounds must satisfy 0 <= lower < upper <= 1.");
        }
        if (cfg_.ray_stride < 0) {
            throw std::runtime_error("Voxel-IG ray_stride must be non-negative.");
        }
        if (cfg_.movement_cost_weight < 0.0 || cfg_.movement_cost_weight > 1.0) {
            throw std::runtime_error("Voxel-IG movement_cost_weight must be in [0, 1].");
        }
        if (cfg_.method == VoxelIgMethod::MCMF || cfg_.method == VoxelIgMethod::GMC) {
            throw std::runtime_error(
                "Voxel-IG methods MCMF/GMC are reserved for future graph-optimization implementations.");
        }
        initializeUnknownMap();
        if (cfg_.backend == VoxelIgBackend::CUDA) {
            // Allocate persistent CUDA scorer resources during algorithm setup so
            // the first timed decision does not absorb one-time backend startup.
            ensureCudaScorer();
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
        const auto start = std::chrono::steady_clock::now();
        if (!cfg_.silent) {
            std::cout << "[voxel_ig] decideNext step=" << ctx.step_index
                      << " method=" << voxelIgMethodName(cfg_.method) << std::endl;
        }
        ObservationFrame observation = readObservationFrame(ctx);
        if (!cfg_.silent) {
            std::cout << "[voxel_ig] readObservationFrame ok" << std::endl;
        }
        updateMapFromObservation(ctx, observation);
        if (!cfg_.silent) {
            std::cout << "[voxel_ig] updateMapFromObservation ok octree_nodes=" << map_.size() << std::endl;
        }

        AlgorithmDecision decision;
        const int next_view_id = chooseBestView(ctx, observation.intrinsics);
        if (!cfg_.silent) {
            std::cout << "[voxel_ig] chooseBestView next=" << next_view_id << std::endl;
        }
        if (next_view_id < 0) {
            decision = AlgorithmDecision::stop("candidate_exhausted");
        } else {
            selected_view_ids_.insert(next_view_id);
            decision = AlgorithmDecision::move(next_view_id);
        }

        const double runtime_sec =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        decision.withRuntime(runtime_sec);
        return decision;
    }

    size_t observedStepCountForTest() const {
        return observed_step_indices_.size();
    }

    size_t mapLeafCountForTest() const {
        return map_.size();
    }

    size_t bboxNodeCountForTest() const {
        const double resolution = voxelResolution();
        const int grid_n = gridAxisCount();
        size_t count = 0;
        for (int ix = 0; ix < grid_n; ++ix) {
            const double x =
                cfg_.map_bbox_min + (static_cast<double>(ix) + 0.5) * resolution;
            for (int iy = 0; iy < grid_n; ++iy) {
                const double y =
                    cfg_.map_bbox_min + (static_cast<double>(iy) + 0.5) * resolution;
                for (int iz = 0; iz < grid_n; ++iz) {
                    const double z =
                        cfg_.map_bbox_min + (static_cast<double>(iz) + 0.5) * resolution;
                    const octomap::ColorOcTreeNode* node = map_.search(
                        octomap::point3d(
                            static_cast<float>(x),
                            static_cast<float>(y),
                            static_cast<float>(z)));
                    if (node != nullptr) {
                        ++count;
                    }
                }
            }
        }
        return count;
    }

    size_t leafCountForTest() const {
        size_t count = 0;
        for (octomap::ColorOcTree::leaf_iterator it = map_.begin_leafs(), end = map_.end_leafs();
             it != end; ++it) {
            ++count;
        }
        return count;
    }

    size_t bboxLeafCountForTest() const {
        size_t count = 0;
        for (octomap::ColorOcTree::leaf_iterator it = map_.begin_leafs(), end = map_.end_leafs();
             it != end; ++it) {
            if (pointInsideBbox(it.getCoordinate())) {
                ++count;
            }
        }
        return count;
    }

private:
    enum class VoxelState {
        Unknown,
        Free,
        Occupied,
    };

    struct ObservationFrame {
        Json::Value frame_meta;
        Pose7d pose;
        CameraIntrinsics intrinsics;
        std::filesystem::path depth_path;
        std::filesystem::path mask_path;
        bool has_depth = false;
    };

    struct CameraBasis {
        Vec3 x_right;
        Vec3 y_down;
        Vec3 z_forward;
        double fx = 0.0;
        double fy = 0.0;
        double cx = 0.0;
        double cy = 0.0;
    };

    struct UnknownVoxelEntry {
        octomap::OcTreeKey key;
        octomap::point3d coord;
        bool is_frontier = false;
    };

    struct RayScoreState {
        double gain = 0.0;
        double visible = 1.0;
        double object_visible = 1.0;
        bool previous_voxel_unknown = false;
        int voxel_num = 0;
    };

    struct VoxelSample {
        VoxelState state = VoxelState::Unknown;
        double occupancy = 0.5;
        double entropy = 1.0;
        double object_weight = 0.0;
        bool is_endpoint = false;
        bool is_occupied = false;
    };

    VoxelIgAlgorithmConfig cfg_;
    octomap::ColorOcTree map_;
    std::unordered_set<int> observed_step_indices_;
    std::unordered_set<int> selected_view_ids_;
    mutable std::unique_ptr<VoxelIgCudaScorer> cuda_scorer_;

    static double computeInitialResolution(const VoxelIgAlgorithmConfig& cfg) {
        if (cfg.map_bbox_min >= cfg.map_bbox_max) {
            throw std::runtime_error("Voxel-IG map bbox min must be smaller than bbox max.");
        }
        if (cfg.octomap_resolution > 0.0) {
            return cfg.octomap_resolution;
        }
        if (cfg.grid_dim <= 0) {
            throw std::runtime_error("Voxel-IG grid_dim must be positive.");
        }
        return (cfg.map_bbox_max - cfg.map_bbox_min) / static_cast<double>(cfg.grid_dim);
    }

    double voxelResolution() const {
        return map_.getResolution();
    }

    int gridAxisCount() const {
        if (cfg_.octomap_resolution > 0.0) {
            const double bbox_extent = cfg_.map_bbox_max - cfg_.map_bbox_min;
            return static_cast<int>(std::llround(bbox_extent / cfg_.octomap_resolution));
        }
        return cfg_.grid_dim;
    }

    int effectiveRayStride() const {
        if (cfg_.ray_stride > 0) return cfg_.ray_stride;
        return cfg_.backend == VoxelIgBackend::CUDA ? 4 : 16;
    }

    DenseGridSpec denseGridSpec() const {
        DenseGridSpec spec{};
        spec.bbox_min[0] = static_cast<float>(cfg_.map_bbox_min);
        spec.bbox_min[1] = static_cast<float>(cfg_.map_bbox_min);
        spec.bbox_min[2] = static_cast<float>(cfg_.map_bbox_min);
        spec.bbox_max[0] = static_cast<float>(cfg_.map_bbox_max);
        spec.bbox_max[1] = static_cast<float>(cfg_.map_bbox_max);
        spec.bbox_max[2] = static_cast<float>(cfg_.map_bbox_max);
        const int grid_n = gridAxisCount();
        spec.grid_dims[0] = grid_n;
        spec.grid_dims[1] = grid_n;
        spec.grid_dims[2] = grid_n;
        return spec;
    }

    DenseGridVolume packDenseGridFromOctomap() const {
        DenseGridVolume volume;
        volume.spec = denseGridSpec();
        const int nx = volume.spec.grid_dims[0];
        const int ny = volume.spec.grid_dims[1];
        const int nz = volume.spec.grid_dims[2];
        const double resolution = voxelResolution();
        volume.occupancy_prob.resize(static_cast<size_t>(nx) * static_cast<size_t>(ny) * static_cast<size_t>(nz), 0.5f);
        for (int iz = 0; iz < nz; ++iz) {
            const double z = cfg_.map_bbox_min + (static_cast<double>(iz) + 0.5) * resolution;
            for (int iy = 0; iy < ny; ++iy) {
                const double y = cfg_.map_bbox_min + (static_cast<double>(iy) + 0.5) * resolution;
                for (int ix = 0; ix < nx; ++ix) {
                    const double x = cfg_.map_bbox_min + (static_cast<double>(ix) + 0.5) * resolution;
                    const octomap::ColorOcTreeNode* node = map_.search(
                        octomap::point3d(
                            static_cast<float>(x),
                            static_cast<float>(y),
                            static_cast<float>(z)));
                    const float occupancy = node == nullptr ? 0.5f : static_cast<float>(node->getOccupancy());
                    const size_t idx =
                        static_cast<size_t>(ix) +
                        static_cast<size_t>(nx) * (
                            static_cast<size_t>(iy) +
                            static_cast<size_t>(ny) * static_cast<size_t>(iz));
                    volume.occupancy_prob[idx] = occupancy;
                }
            }
        }
        return volume;
    }

    VoxelIgCudaScorer& ensureCudaScorer() const {
        const DenseGridSpec spec = denseGridSpec();
        if (!cuda_scorer_) {
            cuda_scorer_ = std::make_unique<VoxelIgCudaScorer>(
                spec,
                static_cast<float>(cfg_.p_unknown_lower_bound),
                static_cast<float>(cfg_.p_unknown_upper_bound));
        }
        return *cuda_scorer_;
    }

    void initializeUnknownMap() {
        const double resolution = voxelResolution();
        const int grid_n = gridAxisCount();
        if (grid_n <= 0) {
            throw std::runtime_error("Voxel-IG grid initialization produced a non-positive grid size.");
        }

        for (int ix = 0; ix < grid_n; ++ix) {
            const double x =
                cfg_.map_bbox_min + (static_cast<double>(ix) + 0.5) * resolution;
            for (int iy = 0; iy < grid_n; ++iy) {
                const double y =
                    cfg_.map_bbox_min + (static_cast<double>(iy) + 0.5) * resolution;
                for (int iz = 0; iz < grid_n; ++iz) {
                    const double z =
                        cfg_.map_bbox_min + (static_cast<double>(iz) + 0.5) * resolution;
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
            throw std::runtime_error("frame_meta pose must contain camera_xyz and lookat_xyz arrays.");
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
            const Json::Value intr_json = ctx.episode_config["camera_intrinsics"];
            if (intr_json.isObject()) {
                observation.intrinsics.width = intr_json.get("image_width", 0).asInt();
                observation.intrinsics.height = intr_json.get("image_height", 0).asInt();
                observation.intrinsics.fov_x_rad = intr_json.get("fov_x_rad", 0.0).asDouble();
                observation.intrinsics.fov_y_rad = intr_json.get("fov_y_rad", 0.0).asDouble();
                observation.intrinsics.principal_x = intr_json.get("principal_x", 0.0).asDouble();
                observation.intrinsics.principal_y = intr_json.get("principal_y", 0.0).asDouble();
            }
        }

        const std::string depth_rel =
            ctx.observation_manifest.get("depth_path", "").asString();
        const std::string mask_rel =
            ctx.observation_manifest.get("mask_path", "").asString();
        observation.has_depth = !depth_rel.empty() && !mask_rel.empty() && !observation.frame_meta.isNull();
        if (observation.has_depth) {
            observation.depth_path = resolveSessionPath(ctx, depth_rel);
            observation.mask_path = resolveSessionPath(ctx, mask_rel);
        }
        return observation;
    }

    void updateMapFromObservation(const AlgorithmContext& ctx, const ObservationFrame& observation) {
        if (ctx.step_index < 0) return;
        if (!observed_step_indices_.insert(ctx.step_index).second) return;

        const Vec3 camera = cameraPosition(observation.pose);
        if (observation.has_depth) {
            if (!cfg_.silent) {
                std::cout << "[voxel_ig] loading depth=" << observation.depth_path
                          << " mask=" << observation.mask_path << std::endl;
            }
            const DepthImage depth = loadDepthNpz(observation.depth_path);
            if (!cfg_.silent) {
                std::cout << "[voxel_ig] depth loaded " << depth.width
                          << "x" << depth.height << std::endl;
            }
            const MaskImage mask = loadMaskImage(observation.mask_path);
            if (!cfg_.silent) {
                std::cout << "[voxel_ig] mask loaded " << mask.width
                          << "x" << mask.height << std::endl;
            }
            const Eigen::Matrix4d camera_to_world =
                parseMatrix4d(observation.frame_meta["camera_to_world"], "camera_to_world");
            if (!cfg_.silent) {
                std::cout << "[voxel_ig] camera_to_world parsed" << std::endl;
            }
            const octomap::Pointcloud cloud = backprojectDepthToWorldPointcloud(
                depth,
                mask,
                observation.intrinsics,
                camera_to_world,
                cfg_.map_bbox_min,
                cfg_.map_bbox_max);
            if (!cfg_.silent) {
                std::cout << "[voxel_ig] backprojected cloud size=" << cloud.size() << std::endl;
            }
            if (cloud.size() == 0) {
                map_.updateNode(octomap::point3d(camera.x(), camera.y(), camera.z()), false);
                if (!cfg_.silent) {
                    std::cout << "[voxel_ig] empty cloud, fell back to updateNode(camera, free)" << std::endl;
                }
            } else {
                map_.insertPointCloud(
                    cloud,
                    octomap::point3d(camera.x(), camera.y(), camera.z()),
                    -1.0,
                    false,
                    false);
                if (!cfg_.silent) {
                    std::cout << "[voxel_ig] insertPointCloud ok" << std::endl;
                }
            }
        } else {
            map_.updateNode(octomap::point3d(camera.x(), camera.y(), camera.z()), false);
            if (!cfg_.silent) {
                std::cout << "[voxel_ig] fallback updateNode ok" << std::endl;
            }
        }
        map_.updateInnerOccupancy();
        if (!cfg_.silent) {
            std::cout << "[voxel_ig] updateInnerOccupancy ok" << std::endl;
        }
        saveDebugOctomap(ctx);
        if (!cfg_.silent) {
            std::cout << "[voxel_ig] saveDebugOctomap ok" << std::endl;
        }

        if (!cfg_.silent) {
            std::cout << "Voxel-IG observed step=" << ctx.step_index
                      << " method=" << voxelIgMethodName(cfg_.method)
                      << " octree_nodes=" << map_.size() << std::endl;
        }
    }

    void saveDebugOctomap(const AlgorithmContext& ctx) {
        if (!cfg_.debug_save_ot) return;

        std::filesystem::path dir = cfg_.debug_ot_dir.empty()
            ? (ctx.session_dir / "voxel_ig_debug")
            : std::filesystem::path(cfg_.debug_ot_dir);
        if (!dir.is_absolute()) dir = ctx.session_dir / dir;
        std::filesystem::create_directories(dir);

        const std::string stem = "map_step_" + std::to_string(ctx.step_index);

        const std::filesystem::path full_path = dir / (stem + "__with_unknown.ot");
        if (!map_.write(full_path.string())) {
            throw std::runtime_error("Failed to save Voxel-IG debug octomap: " + full_path.string());
        }

        octomap::ColorOcTree known_only(voxelResolution());
        for (octomap::ColorOcTree::leaf_iterator it = map_.begin_leafs(), end = map_.end_leafs();
             it != end; ++it) {
            const double occupancy = (*it).getOccupancy();
            if (isUnknownOccupancy(occupancy)) continue;
            const octomap::point3d coord = it.getCoordinate();
            known_only.updateNode(coord, isOccupiedOccupancy(occupancy));
        }
        known_only.updateInnerOccupancy();

        const std::filesystem::path known_only_path = dir / (stem + "__known_only.ot");
        if (!known_only.write(known_only_path.string())) {
            throw std::runtime_error(
                "Failed to save Voxel-IG known-only debug octomap: " + known_only_path.string());
        }
    }

    static double voxelEntropy(double occupancy) {
        const double p = std::clamp(occupancy, 1e-9, 1.0 - 1e-9);
        return -p * std::log(p) - (1.0 - p) * std::log(1.0 - p);
    }

    bool isUnknownOccupancy(double occupancy) const {
        return occupancy > cfg_.p_unknown_lower_bound && occupancy < cfg_.p_unknown_upper_bound;
    }

    bool isFreeOccupancy(double occupancy) const {
        return occupancy <= cfg_.p_unknown_lower_bound;
    }

    bool isOccupiedOccupancy(double occupancy) const {
        return occupancy >= cfg_.p_unknown_upper_bound;
    }

    double voxelVisible(double occupancy) const {
        if (occupancy >= cfg_.p_unknown_upper_bound) return 0.0;
        if (occupancy <= cfg_.p_unknown_lower_bound) return 1.0;
        const double k = (0.0 - 1.0) / (cfg_.p_unknown_upper_bound - cfg_.p_unknown_lower_bound);
        const double b = -k * cfg_.p_unknown_upper_bound;
        return k * occupancy + b;
    }

    static CameraBasis makeCameraBasis(const Pose7d& pose, const CameraIntrinsics& intrinsics) {
        CameraBasis basis;
        basis.fx = 0.5 * static_cast<double>(intrinsics.width) / std::tan(0.5 * intrinsics.fov_x_rad);
        basis.fy = 0.5 * static_cast<double>(intrinsics.height) / std::tan(0.5 * intrinsics.fov_y_rad);
        basis.cx = intrinsics.principal_x;
        basis.cy = intrinsics.principal_y;

        const Vec3 camera(pose.v[0], pose.v[1], pose.v[2]);
        Vec3 lookat(pose.v[3], pose.v[4], pose.v[5]);
        Vec3 z_forward = lookat - camera;
        if (z_forward.norm() < 1e-12) {
            lookat = Vec3(0.0, 0.0, 0.0);
            z_forward = lookat - camera;
        }
        z_forward.normalize();

        if ((z_forward - Vec3(0.0, 0.0, -1.0)).norm() < 1e-6) {
            z_forward = Vec3(1e-8, 1e-8, -1.0).normalized();
        }
        if ((z_forward - Vec3(0.0, 0.0, 1.0)).norm() < 1e-6) {
            z_forward = Vec3(1e-8, 1e-8, 1.0).normalized();
        }

        const Vec3 world_up(0.0, 0.0, 1.0);
        Vec3 x_left = (-z_forward).cross(world_up).normalized();
        Vec3 y_up = x_left.cross(-z_forward).normalized();

        const double c = std::cos(pose.v[6]);
        const double s = std::sin(pose.v[6]);
        const Vec3 x_left_0 = x_left;
        const Vec3 y_up_0 = y_up;
        x_left = c * x_left_0 + s * y_up_0;
        y_up = -s * x_left_0 + c * y_up_0;

        basis.x_right = -x_left;
        basis.y_down = -y_up;
        basis.z_forward = z_forward;
        return basis;
    }

    static Vec3 pixelToWorldRay(int x, int y, const CameraBasis& basis) {
        const double x_cam = (static_cast<double>(x) + 0.5 - basis.cx) / basis.fx;
        const double y_cam = (static_cast<double>(y) + 0.5 - basis.cy) / basis.fy;
        return (basis.x_right * x_cam + basis.y_down * y_cam + basis.z_forward).normalized();
    }

    bool rayIntersectsBbox(const Vec3& origin, const Vec3& dir) const {
        double tmin = -std::numeric_limits<double>::infinity();
        double tmax = std::numeric_limits<double>::infinity();

        auto update_axis = [&](double o, double d) -> bool {
            if (std::abs(d) < 1e-12) {
                return o >= cfg_.map_bbox_min && o <= cfg_.map_bbox_max;
            }
            const double inv_d = 1.0 / d;
            double t1 = (cfg_.map_bbox_min - o) * inv_d;
            double t2 = (cfg_.map_bbox_max - o) * inv_d;
            if (t1 > t2) std::swap(t1, t2);
            tmin = std::max(tmin, t1);
            tmax = std::min(tmax, t2);
            return tmin <= tmax;
        };

        if (!update_axis(origin.x(), dir.x())) return false;
        if (!update_axis(origin.y(), dir.y())) return false;
        if (!update_axis(origin.z(), dir.z())) return false;
        return tmax >= std::max(0.0, tmin);
    }

    double maxRangeForPose(const Pose7d& pose) const {
        const Vec3 camera = cameraPosition(pose);
        const std::array<Vec3, 8> corners = {
            Vec3(cfg_.map_bbox_min, cfg_.map_bbox_min, cfg_.map_bbox_min),
            Vec3(cfg_.map_bbox_min, cfg_.map_bbox_min, cfg_.map_bbox_max),
            Vec3(cfg_.map_bbox_min, cfg_.map_bbox_max, cfg_.map_bbox_min),
            Vec3(cfg_.map_bbox_min, cfg_.map_bbox_max, cfg_.map_bbox_max),
            Vec3(cfg_.map_bbox_max, cfg_.map_bbox_min, cfg_.map_bbox_min),
            Vec3(cfg_.map_bbox_max, cfg_.map_bbox_min, cfg_.map_bbox_max),
            Vec3(cfg_.map_bbox_max, cfg_.map_bbox_max, cfg_.map_bbox_min),
            Vec3(cfg_.map_bbox_max, cfg_.map_bbox_max, cfg_.map_bbox_max),
        };

        double max_range = 0.0;
        for (const Vec3& corner : corners) {
            max_range = std::max(max_range, (corner - camera).norm());
        }
        return max_range + voxelResolution();
    }

    bool pointInsideBbox(const octomap::point3d& p) const {
        return p.x() >= cfg_.map_bbox_min && p.x() <= cfg_.map_bbox_max &&
               p.y() >= cfg_.map_bbox_min && p.y() <= cfg_.map_bbox_max &&
               p.z() >= cfg_.map_bbox_min && p.z() <= cfg_.map_bbox_max;
    }

    VoxelState classifyKey(const octomap::OcTreeKey& key,
                           double* occupancy_out = nullptr,
                           double* entropy_out = nullptr) const {
        octomap::ColorOcTreeNode* node = map_.search(key);
        if (node == nullptr) {
            if (occupancy_out) *occupancy_out = 0.5;
            if (entropy_out) *entropy_out = voxelEntropy(0.5);
            return VoxelState::Unknown;
        }
        const double occupancy = node->getOccupancy();
        if (occupancy_out) *occupancy_out = occupancy;
        if (entropy_out) *entropy_out = voxelEntropy(occupancy);
        if (isUnknownOccupancy(occupancy)) return VoxelState::Unknown;
        if (isOccupiedOccupancy(occupancy)) return VoxelState::Occupied;
        return VoxelState::Free;
    }

    int frontierCheck(const octomap::point3d& node) const {
        int free_cnt = 0;
        int occupied_cnt = 0;
        for (int i = -1; i <= 1; ++i) {
            for (int j = -1; j <= 1; ++j) {
                for (int k = -1; k <= 1; ++k) {
                    if (i == 0 && j == 0 && k == 0) continue;
                    const octomap::point3d neighbour(
                        node.x() + static_cast<float>(i) * static_cast<float>(voxelResolution()),
                        node.y() + static_cast<float>(j) * static_cast<float>(voxelResolution()),
                        node.z() + static_cast<float>(k) * static_cast<float>(voxelResolution()));
                    octomap::OcTreeKey neighbour_key;
                    if (!map_.coordToKeyChecked(neighbour, neighbour_key)) continue;
                    const VoxelState state = classifyKey(neighbour_key);
                    free_cnt += (state == VoxelState::Free) ? 1 : 0;
                    occupied_cnt += (state == VoxelState::Occupied) ? 1 : 0;
                }
            }
        }
        if (free_cnt >= 1 && occupied_cnt >= 1) return 2;
        if (free_cnt >= 1) return 1;
        return 0;
    }

    static double distanceFunction(double distance, double alpha) {
        return std::exp(-(alpha * alpha) * distance);
    }

    std::vector<UnknownVoxelEntry> collectUnknownVoxels() const {
        std::vector<UnknownVoxelEntry> unknowns;
        for (octomap::ColorOcTree::leaf_iterator it = map_.begin_leafs(), end = map_.end_leafs();
             it != end; ++it) {
            const octomap::point3d point = it.getCoordinate();
            if (!pointInsideBbox(point)) continue;

            const double occupancy = (*it).getOccupancy();
            if (!isUnknownOccupancy(occupancy)) continue;

            UnknownVoxelEntry entry;
            entry.key = it.getKey();
            entry.coord = point;
            entry.is_frontier = (frontierCheck(point) == 2);
            unknowns.push_back(entry);
        }
        return unknowns;
    }

    std::unordered_map<octomap::OcTreeKey, double, octomap::OcTreeKey::KeyHash>
    computeObjectWeights() const {
        std::unordered_map<octomap::OcTreeKey, double, octomap::OcTreeKey::KeyHash> weights;
        if (cfg_.method != VoxelIgMethod::APORA && cfg_.method != VoxelIgMethod::PCV) return weights;

        const std::vector<UnknownVoxelEntry> unknowns = collectUnknownVoxels();
        pcl::PointCloud<pcl::PointXYZ>::Ptr frontier(new pcl::PointCloud<pcl::PointXYZ>());
        for (const auto& entry : unknowns) {
            if (!entry.is_frontier) continue;
            frontier->push_back(pcl::PointXYZ(entry.coord.x(), entry.coord.y(), entry.coord.z()));
        }
        if (frontier->empty()) return weights;

        pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
        kdtree.setInputCloud(frontier);

        const int knn = 1;
        const double alpha = 2.0;
        std::vector<int> point_idx_nkn_search(knn);
        std::vector<float> point_nk_squared_distance(knn);

        for (const auto& entry : unknowns) {
            pcl::PointXYZ search_point(entry.coord.x(), entry.coord.y(), entry.coord.z());
            const int found = kdtree.nearestKSearch(
                search_point, knn, point_idx_nkn_search, point_nk_squared_distance);
            if (found <= 0) continue;

            double p_obj = 1.0;
            for (int j = 0; j < found; ++j) {
                p_obj *= distanceFunction(static_cast<double>(point_nk_squared_distance[j]), alpha);
            }
            weights[entry.key] = p_obj;
        }
        return weights;
    }

    static double informationFunction(VoxelIgMethod method,
                                      double ray_information,
                                      double voxel_information,
                                      double visible,
                                      bool is_unknown,
                                      bool previous_voxel_unknown,
                                      bool is_endpoint,
                                      bool is_occupied,
                                      double object_weight,
                                      double object_visible) {
        switch (method) {
            case VoxelIgMethod::OA:
                return ray_information + visible * voxel_information;
            case VoxelIgMethod::UV:
                return is_unknown ? (ray_information + visible * voxel_information) : ray_information;
            case VoxelIgMethod::RSE:
                if (is_endpoint) {
                    if (previous_voxel_unknown) {
                        return is_occupied ? (ray_information + visible * voxel_information) : 0.0;
                    }
                    return 0.0;
                }
                if (is_unknown) {
                    return ray_information + visible * voxel_information;
                }
                return 0.0;
            case VoxelIgMethod::APORA:
                return is_unknown
                    ? (ray_information + object_weight * object_visible * voxel_information)
                    : ray_information;
            case VoxelIgMethod::PCV:
                if (is_endpoint) {
                    return (previous_voxel_unknown && is_occupied) ? ray_information : 0.0;
                }
                if (is_unknown) {
                    return ray_information + object_weight;
                }
                return 0.0;
            case VoxelIgMethod::Kr:
                if (is_endpoint) {
                    return is_occupied ? (ray_information + voxel_information) : 0.0;
                }
                return ray_information + voxel_information;
            case VoxelIgMethod::MCMF:
            case VoxelIgMethod::GMC:
                return is_unknown
                    ? (ray_information + object_weight * visible * voxel_information)
                    : ray_information;
        }
        return ray_information;
    }

    double objectWeightForKey(
        const octomap::OcTreeKey& key,
        const std::unordered_map<octomap::OcTreeKey, double, octomap::OcTreeKey::KeyHash>& object_weights) const {
        const auto it = object_weights.find(key);
        return it == object_weights.end() ? 0.0 : it->second;
    }

    double scoreRay(const Vec3& origin,
                    const Vec3& dir,
                    double max_range,
                    const std::unordered_map<octomap::OcTreeKey, double, octomap::OcTreeKey::KeyHash>& object_weights) const {
        const octomap::point3d octo_origin(origin.x(), origin.y(), origin.z());
        const octomap::point3d end(
            static_cast<float>(origin.x() + dir.x() * max_range),
            static_cast<float>(origin.y() + dir.y() * max_range),
            static_cast<float>(origin.z() + dir.z() * max_range));

        octomap::KeyRay ray;
        if (!map_.computeRayKeys(octo_origin, end, ray)) {
            return 0.0;
        }

        std::vector<octomap::OcTreeKey> inside_keys;
        inside_keys.reserve(ray.size());
        bool endpoint_occupied_found = false;
        for (auto it = ray.begin(); it != ray.end(); ++it) {
            const octomap::point3d coord = map_.keyToCoord(*it);
            if (!pointInsideBbox(coord)) continue;
            if (map_.search(*it) == nullptr) continue;
            inside_keys.push_back(*it);
            if (classifyKey(*it) == VoxelState::Occupied) {
                endpoint_occupied_found = true;
                break;
            }
        }
        if (inside_keys.empty()) return 0.0;

        RayScoreState state;
        for (size_t i = 0; i < inside_keys.size(); ++i) {
            const octomap::OcTreeKey& key = inside_keys[i];
            double occupancy = 0.5;
            double entropy = 1.0;
            const VoxelState voxel_state = classifyKey(key, &occupancy, &entropy);
            const bool is_unknown = voxel_state == VoxelState::Unknown;
            const bool is_occupied = voxel_state == VoxelState::Occupied;
            const bool previous_unknown_before_this = state.previous_voxel_unknown;
            const bool is_endpoint = (i + 1 == inside_keys.size()) || is_occupied;
            const double object_weight = objectWeightForKey(key, object_weights);

            state.gain = informationFunction(
                cfg_.method,
                state.gain,
                entropy,
                state.visible,
                is_unknown,
                previous_unknown_before_this,
                is_endpoint,
                is_occupied,
                object_weight,
                state.object_visible);

            state.object_visible *= (1.0 - object_weight);
            if (cfg_.method == VoxelIgMethod::MCMF || cfg_.method == VoxelIgMethod::GMC) {
                state.visible *= voxelVisible(occupancy);
            } else {
                state.visible *= (1.0 - occupancy);
            }
            state.previous_voxel_unknown = is_unknown;
            state.voxel_num += 1;

            if (is_occupied || is_endpoint) break;
        }

        if (cfg_.method == VoxelIgMethod::RSE && !endpoint_occupied_found) {
            return 0.0;
        }
        return state.gain;
    }

    double scoreViewCpu(const Pose7d& pose,
                        const CameraIntrinsics& intrinsics,
                        const std::unordered_map<octomap::OcTreeKey, double, octomap::OcTreeKey::KeyHash>& object_weights) const {
        const CameraBasis basis = makeCameraBasis(pose, intrinsics);
        const Vec3 origin = cameraPosition(pose);
        const double max_range = maxRangeForPose(pose);
        const int ray_stride = effectiveRayStride();

        double view_gain = 0.0;
        for (int y = 0; y < intrinsics.height; y += ray_stride) {
            for (int x = 0; x < intrinsics.width; x += ray_stride) {
                const Vec3 dir = pixelToWorldRay(x, y, basis);
                if (cfg_.filter_rays_to_bbox && !rayIntersectsBbox(origin, dir)) {
                    continue;
                }
                view_gain += scoreRay(origin, dir, max_range, object_weights);
            }
        }
        return view_gain;
    }

    double scoreViewCuda(const Pose7d& pose,
                         const CameraIntrinsics& intrinsics) const {
        VoxelIgCudaScorer& scorer = ensureCudaScorer();
        const int ray_stride = effectiveRayStride();
        switch (cfg_.method) {
            case VoxelIgMethod::OA:
                return static_cast<double>(scorer.scoreViewOA(pose, intrinsics, ray_stride));
            case VoxelIgMethod::UV:
                return static_cast<double>(scorer.scoreViewUV(pose, intrinsics, ray_stride));
            case VoxelIgMethod::RSE:
                return static_cast<double>(scorer.scoreViewRSE(pose, intrinsics, ray_stride));
            case VoxelIgMethod::Kr:
                return static_cast<double>(scorer.scoreViewKr(pose, intrinsics, ray_stride));
            case VoxelIgMethod::APORA:
            case VoxelIgMethod::PCV:
                throw std::runtime_error(
                    "Voxel-IG CUDA backend currently supports OA/UV/RSE/Kr only.");
            case VoxelIgMethod::MCMF:
            case VoxelIgMethod::GMC:
                throw std::runtime_error(
                    "Voxel-IG methods MCMF/GMC are reserved for future graph-optimization implementations.");
        }
        throw std::runtime_error("Unhandled Voxel-IG CUDA method.");
    }

    double scoreView(const Pose7d& pose,
                     const CameraIntrinsics& intrinsics,
                     const std::unordered_map<octomap::OcTreeKey, double, octomap::OcTreeKey::KeyHash>& object_weights) const {
        if (cfg_.backend == VoxelIgBackend::CUDA) {
            return scoreViewCuda(pose, intrinsics);
        }
        return scoreViewCpu(pose, intrinsics, object_weights);
    }

    int chooseBestView(const AlgorithmContext& ctx, const CameraIntrinsics& intrinsics) const {
        struct CandidateScore {
            int view_id = -1;
            double gain = 0.0;
            double cost = 0.0;
            double utility = 0.0;
        };

        std::unordered_set<int> submitted(ctx.submitted_view_ids.begin(), ctx.submitted_view_ids.end());
        submitted.insert(selected_view_ids_.begin(), selected_view_ids_.end());
        const auto object_weights = computeObjectWeights();
        if (cfg_.backend == VoxelIgBackend::CUDA) {
            DenseGridVolume volume = packDenseGridFromOctomap();
            ensureCudaScorer().uploadVolume(volume);
        }

        std::vector<CandidateScore> candidate_scores;
        candidate_scores.reserve(ctx.candidate_views.size());
        double sum_gain = 0.0;
        double sum_cost = 0.0;
        int best_view_id = -1;
        for (const ViewEntry& candidate : ctx.candidate_views) {
            if (submitted.find(candidate.view_idx) != submitted.end()) continue;
            if (samePose(candidate.pose, ctx.current_pose)) continue;

            if (!cfg_.silent) {
                std::cout << "[voxel_ig] scoring view=" << candidate.view_idx
                          << " pose=(" << candidate.pose.v[0] << ", "
                          << candidate.pose.v[1] << ", "
                          << candidate.pose.v[2] << ")" << std::endl;
            }
            const double gain = scoreView(candidate.pose, intrinsics, object_weights);
            const double cost = (cameraPosition(candidate.pose) - cameraPosition(ctx.current_pose)).norm();
            if (!cfg_.silent) {
                std::cout << "[voxel_ig] view=" << candidate.view_idx
                          << " gain=" << gain
                          << " cost=" << cost << std::endl;
            }
            CandidateScore score;
            score.view_id = candidate.view_idx;
            score.gain = gain;
            score.cost = cost;
            candidate_scores.push_back(score);
            sum_gain += gain;
            sum_cost += cost;
        }

        if (candidate_scores.empty()) {
            return -1;
        }

        const double gain_denom = std::abs(sum_gain) > 1e-12 ? sum_gain : 1.0;
        const double cost_denom = std::abs(sum_cost) > 1e-12 ? sum_cost : 1.0;
        const double gamma = cfg_.use_movement_cost ? cfg_.movement_cost_weight : 0.0;

        double best_utility = -std::numeric_limits<double>::infinity();
        for (CandidateScore& score : candidate_scores) {
            if (cfg_.use_movement_cost) {
                score.utility =
                    (1.0 - gamma) * (score.gain / gain_denom) -
                    gamma * (score.cost / cost_denom);
            } else {
                score.utility = score.gain;
            }
            if (!cfg_.silent) {
                std::cout << "[voxel_ig] view=" << score.view_id
                          << " utility=" << score.utility << std::endl;
            }
            if (score.utility > best_utility) {
                best_utility = score.utility;
                best_view_id = score.view_id;
            }
        }
        return best_view_id;
    }
};

}  // namespace objview

#endif  // OBJVIEWBENCH_VOXEL_IG_ALGORITHM_H_
