#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <Eigen/Dense>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include <octomap/ColorOcTree.h>
#include <octomap/octomap.h>

#include <gurobi_c++.h>

#include "cuda_raycaster.h"

namespace {

using Vec3 = Eigen::Vector3d;

struct Config {
    std::string pcd_path;
    std::string views_path;
    std::string output_path{"set_cover_result.txt"};
    double resolution{0.02};
    double view_radius{3.0};
    int width{512};
    int height{512};
    double fov_deg{45.0};
    double max_range{6.0};
    bool ignore_unknown{true};
    double time_limit_sec{-1.0};
    bool save_vis_pcd{false};
    std::string vis_dir{"vis_pcd"};
    std::string visibility_mode{"inverse_cuda"};  // render_cpu | render_cuda | inverse_cpu | inverse_cuda | membership_cpu | membership_cuda
    Vec3 look_at{0.0, 0.0, 0.0};
    int min_visible_views{1};  // only voxels visible from at least k views enter universe
};

void printUsage(const char* argv0) {
    std::cout
        << "Usage: " << argv0 << " --pcd model.pcd --views 64_xyz.txt [options]\n"
        << "Options:\n"
        << "  --output result.txt             Output summary path (default: set_cover_result.txt)\n"
        << "  --resolution 0.02              Octomap resolution (default: 0.02)\n"
        << "  --view-radius 3.0              Multiply unit view sphere by this radius (default: 3.0)\n"
        << "  --look-at x y z                Camera look-at target (default: 0 0 0)\n"
        << "  --width 512                    Image width (default: 512)\n"
        << "  --height 512                   Image height (default: 512)\n"
        << "  --fov 45                       Horizontal/vertical FOV in degrees (default: 45)\n"
        << "  --max-range 6.0                Ray max range (default: 6.0)\n"
        << "  --ignore-unknown 1             Ignore unknown cells in raycast (default: 1)\n"
        << "  --time-limit 60.0              Time limit in seconds (default: -1.0, no limit)\n"
        << "  --save-vis-pcd 1               Save visualization PCD files (default: 0)\n"
        << "  --vis-dir vis_pcd              Directory for visualization PCD files (default: vis_pcd)\n"
        << "  --visibility-mode inverse_cuda Visibility mode: render_cpu | render_cuda | inverse_cpu | inverse_cuda | membership_cpu | membership_cuda "
        << "(default: inverse_cuda; recommended for observation-oriented view planning benchmark)\n"
        << "  --min-visible-views 2          Keep only voxels visible from at least k views in universe (default: 1)\n";
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

        if (arg == "--pcd") cfg.pcd_path = needValue(arg);
        else if (arg == "--views") cfg.views_path = needValue(arg);
        else if (arg == "--output") cfg.output_path = needValue(arg);
        else if (arg == "--resolution") cfg.resolution = std::stod(needValue(arg));
        else if (arg == "--view-radius") cfg.view_radius = std::stod(needValue(arg));
        else if (arg == "--width") cfg.width = std::stoi(needValue(arg));
        else if (arg == "--height") cfg.height = std::stoi(needValue(arg));
        else if (arg == "--fov") cfg.fov_deg = std::stod(needValue(arg));
        else if (arg == "--max-range") cfg.max_range = std::stod(needValue(arg));
        else if (arg == "--ignore-unknown") cfg.ignore_unknown = (std::stoi(needValue(arg)) != 0);
        else if (arg == "--save-vis-pcd") cfg.save_vis_pcd = (std::stoi(needValue(arg)) != 0);
        else if (arg == "--vis-dir") cfg.vis_dir = needValue(arg);
        else if (arg == "--time-limit") cfg.time_limit_sec = std::stod(needValue(arg));
        else if (arg == "--visibility-mode") cfg.visibility_mode = needValue(arg);
        else if (arg == "--min-visible-views") cfg.min_visible_views = std::stoi(needValue(arg));
        else if (arg == "--look-at") {
            if (i + 3 >= argc) {
                throw std::runtime_error("Missing 3 values for --look-at");
            }
            cfg.look_at = Vec3(std::stod(argv[++i]), std::stod(argv[++i]), std::stod(argv[++i]));
        }
        else if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (cfg.pcd_path.empty() || cfg.views_path.empty()) {
        throw std::runtime_error("Both --pcd and --views are required.");
    }
    if (cfg.width <= 0 || cfg.height <= 0) {
        throw std::runtime_error("Image width/height must be positive.");
    }
    if (cfg.resolution <= 0.0) {
        throw std::runtime_error("Resolution must be positive.");
    }
    if (cfg.view_radius <= 0.0) {
        throw std::runtime_error("View radius must be positive.");
    }
    if (cfg.min_visible_views <= 0) {
        throw std::runtime_error("--min-visible-views must be positive.");
    }

    return cfg;
}

std::vector<Vec3> loadViews(const std::string& path, double radius) {
    std::ifstream fin(path);
    if (!fin) {
        throw std::runtime_error("Failed to open views file: " + path);
    }
    std::vector<Vec3> views;
    std::string line;
    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        Vec3 v;
        if (!(iss >> v.x() >> v.y() >> v.z())) {
            throw std::runtime_error("Failed to parse line in views file: " + line);
        }
        views.push_back(v.normalized() * radius);
    }
    if (views.empty()) {
        throw std::runtime_error("No views loaded from: " + path);
    }
    return views;
}

octomap::ColorOcTree buildOctomapFromPCD(const std::string& pcd_path, double resolution) {
    pcl::PointCloud<pcl::PointXYZRGB> cloud;
    if (pcl::io::loadPCDFile<pcl::PointXYZRGB>(pcd_path, cloud) != 0) {
        throw std::runtime_error("Failed to load PCD as pcl::PointXYZRGB: " + pcd_path);
    }
    if (cloud.empty()) {
        throw std::runtime_error("Loaded empty PCD: " + pcd_path);
    }

    octomap::ColorOcTree tree(resolution);
    std::size_t inserted = 0;
    std::size_t colored = 0;
    for (const auto& p : cloud.points) {
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) continue;
        octomap::OcTreeKey key;
        if (!tree.coordToKeyChecked(octomap::point3d(p.x, p.y, p.z), key)) continue;
        tree.setNodeValue(key, tree.getProbHitLog(), true);
        tree.integrateNodeColor(key, p.r, p.g, p.b);
        ++inserted;
        if (!(p.r == 0 && p.g == 0 && p.b == 0)) ++colored;
    }
    tree.updateInnerOccupancy();

    std::cout << "Loaded PCD points (XYZRGB): " << cloud.size() << "\n";
    std::cout << "Inserted valid points into octomap: " << inserted << "\n";
    std::cout << "Points with nonzero RGB: " << colored << "\n";

    std::size_t occupied = 0;
    for (auto it = tree.begin_leafs(), end = tree.end_leafs(); it != end; ++it) {
        if (tree.isNodeOccupied(*it)) ++occupied;
    }
    std::cout << "Occupied octomap leaf voxels: " << occupied << "\n";
    return tree;
}

struct CameraBasis {
    Vec3 X;
    Vec3 Y;
    Vec3 Z;
    double fx;
    double fy;
    double cx;
    double cy;
};

CameraBasis makeCameraBasis(const Vec3& camera_pos,
                           const Vec3& look_at,
                           int width,
                           int height,
                           double fov_deg) {
    CameraBasis cam;

    cam.fx = (static_cast<double>(width) * 0.5) /
             std::tan(fov_deg * M_PI / 180.0 * 0.5);
    cam.fy = (static_cast<double>(height) * 0.5) /
             std::tan(fov_deg * M_PI / 180.0 * 0.5);
    cam.cx = static_cast<double>(width) * 0.5;
    cam.cy = static_cast<double>(height) * 0.5;

    cam.Z = (look_at - camera_pos).normalized();

    if ((cam.Z - Vec3(0.0, 0.0, -1.0)).norm() < 1e-6) {
        cam.Z = Vec3(1e-8, 1e-8, -1.0).normalized();
    }
    if ((cam.Z - Vec3(0.0, 0.0, 1.0)).norm() < 1e-6) {
        cam.Z = Vec3(1e-8, 1e-8, 1.0).normalized();
    }

    cam.X = ((-cam.Z).cross(Vec3(0.0, 0.0, 1.0))).normalized();
    cam.Y = (cam.X.cross(-cam.Z)).normalized();

    return cam;
}

struct CameraRays {
    std::vector<octomap::point3d> origins;
    std::vector<octomap::point3d> directions;
};

CameraRays generateCameraRays(const Vec3& camera_pos,
                              const Vec3& look_at,
                              int width,
                              int height,
                              double fov_deg) {
    CameraRays rays;
    const std::size_t n = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    rays.origins.reserve(n);
    rays.directions.reserve(n);

    const CameraBasis cam = makeCameraBasis(camera_pos, look_at, width, height, fov_deg);

    for (int v = 0; v < height; ++v) {
        for (int u = 0; u < width; ++u) {
            const double x_cam = (static_cast<double>(u) + 0.5 - cam.cx) / cam.fx;
            const double y_cam = (static_cast<double>(v) + 0.5 - cam.cy) / cam.fy;

            Vec3 dir_world = (cam.X * x_cam + cam.Y * y_cam + cam.Z).normalized();

            rays.origins.emplace_back(camera_pos.x(), camera_pos.y(), camera_pos.z());
            rays.directions.emplace_back(dir_world.x(), dir_world.y(), dir_world.z());
        }
    }

    return rays;
}

bool projectWorldPointToPixel(const Vec3& world_pt,
                              const Vec3& camera_pos,
                              const CameraBasis& cam,
                              int width,
                              int height,
                              int& u,
                              int& v) {
    const Vec3 d = world_pt - camera_pos;

    const double x_cam = d.dot(cam.X);
    const double y_cam = d.dot(cam.Y);
    const double z_cam = d.dot(cam.Z);

    if (z_cam <= 1e-9) return false;

    const double u_f = cam.fx * (x_cam / z_cam) + cam.cx;
    const double v_f = cam.fy * (y_cam / z_cam) + cam.cy;

    if (u_f < 0.0 || u_f >= static_cast<double>(width) ||
        v_f < 0.0 || v_f >= static_cast<double>(height)) {
        return false;
    }

    u = static_cast<int>(std::floor(u_f));
    v = static_cast<int>(std::floor(v_f));
    return true;
}

Vec3 pixelToWorldRay(int u, int v, const CameraBasis& cam) {
    const double x_cam = (static_cast<double>(u) + 0.5 - cam.cx) / cam.fx;
    const double y_cam = (static_cast<double>(v) + 0.5 - cam.cy) / cam.fy;
    return (cam.X * x_cam + cam.Y * y_cam + cam.Z).normalized();
}

bool sameKey(const octomap::OcTreeKey& a, const octomap::OcTreeKey& b) {
    return a.k[0] == b.k[0] && a.k[1] == b.k[1] && a.k[2] == b.k[2];
}

using KeySet = std::unordered_set<octomap::OcTreeKey, octomap::OcTreeKey::KeyHash>;

// ============================
// Render-style visibility
// ============================

std::vector<octomap::OcTreeKey> castAndCollectVisibleKeysRenderCUDA(
    const octomap::ColorOcTree& tree,
    octomap::CudaRayCaster& raycaster,
    const Vec3& camera_pos,
    const Vec3& look_at,
    int width,
    int height,
    double fov_deg,
    double max_range,
    bool ignore_unknown) {

    CameraRays rays = generateCameraRays(camera_pos, look_at, width, height, fov_deg);
    std::vector<double> max_ranges(rays.origins.size(), max_range);
    std::vector<octomap::point3d> end_pts;

    bool* hits = raycaster.castRay(rays.origins, rays.directions, &end_pts, ignore_unknown, max_ranges);
    if (hits == nullptr) {
        throw std::runtime_error("CudaRayCaster::castRay returned null hits pointer.");
    }

    KeySet unique_keys;
    unique_keys.reserve(end_pts.size() / 8 + 1);

    for (std::size_t i = 0; i < end_pts.size(); ++i) {
        if (!hits[i]) continue;
        octomap::OcTreeKey key;
        if (!tree.coordToKeyChecked(end_pts[i], key)) continue;
        auto* node = tree.search(key);
        if (node == nullptr || !tree.isNodeOccupied(node)) continue;
        unique_keys.insert(key);
    }

    delete[] hits;

    std::vector<octomap::OcTreeKey> out;
    out.reserve(unique_keys.size());
    for (const auto& k : unique_keys) out.push_back(k);
    return out;
}

std::vector<octomap::OcTreeKey> castAndCollectVisibleKeysRenderCPU(
    const octomap::ColorOcTree& tree,
    const Vec3& camera_pos,
    const Vec3& look_at,
    int width,
    int height,
    double fov_deg,
    double max_range,
    bool ignore_unknown) {

    CameraRays rays = generateCameraRays(camera_pos, look_at, width, height, fov_deg);

    KeySet unique_keys;
    unique_keys.reserve(rays.origins.size() / 8 + 1);

    for (std::size_t i = 0; i < rays.origins.size(); ++i) {
        octomap::point3d end_pt;
        const bool hit = tree.castRay(
            rays.origins[i],
            rays.directions[i],
            end_pt,
            ignore_unknown,
            max_range);

        if (!hit) continue;

        octomap::OcTreeKey key;
        if (!tree.coordToKeyChecked(end_pt, key)) continue;

        auto* node = tree.search(key);
        if (node == nullptr || !tree.isNodeOccupied(node)) continue;

        unique_keys.insert(key);
    }

    std::vector<octomap::OcTreeKey> out;
    out.reserve(unique_keys.size());
    for (const auto& k : unique_keys) out.push_back(k);
    return out;
}

// ============================
// Inverse visibility (Discrete first-hit visibility under back projection)
// candidate voxel -> pixel -> back-project ray
// no final same-voxel check
// ============================

std::vector<octomap::OcTreeKey> castAndCollectVisibleKeysInverseCPU(
    const octomap::ColorOcTree& tree,
    const Vec3& camera_pos,
    const Vec3& look_at,
    int width,
    int height,
    double fov_deg,
    double max_range,
    bool ignore_unknown) {

    const CameraBasis cam = makeCameraBasis(camera_pos, look_at, width, height, fov_deg);

    KeySet visible_keys;
    visible_keys.reserve(4096);

    for (auto it = tree.begin_leafs(), end = tree.end_leafs(); it != end; ++it) {
        if (!tree.isNodeOccupied(*it)) continue;

        const octomap::point3d coord = it.getCoordinate();
        const Vec3 world_pt(coord.x(), coord.y(), coord.z());

        int u = -1, v = -1;
        if (!projectWorldPointToPixel(world_pt, camera_pos, cam, width, height, u, v)) {
            continue;
        }

        const Vec3 dir = pixelToWorldRay(u, v, cam);

        octomap::point3d end_pt;
        const bool hit = tree.castRay(
            octomap::point3d(camera_pos.x(), camera_pos.y(), camera_pos.z()),
            octomap::point3d(dir.x(), dir.y(), dir.z()),
            end_pt,
            ignore_unknown,
            max_range);

        if (!hit) continue;

        octomap::OcTreeKey hit_key;
        if (!tree.coordToKeyChecked(end_pt, hit_key)) continue;

        auto* node = tree.search(hit_key);
        if (node == nullptr || !tree.isNodeOccupied(node)) continue;

        visible_keys.insert(hit_key);
    }

    std::vector<octomap::OcTreeKey> out;
    out.reserve(visible_keys.size());
    for (const auto& k : visible_keys) out.push_back(k);
    return out;
}

std::vector<octomap::OcTreeKey> castAndCollectVisibleKeysInverseCUDA(
    const octomap::ColorOcTree& tree,
    octomap::CudaRayCaster& raycaster,
    const Vec3& camera_pos,
    const Vec3& look_at,
    int width,
    int height,
    double fov_deg,
    double max_range,
    bool ignore_unknown) {

    const CameraBasis cam = makeCameraBasis(camera_pos, look_at, width, height, fov_deg);

    std::vector<octomap::point3d> origins;
    std::vector<octomap::point3d> dirs;
    std::vector<double> max_ranges;

    origins.reserve(4096);
    dirs.reserve(4096);
    max_ranges.reserve(4096);

    for (auto it = tree.begin_leafs(), end = tree.end_leafs(); it != end; ++it) {
        if (!tree.isNodeOccupied(*it)) continue;

        const octomap::point3d coord = it.getCoordinate();
        const Vec3 world_pt(coord.x(), coord.y(), coord.z());

        int u = -1, v = -1;
        if (!projectWorldPointToPixel(world_pt, camera_pos, cam, width, height, u, v)) {
            continue;
        }

        const Vec3 dir = pixelToWorldRay(u, v, cam);

        origins.emplace_back(camera_pos.x(), camera_pos.y(), camera_pos.z());
        dirs.emplace_back(dir.x(), dir.y(), dir.z());
        max_ranges.push_back(max_range);
    }

    std::vector<octomap::point3d> end_pts;
    bool* hits = raycaster.castRay(origins, dirs, &end_pts, ignore_unknown, max_ranges);
    if (hits == nullptr) {
        throw std::runtime_error("CudaRayCaster::castRay returned null hits pointer.");
    }

    KeySet visible_keys;
    visible_keys.reserve(end_pts.size() / 8 + 1);

    for (std::size_t i = 0; i < end_pts.size(); ++i) {
        if (!hits[i]) continue;

        octomap::OcTreeKey hit_key;
        if (!tree.coordToKeyChecked(end_pts[i], hit_key)) continue;

        auto* node = tree.search(hit_key);
        if (node == nullptr || !tree.isNodeOccupied(node)) continue;

        visible_keys.insert(hit_key);
    }

    delete[] hits;

    std::vector<octomap::OcTreeKey> out;
    out.reserve(visible_keys.size());
    for (const auto& k : visible_keys) out.push_back(k);
    return out;
}

// ============================
// Strict voxel-membership visibility
// candidate voxel -> pixel -> back-project ray
// final same-voxel check required
// ============================

std::vector<octomap::OcTreeKey> castAndCollectVisibleKeysMembershipCPU(
    const octomap::ColorOcTree& tree,
    const Vec3& camera_pos,
    const Vec3& look_at,
    int width,
    int height,
    double fov_deg,
    double max_range,
    bool ignore_unknown) {

    const CameraBasis cam = makeCameraBasis(camera_pos, look_at, width, height, fov_deg);

    std::vector<octomap::OcTreeKey> visible_keys;
    visible_keys.reserve(2048);

    for (auto it = tree.begin_leafs(), end = tree.end_leafs(); it != end; ++it) {
        if (!tree.isNodeOccupied(*it)) continue;

        const octomap::OcTreeKey candidate_key = it.getKey();
        const octomap::point3d coord = it.getCoordinate();
        const Vec3 world_pt(coord.x(), coord.y(), coord.z());

        int u = -1, v = -1;
        if (!projectWorldPointToPixel(world_pt, camera_pos, cam, width, height, u, v)) {
            continue;
        }

        const Vec3 dir = pixelToWorldRay(u, v, cam);

        octomap::point3d end_pt;
        const bool hit = tree.castRay(
            octomap::point3d(camera_pos.x(), camera_pos.y(), camera_pos.z()),
            octomap::point3d(dir.x(), dir.y(), dir.z()),
            end_pt,
            ignore_unknown,
            max_range);

        if (!hit) continue;

        octomap::OcTreeKey hit_key;
        if (!tree.coordToKeyChecked(end_pt, hit_key)) continue;

        if (sameKey(hit_key, candidate_key)) {
            visible_keys.push_back(candidate_key);
        }
    }

    return visible_keys;
}

std::vector<octomap::OcTreeKey> castAndCollectVisibleKeysMembershipCUDA(
    const octomap::ColorOcTree& tree,
    octomap::CudaRayCaster& raycaster,
    const Vec3& camera_pos,
    const Vec3& look_at,
    int width,
    int height,
    double fov_deg,
    double max_range,
    bool ignore_unknown) {

    const CameraBasis cam = makeCameraBasis(camera_pos, look_at, width, height, fov_deg);

    std::vector<octomap::OcTreeKey> candidate_keys;
    std::vector<octomap::point3d> origins;
    std::vector<octomap::point3d> dirs;
    std::vector<double> max_ranges;

    candidate_keys.reserve(4096);
    origins.reserve(4096);
    dirs.reserve(4096);
    max_ranges.reserve(4096);

    for (auto it = tree.begin_leafs(), end = tree.end_leafs(); it != end; ++it) {
        if (!tree.isNodeOccupied(*it)) continue;

        const octomap::OcTreeKey candidate_key = it.getKey();
        const octomap::point3d coord = it.getCoordinate();
        const Vec3 world_pt(coord.x(), coord.y(), coord.z());

        int u = -1, v = -1;
        if (!projectWorldPointToPixel(world_pt, camera_pos, cam, width, height, u, v)) {
            continue;
        }

        const Vec3 dir = pixelToWorldRay(u, v, cam);

        candidate_keys.push_back(candidate_key);
        origins.emplace_back(camera_pos.x(), camera_pos.y(), camera_pos.z());
        dirs.emplace_back(dir.x(), dir.y(), dir.z());
        max_ranges.push_back(max_range);
    }

    std::vector<octomap::point3d> end_pts;
    bool* hits = raycaster.castRay(origins, dirs, &end_pts, ignore_unknown, max_ranges);
    if (hits == nullptr) {
        throw std::runtime_error("CudaRayCaster::castRay returned null hits pointer.");
    }

    std::vector<octomap::OcTreeKey> visible_keys;
    visible_keys.reserve(candidate_keys.size());

    for (std::size_t i = 0; i < candidate_keys.size(); ++i) {
        if (!hits[i]) continue;

        octomap::OcTreeKey hit_key;
        if (!tree.coordToKeyChecked(end_pts[i], hit_key)) continue;

        if (sameKey(hit_key, candidate_keys[i])) {
            visible_keys.push_back(candidate_keys[i]);
        }
    }

    delete[] hits;
    return visible_keys;
}

struct SetCoverResult {
    std::vector<int> selected_views;
    std::size_t raw_universe_size{0};
    std::size_t filtered_universe_size{0};
};

std::vector<std::vector<octomap::OcTreeKey>> filterCoveredKeysByMinVisibleViews(
    const std::vector<std::vector<octomap::OcTreeKey>>& covered_keys_per_view,
    int min_visible_views,
    std::size_t& raw_universe_size,
    std::size_t& filtered_universe_size) {

    std::unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash> voxel_view_count;
    voxel_view_count.reserve(covered_keys_per_view.size() * 1024);

    for (const auto& view_keys : covered_keys_per_view) {
        for (const auto& key : view_keys) {
            voxel_view_count[key] += 1;
        }
    }

    raw_universe_size = voxel_view_count.size();

    std::unordered_set<octomap::OcTreeKey, octomap::OcTreeKey::KeyHash> kept_voxels;
    kept_voxels.reserve(voxel_view_count.size());

    for (const auto& kv : voxel_view_count) {
        if (kv.second >= min_visible_views) {
            kept_voxels.insert(kv.first);
        }
    }

    filtered_universe_size = kept_voxels.size();

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

SetCoverResult solveSetCover(
    const std::vector<std::vector<octomap::OcTreeKey>>& covered_keys_per_view,
    double time_limit_sec = -1.0) {

    std::unordered_map<octomap::OcTreeKey, int, octomap::OcTreeKey::KeyHash> voxel_id_map;
    voxel_id_map.reserve(covered_keys_per_view.size() * 1024);

    int next_id = 0;
    for (const auto& view_keys : covered_keys_per_view) {
        for (const auto& key : view_keys) {
            if (voxel_id_map.find(key) == voxel_id_map.end()) {
                voxel_id_map.emplace(key, next_id++);
            }
        }
    }

    const int num_views = static_cast<int>(covered_keys_per_view.size());
    const int num_voxels = next_id;

    std::vector<std::vector<int>> views_per_voxel(num_voxels);
    for (int view_id = 0; view_id < num_views; ++view_id) {
        for (const auto& key : covered_keys_per_view[view_id]) {
            views_per_voxel[voxel_id_map.at(key)].push_back(view_id);
        }
    }

    GRBEnv env(true);
    env.set("LogToConsole", "1");
    env.start();
    GRBModel model(env);

    if (time_limit_sec > 0.0) {
        model.set(GRB_DoubleParam_TimeLimit, time_limit_sec);
    }

    std::vector<GRBVar> x(num_views);
    for (int i = 0; i < num_views; ++i) {
        x[i] = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, "x_" + std::to_string(i));
    }

    GRBLinExpr obj = 0;
    for (int i = 0; i < num_views; ++i) obj += x[i];
    model.setObjective(obj, GRB_MINIMIZE);

    for (int vid = 0; vid < num_voxels; ++vid) {
        GRBLinExpr cover = 0;
        for (int view_id : views_per_voxel[vid]) {
            cover += x[view_id];
        }
        model.addConstr(cover >= 1.0, "cover_" + std::to_string(vid));
    }

    model.optimize();

    SetCoverResult result;
    result.filtered_universe_size = static_cast<std::size_t>(num_voxels);

    const int status = model.get(GRB_IntAttr_Status);
    if (status != GRB_OPTIMAL && status != GRB_SUBOPTIMAL && status != GRB_TIME_LIMIT) {
        throw std::runtime_error("Gurobi failed with status: " + std::to_string(status));
    }

    for (int i = 0; i < num_views; ++i) {
        if (x[i].get(GRB_DoubleAttr_X) > 0.5) {
            result.selected_views.push_back(i);
        }
    }
    return result;
}

void saveResult(const Config& cfg,
                const std::vector<Vec3>& views,
                const std::vector<std::vector<octomap::OcTreeKey>>& covered_keys_per_view,
                const SetCoverResult& result) {
    std::ofstream fout(cfg.output_path);
    if (!fout) {
        throw std::runtime_error("Failed to open output file: " + cfg.output_path);
    }

    fout << "pcd_path: " << cfg.pcd_path << "\n";
    fout << "views_path: " << cfg.views_path << "\n";
    fout << "resolution: " << cfg.resolution << "\n";
    fout << "view_radius: " << cfg.view_radius << "\n";
    fout << "look_at: " << cfg.look_at.x() << " " << cfg.look_at.y() << " " << cfg.look_at.z() << "\n";
    fout << "width: " << cfg.width << "\n";
    fout << "height: " << cfg.height << "\n";
    fout << "fov_deg: " << cfg.fov_deg << "\n";
    fout << "max_range: " << cfg.max_range << "\n";
    fout << "ignore_unknown: " << (cfg.ignore_unknown ? 1 : 0) << "\n";
    fout << "visibility_mode: " << cfg.visibility_mode << "\n";
    fout << "min_visible_views: " << cfg.min_visible_views << "\n";
    fout << "num_views: " << views.size() << "\n";
    fout << "raw_universe_voxels: " << result.raw_universe_size << "\n";
    fout << "filtered_universe_voxels: " << result.filtered_universe_size << "\n";
    fout << "selected_view_count: " << result.selected_views.size() << "\n";
    fout << "selected_view_ids:";
    for (int id : result.selected_views) fout << ' ' << id;
    fout << "\n\n";

    fout << "per_view_unique_voxel_counts:\n";
    for (std::size_t i = 0; i < covered_keys_per_view.size(); ++i) {
        fout << i << ": " << covered_keys_per_view[i].size() << "\n";
    }
}

using VisCloud = pcl::PointCloud<pcl::PointXYZRGB>;

pcl::PointXYZRGB makePoint(float x, float y, float z, uint8_t r, uint8_t g, uint8_t b) {
    pcl::PointXYZRGB p;
    p.x = x; p.y = y; p.z = z;
    p.r = r; p.g = g; p.b = b;
    return p;
}

void appendVoxelKeyAsPoint(const octomap::ColorOcTree& tree,
                           const octomap::OcTreeKey& key,
                           uint8_t r, uint8_t g, uint8_t b,
                           VisCloud::Ptr cloud) {
    const octomap::point3d coord = tree.keyToCoord(key);
    cloud->points.push_back(makePoint(coord.x(), coord.y(), coord.z(), r, g, b));
}

std::vector<octomap::OcTreeKey> collectAllOccupiedKeys(const octomap::ColorOcTree& tree) {
    std::vector<octomap::OcTreeKey> keys;
    for (auto it = tree.begin_leafs(), end = tree.end_leafs(); it != end; ++it) {
        if (!tree.isNodeOccupied(*it)) continue;
        keys.push_back(it.getKey());
    }
    return keys;
}

void appendCameraAxes(const Vec3& camera_pos,
                      const Vec3& look_at,
                      double axis_length,
                      double step,
                      VisCloud::Ptr cloud) {
    Vec3 Z = (look_at - camera_pos).normalized();
    if ((Z - Vec3(0.0, 0.0, -1.0)).norm() < 1e-6) Z = Vec3(1e-8, 1e-8, -1.0).normalized();
    if ((Z - Vec3(0.0, 0.0, 1.0)).norm() < 1e-6) Z = Vec3(1e-8, 1e-8, 1.0).normalized();

    Vec3 X = ((-Z).cross(Vec3(0.0, 0.0, 1.0))).normalized();
    Vec3 Y = (X.cross(-Z)).normalized();

    cloud->points.push_back(makePoint(camera_pos.x(), camera_pos.y(), camera_pos.z(), 255, 255, 0));

    const int n = std::max(1, static_cast<int>(std::ceil(axis_length / step)));
    for (int i = 1; i <= n; ++i) {
        const double t = std::min(axis_length, i * step);

        Vec3 px = camera_pos + X * t;
        Vec3 py = camera_pos + Y * t;
        Vec3 pz = camera_pos + Z * t;

        cloud->points.push_back(makePoint(px.x(), px.y(), px.z(), 255, 0, 0));
        cloud->points.push_back(makePoint(py.x(), py.y(), py.z(), 0, 255, 0));
        cloud->points.push_back(makePoint(pz.x(), pz.y(), pz.z(), 0, 0, 255));
    }
}

void appendCameraAxes(const Vec3& camera_pos,
                      const Vec3& look_at,
                      double axis_length,
                      double step,
                      uint8_t r,
                      uint8_t g,
                      uint8_t b,
                      VisCloud::Ptr cloud) {
    Vec3 Z = (look_at - camera_pos).normalized();
    if ((Z - Vec3(0.0, 0.0, -1.0)).norm() < 1e-6) Z = Vec3(1e-8, 1e-8, -1.0).normalized();
    if ((Z - Vec3(0.0, 0.0, 1.0)).norm() < 1e-6) Z = Vec3(1e-8, 1e-8, 1.0).normalized();

    Vec3 X = ((-Z).cross(Vec3(0.0, 0.0, 1.0))).normalized();
    Vec3 Y = (X.cross(-Z)).normalized();

    cloud->points.push_back(makePoint(camera_pos.x(), camera_pos.y(), camera_pos.z(), r, g, b));

    const int n = std::max(1, static_cast<int>(std::ceil(axis_length / step)));
    for (int i = 1; i <= n; ++i) {
        const double t = std::min(axis_length, i * step);

        Vec3 px = camera_pos + X * t;
        Vec3 py = camera_pos + Y * t;
        Vec3 pz = camera_pos + Z * t;

        cloud->points.push_back(makePoint(px.x(), px.y(), px.z(), r, g, b));
        cloud->points.push_back(makePoint(py.x(), py.y(), py.z(), r, g, b));
        cloud->points.push_back(makePoint(pz.x(), pz.y(), pz.z(), r, g, b));
    }
}

void savePerViewVisualizationPCD(const octomap::ColorOcTree& tree,
                                 const std::vector<octomap::OcTreeKey>& all_keys,
                                 const std::vector<octomap::OcTreeKey>& covered_keys,
                                 const Vec3& view_pos,
                                 const Vec3& look_at,
                                 const std::string& path) {
    VisCloud::Ptr cloud(new VisCloud);

    std::unordered_set<octomap::OcTreeKey, octomap::OcTreeKey::KeyHash> covered_set;
    covered_set.reserve(covered_keys.size() * 2 + 1);
    for (const auto& k : covered_keys) covered_set.insert(k);

    for (const auto& key : all_keys) {
        if (covered_set.find(key) != covered_set.end()) {
            appendVoxelKeyAsPoint(tree, key, 255, 64, 64, cloud);
        } else {
            appendVoxelKeyAsPoint(tree, key, 180, 180, 180, cloud);
        }
    }

    appendCameraAxes(view_pos, look_at, 0.25, 0.01, cloud);

    cloud->width = static_cast<uint32_t>(cloud->points.size());
    cloud->height = 1;
    cloud->is_dense = false;

    if (pcl::io::savePCDFileBinary(path, *cloud) != 0) {
        throw std::runtime_error("Failed to save per-view visualization PCD: " + path);
    }
}

struct RGB {
    uint8_t r, g, b;
};

RGB colorFromViewId(int view_id) {
    static const std::vector<RGB> palette = {
        {230, 25, 75}, {60, 180, 75}, {0, 130, 200}, {245, 130, 48},
        {145, 30, 180}, {70, 240, 240}, {240, 50, 230}, {210, 245, 60},
        {250, 190, 190}, {0, 128, 128}, {230, 190, 255}, {170, 110, 40},
        {255, 250, 200}, {128, 0, 0}, {170, 255, 195}, {128, 128, 0},
        {255, 215, 180}, {0, 0, 128}, {128, 128, 128}, {255, 255, 255}
    };
    return palette[view_id % palette.size()];
}

void saveSelectedViewsVisualizationPCD(
    const octomap::ColorOcTree& tree,
    const std::vector<octomap::OcTreeKey>& all_keys,
    const std::vector<std::vector<octomap::OcTreeKey>>& covered_keys_per_view,
    const std::vector<Vec3>& views,
    const Vec3& look_at,
    const std::vector<int>& selected_view_ids,
    const std::string& path) {

    VisCloud::Ptr cloud(new VisCloud);

    std::unordered_map<octomap::OcTreeKey, RGB, octomap::OcTreeKey::KeyHash> voxel_color;
    voxel_color.reserve(all_keys.size() * 2 + 1);

    std::unordered_set<int> selected_ids(selected_view_ids.begin(), selected_view_ids.end());

    for (int vid : selected_view_ids) {
        const RGB c = colorFromViewId(vid);
        for (const auto& k : covered_keys_per_view[vid]) {
            if (voxel_color.find(k) == voxel_color.end()) {
                voxel_color.emplace(k, c);
            }
        }
    }

    for (const auto& key : all_keys) {
        auto it = voxel_color.find(key);
        if (it != voxel_color.end()) {
            appendVoxelKeyAsPoint(tree, key, it->second.r, it->second.g, it->second.b, cloud);
        } else {
            appendVoxelKeyAsPoint(tree, key, 180, 180, 180, cloud);
        }
    }

    for (std::size_t i = 0; i < views.size(); ++i) {
        if (selected_ids.find(static_cast<int>(i)) != selected_ids.end()) {
            RGB c = colorFromViewId(static_cast<int>(i));
            appendCameraAxes(views[i], look_at, 0.25, 0.01, c.r, c.g, c.b, cloud);
        } else {
            cloud->points.push_back(makePoint(views[i].x(), views[i].y(), views[i].z(), 80, 80, 80));
        }
    }

    cloud->width = static_cast<uint32_t>(cloud->points.size());
    cloud->height = 1;
    cloud->is_dense = false;

    if (pcl::io::savePCDFileBinary(path, *cloud) != 0) {
        throw std::runtime_error("Failed to save selected-views visualization PCD: " + path);
    }
}

std::vector<octomap::OcTreeKey> runVisibility(
    const Config& cfg,
    const octomap::ColorOcTree& tree,
    octomap::CudaRayCaster& raycaster,
    const Vec3& view) {

    if (cfg.visibility_mode == "render_cpu") {
        return castAndCollectVisibleKeysRenderCPU(
            tree, view, cfg.look_at, cfg.width, cfg.height, cfg.fov_deg,
            cfg.max_range, cfg.ignore_unknown);
    }
    if (cfg.visibility_mode == "render_cuda") {
        return castAndCollectVisibleKeysRenderCUDA(
            tree, raycaster, view, cfg.look_at, cfg.width, cfg.height, cfg.fov_deg,
            cfg.max_range, cfg.ignore_unknown);
    }
    if (cfg.visibility_mode == "inverse_cpu") {
        return castAndCollectVisibleKeysInverseCPU(
            tree, view, cfg.look_at, cfg.width, cfg.height, cfg.fov_deg,
            cfg.max_range, cfg.ignore_unknown);
    }
    if (cfg.visibility_mode == "inverse_cuda") {
        return castAndCollectVisibleKeysInverseCUDA(
            tree, raycaster, view, cfg.look_at, cfg.width, cfg.height, cfg.fov_deg,
            cfg.max_range, cfg.ignore_unknown);
    }
    if (cfg.visibility_mode == "membership_cpu") {
        return castAndCollectVisibleKeysMembershipCPU(
            tree, view, cfg.look_at, cfg.width, cfg.height, cfg.fov_deg,
            cfg.max_range, cfg.ignore_unknown);
    }
    if (cfg.visibility_mode == "membership_cuda") {
        return castAndCollectVisibleKeysMembershipCUDA(
            tree, raycaster, view, cfg.look_at, cfg.width, cfg.height, cfg.fov_deg,
            cfg.max_range, cfg.ignore_unknown);
    }

    throw std::runtime_error("Unknown visibility mode: " + cfg.visibility_mode);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Config cfg = parseArgs(argc, argv);

        if (cfg.save_vis_pcd) {
            std::filesystem::create_directories(cfg.vis_dir);
        }

        auto tree = buildOctomapFromPCD(cfg.pcd_path, cfg.resolution);
        const auto views = loadViews(cfg.views_path, cfg.view_radius);
        const auto all_keys = collectAllOccupiedKeys(tree);

        std::cout << "Loaded views: " << views.size() << "\n";
        std::cout << "Look-at: " << cfg.look_at.x() << " " << cfg.look_at.y() << " " << cfg.look_at.z() << "\n";
        std::cout << "Visibility mode: " << cfg.visibility_mode << "\n";
        std::cout << "min_visible_views: " << cfg.min_visible_views << "\n";
        std::cout << "Constructing CUDA ray caster..." << std::endl;
        octomap::CudaRayCaster raycaster(tree);

        std::vector<std::vector<octomap::OcTreeKey>> covered_keys_per_view;
        covered_keys_per_view.reserve(views.size());

        for (std::size_t i = 0; i < views.size(); ++i) {
            const auto keys = runVisibility(cfg, tree, raycaster, views[i]);

            std::cout << "View " << i
                      << " @ (" << views[i].x() << ", " << views[i].y() << ", " << views[i].z()
                      << ") covers " << keys.size() << " unique occupied voxels\n";

            covered_keys_per_view.push_back(keys);

            if (cfg.save_vis_pcd) {
                const std::string path = cfg.vis_dir + "/view_" + std::to_string(i) + ".pcd";
                savePerViewVisualizationPCD(tree, all_keys, keys, views[i], cfg.look_at, path);
            }
        }

        std::size_t raw_universe_size = 0;
        std::size_t filtered_universe_size = 0;

        const auto filtered_covered_keys_per_view =
            filterCoveredKeysByMinVisibleViews(
                covered_keys_per_view,
                cfg.min_visible_views,
                raw_universe_size,
                filtered_universe_size);

        std::cout << "Raw universe voxel count: " << raw_universe_size << "\n";
        std::cout << "Filtered universe voxel count (visible by >= "
                  << cfg.min_visible_views << " views): "
                  << filtered_universe_size << "\n";

        auto result = solveSetCover(filtered_covered_keys_per_view, cfg.time_limit_sec);
        result.raw_universe_size = raw_universe_size;
        result.filtered_universe_size = filtered_universe_size;

        std::cout << "Selected " << result.selected_views.size() << " views:";
        for (int id : result.selected_views) std::cout << ' ' << id;
        std::cout << std::endl;

        saveResult(cfg, views, filtered_covered_keys_per_view, result);
        std::cout << "Saved result to: " << cfg.output_path << std::endl;

        if (cfg.save_vis_pcd) {
            saveSelectedViewsVisualizationPCD(
                tree, all_keys, filtered_covered_keys_per_view, views, cfg.look_at,
                result.selected_views,
                cfg.vis_dir + "/selected_views_union.pcd");
        }

        return 0;
    } catch (const GRBException& e) {
        std::cerr << "Gurobi error: code " << e.getErrorCode() << ", " << e.getMessage() << std::endl;
        return 2;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}

