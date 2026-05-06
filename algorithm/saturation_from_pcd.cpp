#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <numeric>
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

#include "cuda_raycaster.h"

namespace {

using Vec3 = Eigen::Vector3d;

struct Config {
    std::string pcd_path;
    std::string tammes_dir;
    std::string output_path{"saturation_result.txt"};

    double resolution{0.02};
    double view_radius{3.0};
    int width{512};
    int height{512};
    double fov_deg{45.0};
    double max_range{6.0};
    bool ignore_unknown{true};
    std::string visibility_mode{"inverse_cuda"};
    Vec3 look_at{0.0, 0.0, 0.0};

    // Saturation rule:
    // N* = min N such that for all N' in [N, N+window],
    // Y_hat(N'+delta)-Y_hat(N') <= epsilon_ratio * gt_surface_voxel_count
    double saturation_delta{10.0};
    double saturation_window{20.0};
    double saturation_epsilon_ratio{2e-4};

    std::vector<int> n_values;
};

void printUsage(const char* argv0) {
    std::cout
        << "Usage: " << argv0 << " --pcd model.pcd --tammes-dir DIR [options]\n"
        << "Options:\n"
        << "  --output result.txt               Output summary path (default: saturation_result.txt)\n"
        << "  --resolution 0.02                Octomap resolution (default: 0.02)\n"
        << "  --view-radius 3.0                Multiply unit view sphere by this radius (default: 3.0)\n"
        << "  --look-at x y z                  Camera look-at target (default: 0 0 0)\n"
        << "  --width 512                      Image width (default: 512)\n"
        << "  --height 512                     Image height (default: 512)\n"
        << "  --fov 45                         Horizontal/vertical FOV in degrees (default: 45)\n"
        << "  --max-range 6.0                  Ray max range (default: 6.0)\n"
        << "  --ignore-unknown 1               Ignore unknown cells in raycast (default: 1)\n"
        << "  --visibility-mode inverse_cuda   Visibility mode (default: inverse_cuda)\n"
        << "  --saturation-delta 10            Delta in N for saturation condition (default: 10)\n"
        << "  --saturation-window 20           Continuous window size for saturation condition (default: 20)\n"
        << "  --saturation-eps-ratio 2e-4      Epsilon ratio wrt GT surface voxel count (default: 2e-4)\n"
        << "  --n-list 6,8,10,...              Optional comma-separated Tammes N list\n";
}

std::vector<int> defaultNValues() {
    return {
        6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 64,
        80, 96, 112, 128, 144, 160, 180, 200,
        270, 360, 432, 492
    };
}

std::vector<int> parseCommaSeparatedInts(const std::string& s) {
    std::vector<int> vals;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) continue;
        vals.push_back(std::stoi(item));
    }
    if (vals.empty()) {
        throw std::runtime_error("Parsed empty integer list from --n-list");
    }
    std::sort(vals.begin(), vals.end());
    vals.erase(std::unique(vals.begin(), vals.end()), vals.end());
    return vals;
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
        else if (arg == "--tammes-dir") cfg.tammes_dir = needValue(arg);
        else if (arg == "--output") cfg.output_path = needValue(arg);
        else if (arg == "--resolution") cfg.resolution = std::stod(needValue(arg));
        else if (arg == "--view-radius") cfg.view_radius = std::stod(needValue(arg));
        else if (arg == "--width") cfg.width = std::stoi(needValue(arg));
        else if (arg == "--height") cfg.height = std::stoi(needValue(arg));
        else if (arg == "--fov") cfg.fov_deg = std::stod(needValue(arg));
        else if (arg == "--max-range") cfg.max_range = std::stod(needValue(arg));
        else if (arg == "--ignore-unknown") cfg.ignore_unknown = (std::stoi(needValue(arg)) != 0);
        else if (arg == "--visibility-mode") cfg.visibility_mode = needValue(arg);
        else if (arg == "--saturation-delta") cfg.saturation_delta = std::stod(needValue(arg));
        else if (arg == "--saturation-window") cfg.saturation_window = std::stod(needValue(arg));
        else if (arg == "--saturation-eps-ratio") cfg.saturation_epsilon_ratio = std::stod(needValue(arg));
        else if (arg == "--n-list") cfg.n_values = parseCommaSeparatedInts(needValue(arg));
        else if (arg == "--look-at") {
            if (i + 3 >= argc) {
                throw std::runtime_error("Missing 3 values for --look-at");
            }
            cfg.look_at = Vec3(std::stod(argv[++i]), std::stod(argv[++i]), std::stod(argv[++i]));
        } else if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (cfg.pcd_path.empty()) throw std::runtime_error("--pcd is required.");
    if (cfg.tammes_dir.empty()) throw std::runtime_error("--tammes-dir is required.");
    if (cfg.width <= 0 || cfg.height <= 0) throw std::runtime_error("Image width/height must be positive.");
    if (cfg.resolution <= 0.0) throw std::runtime_error("Resolution must be positive.");
    if (cfg.view_radius <= 0.0) throw std::runtime_error("View radius must be positive.");
    if (cfg.saturation_delta <= 0.0) throw std::runtime_error("--saturation-delta must be positive.");
    if (cfg.saturation_window < 0.0) throw std::runtime_error("--saturation-window must be nonnegative.");
    if (cfg.saturation_epsilon_ratio <= 0.0) throw std::runtime_error("--saturation-eps-ratio must be positive.");

    if (cfg.n_values.empty()) {
        cfg.n_values = defaultNValues();
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
    for (const auto& p : cloud.points) {
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) continue;
        octomap::OcTreeKey key;
        if (!tree.coordToKeyChecked(octomap::point3d(p.x, p.y, p.z), key)) continue;
        tree.setNodeValue(key, tree.getProbHitLog(), true);
        tree.integrateNodeColor(key, p.r, p.g, p.b);
        ++inserted;
    }
    tree.updateInnerOccupancy();

    std::cout << "Loaded PCD points (XYZRGB): " << cloud.size() << "\n";
    std::cout << "Inserted valid points into octomap: " << inserted << "\n";

    std::size_t occupied = 0;
    for (auto it = tree.begin_leafs(), end = tree.end_leafs(); it != end; ++it) {
        if (tree.isNodeOccupied(*it)) ++occupied;
    }
    std::cout << "Occupied octomap leaf voxels: " << occupied << "\n";
    return tree;
}

std::vector<octomap::OcTreeKey> collectAllOccupiedKeys(const octomap::ColorOcTree& tree) {
    std::vector<octomap::OcTreeKey> keys;
    for (auto it = tree.begin_leafs(), end = tree.end_leafs(); it != end; ++it) {
        if (!tree.isNodeOccupied(*it)) continue;
        keys.push_back(it.getKey());
    }
    return keys;
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
// Inverse visibility
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

// ============================
// Lognormal fit helpers
// ============================

struct LognormalFitResult {
    std::vector<int> n_values;
    std::vector<double> y_raw;
    std::vector<double> y_mono;

    double ymax_fit{0.0};
    double mu{0.0};
    double sigma{1.0};

    double n_star_continuous{0.0};
    int n_star_discrete{-1};

    double y_star_fit{0.0};
    double y_star_discrete{0.0};

    double a_fit{0.0};
    double a_discrete{0.0};
};

std::vector<double> cumulativeMax(const std::vector<double>& y) {
    std::vector<double> out = y;
    for (std::size_t i = 1; i < out.size(); ++i) {
        out[i] = std::max(out[i], out[i - 1]);
    }
    return out;
}

double normalCDF(double x) {
    return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
}

double inverseNormalCDF(double p) {
    if (p <= 0.0 || p >= 1.0) {
        throw std::runtime_error("inverseNormalCDF requires p in (0,1)");
    }

    static const double a1 = -3.969683028665376e+01;
    static const double a2 =  2.209460984245205e+02;
    static const double a3 = -2.759285104469687e+02;
    static const double a4 =  1.383577518672690e+02;
    static const double a5 = -3.066479806614716e+01;
    static const double a6 =  2.506628277459239e+00;

    static const double b1 = -5.447609879822406e+01;
    static const double b2 =  1.615858368580409e+02;
    static const double b3 = -1.556989798598866e+02;
    static const double b4 =  6.680131188771972e+01;
    static const double b5 = -1.328068155288572e+01;

    static const double c1 = -7.784894002430293e-03;
    static const double c2 = -3.223964580411365e-01;
    static const double c3 = -2.400758277161838e+00;
    static const double c4 = -2.549732539343734e+00;
    static const double c5 =  4.374664141464968e+00;
    static const double c6 =  2.938163982698783e+00;

    static const double d1 =  7.784695709041462e-03;
    static const double d2 =  3.224671290700398e-01;
    static const double d3 =  2.445134137142996e+00;
    static const double d4 =  3.754408661907416e+00;

    static const double p_low  = 0.02425;
    static const double p_high = 1.0 - p_low;

    double q, r;
    if (p < p_low) {
        q = std::sqrt(-2.0 * std::log(p));
        return (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
               ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
    }
    if (p <= p_high) {
        q = p - 0.5;
        r = q * q;
        return (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q /
               (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0);
    }

    q = std::sqrt(-2.0 * std::log(1.0 - p));
    return -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
             ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
}

double lognormalCDF(double N, double ymax, double mu, double sigma) {
    if (N <= 0.0) return 0.0;
    sigma = std::max(sigma, 1e-8);
    const double z = (std::log(N) - mu) / sigma;
    return ymax * normalCDF(z);
}

LognormalFitResult fitLognormalCurve(
    const std::vector<int>& n_values,
    const std::vector<double>& y_values,
    double gt_surface_count) {

    if (n_values.size() != y_values.size() || n_values.empty()) {
        throw std::runtime_error("fitLognormalCurve: invalid input sizes");
    }
    if (gt_surface_count <= 0.0) {
        throw std::runtime_error("fitLognormalCurve: gt_surface_count must be positive");
    }

    LognormalFitResult result;
    result.n_values = n_values;
    result.y_raw = y_values;
    result.y_mono = cumulativeMax(y_values);

    result.ymax_fit = result.y_mono.back();
    if (result.ymax_fit <= 0.0) {
        throw std::runtime_error("fitLognormalCurve: ymax_fit <= 0");
    }

    std::vector<double> xs;
    std::vector<double> zs;
    xs.reserve(n_values.size());
    zs.reserve(n_values.size());

    for (std::size_t i = 0; i < n_values.size(); ++i) {
        double p = result.y_mono[i] / result.ymax_fit;
        p = std::max(1e-4, std::min(1.0 - 1e-4, p));

        const double z = inverseNormalCDF(p);
        const double x = std::log(static_cast<double>(n_values[i]));

        xs.push_back(x);
        zs.push_back(z);
    }

    const double mean_z = std::accumulate(zs.begin(), zs.end(), 0.0) / zs.size();
    const double mean_x = std::accumulate(xs.begin(), xs.end(), 0.0) / xs.size();

    double num = 0.0;
    double den = 0.0;
    for (std::size_t i = 0; i < xs.size(); ++i) {
        num += (zs[i] - mean_z) * (xs[i] - mean_x);
        den += (zs[i] - mean_z) * (zs[i] - mean_z);
    }

    if (den <= 1e-12) {
        result.sigma = 1.0;
        result.mu = mean_x;
    } else {
        result.sigma = num / den;
        result.mu = mean_x - result.sigma * mean_z;
    }

    result.sigma = std::max(result.sigma, 1e-6);
    return result;
}

double findSaturationNStarContinuous(
    double ymax,
    double mu,
    double sigma,
    double delta,
    double window,
    double epsilon_abs,
    double n_min,
    double n_max) {

    auto diffFn = [&](double N) -> double {
        const double y1 = lognormalCDF(N, ymax, mu, sigma);
        const double y2 = lognormalCDF(N + delta, ymax, mu, sigma);
        return y2 - y1;
    };

    auto sustainedSatisfied = [&](double N) -> bool {
        const double end = std::min(n_max, N + window);
        const double step = std::max(1.0, delta * 0.5);
        for (double cur = N; cur <= end + 1e-9; cur += step) {
            if (diffFn(cur) > epsilon_abs) {
                return false;
            }
        }
        return true;
    };

    if (sustainedSatisfied(n_min)) return n_min;
    if (!sustainedSatisfied(n_max)) return n_max;

    double lo = n_min;
    double hi = n_max;
    for (int iter = 0; iter < 80; ++iter) {
        const double mid = 0.5 * (lo + hi);
        if (sustainedSatisfied(mid)) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    return hi;
}

int snapToCeilAvailableN(double n_star_continuous, const std::vector<int>& n_values) {
    if (n_values.empty()) {
        throw std::runtime_error("snapToCeilAvailableN: empty n_values");
    }
    for (int n : n_values) {
        if (static_cast<double>(n) >= n_star_continuous) {
            return n;
        }
    }
    return n_values.back();
}

std::size_t findYAtDiscreteN(int target_n,
                             const std::vector<int>& n_values,
                             const std::vector<double>& y_values) {
    for (std::size_t i = 0; i < n_values.size(); ++i) {
        if (n_values[i] == target_n) {
            return static_cast<std::size_t>(std::llround(y_values[i]));
        }
    }
    throw std::runtime_error("findYAtDiscreteN: target_n not found");
}

// ============================
// Observable-surface curve computation
// ============================

std::size_t computeObservableSurfaceCountForViewSet(
    const Config& cfg,
    const octomap::ColorOcTree& tree,
    octomap::CudaRayCaster& raycaster,
    const std::vector<Vec3>& views) {

    KeySet union_keys;
    union_keys.reserve(8192);

    for (const auto& view : views) {
        const auto keys = runVisibility(cfg, tree, raycaster, view);
        for (const auto& k : keys) {
            union_keys.insert(k);
        }
    }
    return union_keys.size();
}

std::string tammesFilePath(const std::string& dir, int n) {
    return (std::filesystem::path(dir) / (std::to_string(n) + "_xyz.txt")).string();
}

void saveResult(
    const Config& cfg,
    std::size_t gt_surface_voxel_count,
    const LognormalFitResult& fit_result) {

    std::ofstream fout(cfg.output_path);
    if (!fout) {
        throw std::runtime_error("Failed to open output file: " + cfg.output_path);
    }

    fout << std::fixed << std::setprecision(8);

    fout << "pcd_path: " << cfg.pcd_path << "\n";
    fout << "tammes_dir: " << cfg.tammes_dir << "\n";
    fout << "resolution: " << cfg.resolution << "\n";
    fout << "view_radius: " << cfg.view_radius << "\n";
    fout << "look_at: " << cfg.look_at.x() << " " << cfg.look_at.y() << " " << cfg.look_at.z() << "\n";
    fout << "width: " << cfg.width << "\n";
    fout << "height: " << cfg.height << "\n";
    fout << "fov_deg: " << cfg.fov_deg << "\n";
    fout << "max_range: " << cfg.max_range << "\n";
    fout << "ignore_unknown: " << (cfg.ignore_unknown ? 1 : 0) << "\n";
    fout << "visibility_mode: " << cfg.visibility_mode << "\n";
    fout << "saturation_delta: " << cfg.saturation_delta << "\n";
    fout << "saturation_window: " << cfg.saturation_window << "\n";
    fout << "saturation_epsilon_ratio: " << cfg.saturation_epsilon_ratio << "\n";

    fout << "gt_surface_voxel_count: " << gt_surface_voxel_count << "\n";
    fout << "fit_type: lognormal_cdf\n";
    fout << "fit_ymax: " << fit_result.ymax_fit << "\n";
    fout << "fit_mu: " << fit_result.mu << "\n";
    fout << "fit_sigma: " << fit_result.sigma << "\n";

    fout << "n_star_continuous: " << fit_result.n_star_continuous << "\n";
    fout << "n_star_discrete: " << fit_result.n_star_discrete << "\n";
    fout << "y_star_fit: " << fit_result.y_star_fit << "\n";
    fout << "y_star_discrete: " << fit_result.y_star_discrete << "\n";
    fout << "A_fit: " << fit_result.a_fit << "\n";
    fout << "A_discrete: " << fit_result.a_discrete << "\n";
    fout << "B_continuous: " << fit_result.n_star_continuous << "\n";
    fout << "B_discrete: " << fit_result.n_star_discrete << "\n";
    fout << "\n";

    fout << "curve_raw:\n";
    for (std::size_t i = 0; i < fit_result.n_values.size(); ++i) {
        fout << fit_result.n_values[i] << ": " << fit_result.y_raw[i] << "\n";
    }
    fout << "\n";

    fout << "curve_monotonicized:\n";
    for (std::size_t i = 0; i < fit_result.n_values.size(); ++i) {
        fout << fit_result.n_values[i] << ": " << fit_result.y_mono[i] << "\n";
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Config cfg = parseArgs(argc, argv);

        auto tree = buildOctomapFromPCD(cfg.pcd_path, cfg.resolution);
        const auto all_keys = collectAllOccupiedKeys(tree);
        const std::size_t gt_surface_voxel_count = all_keys.size();

        std::cout << "GT surface voxel count (all occupied leaf voxels): "
                  << gt_surface_voxel_count << "\n";

        std::cout << "Constructing CUDA ray caster..." << std::endl;
        octomap::CudaRayCaster raycaster(tree);

        std::vector<double> y_values;
        y_values.reserve(cfg.n_values.size());

        for (int n : cfg.n_values) {
            const std::string view_file = tammesFilePath(cfg.tammes_dir, n);
            const auto views = loadViews(view_file, cfg.view_radius);

            if (static_cast<int>(views.size()) != n) {
                std::cerr << "Warning: file " << view_file
                          << " contains " << views.size()
                          << " views, expected " << n << "\n";
            }

            const std::size_t y_n =
                computeObservableSurfaceCountForViewSet(cfg, tree, raycaster, views);

            y_values.push_back(static_cast<double>(y_n));

            std::cout << "Tammes N = " << n
                      << ", observable surface voxels Y(N) = "
                      << y_n << "\n";
        }

        auto fit_result = fitLognormalCurve(cfg.n_values, y_values,
                                            static_cast<double>(gt_surface_voxel_count));

        const double epsilon_abs =
            cfg.saturation_epsilon_ratio * static_cast<double>(gt_surface_voxel_count);

        const double n_min = static_cast<double>(cfg.n_values.front());
        const double n_max = static_cast<double>(cfg.n_values.back());

        fit_result.n_star_continuous = findSaturationNStarContinuous(
            fit_result.ymax_fit,
            fit_result.mu,
            fit_result.sigma,
            cfg.saturation_delta,
            cfg.saturation_window,
            epsilon_abs,
            n_min,
            n_max);

        fit_result.n_star_discrete =
            snapToCeilAvailableN(fit_result.n_star_continuous, cfg.n_values);

        fit_result.y_star_fit =
            lognormalCDF(fit_result.n_star_continuous,
                         fit_result.ymax_fit,
                         fit_result.mu,
                         fit_result.sigma);

        fit_result.y_star_discrete =
            static_cast<double>(findYAtDiscreteN(fit_result.n_star_discrete,
                                                 fit_result.n_values,
                                                 fit_result.y_mono));

        fit_result.a_fit =
            fit_result.y_star_fit / static_cast<double>(gt_surface_voxel_count);
        fit_result.a_discrete =
            fit_result.y_star_discrete / static_cast<double>(gt_surface_voxel_count);

        std::cout << "\n=== Saturation result ===\n";
        std::cout << "n_star_continuous = " << fit_result.n_star_continuous << "\n";
        std::cout << "n_star_discrete   = " << fit_result.n_star_discrete << "\n";
        std::cout << "y_star_fit        = " << fit_result.y_star_fit << "\n";
        std::cout << "y_star_discrete   = " << fit_result.y_star_discrete << "\n";
        std::cout << "A_fit             = " << fit_result.a_fit << "\n";
        std::cout << "A_discrete        = " << fit_result.a_discrete << "\n";
        std::cout << "B_continuous      = " << fit_result.n_star_continuous << "\n";
        std::cout << "B_discrete        = " << fit_result.n_star_discrete << "\n";

        saveResult(cfg, gt_surface_voxel_count, fit_result);
        std::cout << "Saved result to: " << cfg.output_path << std::endl;

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
