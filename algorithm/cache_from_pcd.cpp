#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
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
using PointT = pcl::PointXYZRGB;
using CloudT = pcl::PointCloud<PointT>;

struct Config {
    std::string pcd_path;
    std::string views_path;
    std::string output_dir;

    double resolution{0.02};
    double view_radius{3.0};
    int width{512};
    int height{512};
    double fov_deg{45.0};
    double max_range{6.0};
    bool ignore_unknown{true};

    // render_cpu | render_cuda | inverse_cpu | inverse_cuda | membership_cpu | membership_cuda
    std::string visibility_mode{"render_cuda"};
    Vec3 look_at{0.0, 0.0, 0.0};

    bool save_meta{true};
};

void printUsage(const char* argv0) {
    std::cout
        << "Usage: " << argv0 << " --pcd model.pcd --views 128_xyz.txt --output_dir out_dir [options]\n"
        << "Options:\n"
        << "  --resolution 0.02\n"
        << "  --view-radius 3.0\n"
        << "  --look-at x y z\n"
        << "  --width 512\n"
        << "  --height 512\n"
        << "  --fov 45\n"
        << "  --max-range 6.0\n"
        << "  --ignore-unknown 1\n"
        << "  --visibility-mode render_cuda\n"
        << "  --save-meta 1\n";
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
        else if (arg == "--output_dir") cfg.output_dir = needValue(arg);
        else if (arg == "--resolution") cfg.resolution = std::stod(needValue(arg));
        else if (arg == "--view-radius") cfg.view_radius = std::stod(needValue(arg));
        else if (arg == "--width") cfg.width = std::stoi(needValue(arg));
        else if (arg == "--height") cfg.height = std::stoi(needValue(arg));
        else if (arg == "--fov") cfg.fov_deg = std::stod(needValue(arg));
        else if (arg == "--max-range") cfg.max_range = std::stod(needValue(arg));
        else if (arg == "--ignore-unknown") cfg.ignore_unknown = (std::stoi(needValue(arg)) != 0);
        else if (arg == "--visibility-mode") cfg.visibility_mode = needValue(arg);
        else if (arg == "--save-meta") cfg.save_meta = (std::stoi(needValue(arg)) != 0);
        else if (arg == "--look-at") {
            if (i + 3 >= argc) {
                throw std::runtime_error("Missing 3 values for --look-at");
            }
            cfg.look_at = Vec3(std::stod(argv[++i]), std::stod(argv[++i]), std::stod(argv[++i]));
        }
        else if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            std::exit(0);
        }
        else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (cfg.pcd_path.empty() || cfg.views_path.empty() || cfg.output_dir.empty()) {
        throw std::runtime_error("--pcd, --views, and --output_dir are required.");
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

    return cfg;
}

bool sameKey(const octomap::OcTreeKey& a, const octomap::OcTreeKey& b) {
    return a.k[0] == b.k[0] && a.k[1] == b.k[1] && a.k[2] == b.k[2];
}

struct KeyHash {
    std::size_t operator()(const octomap::OcTreeKey& k) const {
        return octomap::OcTreeKey::KeyHash()(k);
    }
};

struct KeyEqual {
    bool operator()(const octomap::OcTreeKey& a, const octomap::OcTreeKey& b) const {
        return sameKey(a, b);
    }
};

using KeySet = std::unordered_set<octomap::OcTreeKey, KeyHash, KeyEqual>;

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

std::unique_ptr<octomap::ColorOcTree> buildOctomapFromPCD(const std::string& pcd_path, double resolution) {
    CloudT cloud;
    if (pcl::io::loadPCDFile<PointT>(pcd_path, cloud) != 0) {
        throw std::runtime_error("Failed to load PCD: " + pcd_path);
    }
    if (cloud.empty()) {
        throw std::runtime_error("Loaded empty PCD: " + pcd_path);
    }

    auto tree = std::make_unique<octomap::ColorOcTree>(resolution);

    std::size_t inserted = 0;
    std::size_t colored = 0;

    for (const auto& p : cloud.points) {
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) continue;

        octomap::OcTreeKey key;
        if (!tree->coordToKeyChecked(octomap::point3d(p.x, p.y, p.z), key)) continue;

        tree->setNodeValue(key, tree->getProbHitLog(), true);
        tree->integrateNodeColor(key, p.r, p.g, p.b);

        ++inserted;
        if (!(p.r == 0 && p.g == 0 && p.b == 0)) ++colored;
    }

    tree->updateInnerOccupancy();

    std::size_t occupied = 0;
    for (auto it = tree->begin_leafs(), end = tree->end_leafs(); it != end; ++it) {
        if (tree->isNodeOccupied(*it)) ++occupied;
    }

    std::cout << "Loaded PCD points (XYZRGB): " << cloud.size() << "\n";
    std::cout << "Inserted valid points into octomap: " << inserted << "\n";
    std::cout << "Points with nonzero RGB: " << colored << "\n";
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

PointT makeVoxelCenterPoint(const octomap::ColorOcTree& tree, const octomap::OcTreeKey& key) {
    const octomap::point3d coord = tree.keyToCoord(key);

    PointT p;
    p.x = coord.x();
    p.y = coord.y();
    p.z = coord.z();

    auto* node = tree.search(key);
    if (node != nullptr) {
        p.r = node->getColor().r;
        p.g = node->getColor().g;
        p.b = node->getColor().b;
    } else {
        p.r = 255;
        p.g = 255;
        p.b = 255;
    }
    return p;
}

std::string makeViewFilename(int idx) {
    std::ostringstream oss;
    oss << "view_" << std::setw(3) << std::setfill('0') << idx << ".pcd";
    return oss.str();
}

void saveVisibleVoxelCentersAsPCD(
    const octomap::ColorOcTree& tree,
    const std::vector<octomap::OcTreeKey>& visible_keys,
    const std::string& out_path) {

    CloudT out;
    out.points.reserve(visible_keys.size());

    for (const auto& key : visible_keys) {
        out.points.push_back(makeVoxelCenterPoint(tree, key));
    }

    out.width = static_cast<uint32_t>(out.points.size());
    out.height = 1;
    out.is_dense = false;

    if (pcl::io::savePCDFileBinary(out_path, out) != 0) {
        throw std::runtime_error("Failed to save PCD: " + out_path);
    }
}

void saveMeta(const Config& cfg,
              const std::vector<Vec3>& views,
              const std::vector<std::size_t>& visible_counts,
              const std::string& out_path) {
    std::ofstream fout(out_path);
    if (!fout) {
        throw std::runtime_error("Failed to open meta file: " + out_path);
    }

    fout << "pcd_path: " << cfg.pcd_path << "\n";
    fout << "views_path: " << cfg.views_path << "\n";
    fout << "output_dir: " << cfg.output_dir << "\n";
    fout << "resolution: " << cfg.resolution << "\n";
    fout << "view_radius: " << cfg.view_radius << "\n";
    fout << "look_at: " << cfg.look_at.x() << " " << cfg.look_at.y() << " " << cfg.look_at.z() << "\n";
    fout << "width: " << cfg.width << "\n";
    fout << "height: " << cfg.height << "\n";
    fout << "fov_deg: " << cfg.fov_deg << "\n";
    fout << "max_range: " << cfg.max_range << "\n";
    fout << "ignore_unknown: " << (cfg.ignore_unknown ? 1 : 0) << "\n";
    fout << "visibility_mode: " << cfg.visibility_mode << "\n";
    fout << "num_views: " << views.size() << "\n\n";

    fout << "per_view_visible_voxel_counts:\n";
    for (std::size_t i = 0; i < views.size(); ++i) {
        fout << i << ": " << visible_counts[i] << "\n";
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Config cfg = parseArgs(argc, argv);

        std::filesystem::create_directories(cfg.output_dir);

        auto tree = buildOctomapFromPCD(cfg.pcd_path, cfg.resolution);
        const auto views = loadViews(cfg.views_path, cfg.view_radius);

        std::cout << "Loaded views: " << views.size() << "\n";
        std::cout << "Look-at: " << cfg.look_at.x() << " " << cfg.look_at.y() << " " << cfg.look_at.z() << "\n";
        std::cout << "Visibility mode: " << cfg.visibility_mode << "\n";
        std::cout << "Constructing CUDA ray caster..." << std::endl;

        octomap::CudaRayCaster raycaster(*tree);

        std::vector<std::size_t> visible_counts;
        visible_counts.reserve(views.size());

        for (std::size_t i = 0; i < views.size(); ++i) {
            const auto visible_keys = runVisibility(cfg, *tree, raycaster, views[i]);
            visible_counts.push_back(visible_keys.size());

            const std::string out_path =
                (std::filesystem::path(cfg.output_dir) / makeViewFilename(static_cast<int>(i))).string();

            saveVisibleVoxelCentersAsPCD(*tree, visible_keys, out_path);

            std::cout << "Saved view " << i
                      << " -> " << out_path
                      << " | visible voxels = " << visible_keys.size()
                      << "\n";
        }

        if (cfg.save_meta) {
            saveMeta(cfg, views, visible_counts,
                     (std::filesystem::path(cfg.output_dir) / "meta.txt").string());
        }

        std::cout << "Done." << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}

