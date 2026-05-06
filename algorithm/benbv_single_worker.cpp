#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <Eigen/Dense>

#include <json/json.h>

#include <pcl/features/boundary.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>

#include <octomap/ColorOcTree.h>
#include <octomap/octomap.h>

#include <cnpy.h>

#include "cuda_raycaster.h"

namespace fs = std::filesystem;

namespace {

using Vec3 = Eigen::Vector3d;
using PointXYZ = pcl::PointXYZ;
using PointNormal = pcl::PointNormal;
using CloudXYZ = pcl::PointCloud<PointXYZ>;
using CloudNormal = pcl::PointCloud<pcl::Normal>;
using CloudPN = pcl::PointCloud<PointNormal>;

PointXYZ makePointXYZ(float x, float y, float z) {
    PointXYZ p;
    p.x = x;
    p.y = y;
    p.z = z;
    return p;
}

struct Config {
    std::string uid;
    std::string pcd_path;
    std::string view_cache_dir;
    std::string views_path;
    std::string output_dir;

    int start_view_id = -1;
    int start_id = 0;
    int random_seed = 42;

    double resolution = 0.02;
    double camera_distance = 2.0;
    int width = 512;
    int height = 512;
    double fov_deg = 45.0;
    double max_range = 6.0;
    bool ignore_unknown = true;

    int max_steps = 128;
    double coverage_threshold = 0.99;
    int min_gain = 10;

    int candidate_count = 20;
    int point_sample_count = 4096;
    int knn = 30;
    double boundary_angle_deg = 120.0;

    bool debug = false;
};

struct KeyHash {
    std::size_t operator()(const octomap::OcTreeKey& k) const {
        return octomap::OcTreeKey::KeyHash()(k);
    }
};

struct KeyEqual {
    bool operator()(const octomap::OcTreeKey& a, const octomap::OcTreeKey& b) const {
        return a.k[0] == b.k[0] && a.k[1] == b.k[1] && a.k[2] == b.k[2];
    }
};

using KeySet = std::unordered_set<octomap::OcTreeKey, KeyHash, KeyEqual>;

struct KeySetHash {
    std::size_t operator()(const octomap::OcTreeKey& k) const {
        return octomap::OcTreeKey::KeyHash()(k);
    }
};

struct ObjectData {
    std::unique_ptr<octomap::ColorOcTree> tree;
    KeySet universe;
    std::vector<Vec3> views;
};

struct CameraBasis {
    Vec3 x;
    Vec3 y;
    Vec3 z;
    double fx = 0.0;
    double fy = 0.0;
    double cx = 0.0;
    double cy = 0.0;
};

struct Candidate {
    Vec3 target = Vec3::Zero();
    Vec3 direction = Vec3::Zero();
    Vec3 camera = Vec3::Zero();
    bool valid = false;
};

struct StepArrays {
    std::vector<float> P;
    std::vector<float> S;
    std::vector<float> C;
    std::vector<float> y;
    std::vector<int32_t> best_idx;
    std::vector<float> cur_cov;
    std::vector<float> coverage;
    std::vector<float> overlap;
    std::vector<int32_t> gain;
    std::vector<uint8_t> valid_mask;
};

void printUsage(const char* argv0) {
    std::cout
        << "Usage: " << argv0
        << " --uid UID --pcd model.pcd --view_cache_dir DIR --views Tammes_sphere/128_xyz.txt "
        << "--output_dir benbv_training_npz/UID [options]\n"
        << "Options:\n"
        << "  --start_view_id -1        -1 selects one random Tammes128 view\n"
        << "  --start_id 0\n"
        << "  --random_seed 42\n"
        << "  --resolution 0.02\n"
        << "  --camera-distance 2.0\n"
        << "  --width 512 --height 512 --fov 45\n"
        << "  --max-range 6 --ignore-unknown 1\n"
        << "  --max-steps 128 --coverage-threshold 0.99 --min-gain 10\n"
        << "  --candidate-count 20 --point-sample-count 4096 --knn 30\n"
        << "  --debug 0\n";
}

Config parseArgs(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto needValue = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("Missing value for " + name);
            return argv[++i];
        };

        if (arg == "--uid") cfg.uid = needValue(arg);
        else if (arg == "--pcd") cfg.pcd_path = needValue(arg);
        else if (arg == "--view_cache_dir") cfg.view_cache_dir = needValue(arg);
        else if (arg == "--views") cfg.views_path = needValue(arg);
        else if (arg == "--output_dir") cfg.output_dir = needValue(arg);
        else if (arg == "--start_view_id") cfg.start_view_id = std::stoi(needValue(arg));
        else if (arg == "--start_id") cfg.start_id = std::stoi(needValue(arg));
        else if (arg == "--random_seed") cfg.random_seed = std::stoi(needValue(arg));
        else if (arg == "--resolution") cfg.resolution = std::stod(needValue(arg));
        else if (arg == "--camera-distance") cfg.camera_distance = std::stod(needValue(arg));
        else if (arg == "--width") cfg.width = std::stoi(needValue(arg));
        else if (arg == "--height") cfg.height = std::stoi(needValue(arg));
        else if (arg == "--fov") cfg.fov_deg = std::stod(needValue(arg));
        else if (arg == "--max-range") cfg.max_range = std::stod(needValue(arg));
        else if (arg == "--ignore-unknown") cfg.ignore_unknown = std::stoi(needValue(arg)) != 0;
        else if (arg == "--max-steps") cfg.max_steps = std::stoi(needValue(arg));
        else if (arg == "--coverage-threshold") cfg.coverage_threshold = std::stod(needValue(arg));
        else if (arg == "--min-gain") cfg.min_gain = std::stoi(needValue(arg));
        else if (arg == "--candidate-count") cfg.candidate_count = std::stoi(needValue(arg));
        else if (arg == "--point-sample-count") cfg.point_sample_count = std::stoi(needValue(arg));
        else if (arg == "--knn") cfg.knn = std::stoi(needValue(arg));
        else if (arg == "--debug") cfg.debug = std::stoi(needValue(arg)) != 0;
        else if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (cfg.uid.empty() || cfg.pcd_path.empty() || cfg.view_cache_dir.empty() ||
        cfg.views_path.empty() || cfg.output_dir.empty()) {
        throw std::runtime_error("--uid, --pcd, --view_cache_dir, --views, and --output_dir are required.");
    }
    if (cfg.candidate_count != 20) {
        throw std::runtime_error("This first BENBV-Net worker expects --candidate-count 20.");
    }
    if (cfg.width <= 0 || cfg.height <= 0 || cfg.resolution <= 0.0 || cfg.camera_distance <= 0.0) {
        throw std::runtime_error("Invalid camera or resolution parameters.");
    }
    return cfg;
}

Vec3 unit(const Vec3& v, const Vec3& fallback = Vec3(0.0, 0.0, 1.0)) {
    const double n = v.norm();
    if (n < 1e-12) return fallback;
    return v / n;
}

std::string zeroPad3(int x) {
    std::ostringstream oss;
    oss << std::setw(3) << std::setfill('0') << x;
    return oss.str();
}

uint64_t stableStringHash64(const std::string& s) {
    uint64_t h = 14695981039346656037ULL;
    for (const unsigned char c : s) {
        h ^= static_cast<uint64_t>(c);
        h *= 1099511628211ULL;
    }
    return h;
}

uint32_t makeUidSeed(const std::string& uid, int base_seed, uint32_t salt) {
    const uint64_t h = stableStringHash64(uid);
    uint64_t x = h ^ (static_cast<uint64_t>(static_cast<uint32_t>(base_seed)) << 32) ^ static_cast<uint64_t>(salt);
    x ^= (x >> 33);
    x *= 0xff51afd7ed558ccdULL;
    x ^= (x >> 33);
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= (x >> 33);
    return static_cast<uint32_t>(x & 0xffffffffu);
}

std::vector<Vec3> loadViews(const std::string& path, double radius) {
    std::ifstream fin(path);
    if (!fin) throw std::runtime_error("Failed to open views file: " + path);

    std::vector<Vec3> views;
    std::string line;
    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        Vec3 v;
        if (!(iss >> v.x() >> v.y() >> v.z())) {
            throw std::runtime_error("Failed to parse views line: " + line);
        }
        views.push_back(unit(v) * radius);
    }
    if (views.empty()) throw std::runtime_error("No views loaded from: " + path);
    return views;
}

bool loadPcdXYZ(const std::string& path, CloudXYZ::Ptr out) {
    pcl::PointCloud<pcl::PointXYZRGB> rgb;
    if (pcl::io::loadPCDFile<pcl::PointXYZRGB>(path, rgb) == 0 && !rgb.empty()) {
        out->clear();
        out->reserve(rgb.size());
        for (const auto& p : rgb.points) {
            if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) continue;
            out->push_back(makePointXYZ(p.x, p.y, p.z));
        }
        out->width = static_cast<uint32_t>(out->size());
        out->height = 1;
        out->is_dense = false;
        return !out->empty();
    }

    CloudXYZ xyz;
    if (pcl::io::loadPCDFile<PointXYZ>(path, xyz) == 0 && !xyz.empty()) {
        *out = xyz;
        return true;
    }
    return false;
}

std::unique_ptr<octomap::ColorOcTree> buildOctomapFromPCD(const std::string& pcd_path, double resolution) {
    CloudXYZ::Ptr cloud(new CloudXYZ);
    if (!loadPcdXYZ(pcd_path, cloud)) {
        throw std::runtime_error("Failed to load PCD as XYZ/XYZRGB: " + pcd_path);
    }

    auto tree = std::make_unique<octomap::ColorOcTree>(resolution);
    std::size_t inserted = 0;
    for (const auto& p : cloud->points) {
        octomap::OcTreeKey key;
        if (!tree->coordToKeyChecked(octomap::point3d(p.x, p.y, p.z), key)) continue;
        tree->setNodeValue(key, tree->getProbHitLog(), true);
        ++inserted;
    }
    tree->updateInnerOccupancy();
    std::cout << "Loaded object cloud points: " << cloud->size() << "\n";
    std::cout << "Inserted valid occupied samples: " << inserted << "\n";
    return tree;
}

KeySet loadUniverseFromCache(const std::string& view_cache_dir, int expected_num_views, double resolution) {
    octomap::OcTree tmp_tree(resolution);
    KeySet universe;
    universe.reserve(200000);

    for (int vid = 0; vid < expected_num_views; ++vid) {
        const fs::path p = fs::path(view_cache_dir) / ("view_" + zeroPad3(vid) + ".pcd");
        if (!fs::exists(p)) {
            throw std::runtime_error("Missing cached view PCD for universe: " + p.string());
        }
        CloudXYZ::Ptr cloud(new CloudXYZ);
        if (!loadPcdXYZ(p.string(), cloud)) {
            throw std::runtime_error("Failed to load cached view PCD: " + p.string());
        }
        for (const auto& q : cloud->points) {
            octomap::OcTreeKey key;
            if (tmp_tree.coordToKeyChecked(octomap::point3d(q.x, q.y, q.z), key)) {
                universe.insert(key);
            }
        }
    }
    if (universe.empty()) throw std::runtime_error("U_128 universe is empty.");
    return universe;
}

CameraBasis makeCameraBasis(const Vec3& camera_pos,
                            const Vec3& look_at,
                            int width,
                            int height,
                            double fov_deg) {
    CameraBasis cam;
    cam.fx = (static_cast<double>(width) * 0.5) / std::tan(fov_deg * M_PI / 180.0 * 0.5);
    cam.fy = (static_cast<double>(height) * 0.5) / std::tan(fov_deg * M_PI / 180.0 * 0.5);
    cam.cx = static_cast<double>(width) * 0.5;
    cam.cy = static_cast<double>(height) * 0.5;

    cam.z = unit(look_at - camera_pos, Vec3(0.0, 0.0, -1.0));
    if (std::abs(cam.z.dot(Vec3(0.0, 0.0, 1.0))) > 0.999) {
        cam.x = unit(Vec3(1.0, 0.0, 0.0).cross(cam.z), Vec3(0.0, 1.0, 0.0));
    } else {
        cam.x = unit((-cam.z).cross(Vec3(0.0, 0.0, 1.0)), Vec3(1.0, 0.0, 0.0));
    }
    cam.y = unit(cam.x.cross(-cam.z), Vec3(0.0, 1.0, 0.0));
    return cam;
}

std::vector<octomap::OcTreeKey> runVisibilityDynamic(const Config& cfg,
                                                     const octomap::ColorOcTree& tree,
                                                     octomap::CudaRayCaster& raycaster,
                                                     const Vec3& camera_pos,
                                                     const Vec3& target_pos) {
    const CameraBasis cam = makeCameraBasis(camera_pos, target_pos, cfg.width, cfg.height, cfg.fov_deg);

    std::vector<octomap::point3d> origins;
    std::vector<octomap::point3d> dirs;
    std::vector<double> max_ranges;
    const std::size_t ray_count = static_cast<std::size_t>(cfg.width) * static_cast<std::size_t>(cfg.height);
    origins.reserve(ray_count);
    dirs.reserve(ray_count);
    max_ranges.reserve(ray_count);

    for (int v = 0; v < cfg.height; ++v) {
        for (int u = 0; u < cfg.width; ++u) {
            const double x_cam = (static_cast<double>(u) + 0.5 - cam.cx) / cam.fx;
            const double y_cam = (static_cast<double>(v) + 0.5 - cam.cy) / cam.fy;
            const Vec3 dir = unit(cam.x * x_cam + cam.y * y_cam + cam.z);
            origins.emplace_back(camera_pos.x(), camera_pos.y(), camera_pos.z());
            dirs.emplace_back(dir.x(), dir.y(), dir.z());
            max_ranges.push_back(cfg.max_range);
        }
    }

    std::vector<octomap::point3d> end_pts;
    bool* hits = raycaster.castRay(origins, dirs, &end_pts, cfg.ignore_unknown, max_ranges);
    if (hits == nullptr) {
        throw std::runtime_error("CudaRayCaster::castRay returned null hits pointer.");
    }

    KeySet unique;
    unique.reserve(end_pts.size() / 8 + 1);
    for (std::size_t i = 0; i < end_pts.size(); ++i) {
        if (!hits[i]) continue;
        octomap::OcTreeKey key;
        if (!tree.coordToKeyChecked(end_pts[i], key)) continue;
        auto* node = tree.search(key);
        if (node == nullptr || !tree.isNodeOccupied(node)) continue;
        unique.insert(key);
    }
    delete[] hits;

    std::vector<octomap::OcTreeKey> out;
    out.reserve(unique.size());
    for (const auto& k : unique) out.push_back(k);
    return out;
}

KeySet intersectUniverse(const std::vector<octomap::OcTreeKey>& keys, const KeySet& universe) {
    KeySet out;
    out.reserve(keys.size());
    for (const auto& k : keys) {
        if (universe.find(k) != universe.end()) out.insert(k);
    }
    return out;
}

std::vector<Vec3> keysToPoints(const octomap::ColorOcTree& tree, const KeySet& keys) {
    std::vector<Vec3> pts;
    pts.reserve(keys.size());
    for (const auto& k : keys) {
        const octomap::point3d p = tree.keyToCoord(k);
        pts.emplace_back(p.x(), p.y(), p.z());
    }
    return pts;
}

CloudXYZ::Ptr makeCloud(const std::vector<Vec3>& pts) {
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

void saveCloud(const std::vector<Vec3>& pts, const fs::path& path) {
    fs::create_directories(path.parent_path());
    CloudXYZ::Ptr cloud = makeCloud(pts);
    if (pcl::io::savePCDFileBinary(path.string(), *cloud) != 0) {
        throw std::runtime_error("Failed to save PCD: " + path.string());
    }
}

CloudPN::Ptr estimateNormalsOutward(const std::vector<Vec3>& pts, int knn) {
    CloudXYZ::Ptr cloud = makeCloud(pts);
    CloudNormal::Ptr normals(new CloudNormal);
    CloudPN::Ptr out(new CloudPN);
    if (cloud->empty()) return out;

    pcl::NormalEstimationOMP<PointXYZ, pcl::Normal> ne;
    ne.setInputCloud(cloud);
    ne.setSearchMethod(pcl::search::KdTree<PointXYZ>::Ptr(new pcl::search::KdTree<PointXYZ>));
    ne.setKSearch(std::max(3, std::min(knn, static_cast<int>(cloud->size()))));
    ne.compute(*normals);

    out->reserve(cloud->size());
    for (std::size_t i = 0; i < cloud->size(); ++i) {
        const auto& p = (*cloud)[i];
        const auto& n = (*normals)[i];
        Vec3 normal(n.normal_x, n.normal_y, n.normal_z);
        if (!std::isfinite(normal.x()) || !std::isfinite(normal.y()) || !std::isfinite(normal.z()) ||
            normal.norm() < 1e-12) {
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

std::vector<int> computeBoundaryIndices(const CloudPN::Ptr& cloud_pn, int knn, double angle_deg) {
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
    be.setSearchMethod(pcl::search::KdTree<PointXYZ>::Ptr(new pcl::search::KdTree<PointXYZ>));
    be.setKSearch(std::max(3, std::min(knn, static_cast<int>(points->size()))));
    be.setAngleThreshold(angle_deg * M_PI / 180.0);

    pcl::PointCloud<pcl::Boundary> boundaries;
    be.compute(boundaries);
    for (std::size_t i = 0; i < boundaries.size(); ++i) {
        if (boundaries[i].boundary_point != 0) indices.push_back(static_cast<int>(i));
    }
    return indices;
}

std::vector<int> runKMeansSelect(const CloudPN::Ptr& cloud,
                                 const std::vector<int>& boundary_indices,
                                 int k,
                                 int seed) {
    if (boundary_indices.empty()) return {};
    k = std::min(k, static_cast<int>(boundary_indices.size()));
    if (k <= 0) return {};

    std::mt19937 rng(seed);
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
                const double d = (x - centroids[c]).squaredNorm();
                if (d < best_d) {
                    best_d = d;
                    best = c;
                }
            }
            labels[i] = best;
        }

        std::vector<Vec3> sums(k, Vec3::Zero());
        std::vector<int> counts(k, 0);
        for (std::size_t i = 0; i < boundary_indices.size(); ++i) {
            const auto& p = (*cloud)[boundary_indices[i]];
            sums[labels[i]] += Vec3(p.x, p.y, p.z);
            counts[labels[i]] += 1;
        }
        for (int c = 0; c < k; ++c) {
            if (counts[c] > 0) centroids[c] = sums[c] / static_cast<double>(counts[c]);
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
            const double d = (Vec3(p.x, p.y, p.z) - centroids[c]).squaredNorm();
            if (d < best_d) {
                best_d = d;
                best_idx = boundary_indices[i];
            }
        }
        if (best_idx >= 0) selected.push_back(best_idx);
    }
    return selected;
}

Vec3 rotateAroundAxis(const Vec3& v, const Vec3& axis, double rad) {
    const Vec3 a = unit(axis, Vec3(0.0, 0.0, 1.0));
    return v * std::cos(rad) + a.cross(v) * std::sin(rad) + a * (a.dot(v)) * (1.0 - std::cos(rad));
}

std::vector<Candidate> generateCandidates(const CloudPN::Ptr& cloud_pn,
                                          const std::vector<int>& selected_indices,
                                          int candidate_count,
                                          double camera_distance,
                                          int knn) {
    std::vector<Candidate> candidates(candidate_count);
    if (cloud_pn->empty()) return candidates;

    CloudXYZ::Ptr xyz(new CloudXYZ);
    xyz->reserve(cloud_pn->size());
    for (const auto& p : cloud_pn->points) xyz->push_back(makePointXYZ(p.x, p.y, p.z));

    pcl::KdTreeFLANN<PointXYZ> tree;
    tree.setInputCloud(xyz);

    for (std::size_t ci = 0; ci < selected_indices.size() && ci < static_cast<std::size_t>(candidate_count); ++ci) {
        const int idx = selected_indices[ci];
        const auto& p = (*cloud_pn)[idx];
        const Vec3 target(p.x, p.y, p.z);
        Vec3 normal(p.normal_x, p.normal_y, p.normal_z);
        normal = unit(normal, unit(target));

        PointXYZ query{p.x, p.y, p.z};
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

std::vector<float> buildSampleP(const CloudPN::Ptr& cloud_pn, int sample_count, std::mt19937& rng) {
    std::vector<float> out(static_cast<std::size_t>(sample_count) * 6, 0.0f);
    if (cloud_pn->empty()) return out;

    std::vector<int> ids(cloud_pn->size());
    std::iota(ids.begin(), ids.end(), 0);
    if (static_cast<int>(ids.size()) > sample_count) {
        std::shuffle(ids.begin(), ids.end(), rng);
        ids.resize(sample_count);
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

std::vector<float> buildS(const std::vector<Candidate>& candidates) {
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

std::vector<float> computeDensityC(const CloudPN::Ptr& cloud_pn,
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
        PointXYZ q{
            static_cast<float>(candidates[i].target.x()),
            static_cast<float>(candidates[i].target.y()),
            static_cast<float>(candidates[i].target.z())};
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

double overlapQuality(double x) {
    if (x > 0.4 && x < 0.5) return 1.0;
    if (x <= 0.4) return -6.25 * x * x + 5.0 * x;
    return std::max(0.0, -4.0 * x * x + 4.0 * x);
}

double sigmoidCoverageWeight(double cur_cov) {
    return 1.0 / (1.0 + std::exp(-10.0 * (cur_cov - 0.5)));
}

int countGain(const KeySet& visible, const KeySet& covered) {
    int gain = 0;
    for (const auto& k : visible) {
        if (covered.find(k) == covered.end()) ++gain;
    }
    return gain;
}

void mergeInto(KeySet& covered, const KeySet& visible) {
    for (const auto& k : visible) covered.insert(k);
}

std::vector<Vec3> boundaryPointsForDebug(const CloudPN::Ptr& cloud_pn, const std::vector<int>& idxs) {
    std::vector<Vec3> pts;
    pts.reserve(idxs.size());
    for (int idx : idxs) {
        const auto& p = (*cloud_pn)[idx];
        pts.emplace_back(p.x, p.y, p.z);
    }
    return pts;
}

std::vector<Vec3> candidatePointsForDebug(const std::vector<Candidate>& candidates) {
    std::vector<Vec3> pts;
    for (const auto& c : candidates) {
        if (!c.valid) continue;
        pts.push_back(c.target);
        pts.push_back(c.camera);
    }
    return pts;
}

void appendCameraFrameForDebug(std::vector<Vec3>& pts,
                               const Vec3& camera_pos,
                               const Vec3& target_pos,
                               double axis_length = 0.25,
                               int samples_per_axis = 24) {
    const CameraBasis cam = makeCameraBasis(camera_pos, target_pos, 512, 512, 45.0);
    pts.push_back(camera_pos);

    const std::array<Vec3, 3> axes = {cam.x, cam.y, cam.z};
    for (const Vec3& axis : axes) {
        for (int i = 1; i <= samples_per_axis; ++i) {
            const double t = axis_length * static_cast<double>(i) / static_cast<double>(samples_per_axis);
            pts.push_back(camera_pos + axis * t);
        }
    }
}

void saveStepScoreJson(const fs::path& path,
                       int step,
                       double cur_cov,
                       int best_idx,
                       int best_gain,
                       const std::vector<float>& coverage,
                       const std::vector<float>& overlap,
                       const std::vector<int32_t>& gain,
                       const std::vector<float>& score) {
    fs::create_directories(path.parent_path());
    Json::Value root(Json::objectValue);
    root["step"] = step;
    root["cur_cov"] = cur_cov;
    root["best_idx"] = best_idx;
    root["best_gain"] = best_gain;
    root["coverage"] = Json::arrayValue;
    root["overlap"] = Json::arrayValue;
    root["gain"] = Json::arrayValue;
    root["score"] = Json::arrayValue;
    for (float v : coverage) root["coverage"].append(v);
    for (float v : overlap) root["overlap"].append(v);
    for (int32_t v : gain) root["gain"].append(v);
    for (float v : score) root["score"].append(v);

    Json::StreamWriterBuilder builder;
    builder["indentation"] = "  ";
    std::ofstream fout(path, std::ios::binary);
    if (!fout) throw std::runtime_error("Failed to write debug json: " + path.string());
    std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());
    writer->write(root, &fout);
}

void saveMetaJson(const fs::path& path,
                  const Config& cfg,
                  int start_view_id,
                  int steps,
                  const std::string& stop_reason,
                  double final_coverage) {
    fs::create_directories(path.parent_path());
    Json::Value root(Json::objectValue);
    root["uid"] = cfg.uid;
    root["start_id"] = cfg.start_id;
    root["start_view_id"] = start_view_id;
    root["steps"] = steps;
    root["stop_reason"] = stop_reason;
    root["final_coverage"] = final_coverage;
    root["camera_distance"] = cfg.camera_distance;
    root["fov"] = cfg.fov_deg;
    root["width"] = cfg.width;
    root["height"] = cfg.height;
    root["resolution"] = cfg.resolution;
    root["coverage_threshold"] = cfg.coverage_threshold;
    root["min_gain"] = cfg.min_gain;
    root["max_steps"] = cfg.max_steps;

    Json::StreamWriterBuilder builder;
    builder["indentation"] = "  ";
    std::ofstream fout(path, std::ios::binary);
    if (!fout) throw std::runtime_error("Failed to write meta json: " + path.string());
    std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());
    writer->write(root, &fout);
}

void appendStepArrays(StepArrays& arrays,
                      const std::vector<float>& P,
                      const std::vector<float>& S,
                      const std::vector<float>& C,
                      const std::vector<float>& y,
                      int best_idx,
                      float cur_cov,
                      const std::vector<float>& coverage,
                      const std::vector<float>& overlap,
                      const std::vector<int32_t>& gain,
                      const std::vector<uint8_t>& valid_mask) {
    arrays.P.insert(arrays.P.end(), P.begin(), P.end());
    arrays.S.insert(arrays.S.end(), S.begin(), S.end());
    arrays.C.insert(arrays.C.end(), C.begin(), C.end());
    arrays.y.insert(arrays.y.end(), y.begin(), y.end());
    arrays.best_idx.push_back(best_idx);
    arrays.cur_cov.push_back(cur_cov);
    arrays.coverage.insert(arrays.coverage.end(), coverage.begin(), coverage.end());
    arrays.overlap.insert(arrays.overlap.end(), overlap.begin(), overlap.end());
    arrays.gain.insert(arrays.gain.end(), gain.begin(), gain.end());
    arrays.valid_mask.insert(arrays.valid_mask.end(), valid_mask.begin(), valid_mask.end());
}

void saveNpz(const fs::path& path,
             const StepArrays& arrays,
             int T,
             int sample_count,
             int start_view_id) {
    fs::create_directories(path.parent_path());
    cnpy::npz_save(path.string(), "P", arrays.P.data(), {static_cast<size_t>(T), static_cast<size_t>(sample_count), 6}, "w");
    cnpy::npz_save(path.string(), "S", arrays.S.data(), {static_cast<size_t>(T), 20, 6}, "a");
    cnpy::npz_save(path.string(), "C", arrays.C.data(), {static_cast<size_t>(T), 21, 1}, "a");
    cnpy::npz_save(path.string(), "y", arrays.y.data(), {static_cast<size_t>(T), 20, 1}, "a");
    cnpy::npz_save(path.string(), "best_idx", arrays.best_idx.data(), {static_cast<size_t>(T)}, "a");
    cnpy::npz_save(path.string(), "cur_cov", arrays.cur_cov.data(), {static_cast<size_t>(T)}, "a");
    cnpy::npz_save(path.string(), "coverage", arrays.coverage.data(), {static_cast<size_t>(T), 20}, "a");
    cnpy::npz_save(path.string(), "overlap", arrays.overlap.data(), {static_cast<size_t>(T), 20}, "a");
    cnpy::npz_save(path.string(), "gain", arrays.gain.data(), {static_cast<size_t>(T), 20}, "a");
    cnpy::npz_save(path.string(), "valid_mask", arrays.valid_mask.data(), {static_cast<size_t>(T), 20}, "a");
    int32_t start = start_view_id;
    cnpy::npz_save(path.string(), "start_view_id", &start, {1}, "a");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Config cfg = parseArgs(argc, argv);
        fs::create_directories(cfg.output_dir);

        const uint32_t start_seed = makeUidSeed(cfg.uid, cfg.random_seed, 0xBEEB1234u);
        const uint32_t rollout_seed = makeUidSeed(
            cfg.uid,
            cfg.random_seed,
            0xBEEB5678u ^ static_cast<uint32_t>(cfg.start_id * 0x9E3779B9u));
        std::mt19937 rng_start(start_seed);
        std::mt19937 rng_rollout(rollout_seed);

        ObjectData data;
        data.views = loadViews(cfg.views_path, cfg.camera_distance);
        if (data.views.size() != 128) {
            std::cerr << "[WARN] views file count is " << data.views.size() << ", expected 128 for U_128 semantics.\n";
        }
        data.universe = loadUniverseFromCache(cfg.view_cache_dir, static_cast<int>(data.views.size()), cfg.resolution);
        data.tree = buildOctomapFromPCD(cfg.pcd_path, cfg.resolution);

        int start_view_id = cfg.start_view_id;
        if (start_view_id < 0) {
            std::vector<int> start_pool(data.views.size());
            std::iota(start_pool.begin(), start_pool.end(), 0);
            std::shuffle(start_pool.begin(), start_pool.end(), rng_start);
            start_view_id = start_pool[static_cast<std::size_t>(cfg.start_id) % start_pool.size()];
        }
        if (start_view_id < 0 || start_view_id >= static_cast<int>(data.views.size())) {
            throw std::runtime_error("start_view_id out of range.");
        }

        std::cout << "UID: " << cfg.uid << "\n";
        std::cout << "U_128 universe voxels: " << data.universe.size() << "\n";
        std::cout << "Start seed: " << start_seed << "\n";
        std::cout << "Rollout seed: " << rollout_seed << "\n";
        std::cout << "Start id: " << cfg.start_id << "\n";
        std::cout << "Start view id: " << start_view_id << "\n";
        std::cout << "Constructing CUDA raycaster..." << std::endl;
        octomap::CudaRayCaster raycaster(*data.tree);

        const Vec3 center = Vec3::Zero();
        const auto init_raw = runVisibilityDynamic(cfg, *data.tree, raycaster, data.views[start_view_id], center);
        KeySet covered = intersectUniverse(init_raw, data.universe);
        KeySet partial_keys = covered;

        std::string stop_reason = "unknown";
        StepArrays arrays;
        int steps = 0;

        if (cfg.debug) {
            saveCloud(keysToPoints(*data.tree, covered),
                      fs::path(cfg.output_dir) / "debug" / ("start" + zeroPad3(cfg.start_id)) / "initial_visible.pcd");
        }

        for (int step = 0; step < cfg.max_steps; ++step) {
            const double cur_cov = static_cast<double>(covered.size()) / static_cast<double>(data.universe.size());
            if (cur_cov >= cfg.coverage_threshold) {
                stop_reason = "ms_099";
                break;
            }

            const std::vector<Vec3> partial_points = keysToPoints(*data.tree, partial_keys);
            CloudPN::Ptr cloud_pn = estimateNormalsOutward(partial_points, cfg.knn);
            if (cloud_pn->size() < 8) {
                stop_reason = "too_few_points";
                break;
            }

            std::vector<int> boundary = computeBoundaryIndices(cloud_pn, cfg.knn, cfg.boundary_angle_deg);
            if (boundary.empty()) {
                stop_reason = "candidate_empty";
                break;
            }

            const std::vector<int> selected = runKMeansSelect(
                cloud_pn, boundary, cfg.candidate_count, cfg.random_seed + cfg.start_id * 4099 + step);
            std::vector<Candidate> candidates = generateCandidates(
                cloud_pn, selected, cfg.candidate_count, cfg.camera_distance, cfg.knn);

            std::vector<KeySet> visible_sets(20);
            std::vector<float> coverage(20, 0.0f);
            std::vector<float> overlap(20, 0.0f);
            std::vector<float> score(20, 0.0f);
            std::vector<int32_t> gain(20, 0);
            std::vector<uint8_t> valid_mask(20, 0);

            const double w = sigmoidCoverageWeight(cur_cov);
            int best_idx = -1;
            double best_score = -std::numeric_limits<double>::infinity();
            int best_gain = 0;

            for (int ci = 0; ci < 20; ++ci) {
                if (ci >= static_cast<int>(candidates.size()) || !candidates[ci].valid) continue;
                const auto raw = runVisibilityDynamic(
                    cfg, *data.tree, raycaster, candidates[ci].camera, candidates[ci].target);
                visible_sets[ci] = intersectUniverse(raw, data.universe);
                valid_mask[ci] = 1;

                const int g = countGain(visible_sets[ci], covered);
                const int overlap_count = static_cast<int>(visible_sets[ci].size()) - g;
                const double cov_after =
                    static_cast<double>(covered.size() + g) / static_cast<double>(data.universe.size());
                const double ov =
                    visible_sets[ci].empty() ? 0.0 : static_cast<double>(overlap_count) / static_cast<double>(visible_sets[ci].size());
                const double s = w * cov_after + (1.0 - w) * overlapQuality(ov);

                gain[ci] = g;
                coverage[ci] = static_cast<float>(cov_after);
                overlap[ci] = static_cast<float>(ov);
                score[ci] = static_cast<float>(s);

                if (s > best_score) {
                    best_score = s;
                    best_idx = ci;
                    best_gain = g;
                }
            }

            if (best_idx < 0) {
                stop_reason = "candidate_empty";
                break;
            }

            const auto P = buildSampleP(cloud_pn, cfg.point_sample_count, rng_rollout);
            const auto S = buildS(candidates);
            const auto C = computeDensityC(cloud_pn, candidates, step, cfg.knn);
            appendStepArrays(arrays, P, S, C, score, best_idx, static_cast<float>(cur_cov),
                             coverage, overlap, gain, valid_mask);
            ++steps;

            if (cfg.debug) {
                const fs::path step_dir = fs::path(cfg.output_dir) / "debug" /
                                          ("start" + zeroPad3(cfg.start_id)) / ("step_" + zeroPad3(step));
                saveCloud(partial_points, step_dir / "cloud.pcd");
                saveCloud(boundaryPointsForDebug(cloud_pn, boundary), step_dir / "boundary.pcd");
                saveCloud(candidatePointsForDebug(candidates), step_dir / "candidates_target_camera.pcd");
                std::vector<Vec3> best_visible_debug = keysToPoints(*data.tree, visible_sets[best_idx]);
                appendCameraFrameForDebug(
                    best_visible_debug,
                    candidates[best_idx].camera,
                    candidates[best_idx].target);
                saveCloud(best_visible_debug, step_dir / "best_visible.pcd");
                saveStepScoreJson(step_dir / "score.json", step, cur_cov, best_idx, best_gain,
                                  coverage, overlap, gain, score);
            }

            if (best_gain <= cfg.min_gain) {
                stop_reason = "low_gain";
                break;
            }

            mergeInto(covered, visible_sets[best_idx]);
            mergeInto(partial_keys, visible_sets[best_idx]);
        }

        if (stop_reason == "unknown") {
            stop_reason = "max_cap";
        }

        const double final_cov = static_cast<double>(covered.size()) / static_cast<double>(data.universe.size());
        const fs::path npz_path = fs::path(cfg.output_dir) / ("start" + zeroPad3(cfg.start_id) + ".npz");
        const fs::path meta_path = fs::path(cfg.output_dir) / ("start" + zeroPad3(cfg.start_id) + "_meta.json");
        if (steps > 0) {
            saveNpz(npz_path, arrays, steps, cfg.point_sample_count, start_view_id);
        } else {
            std::cerr << "[WARN] No training steps generated; npz not written.\n";
        }
        saveMetaJson(meta_path, cfg, start_view_id, steps, stop_reason, final_cov);

        std::cout << "Done.\n";
        std::cout << "Steps: " << steps << "\n";
        std::cout << "Final coverage: " << final_cov << "\n";
        std::cout << "Stop reason: " << stop_reason << "\n";
        std::cout << "Output: " << cfg.output_dir << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}

