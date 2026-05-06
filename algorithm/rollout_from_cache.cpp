#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
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
#include <pcl/point_cloud.h>

#include <octomap/octomap.h>
#include <octomap/ColorOcTree.h>

#include <json/json.h>
#include <gurobi_c++.h>

namespace fs = std::filesystem;

using Vec3 = Eigen::Vector3d;
using PointT = pcl::PointXYZRGB;
using CloudT = pcl::PointCloud<PointT>;

struct Config {
    std::string view_cache_dir;     // .../view_pcd_cache/<uid>
    std::string views_path;         // Tammes 128 view file
    std::string output_json;        // .../oracle_rollout/<uid>.json

    double resolution = 0.02;
    int expected_num_views = 128;
    int max_steps = -1;             // -1 = rollout until no gain / fully covered
    double time_limit_sec = -1.0;   // Gurobi time limit
    int skip_existing = 0;

    // visualization
    int save_vis = 0;
    std::string vis_dir = "oracle_rollout_vis";
    int vis_start_view = -1;        // -1 = all
    int vis_max_steps = -1;         // -1 = all
    double axis_length = 0.25;
    double axis_step = 0.01;
    Vec3 look_at{0.0, 0.0, 0.0};
    double view_radius = 3.0;
};

struct StepRecord {
    int step_id = 0;
    std::vector<int> visited_view_ids;
    std::vector<int> candidate_view_ids;
    std::vector<int> candidate_marginal_gains;

    int oracle_nbv_view_id = -1;
    int oracle_nbv_gain = 0;

    int covered_voxel_count = 0;
    int residual_voxel_count = 0;
    double covered_ratio = 0.0;

    std::vector<int> residual_set_cover_view_ids;
    int residual_set_cover_size = 0;
};

struct RolloutRecord {
    int start_view_id = -1;
    int universe_voxel_count = 0;
    std::vector<StepRecord> steps;
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
using KeyToIdMap = std::unordered_map<octomap::OcTreeKey, int, KeyHash, KeyEqual>;

struct StaticObjectData {
    std::string uid;
    std::vector<std::vector<int>> coverage_ids_per_view; // per-view covered universe ids
    std::vector<octomap::OcTreeKey> id_to_key;           // universe id -> key
    std::vector<Vec3> views;
    int universe_size = 0;
};

void printUsage(const char* argv0) {
    std::cout
        << "Usage: " << argv0 << " --view_cache_dir DIR --views PATH --output_json PATH [options]\n"
        << "Options:\n"
        << "  --resolution 0.02\n"
        << "  --expected_num_views 128\n"
        << "  --max_steps -1\n"
        << "  --time-limit -1\n"
        << "  --skip_existing 0\n"
        << "  --look-at x y z\n"
        << "  --view-radius 3.0\n"
        << "  --save_vis 0\n"
        << "  --vis_dir DIR\n"
        << "  --vis_start_view -1\n"
        << "  --vis_max_steps -1\n"
        << "  --axis_length 0.25\n"
        << "  --axis_step 0.01\n";
}

Config parseArgs(int argc, char** argv) {
    Config cfg;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        auto needValue = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("Missing value for " + name);
            return argv[++i];
        };

        if (arg == "--view_cache_dir") cfg.view_cache_dir = needValue(arg);
        else if (arg == "--views") cfg.views_path = needValue(arg);
        else if (arg == "--output_json") cfg.output_json = needValue(arg);
        else if (arg == "--resolution") cfg.resolution = std::stod(needValue(arg));
        else if (arg == "--expected_num_views") cfg.expected_num_views = std::stoi(needValue(arg));
        else if (arg == "--max_steps") cfg.max_steps = std::stoi(needValue(arg));
        else if (arg == "--time-limit") cfg.time_limit_sec = std::stod(needValue(arg));
        else if (arg == "--skip_existing") cfg.skip_existing = std::stoi(needValue(arg))
;
        else if (arg == "--save_vis") cfg.save_vis = std::stoi(needValue(arg));
        else if (arg == "--vis_dir") cfg.vis_dir = needValue(arg);
        else if (arg == "--vis_start_view") cfg.vis_start_view = std::stoi(needValue(arg));
        else if (arg == "--vis_max_steps") cfg.vis_max_steps = std::stoi(needValue(arg));
        else if (arg == "--axis_length") cfg.axis_length = std::stod(needValue(arg));
        else if (arg == "--axis_step") cfg.axis_step = std::stod(needValue(arg));
        else if (arg == "--look-at") {
            if (i + 3 >= argc) throw std::runtime_error("Missing 3 values for --look-at");
            cfg.look_at = Vec3(std::stod(argv[++i]), std::stod(argv[++i]), std::stod(argv[++i]));
        }
        else if (arg == "--view-radius") cfg.view_radius = std::stod(needValue(arg));
        else if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            std::exit(0);
        }
        else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (cfg.view_cache_dir.empty() || cfg.views_path.empty() || cfg.output_json.empty()) {
        throw std::runtime_error("--view_cache_dir, --views, and --output_json are required.");
    }
    if (cfg.expected_num_views <= 0) {
        throw std::runtime_error("--expected_num_views must be positive.");
    }

    return cfg;
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
        views.push_back(v.normalized() * radius);
    }
    if (views.empty()) throw std::runtime_error("No views loaded from: " + path);
    return views;
}

std::string makeViewFilename(int idx) {
    std::ostringstream oss;
    oss << "view_" << std::setw(3) << std::setfill('0') << idx << ".pcd";
    return oss.str();
}

std::vector<octomap::OcTreeKey> loadViewCoverageKeys(const fs::path& pcd_path, double resolution) {
    pcl::PointCloud<PointT> cloud;
    if (pcl::io::loadPCDFile<PointT>(pcd_path.string(), cloud) != 0) {
        throw std::runtime_error("Failed to load PCD: " + pcd_path.string());
    }

    octomap::OcTree tmp_tree(resolution);

    KeySet keyset;
    keyset.reserve(cloud.size());

    for (const auto& p : cloud.points) {
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) continue;
        octomap::OcTreeKey key;
        if (!tmp_tree.coordToKeyChecked(octomap::point3d(p.x, p.y, p.z), key)) continue;
        keyset.insert(key);
    }

    std::vector<octomap::OcTreeKey> keys;
    keys.reserve(keyset.size());
    for (const auto& k : keyset) keys.push_back(k);
    return keys;
}

StaticObjectData buildStaticObjectData(const Config& cfg) {
    if (!fs::exists(cfg.view_cache_dir) || !fs::is_directory(cfg.view_cache_dir)) {
        throw std::runtime_error("Invalid view_cache_dir: " + cfg.view_cache_dir);
    }

    StaticObjectData data;
    data.uid = fs::path(cfg.view_cache_dir).filename().string();
    data.coverage_ids_per_view.resize(cfg.expected_num_views);
    data.views = loadViews(cfg.views_path, cfg.view_radius);

    if (static_cast<int>(data.views.size()) != cfg.expected_num_views) {
        throw std::runtime_error("views_path count does not match expected_num_views.");
    }

    KeyToIdMap voxel_to_id;
    voxel_to_id.reserve(500000);

    int next_id = 0;

    for (int vid = 0; vid < cfg.expected_num_views; ++vid) {
        fs::path pcd_path = fs::path(cfg.view_cache_dir) / makeViewFilename(vid);
        if (!fs::exists(pcd_path)) {
            throw std::runtime_error("Missing cached view PCD: " + pcd_path.string());
        }

        auto keys = loadViewCoverageKeys(pcd_path, cfg.resolution);

        std::vector<int> ids;
        ids.reserve(keys.size());

        for (const auto& key : keys) {
            auto it = voxel_to_id.find(key);
            if (it == voxel_to_id.end()) {
                voxel_to_id.emplace(key, next_id);
                data.id_to_key.push_back(key);
                ids.push_back(next_id);
                ++next_id;
            } else {
                ids.push_back(it->second);
            }
        }

        std::sort(ids.begin(), ids.end());
        ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
        data.coverage_ids_per_view[vid] = std::move(ids);

        std::cout << "Loaded view " << vid
                  << " | unique covered voxels = "
                  << data.coverage_ids_per_view[vid].size() << "\n";
    }

    data.universe_size = next_id;
    std::cout << "Universe size: " << data.universe_size << "\n";

    return data;
}

void unionCoverageInto(std::vector<uint8_t>& covered, const std::vector<int>& coverage_ids, int& covered_count) {
    for (int id : coverage_ids) {
        if (!covered[id]) {
            covered[id] = 1;
            ++covered_count;
        }
    }
}

int marginalGain(const std::vector<uint8_t>& covered, const std::vector<int>& coverage_ids) {
    int gain = 0;
    for (int id : coverage_ids) {
        if (!covered[id]) ++gain;
    }
    return gain;
}

struct ResidualSetCoverResult {
    std::vector<int> selected_view_ids;
    int residual_universe_size = 0;
};

ResidualSetCoverResult solveResidualSetCover(
    const std::vector<std::vector<int>>& coverage_ids_per_view,
    const std::vector<uint8_t>& covered,
    const std::vector<uint8_t>& visited,
    double time_limit_sec) {

    const int num_views = static_cast<int>(coverage_ids_per_view.size());
    const int universe_size = static_cast<int>(covered.size());

    std::vector<int> uncovered_ids;
    uncovered_ids.reserve(universe_size);
    for (int i = 0; i < universe_size; ++i) {
        if (!covered[i]) uncovered_ids.push_back(i);
    }

    ResidualSetCoverResult result;
    result.residual_universe_size = static_cast<int>(uncovered_ids.size());

    if (uncovered_ids.empty()) {
        return result;
    }

    std::unordered_map<int, int> residual_index;
    residual_index.reserve(uncovered_ids.size());
    for (int i = 0; i < static_cast<int>(uncovered_ids.size()); ++i) {
        residual_index[uncovered_ids[i]] = i;
    }

    std::vector<std::vector<int>> views_per_residual_voxel(uncovered_ids.size());

    for (int vid = 0; vid < num_views; ++vid) {
        if (visited[vid]) continue;

        for (int voxel_id : coverage_ids_per_view[vid]) {
            auto it = residual_index.find(voxel_id);
            if (it != residual_index.end()) {
                views_per_residual_voxel[it->second].push_back(vid);
            }
        }
    }

    for (const auto& v : views_per_residual_voxel) {
        if (v.empty()) {
            return result;
        }
    }

    GRBEnv env(true);
    env.set("LogToConsole", "0");
    env.start();
    GRBModel model(env);

    if (time_limit_sec > 0.0) {
        model.set(GRB_DoubleParam_TimeLimit, time_limit_sec);
    }

    std::vector<GRBVar> x(num_views);
    for (int vid = 0; vid < num_views; ++vid) {
        if (visited[vid]) continue;
        x[vid] = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, "x_" + std::to_string(vid));
    }

    GRBLinExpr obj = 0;
    for (int vid = 0; vid < num_views; ++vid) {
        if (visited[vid]) continue;
        obj += x[vid];
    }
    model.setObjective(obj, GRB_MINIMIZE);

    for (int rid = 0; rid < static_cast<int>(views_per_residual_voxel.size()); ++rid) {
        GRBLinExpr cover = 0;
        for (int vid : views_per_residual_voxel[rid]) {
            cover += x[vid];
        }
        model.addConstr(cover >= 1.0, "cover_" + std::to_string(rid));
    }

    model.optimize();

    const int status = model.get(GRB_IntAttr_Status);
    if (status != GRB_OPTIMAL && status != GRB_SUBOPTIMAL && status != GRB_TIME_LIMIT) {
        throw std::runtime_error("Gurobi failed with status: " + std::to_string(status));
    }

    for (int vid = 0; vid < num_views; ++vid) {
        if (visited[vid]) continue;
        if (x[vid].get(GRB_DoubleAttr_X) > 0.5) {
            result.selected_view_ids.push_back(vid);
        }
    }

    return result;
}

pcl::PointXYZRGB makePoint(float x, float y, float z, uint8_t r, uint8_t g, uint8_t b) {
    pcl::PointXYZRGB p;
    p.x = x; p.y = y; p.z = z;
    p.r = r; p.g = g; p.b = b;
    return p;
}

using VisCloud = pcl::PointCloud<pcl::PointXYZRGB>;

struct RGB {
    uint8_t r, g, b;
};

RGB colorFromIndex(int idx) {
    static const std::vector<RGB> palette = {
        {230, 25, 75}, {60, 180, 75}, {0, 130, 200}, {245, 130, 48},
        {145, 30, 180}, {70, 240, 240}, {240, 50, 230}, {210, 245, 60},
        {250, 190, 190}, {0, 128, 128}, {230, 190, 255}, {170, 110, 40},
        {255, 250, 200}, {128, 0, 0}, {170, 255, 195}, {128, 128, 0},
        {255, 215, 180}, {0, 0, 128}, {128, 128, 128}, {255, 255, 255}
    };
    return palette[idx % palette.size()];
}

void appendVoxelIdsAsPoints(
    const StaticObjectData& data,
    const std::vector<int>& voxel_ids,
    uint8_t r, uint8_t g, uint8_t b,
    VisCloud::Ptr cloud,
    double resolution) {

    octomap::OcTree tmp_tree(resolution);
    for (int id : voxel_ids) {
        const auto& key = data.id_to_key[id];
        const octomap::point3d coord = tmp_tree.keyToCoord(key);
        cloud->points.push_back(makePoint(coord.x(), coord.y(), coord.z(), r, g, b));
    }
}

void appendAllUniverseAsPoints(
    const StaticObjectData& data,
    uint8_t r, uint8_t g, uint8_t b,
    VisCloud::Ptr cloud,
    double resolution) {

    octomap::OcTree tmp_tree(resolution);
    for (int id = 0; id < data.universe_size; ++id) {
        const auto& key = data.id_to_key[id];
        const octomap::point3d coord = tmp_tree.keyToCoord(key);
        cloud->points.push_back(makePoint(coord.x(), coord.y(), coord.z(), r, g, b));
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

std::vector<int> differenceCoverage(const std::vector<int>& coverage_ids, const std::vector<uint8_t>& covered) {
    std::vector<int> out;
    out.reserve(coverage_ids.size());
    for (int id : coverage_ids) {
        if (!covered[id]) out.push_back(id);
    }
    return out;
}

void saveNBVVisualizationPCD(
    const Config& cfg,
    const StaticObjectData& data,
    const std::vector<uint8_t>& covered,
    const std::vector<uint8_t>& visited,
    int oracle_nbv_view_id,
    const std::string& out_path) {

    VisCloud::Ptr cloud(new VisCloud);

    appendAllUniverseAsPoints(data, 0, 0, 0, cloud, cfg.resolution);

    for (int vid = 0; vid < static_cast<int>(visited.size()); ++vid) {
        if (visited[vid]) {
            appendCameraAxes(data.views[vid], cfg.look_at, cfg.axis_length, cfg.axis_step, 0, 0, 0, cloud);
        }
    }

    if (oracle_nbv_view_id >= 0) {
        auto new_surface = differenceCoverage(data.coverage_ids_per_view[oracle_nbv_view_id], covered);
        appendVoxelIdsAsPoints(data, new_surface, 255, 0, 0, cloud, cfg.resolution);
        appendCameraAxes(data.views[oracle_nbv_view_id], cfg.look_at, cfg.axis_length, cfg.axis_step, 255, 0, 0, cloud);
    }

    cloud->width = static_cast<uint32_t>(cloud->points.size());
    cloud->height = 1;
    cloud->is_dense = false;

    fs::create_directories(fs::path(out_path).parent_path());
    if (pcl::io::savePCDFileBinary(out_path, *cloud) != 0) {
        throw std::runtime_error("Failed to save NBV vis PCD: " + out_path);
    }
}

void saveSCVPVisualizationPCD(
    const Config& cfg,
    const StaticObjectData& data,
    const std::vector<uint8_t>& covered,
    const std::vector<uint8_t>& visited,
    const std::vector<int>& residual_set_cover_view_ids,
    const std::string& out_path) {

    VisCloud::Ptr cloud(new VisCloud);

    appendAllUniverseAsPoints(data, 0, 0, 0, cloud, cfg.resolution);

    for (int vid = 0; vid < static_cast<int>(visited.size()); ++vid) {
        if (visited[vid]) {
            appendCameraAxes(data.views[vid], cfg.look_at, cfg.axis_length, cfg.axis_step, 0, 0, 0, cloud);
        }
    }

    std::vector<uint8_t> tmp_covered = covered;

    for (std::size_t i = 0; i < residual_set_cover_view_ids.size(); ++i) {
        const int vid = residual_set_cover_view_ids[i];
        RGB c = colorFromIndex(static_cast<int>(i));

        std::vector<int> new_ids;
        new_ids.reserve(data.coverage_ids_per_view[vid].size());
        for (int id : data.coverage_ids_per_view[vid]) {
            if (!tmp_covered[id]) {
                new_ids.push_back(id);
                tmp_covered[id] = 1;
            }
        }

        appendVoxelIdsAsPoints(data, new_ids, c.r, c.g, c.b, cloud, cfg.resolution);
        appendCameraAxes(data.views[vid], cfg.look_at, cfg.axis_length, cfg.axis_step, c.r, c.g, c.b, cloud);
    }

    cloud->width = static_cast<uint32_t>(cloud->points.size());
    cloud->height = 1;
    cloud->is_dense = false;

    fs::create_directories(fs::path(out_path).parent_path());
    if (pcl::io::savePCDFileBinary(out_path, *cloud) != 0) {
        throw std::runtime_error("Failed to save SCVP vis PCD: " + out_path);
    }
}

Json::Value toJson(const StepRecord& s) {
    Json::Value x(Json::objectValue);

    x["step_id"] = s.step_id;
    x["oracle_nbv_view_id"] = s.oracle_nbv_view_id;
    x["oracle_nbv_gain"] = s.oracle_nbv_gain;
    x["covered_voxel_count"] = s.covered_voxel_count;
    x["residual_voxel_count"] = s.residual_voxel_count;
    x["covered_ratio"] = s.covered_ratio;
    x["residual_set_cover_size"] = s.residual_set_cover_size;

    x["visited_view_ids"] = Json::arrayValue;
    for (int v : s.visited_view_ids) x["visited_view_ids"].append(v);

    x["candidate_view_ids"] = Json::arrayValue;
    for (int v : s.candidate_view_ids) x["candidate_view_ids"].append(v);

    x["candidate_marginal_gains"] = Json::arrayValue;
    for (int g : s.candidate_marginal_gains) x["candidate_marginal_gains"].append(g);

    x["residual_set_cover_view_ids"] = Json::arrayValue;
    for (int v : s.residual_set_cover_view_ids) x["residual_set_cover_view_ids"].append(v);

    return x;
}

Json::Value toJson(const RolloutRecord& r) {
    Json::Value x(Json::objectValue);

    x["start_view_id"] = r.start_view_id;
    x["universe_voxel_count"] = r.universe_voxel_count;
    x["steps"] = Json::arrayValue;

    for (const auto& s : r.steps) {
        x["steps"].append(toJson(s));
    }

    return x;
}

RolloutRecord runSingleStartViewRollout(
    const Config& cfg,
    const StaticObjectData& data,
    int start_view_id) {

    const int num_views = static_cast<int>(data.coverage_ids_per_view.size());

    std::vector<uint8_t> visited(num_views, 0);
    std::vector<uint8_t> covered(data.universe_size, 0);

    int covered_count = 0;
    visited[start_view_id] = 1;
    unionCoverageInto(covered, data.coverage_ids_per_view[start_view_id], covered_count);

    RolloutRecord rollout;
    rollout.start_view_id = start_view_id;
    rollout.universe_voxel_count = data.universe_size;

    int step_id = 0;

    while (true) {
        StepRecord step;
        step.step_id = step_id;

        for (int vid = 0; vid < num_views; ++vid) {
            if (visited[vid]) step.visited_view_ids.push_back(vid);
        }

        step.covered_voxel_count = covered_count;
        step.residual_voxel_count = data.universe_size - covered_count;
        step.covered_ratio = (data.universe_size > 0)
            ? static_cast<double>(covered_count) / static_cast<double>(data.universe_size)
            : 0.0;

        int best_view = -1;
        int best_gain = -1;

        for (int vid = 0; vid < num_views; ++vid) {
            if (visited[vid]) continue;

            const int gain = marginalGain(covered, data.coverage_ids_per_view[vid]);
            step.candidate_view_ids.push_back(vid);
            step.candidate_marginal_gains.push_back(gain);

            if (gain > best_gain) {
                best_gain = gain;
                best_view = vid;
            }
        }

        step.oracle_nbv_view_id = best_view;
        step.oracle_nbv_gain = std::max(0, best_gain);

        auto residual = solveResidualSetCover(
            data.coverage_ids_per_view,
            covered,
            visited,
            cfg.time_limit_sec);

        step.residual_set_cover_view_ids = residual.selected_view_ids;
        step.residual_set_cover_size = static_cast<int>(residual.selected_view_ids.size());

        rollout.steps.push_back(step);

        const bool do_vis =
            (cfg.save_vis == 1) &&
            (cfg.vis_start_view < 0 || cfg.vis_start_view == start_view_id) &&
            (cfg.vis_max_steps < 0 || step_id < cfg.vis_max_steps);

        if (do_vis) {
            std::ostringstream nbv_path, scvp_path;
            nbv_path << cfg.vis_dir << "/" << data.uid
                     << "/start_" << std::setw(3) << std::setfill('0') << start_view_id
                     << "/step_" << std::setw(3) << std::setfill('0') << step_id
                     << "_nbv.pcd";
            scvp_path << cfg.vis_dir << "/" << data.uid
                      << "/start_" << std::setw(3) << std::setfill('0') << start_view_id
                      << "/step_" << std::setw(3) << std::setfill('0') << step_id
                      << "_scvp.pcd";

            saveNBVVisualizationPCD(cfg, data, covered, visited, step.oracle_nbv_view_id, nbv_path.str());
            saveSCVPVisualizationPCD(cfg, data, covered, visited, step.residual_set_cover_view_ids, scvp_path.str());
        }

        if (covered_count >= data.universe_size) {
            break;
        }
        if (best_view < 0 || best_gain <= 0) {
            break;
        }
        if (cfg.max_steps > 0 && static_cast<int>(rollout.steps.size()) >= cfg.max_steps) {
            break;
        }

        visited[best_view] = 1;
        unionCoverageInto(covered, data.coverage_ids_per_view[best_view], covered_count);
        ++step_id;
    }

    return rollout;
}

void saveObjectRollouts(const Config& cfg,
                        const StaticObjectData& data,
                        const std::vector<RolloutRecord>& rollouts) {
    Json::Value root(Json::objectValue);
    root["uid"] = data.uid;
    root["view_cache_dir"] = cfg.view_cache_dir;
    root["views_path"] = cfg.views_path;
    root["resolution"] = cfg.resolution;
    root["expected_num_views"] = cfg.expected_num_views;
    root["universe_voxel_count"] = data.universe_size;

    root["rollouts"] = Json::arrayValue;
    for (const auto& r : rollouts) {
        root["rollouts"].append(toJson(r));
    }

    fs::create_directories(fs::path(cfg.output_json).parent_path());

    Json::StreamWriterBuilder builder;
    builder["indentation"] = "  ";

    std::ofstream fout(cfg.output_json, std::ios::binary);
    if (!fout) {
        throw std::runtime_error("Failed to open output json: " + cfg.output_json);
    }

    std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());
    writer->write(root, &fout);
}

int main(int argc, char** argv) {
    try {
        Config cfg = parseArgs(argc, argv);

        if (cfg.skip_existing == 1 && fs::exists(cfg.output_json)) {
            std::cout << "Skip existing: " << cfg.output_json << std::endl;
            return 0;
        }

        auto data = buildStaticObjectData(cfg);

        std::vector<RolloutRecord> rollouts;
        rollouts.reserve(cfg.expected_num_views);

        for (int start_view = 0; start_view < cfg.expected_num_views; ++start_view) {
            std::cout << "Running rollout for start_view = " << start_view << std::endl;
            auto rollout = runSingleStartViewRollout(cfg, data, start_view);
            rollouts.push_back(std::move(rollout));
        }

        saveObjectRollouts(cfg, data, rollouts);

        std::cout << "Saved rollout json to: " << cfg.output_json << std::endl;
        std::cout << "Done." << std::endl;
        return 0;
    }
    catch (const GRBException& e) {
        std::cerr << "Gurobi error: code " << e.getErrorCode()
                  << ", " << e.getMessage() << std::endl;
        return 2;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
