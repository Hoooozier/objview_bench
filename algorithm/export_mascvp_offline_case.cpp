#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>

#include <json/json.h>

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <octomap/OcTree.h>
#include <octomap/Pointcloud.h>
#include <octomap/octomap_types.h>

#include <cnpy.h>

namespace fs = std::filesystem;

namespace {

using Vec3 = Eigen::Vector3d;
using PointXYZ = pcl::PointXYZ;
using CloudXYZ = pcl::PointCloud<PointXYZ>;

struct Config {
    std::string analysis_split_json;
    std::string rollout_root;
    std::string view_cache_root;
    std::string views_path;
    std::string output_root;
    std::string constraint_name;
    std::string uid;
    std::string debug_dir = "mascvp_offline_debug";

    int expected_num_views = 128;
    int grid_size = 64;
    int start_view_id = -1;
    int step_id = 0;
    int max_step_id = -1;
    int limit_uids = -1;
    bool skip_existing = true;
    bool debug_save_ot = false;

    double bbox_min = -1.0;
    double bbox_max = 1.0;
    double unknown_occ = 0.5;
    double max_range = -1.0;
    double view_radius = 3.0;
}
;

void printUsage(const char* argv0) {
    std::cout
        << "Usage: " << argv0 << " --analysis_split_json SPLIT.json"
        << " --rollout_root DIR --view_cache_root DIR --views Tammes_sphere/128_xyz.txt --output_root DIR [options]\n"
        << "Options:\n"
        << "  --uid UID                  optional single-object override\n"
        << "  --constraint_name NAME      optional metadata only\n"
        << "  --debug_dir DIR             relative or absolute debug output dir\n"
        << "  --debug_save_ot 0           save per-case octomap .ot\n"
        << "  --expected_num_views 128\n"
        << "  --grid_size 64\n"
        << "  --step_id 0\n"
        << "  --max_step_id -1         -1 means export only step_id\n"
        << "  --bbox_min -1.0\n"
        << "  --bbox_max 1.0\n"
        << "  --unknown_occ 0.5\n"
        << "  --max_range -1             octomap insertPointCloud max range, <=0 disables\n"
        << "  --view_radius 3.0\n"
        << "  --start_view_id -1         -1 exports all Tammes views, otherwise export one\n"
        << "  --limit_uids -1            debug aid\n"
        << "  --skip_existing 1\n";
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

        if (arg == "--analysis_split_json") cfg.analysis_split_json = needValue(arg);
        else if (arg == "--rollout_root") cfg.rollout_root = needValue(arg);
        else if (arg == "--view_cache_root") cfg.view_cache_root = needValue(arg);
        else if (arg == "--views") cfg.views_path = needValue(arg);
        else if (arg == "--output_root") cfg.output_root = needValue(arg);
        else if (arg == "--uid") cfg.uid = needValue(arg);
        else if (arg == "--constraint_name") cfg.constraint_name = needValue(arg);
        else if (arg == "--debug_dir") cfg.debug_dir = needValue(arg);
        else if (arg == "--debug_save_ot") cfg.debug_save_ot = std::stoi(needValue(arg)) != 0;
        else if (arg == "--expected_num_views") cfg.expected_num_views = std::stoi(needValue(arg));
        else if (arg == "--grid_size") cfg.grid_size = std::stoi(needValue(arg));
        else if (arg == "--start_view_id") cfg.start_view_id = std::stoi(needValue(arg));
        else if (arg == "--step_id") cfg.step_id = std::stoi(needValue(arg));
        else if (arg == "--max_step_id") cfg.max_step_id = std::stoi(needValue(arg));
        else if (arg == "--limit_uids") cfg.limit_uids = std::stoi(needValue(arg));
        else if (arg == "--skip_existing") cfg.skip_existing = std::stoi(needValue(arg)) != 0;
        else if (arg == "--bbox_min") cfg.bbox_min = std::stod(needValue(arg));
        else if (arg == "--bbox_max") cfg.bbox_max = std::stod(needValue(arg));
        else if (arg == "--unknown_occ") cfg.unknown_occ = std::stod(needValue(arg));
        else if (arg == "--max_range") cfg.max_range = std::stod(needValue(arg));
        else if (arg == "--view_radius") cfg.view_radius = std::stod(needValue(arg));
        else if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (cfg.analysis_split_json.empty() || cfg.rollout_root.empty() || cfg.view_cache_root.empty() ||
        cfg.views_path.empty() || cfg.output_root.empty()) {
        throw std::runtime_error(
            "--analysis_split_json, --rollout_root, --view_cache_root, --views, and --output_root are required.");
    }
    if (cfg.expected_num_views <= 0 || cfg.grid_size <= 0) {
        throw std::runtime_error("expected_num_views and grid_size must be positive.");
    }
    if (cfg.step_id < 0) {
        throw std::runtime_error("step_id must be >= 0.");
    }
    if (cfg.max_step_id != -1 && cfg.max_step_id < cfg.step_id) {
        throw std::runtime_error("max_step_id must be -1 or >= step_id.");
    }
    if (!(cfg.bbox_min < cfg.bbox_max)) {
        throw std::runtime_error("bbox_min must be smaller than bbox_max.");
    }
    if (cfg.start_view_id >= cfg.expected_num_views) {
        throw std::runtime_error("start_view_id exceeds expected_num_views.");
    }
    return cfg;
}

std::string trim(const std::string& s) {
    const auto first = s.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    const auto last = s.find_last_not_of(" \t\r\n");
    return s.substr(first, last - first + 1);
}

std::string zeroPadInt(int value, int width) {
    std::ostringstream oss;
    oss << std::setw(width) << std::setfill('0') << value;
    return oss.str();
}

std::vector<Vec3> loadViews(const std::string& path, double view_radius) {
    std::ifstream ifs(path);
    if (!ifs) {
        throw std::runtime_error("Failed to open views file: " + path);
    }
    std::vector<Vec3> views;
    std::string line;
    while (std::getline(ifs, line)) {
        line = trim(line);
        if (line.empty()) continue;
        std::istringstream iss(line);
        double x = 0.0, y = 0.0, z = 0.0;
        if (!(iss >> x >> y >> z)) {
            throw std::runtime_error("Invalid line in views file: " + line);
        }
        Vec3 v(x, y, z);
        if (v.norm() <= 1e-12) {
            throw std::runtime_error("Encountered zero-length view vector in views file: " + path);
        }
        views.emplace_back(v.normalized() * view_radius);
    }
    return views;
}

std::vector<std::string> loadUidsFromSplit(const std::string& path, int limit_uids) {
    std::ifstream ifs(path);
    if (!ifs) {
        throw std::runtime_error("Failed to open split json: " + path);
    }

    Json::Value root;
    ifs >> root;
    if (!root.isArray()) {
        throw std::runtime_error("Expected top-level array in split json: " + path);
    }

    std::vector<std::string> uids;
    uids.reserve(root.size());
    for (const Json::Value& item : root) {
        if (!item.isObject() || !item.isMember("uid") || !item["uid"].isString()) {
            throw std::runtime_error("Each split entry must contain a string uid.");
        }
        uids.push_back(item["uid"].asString());
        if (limit_uids > 0 && static_cast<int>(uids.size()) >= limit_uids) {
            break;
        }
    }
    return uids;
}

std::vector<std::string> resolveUids(const Config& cfg) {
    if (!cfg.uid.empty()) {
        return {cfg.uid};
    }
    return loadUidsFromSplit(cfg.analysis_split_json, cfg.limit_uids);
}

std::unordered_map<int, std::vector<int>> loadVisitedViewsPerStartView(
    const fs::path& rollout_json,
    int step_id) {
    std::ifstream ifs(rollout_json, std::ios::binary);
    if (!ifs) {
        throw std::runtime_error("Failed to open rollout json: " + rollout_json.string());
    }

    Json::CharReaderBuilder builder;
    builder["collectComments"] = false;
    Json::Value root;
    std::string errs;
    if (!Json::parseFromStream(builder, ifs, &root, &errs)) {
        throw std::runtime_error("Failed to parse rollout json " + rollout_json.string() + ": " + errs);
    }
    if (!root.isObject() || !root.isMember("rollouts") || !root["rollouts"].isArray()) {
        throw std::runtime_error("Invalid rollout json structure: " + rollout_json.string());
    }

    std::unordered_map<int, std::vector<int>> out;
    for (const Json::Value& rollout : root["rollouts"]) {
        if (!rollout.isObject() || !rollout.isMember("start_view_id") || !rollout["start_view_id"].isInt()) {
            continue;
        }
        const int start_view_id = rollout["start_view_id"].asInt();
        if (!rollout.isMember("steps") || !rollout["steps"].isArray()) {
            throw std::runtime_error("Missing steps array for rollout start_view_id=" + std::to_string(start_view_id));
        }
        const Json::Value& steps = rollout["steps"];
        if (step_id >= static_cast<int>(steps.size())) {
            throw std::runtime_error(
                "Requested step_id out of range for start_view_id=" + std::to_string(start_view_id));
        }
        const Json::Value& step = steps[step_id];
        if (!step.isObject() || !step.isMember("visited_view_ids") || !step["visited_view_ids"].isArray()) {
            throw std::runtime_error("Missing visited_view_ids in rollout step.");
        }

        std::vector<int> visited_view_ids;
        visited_view_ids.reserve(step["visited_view_ids"].size());
        for (const Json::Value& x : step["visited_view_ids"]) {
            if (x.isInt()) visited_view_ids.push_back(x.asInt());
        }
        std::sort(visited_view_ids.begin(), visited_view_ids.end());
        visited_view_ids.erase(std::unique(visited_view_ids.begin(), visited_view_ids.end()), visited_view_ids.end());
        out.emplace(start_view_id, std::move(visited_view_ids));
    }

    return out;
}

CloudXYZ::Ptr loadPcd(const fs::path& path) {
    CloudXYZ::Ptr cloud(new CloudXYZ);
    if (pcl::io::loadPCDFile<PointXYZ>(path.string(), *cloud) != 0) {
        throw std::runtime_error("Failed to load PCD: " + path.string());
    }
    return cloud;
}

octomap::Pointcloud pclToOctoCloudClipped(
    const CloudXYZ::Ptr& cloud,
    double bbox_min,
    double bbox_max) {
    octomap::Pointcloud out;
    for (const auto& p : cloud->points) {
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) continue;
        if (p.x < bbox_min || p.x > bbox_max ||
            p.y < bbox_min || p.y > bbox_max ||
            p.z < bbox_min || p.z > bbox_max) {
            continue;
        }
        out.push_back(static_cast<float>(p.x), static_cast<float>(p.y), static_cast<float>(p.z));
    }
    return out;
}

std::vector<float> buildDenseOccupancyGrid(
    const octomap::OcTree& tree,
    int grid_size,
    double bbox_min,
    double bbox_max,
    double unknown_occ,
    double* out_min,
    double* out_max) {
    const double voxel = (bbox_max - bbox_min) / static_cast<double>(grid_size);
    std::vector<float> grid;
    grid.resize(static_cast<std::size_t>(grid_size) * grid_size * grid_size, static_cast<float>(unknown_occ));

    float min_val = std::numeric_limits<float>::infinity();
    float max_val = -std::numeric_limits<float>::infinity();
    std::size_t idx = 0;
    for (int ix = 0; ix < grid_size; ++ix) {
        const double x = bbox_min + (static_cast<double>(ix) + 0.5) * voxel;
        for (int iy = 0; iy < grid_size; ++iy) {
            const double y = bbox_min + (static_cast<double>(iy) + 0.5) * voxel;
            for (int iz = 0; iz < grid_size; ++iz, ++idx) {
                const double z = bbox_min + (static_cast<double>(iz) + 0.5) * voxel;
                const octomap::OcTreeNode* node = tree.search(x, y, z);
                float occ = static_cast<float>(unknown_occ);
                if (node != nullptr) {
                    occ = static_cast<float>(node->getOccupancy());
                }
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

std::vector<uint8_t> makeViewState(int expected_num_views, const std::vector<int>& visited_view_ids) {
    std::vector<uint8_t> vs(expected_num_views, static_cast<uint8_t>(0));
    for (int visited_view_id : visited_view_ids) {
        if (visited_view_id >= 0 && visited_view_id < expected_num_views) {
            vs[visited_view_id] = static_cast<uint8_t>(1);
        }
    }
    return vs;
}

std::vector<uint8_t> toAsciiBytes(const std::string& s) {
    return std::vector<uint8_t>(s.begin(), s.end());
}

Json::Value toJsonArray(const std::vector<int>& xs) {
    Json::Value arr(Json::arrayValue);
    for (int x : xs) arr.append(x);
    return arr;
}

void saveNpz(
    const fs::path& npz_path,
    const std::vector<float>& grid,
    int grid_size,
    const std::vector<uint8_t>& vs,
    const std::string& uid,
    int start_view_id,
    int step_id,
    const std::vector<int>& visited_view_ids,
    const std::string& constraint_name,
    const std::string& views_path,
    const Vec3& view_origin,
    double bbox_min,
    double bbox_max) {
    const std::vector<std::size_t> grid_shape{
        static_cast<std::size_t>(grid_size),
        static_cast<std::size_t>(grid_size),
        static_cast<std::size_t>(grid_size)
    };
    const std::vector<std::size_t> vs_shape{vs.size()};
    const std::vector<std::size_t> scalar_shape{1};
    const std::vector<std::size_t> vec3_shape{3};

    const int32_t start_view_id_i32 = static_cast<int32_t>(start_view_id);
    const int32_t step_id_i32 = static_cast<int32_t>(step_id);
    const int32_t grid_size_i32 = static_cast<int32_t>(grid_size);
    std::vector<int32_t> visited_view_ids_i32;
    visited_view_ids_i32.reserve(visited_view_ids.size());
    for (int x : visited_view_ids) visited_view_ids_i32.push_back(static_cast<int32_t>(x));
    const std::array<float, 3> bbox_min_arr{
        static_cast<float>(bbox_min),
        static_cast<float>(bbox_min),
        static_cast<float>(bbox_min)
    };
    const std::array<float, 3> bbox_max_arr{
        static_cast<float>(bbox_max),
        static_cast<float>(bbox_max),
        static_cast<float>(bbox_max)
    };
    const std::array<float, 3> view_origin_arr{
        static_cast<float>(view_origin.x()),
        static_cast<float>(view_origin.y()),
        static_cast<float>(view_origin.z())
    };
    const std::vector<uint8_t> uid_ascii = toAsciiBytes(uid);
    const std::vector<uint8_t> views_ascii = toAsciiBytes(views_path);
    const std::vector<uint8_t> constraint_ascii = toAsciiBytes(constraint_name);

    cnpy::npz_save(npz_path.string(), "grid", grid.data(), grid_shape, "w");
    cnpy::npz_save(npz_path.string(), "vs", vs.data(), vs_shape, "a");
    cnpy::npz_save(npz_path.string(), "start_view_id", &start_view_id_i32, scalar_shape, "a");
    cnpy::npz_save(npz_path.string(), "step_id", &step_id_i32, scalar_shape, "a");
    cnpy::npz_save(npz_path.string(), "grid_size", &grid_size_i32, scalar_shape, "a");
    cnpy::npz_save(npz_path.string(), "bbox_min", bbox_min_arr.data(), vec3_shape, "a");
    cnpy::npz_save(npz_path.string(), "bbox_max", bbox_max_arr.data(), vec3_shape, "a");
    cnpy::npz_save(npz_path.string(), "view_origin", view_origin_arr.data(), vec3_shape, "a");
    if (!visited_view_ids_i32.empty()) {
        cnpy::npz_save(
            npz_path.string(),
            "visited_view_ids",
            visited_view_ids_i32.data(),
            std::vector<std::size_t>{visited_view_ids_i32.size()},
            "a");
    }
    if (!uid_ascii.empty()) {
        cnpy::npz_save(npz_path.string(), "uid_ascii", uid_ascii.data(), std::vector<std::size_t>{uid_ascii.size()}, "a");
    }
    if (!views_ascii.empty()) {
        cnpy::npz_save(npz_path.string(), "views_path_ascii", views_ascii.data(), std::vector<std::size_t>{views_ascii.size()}, "a");
    }
    if (!constraint_ascii.empty()) {
        cnpy::npz_save(npz_path.string(), "constraint_name_ascii", constraint_ascii.data(), std::vector<std::size_t>{constraint_ascii.size()}, "a");
    }
}

void saveSidecarJson(
    const fs::path& json_path,
    const std::string& uid,
    int start_view_id,
    int step_id,
    const std::vector<int>& visited_view_ids,
    int grid_size,
    double bbox_min,
    double bbox_max,
    double grid_min,
    double grid_max,
    int visited_count,
    const std::string& views_path,
    const std::string& constraint_name,
    const fs::path& npz_path,
    bool debug_save_ot,
    const fs::path& octomap_ot_path,
    const fs::path& octomap_with_unknown_ot_path) {
    Json::Value root(Json::objectValue);
    root["uid"] = uid;
    root["start_view_id"] = start_view_id;
    root["step_id"] = step_id;
    root["grid_size"] = grid_size;
    root["visited_view_count"] = visited_count;
    root["visited_view_ids"] = toJsonArray(visited_view_ids);
    root["grid_min"] = grid_min;
    root["grid_max"] = grid_max;
    root["views_path"] = views_path;
    root["npz_path"] = npz_path.string();
    root["debug_save_ot"] = debug_save_ot;
    if (debug_save_ot) {
        root["octomap_ot_path"] = octomap_ot_path.string();
        root["octomap_with_unknown_ot_path"] = octomap_with_unknown_ot_path.string();
    }
    if (!constraint_name.empty()) {
        root["constraint_name"] = constraint_name;
    }

    Json::Value bbox_min_json(Json::arrayValue);
    Json::Value bbox_max_json(Json::arrayValue);
    for (int axis = 0; axis < 3; ++axis) {
        bbox_min_json.append(bbox_min);
        bbox_max_json.append(bbox_max);
    }
    root["bbox_min"] = bbox_min_json;
    root["bbox_max"] = bbox_max_json;

    Json::StreamWriterBuilder builder;
    builder["indentation"] = "  ";
    std::ofstream ofs(json_path);
    if (!ofs) {
        throw std::runtime_error("Failed to open metadata json for write: " + json_path.string());
    }
    ofs << Json::writeString(builder, root);
}

struct OctomapDebugPaths {
    fs::path observed_only_ot_path;
    fs::path with_unknown_ot_path;
};

OctomapDebugPaths saveOctomapDebug(
    const Config& cfg,
    const std::string& uid,
    const std::string& stem,
    octomap::OcTree& tree,
    double bbox_min,
    double bbox_max,
    double unknown_occ) {
    OctomapDebugPaths out;
    if (!cfg.debug_save_ot) return out;

    fs::path debug_root = fs::path(cfg.debug_dir);
    if (debug_root.is_relative()) {
        debug_root = fs::path(cfg.output_root) / debug_root;
    }
    const fs::path uid_dir = debug_root / uid;
    fs::create_directories(uid_dir);
    out.observed_only_ot_path = uid_dir / (stem + "__observed_only.ot");
    out.with_unknown_ot_path = uid_dir / (stem + "__with_unknown.ot");

    if (!tree.write(out.observed_only_ot_path.string())) {
        throw std::runtime_error("Failed to save debug octomap: " + out.observed_only_ot_path.string());
    }

    const double resolution = tree.getResolution();
    const int grid_size = static_cast<int>(std::llround((bbox_max - bbox_min) / resolution));
    octomap::OcTree dense_tree(resolution);
    for (int ix = 0; ix < grid_size; ++ix) {
        const double x = bbox_min + (static_cast<double>(ix) + 0.5) * resolution;
        for (int iy = 0; iy < grid_size; ++iy) {
            const double y = bbox_min + (static_cast<double>(iy) + 0.5) * resolution;
            for (int iz = 0; iz < grid_size; ++iz) {
                const double z = bbox_min + (static_cast<double>(iz) + 0.5) * resolution;
                const octomap::OcTreeNode* src = tree.search(x, y, z);
                const double occ = (src != nullptr) ? static_cast<double>(src->getOccupancy()) : unknown_occ;
                octomap::OcTreeNode* node =
                    dense_tree.updateNode(octomap::point3d(x, y, z), true, true);
                if (node == nullptr) {
                    throw std::runtime_error("Failed to allocate debug octomap node.");
                }
                node->setLogOdds(octomap::logodds(occ));
            }
        }
    }
    dense_tree.updateInnerOccupancy();
    if (!dense_tree.write(out.with_unknown_ot_path.string())) {
        throw std::runtime_error("Failed to save debug octomap: " + out.with_unknown_ot_path.string());
    }

    return out;
}

void exportOneCase(
    const Config& cfg,
    const std::string& uid,
    const std::vector<Vec3>& views,
    int start_view_id,
    int step_id,
    const std::vector<int>& visited_view_ids) {
    const fs::path uid_cache_dir = fs::path(cfg.view_cache_root) / uid;
    if (visited_view_ids.empty()) {
        throw std::runtime_error("Visited view list is empty for start_view_id=" + std::to_string(start_view_id));
    }

    const fs::path out_dir = fs::path(cfg.output_root) / uid;
    fs::create_directories(out_dir);
    const std::string stem =
        uid + "__view_" + zeroPadInt(start_view_id, 3) + "__step_" + zeroPadInt(step_id, 3);
    const fs::path npz_path = out_dir / (stem + ".npz");
    const fs::path meta_path = out_dir / (stem + ".json");

    if (cfg.skip_existing && fs::exists(npz_path) && fs::exists(meta_path)) {
        std::cout << "[skip] " << uid << " start_view=" << start_view_id << " step=" << step_id << "\n";
        return;
    }

    const double resolution = (cfg.bbox_max - cfg.bbox_min) / static_cast<double>(cfg.grid_size);
    octomap::OcTree tree(resolution);
    const Vec3& origin = views.at(static_cast<std::size_t>(start_view_id));
    for (int visited_view_id : visited_view_ids) {
        const fs::path pcd_path = uid_cache_dir / ("view_" + zeroPadInt(visited_view_id, 3) + ".pcd");
        if (!fs::exists(pcd_path)) {
            throw std::runtime_error("Missing cached view PCD: " + pcd_path.string());
        }
        CloudXYZ::Ptr cloud = loadPcd(pcd_path);
        const octomap::Pointcloud octo_cloud = pclToOctoCloudClipped(cloud, cfg.bbox_min, cfg.bbox_max);
        const Vec3& sensor = views.at(static_cast<std::size_t>(visited_view_id));
        const octomap::point3d sensor_origin(
            static_cast<float>(sensor.x()),
            static_cast<float>(sensor.y()),
            static_cast<float>(sensor.z()));
        tree.insertPointCloud(octo_cloud, sensor_origin, cfg.max_range, true, true);
    }
    tree.updateInnerOccupancy();

    double grid_min = 0.0;
    double grid_max = 0.0;
    const std::vector<float> grid = buildDenseOccupancyGrid(
        tree, cfg.grid_size, cfg.bbox_min, cfg.bbox_max, cfg.unknown_occ, &grid_min, &grid_max);
    const std::vector<uint8_t> vs = makeViewState(cfg.expected_num_views, visited_view_ids);
    const OctomapDebugPaths debug_paths =
        saveOctomapDebug(cfg, uid, stem, tree, cfg.bbox_min, cfg.bbox_max, cfg.unknown_occ);

    saveNpz(
        npz_path, grid, cfg.grid_size, vs, uid, start_view_id, step_id, visited_view_ids, cfg.constraint_name,
        cfg.views_path, origin, cfg.bbox_min, cfg.bbox_max);
    saveSidecarJson(
        meta_path, uid, start_view_id, step_id, visited_view_ids, cfg.grid_size, cfg.bbox_min, cfg.bbox_max,
        grid_min, grid_max, static_cast<int>(visited_view_ids.size()), cfg.views_path, cfg.constraint_name, npz_path,
        cfg.debug_save_ot, debug_paths.observed_only_ot_path, debug_paths.with_unknown_ot_path);

    std::cout
        << "[ok] uid=" << uid
        << " start_view=" << start_view_id
        << " step=" << step_id
        << " visited=" << visited_view_ids.size()
        << " grid_min=" << grid_min
        << " grid_max=" << grid_max
        << (cfg.debug_save_ot ? " ot_obs=" + debug_paths.observed_only_ot_path.string() : "")
        << (cfg.debug_save_ot ? " ot_unk=" + debug_paths.with_unknown_ot_path.string() : "")
        << " npz=" << npz_path.string()
        << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Config cfg = parseArgs(argc, argv);
        const std::vector<Vec3> views = loadViews(cfg.views_path, cfg.view_radius);
        if (static_cast<int>(views.size()) != cfg.expected_num_views) {
            throw std::runtime_error(
                "View count mismatch: expected " + std::to_string(cfg.expected_num_views) +
                ", got " + std::to_string(views.size()) + " from " + cfg.views_path);
        }

        const std::vector<std::string> uids = resolveUids(cfg);
        if (uids.empty()) {
            throw std::runtime_error("No UIDs found in split json.");
        }

        std::vector<int> start_views;
        if (cfg.start_view_id >= 0) {
            start_views.push_back(cfg.start_view_id);
        } else {
            start_views.reserve(cfg.expected_num_views);
            for (int i = 0; i < cfg.expected_num_views; ++i) {
                start_views.push_back(i);
            }
        }

        std::cout
            << "Exporting MASCVP offline cases"
            << " uids=" << uids.size()
            << " start_views_per_uid=" << start_views.size()
            << " step_id=" << cfg.step_id
            << " max_step_id=" << (cfg.max_step_id == -1 ? cfg.step_id : cfg.max_step_id)
            << " grid_size=" << cfg.grid_size
            << " bbox=[" << cfg.bbox_min << ", " << cfg.bbox_max << "]"
            << " view_radius=" << cfg.view_radius
            << " views=" << cfg.views_path
            << "\n";

        for (const std::string& uid : uids) {
            const int max_step_id = (cfg.max_step_id == -1) ? cfg.step_id : cfg.max_step_id;
            const fs::path rollout_json = fs::path(cfg.rollout_root) / (uid + ".json");
            for (int step_id = cfg.step_id; step_id <= max_step_id; ++step_id) {
                const auto visited_map = loadVisitedViewsPerStartView(rollout_json, step_id);
                for (int start_view_id : start_views) {
                    auto it = visited_map.find(start_view_id);
                    if (it == visited_map.end()) {
                        throw std::runtime_error(
                            "Missing rollout step state for uid=" + uid +
                            " start_view_id=" + std::to_string(start_view_id) +
                            " step_id=" + std::to_string(step_id));
                    }
                    exportOneCase(cfg, uid, views, start_view_id, step_id, it->second);
                }
            }
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}

