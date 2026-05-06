#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <cstdint>

#include <Eigen/Dense>

#include <json/json.h>

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <octomap/octomap.h>
#include <octomap/OcTree.h>

namespace fs = std::filesystem;

using Vec3 = Eigen::Vector3d;
using CloudXYZ = pcl::PointCloud<pcl::PointXYZ>;

struct Config {
    std::string rollout_json;
    std::string view_cache_dir;   // contains view_000.pcd ...
    std::string views_path;       // 128_xyz.txt
    std::string output_root;

    int expected_num_views = 128;
    double view_radius = 3.0;

    // cache resolution: used for point dedup and stable keying
    double cache_resolution = 0.02;

    // octomap export
    int grid_size = 64;
    double bbox_min = -1.0;
    double bbox_max = 1.0;
    double unknown_occ = 0.5;

    // SCVP sampling
    int scvp_num_start_views = 64;

    // long-tail
    int longtail_need_case_1 = 64;
    int random_seed = 42;

    int skip_existing = 0;
};

struct StepRecord {
    int step_id = -1;
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

struct RolloutData {
    std::string uid;
    int expected_num_views = 0;
    double resolution = 0.0;
    int universe_voxel_count = 0;
    std::vector<RolloutRecord> rollouts;
};

struct CaseRef {
    std::string uid;
    int view_id = -1;   // start view id
    int step_id = -1;
    int m = -1;         // visited_view_ids.size()
    int oracle_nbv_gain = 0;
    double covered_ratio = 0.0;
    int residual_set_cover_size = 0;
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

struct StepKey {
    int view_id = -1;
    int step_id = -1;

    bool operator==(const StepKey& other) const {
        return view_id == other.view_id && step_id == other.step_id;
    }
};

struct StepKeyHash {
    std::size_t operator()(const StepKey& k) const {
        return (static_cast<std::size_t>(static_cast<uint32_t>(k.view_id)) << 32) ^
               static_cast<std::size_t>(static_cast<uint32_t>(k.step_id));
    }
};

static std::string zeroPad3(int x) {
    std::ostringstream oss;
    oss << std::setw(3) << std::setfill('0') << x;
    return oss.str();
}

static void ensureDir(const fs::path& p) {
    fs::create_directories(p);
}

static Json::Value loadJsonFile(const std::string& path) {
    std::ifstream fin(path, std::ios::binary);
    if (!fin) {
        throw std::runtime_error("Failed to open json: " + path);
    }

    Json::CharReaderBuilder builder;
    builder["collectComments"] = false;

    Json::Value root;
    std::string errs;
    if (!Json::parseFromStream(builder, fin, &root, &errs)) {
        throw std::runtime_error("Failed to parse json: " + errs);
    }
    return root;
}

static void saveJsonFile(const Json::Value& root, const fs::path& path) {
    ensureDir(path.parent_path());

    Json::StreamWriterBuilder builder;
    builder["indentation"] = "  ";

    std::ofstream fout(path, std::ios::binary);
    if (!fout) {
        throw std::runtime_error("Failed to open output json: " + path.string());
    }
    std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());
    writer->write(root, &fout);
}

static std::vector<Vec3> loadViews(const std::string& path, double radius) {
    std::ifstream fin(path);
    if (!fin) throw std::runtime_error("Failed to open views file: " + path);

    std::vector<Vec3> views;
    std::string line;
    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        Vec3 v;
        if (!(iss >> v.x() >> v.y() >> v.z())) {
            throw std::runtime_error("Failed to parse line in views file: " + line);
        }
        if (v.norm() < 1e-12) {
            throw std::runtime_error("Zero-norm view in views file.");
        }
        views.push_back(v.normalized() * radius);
    }
    return views;
}

static bool loadPcdAsXYZ(const fs::path& path, CloudXYZ::Ptr out) {
    CloudXYZ cloud;
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(path.string(), cloud) == 0) {
        *out = cloud;
        return true;
    }
    return false;
}

static RolloutData loadRolloutData(const std::string& path) {
    Json::Value root = loadJsonFile(path);

    RolloutData data;
    data.uid = root["uid"].asString();
    data.expected_num_views = root.get("expected_num_views", 0).asInt();
    data.resolution = root.get("resolution", 0.0).asDouble();
    data.universe_voxel_count = root.get("universe_voxel_count", 0).asInt();

    const Json::Value& rollouts = root["rollouts"];
    if (!rollouts.isArray()) {
        throw std::runtime_error("rollouts is not an array in rollout json.");
    }

    for (const auto& r : rollouts) {
        RolloutRecord rr;
        rr.start_view_id = r["start_view_id"].asInt();
        rr.universe_voxel_count = r["universe_voxel_count"].asInt();

        const Json::Value& steps = r["steps"];
        if (!steps.isArray()) throw std::runtime_error("steps is not array.");

        for (const auto& s : steps) {
            StepRecord sr;
            sr.step_id = s["step_id"].asInt();
            sr.oracle_nbv_view_id = s["oracle_nbv_view_id"].asInt();
            sr.oracle_nbv_gain = s["oracle_nbv_gain"].asInt();
            sr.covered_voxel_count = s["covered_voxel_count"].asInt();
            sr.residual_voxel_count = s["residual_voxel_count"].asInt();
            sr.covered_ratio = s["covered_ratio"].asDouble();
            sr.residual_set_cover_size = s["residual_set_cover_size"].asInt();

            for (const auto& x : s["visited_view_ids"]) sr.visited_view_ids.push_back(x.asInt());
            for (const auto& x : s["candidate_view_ids"]) sr.candidate_view_ids.push_back(x.asInt());
            for (const auto& x : s["candidate_marginal_gains"]) sr.candidate_marginal_gains.push_back(x.asInt());
            for (const auto& x : s["residual_set_cover_view_ids"]) sr.residual_set_cover_view_ids.push_back(x.asInt());

            rr.steps.push_back(std::move(sr));
        }

        data.rollouts.push_back(std::move(rr));
    }

    return data;
}

static void writeSingleInt(const fs::path& path, int value) {
    ensureDir(path.parent_path());
    std::ofstream fout(path);
    if (!fout) throw std::runtime_error("Failed to open " + path.string());
    fout << value << "\n";
}

static void writeIntListOnePerLine(const fs::path& path, const std::vector<int>& vals) {
    ensureDir(path.parent_path());
    std::ofstream fout(path);
    if (!fout) throw std::runtime_error("Failed to open " + path.string());
    for (int v : vals) fout << v << "\n";
}

static void writeFloatListOnePerLine(const fs::path& path, const std::vector<double>& vals) {
    ensureDir(path.parent_path());
    std::ofstream fout(path);
    if (!fout) throw std::runtime_error("Failed to open " + path.string());
    fout << std::setprecision(10);
    for (double v : vals) fout << v << "\n";
}

static void saveCloudPcd(const std::vector<Vec3>& pts, const fs::path& path) {
    ensureDir(path.parent_path());

    CloudXYZ cloud;
    cloud.reserve(pts.size());

    for (const auto& p : pts) {
        pcl::PointXYZ q;
        q.x = static_cast<float>(p.x());
        q.y = static_cast<float>(p.y());
        q.z = static_cast<float>(p.z());
        cloud.push_back(q);
    }

    cloud.width = static_cast<uint32_t>(cloud.size());
    cloud.height = 1;
    cloud.is_dense = false;

    if (pcl::io::savePCDFileBinary(path.string(), cloud) != 0) {
        throw std::runtime_error("Failed to save .pcd file: " + path.string());
    }
}

static Json::Value makeCaseJson(const CaseRef& c) {
    Json::Value x(Json::objectValue);
    x["uid"] = c.uid;
    x["view_id"] = c.view_id;
    x["step_id"] = c.step_id;
    return x;
}

static void insertViewCloudIntoTree(
    octomap::OcTree& tree,
    const CloudXYZ::Ptr& cloud,
    const Vec3& origin,
    double bbox_min,
    double bbox_max) {

    octomap::Pointcloud octo_cloud;
    octo_cloud.reserve(cloud->size());

    for (const auto& p : cloud->points) {
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) continue;
        if (p.x < bbox_min || p.x > bbox_max ||
            p.y < bbox_min || p.y > bbox_max ||
            p.z < bbox_min || p.z > bbox_max) {
            continue;
        }
        octo_cloud.push_back(p.x, p.y, p.z);
    }

    tree.insertPointCloud(
        octo_cloud,
        octomap::point3d(origin.x(), origin.y(), origin.z()),
        -1.0,
        false,
        false);
}

static void saveTreeOT(octomap::OcTree& tree, const fs::path& path) {
    ensureDir(path.parent_path());
    if (!tree.write(path.string())) {
        throw std::runtime_error("Failed to save octomap .ot file: " + path.string());
    }
}

static uint32_t makeUidSeed(const std::string& uid, int base_seed, uint32_t salt) {
    const uint64_t h = static_cast<uint64_t>(std::hash<std::string>{}(uid));
    uint64_t x = h ^ (static_cast<uint64_t>(static_cast<uint32_t>(base_seed)) << 1) ^ salt;
    x ^= (x >> 33);
    x *= 0xff51afd7ed558ccdULL;
    x ^= (x >> 33);
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= (x >> 33);
    return static_cast<uint32_t>(x & 0xffffffffu);
}

static void printUsage(const char* argv0) {
    std::cout
        << "Usage: " << argv0
        << " --rollout_json PATH --view_cache_dir DIR --views PATH --output_root DIR [options]\n"
        << "Options:\n"
        << "  --expected_num_views 128\n"
        << "  --view-radius 3.0\n"
        << "  --cache-resolution 0.02\n"
        << "  --grid-size 64\n"
        << "  --bbox-min -1.0\n"
        << "  --bbox-max 1.0\n"
        << "  --unknown-occ 0.5\n"
        << "  --scvp-num-start-views 64\n"
        << "  --longtail-need-case-1 64\n"
        << "  --random-seed 42\n"
        << "  --skip_existing 0\n";
}

static Config parseArgs(int argc, char** argv) {
    Config cfg;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto needValue = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("Missing value for " + name);
            return argv[++i];
        };

        if (arg == "--rollout_json") cfg.rollout_json = needValue(arg);
        else if (arg == "--view_cache_dir") cfg.view_cache_dir = needValue(arg);
        else if (arg == "--views") cfg.views_path = needValue(arg);
        else if (arg == "--output_root") cfg.output_root = needValue(arg);
        else if (arg == "--expected_num_views") cfg.expected_num_views = std::stoi(needValue(arg));
        else if (arg == "--view-radius") cfg.view_radius = std::stod(needValue(arg));
        else if (arg == "--cache-resolution") cfg.cache_resolution = std::stod(needValue(arg));
        else if (arg == "--grid-size") cfg.grid_size = std::stoi(needValue(arg));
        else if (arg == "--bbox-min") cfg.bbox_min = std::stod(needValue(arg));
        else if (arg == "--bbox-max") cfg.bbox_max = std::stod(needValue(arg));
        else if (arg == "--unknown-occ") cfg.unknown_occ = std::stod(needValue(arg));
        else if (arg == "--scvp-num-start-views") cfg.scvp_num_start_views = std::stoi(needValue(arg));
        else if (arg == "--longtail-need-case-1") cfg.longtail_need_case_1 = std::stoi(needValue(arg));
        else if (arg == "--random-seed") cfg.random_seed = std::stoi(needValue(arg));
        else if (arg == "--skip_existing") cfg.skip_existing = std::stoi(needValue(arg));
        else if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (cfg.rollout_json.empty() || cfg.view_cache_dir.empty() ||
        cfg.views_path.empty() || cfg.output_root.empty()) {
        throw std::runtime_error("--rollout_json, --view_cache_dir, --views, --output_root are required.");
    }
    return cfg;
}

static fs::path makeStepDir(const fs::path& object_root, int view_id, int step_id) {
    return object_root / ("view_" + zeroPad3(view_id)) / ("step_" + zeroPad3(step_id));
}

static bool isValidNonTerminalCase(const StepRecord& step) {
    return (step.oracle_nbv_gain > 0) || (!step.residual_set_cover_view_ids.empty());
}

static std::vector<double> makePcnbvScoreVector(
    const StepRecord& step,
    int expected_num_views,
    int universe_voxel_count) {

    std::vector<double> score(expected_num_views, 0.0);
    const double denom = std::max(1, universe_voxel_count);

    for (size_t i = 0; i < step.candidate_view_ids.size() && i < step.candidate_marginal_gains.size(); ++i) {
        const int vid = step.candidate_view_ids[i];
        if (vid < 0 || vid >= expected_num_views) continue;
        score[vid] = static_cast<double>(step.candidate_marginal_gains[i]) / denom;
    }
    return score;
}

static std::vector<int> makeStateVector(const StepRecord& step, int expected_num_views) {
    std::vector<int> state(expected_num_views, 0);
    for (int vid : step.visited_view_ids) {
        if (vid >= 0 && vid < expected_num_views) state[vid] = 1;
    }
    return state;
}

static Json::Value makeObjectMeta(const RolloutData& rollout, const Config& cfg) {
    Json::Value x(Json::objectValue);
    x["uid"] = rollout.uid;
    x["expected_num_views"] = cfg.expected_num_views;
    x["view_radius"] = cfg.view_radius;
    x["cache_resolution"] = cfg.cache_resolution;
    x["grid_size"] = cfg.grid_size;
    x["bbox_min"] = cfg.bbox_min;
    x["bbox_max"] = cfg.bbox_max;
    x["unknown_occ"] = cfg.unknown_occ;
    x["scvp_num_start_views"] = cfg.scvp_num_start_views;
    x["longtail_need_case_1"] = cfg.longtail_need_case_1;
    x["random_seed"] = cfg.random_seed;
    x["universe_voxel_count"] = rollout.universe_voxel_count;
    x["source_rollout_json"] = cfg.rollout_json;
    x["source_view_cache_dir"] = cfg.view_cache_dir;
    x["views_path"] = cfg.views_path;
    return x;
}

static void saveLongtailSamplingJson(
    const fs::path& path,
    const std::string& uid,
    int random_seed,
    int need_case_1,
    const std::unordered_map<int, int>& all_counts,
    const std::unordered_map<int, double>& distribution_gain,
    const std::unordered_map<int, int>& longtail_counts,
    const std::unordered_map<int, std::vector<CaseRef>>& selected_cases_by_bucket) {

    Json::Value root(Json::objectValue);
    root["uid"] = uid;
    root["random_seed"] = random_seed;
    root["need_case_1"] = need_case_1;

    std::set<int> ms;
    for (const auto& kv : all_counts) ms.insert(kv.first);
    for (const auto& kv : distribution_gain) ms.insert(kv.first);
    for (const auto& kv : longtail_counts) ms.insert(kv.first);

    root["distribution"] = Json::arrayValue;
    for (int m : ms) {
        Json::Value row(Json::objectValue);
        row["m"] = m;
        row["all_cases"] = all_counts.count(m) ? all_counts.at(m) : 0;
        row["avg_oracle_nbv_gain"] = distribution_gain.count(m) ? distribution_gain.at(m) : 0.0;
        row["target_cases"] = longtail_counts.count(m) ? longtail_counts.at(m) : 0;
        row["selected_cases"] = longtail_counts.count(m) ? longtail_counts.at(m) : 0;
        root["distribution"].append(row);
    }

    root["selected_cases_by_bucket"] = Json::arrayValue;
    for (int m : ms) {
        Json::Value bucket(Json::objectValue);
        bucket["m"] = m;
        bucket["cases"] = Json::arrayValue;

        auto it = selected_cases_by_bucket.find(m);
        if (it != selected_cases_by_bucket.end()) {
            for (const auto& c : it->second) {
                bucket["cases"].append(makeCaseJson(c));
            }
        }
        root["selected_cases_by_bucket"].append(bucket);
    }

    saveJsonFile(root, path);
}

static void saveNbvSamplingJson(
    const fs::path& path,
    const std::string& uid,
    int random_seed,
    int target_case_num,
    int selected_start_views,
    int selected_case_num,
    const std::vector<std::pair<int, int>>& group_sizes,
    const std::vector<int>& selected_start_view_ids,
    const std::vector<CaseRef>& selected_cases) {

    Json::Value root(Json::objectValue);
    root["uid"] = uid;
    root["random_seed"] = random_seed;
    root["target_case_num"] = target_case_num;
    root["selected_start_views"] = selected_start_views;
    root["selected_case_num"] = selected_case_num;

    root["group_sizes"] = Json::arrayValue;
    for (const auto& kv : group_sizes) {
        Json::Value x(Json::objectValue);
        x["start_view_id"] = kv.first;
        x["case_count"] = kv.second;
        root["group_sizes"].append(x);
    }

    root["selected_start_view_ids"] = Json::arrayValue;
    for (int vid : selected_start_view_ids) {
        root["selected_start_view_ids"].append(vid);
    }

    root["selected_cases"] = Json::arrayValue;
    for (const auto& c : selected_cases) {
        root["selected_cases"].append(makeCaseJson(c));
    }

    saveJsonFile(root, path);
}

static void saveScvpSamplingJson(
    const fs::path& path,
    const std::string& uid,
    int random_seed,
    int requested_start_views,
    int selected_start_views,
    const std::vector<int>& candidate_start_view_ids,
    const std::vector<int>& selected_start_view_ids,
    const std::vector<CaseRef>& selected_cases) {

    Json::Value root(Json::objectValue);
    root["uid"] = uid;
    root["random_seed"] = random_seed;
    root["requested_start_views"] = requested_start_views;
    root["selected_start_views"] = selected_start_views;

    root["candidate_start_view_ids"] = Json::arrayValue;
    for (int vid : candidate_start_view_ids) {
        root["candidate_start_view_ids"].append(vid);
    }

    root["selected_start_view_ids"] = Json::arrayValue;
    for (int vid : selected_start_view_ids) {
        root["selected_start_view_ids"].append(vid);
    }

    root["selected_cases"] = Json::arrayValue;
    for (const auto& c : selected_cases) {
        root["selected_cases"].append(makeCaseJson(c));
    }

    saveJsonFile(root, path);
}

int main(int argc, char** argv) {
    try {
        Config cfg = parseArgs(argc, argv);

        RolloutData rollout = loadRolloutData(cfg.rollout_json);
        if (rollout.expected_num_views > 0 && cfg.expected_num_views != rollout.expected_num_views) {
            std::cerr << "[WARN] rollout expected_num_views=" << rollout.expected_num_views
                      << " but cfg expected_num_views=" << cfg.expected_num_views << "\n";
        }

        const fs::path object_root = fs::path(cfg.output_root) / "method_cases" / rollout.uid;
        const fs::path case_out_root = fs::path(cfg.output_root) / "cases" / rollout.uid;

        if (cfg.skip_existing == 1 &&
            fs::exists(case_out_root / "nbvnet_nbv_sampled_cases.json") &&
            fs::exists(case_out_root / "pcnbv_nbv_sampled_cases.json") &&
            fs::exists(case_out_root / "scvp_cases.json") &&
            fs::exists(case_out_root / "pcnbv_longtail_cases.json") &&
            fs::exists(case_out_root / "mascvp_longtail_cases.json") &&
            fs::exists(case_out_root / "nbv_sampling.json") &&
            fs::exists(case_out_root / "scvp_sampling.json") &&
            fs::exists(case_out_root / "longtail_sampling.json")) {
            std::cout << "Skip existing uid: " << rollout.uid << std::endl;
            return 0;
        }

        ensureDir(object_root);
        ensureDir(case_out_root);

        saveJsonFile(makeObjectMeta(rollout, cfg), object_root / "meta.json");

        std::vector<Vec3> view_origins = loadViews(cfg.views_path, cfg.view_radius);
        if (static_cast<int>(view_origins.size()) != cfg.expected_num_views) {
            throw std::runtime_error("views_path count != expected_num_views");
        }

        std::vector<CloudXYZ::Ptr> cached_view_clouds(cfg.expected_num_views);
        for (int vid = 0; vid < cfg.expected_num_views; ++vid) {
            auto cloud = CloudXYZ::Ptr(new CloudXYZ);
            fs::path pcd_path = fs::path(cfg.view_cache_dir) / ("view_" + zeroPad3(vid) + ".pcd");
            if (!fs::exists(pcd_path)) {
                throw std::runtime_error("Missing cached view pcd: " + pcd_path.string());
            }
            if (!loadPcdAsXYZ(pcd_path, cloud)) {
                throw std::runtime_error("Failed to load cached pcd: " + pcd_path.string());
            }
            cached_view_clouds[vid] = cloud;
        }

        // Pools
        std::vector<CaseRef> scvp_all_cases;
        std::unordered_map<int, std::vector<CaseRef>> nbv_groups_by_start_view;
        std::unordered_map<int, std::vector<CaseRef>> longtail_buckets_by_m;
        std::unordered_map<int, std::vector<CaseRef>> all_valid_by_start_view;

        for (const auto& rr : rollout.rollouts) {
            for (const auto& step : rr.steps) {
                if (!isValidNonTerminalCase(step)) continue;

                CaseRef cref;
                cref.uid = rollout.uid;
                cref.view_id = rr.start_view_id;
                cref.step_id = step.step_id;
                cref.m = static_cast<int>(step.visited_view_ids.size());
                cref.oracle_nbv_gain = step.oracle_nbv_gain;
                cref.covered_ratio = step.covered_ratio;
                cref.residual_set_cover_size = step.residual_set_cover_size;

                all_valid_by_start_view[rr.start_view_id].push_back(cref);
                longtail_buckets_by_m[cref.m].push_back(cref);

                if (step.step_id == 0) {
                    scvp_all_cases.push_back(cref);
                }
            }
        }

        // Build NBV groups: each start_view is one rollout group
        for (auto& kv : all_valid_by_start_view) {
            auto& group = kv.second;
            std::sort(group.begin(), group.end(),
                      [](const CaseRef& a, const CaseRef& b) {
                          return a.step_id < b.step_id;
                      });
            nbv_groups_by_start_view[kv.first] = group;
        }

        // uid-dependent RNGs
        std::mt19937 rng_longtail(makeUidSeed(rollout.uid, cfg.random_seed, 0x13579BDFu));
        std::mt19937 rng_nbv(makeUidSeed(rollout.uid, cfg.random_seed, 0x2468ACE0u));
        std::mt19937 rng_scvp(makeUidSeed(rollout.uid, cfg.random_seed, 0xA5A5A5A5u));

        // -----------------------------
        // Long-tail protocol first
        // -----------------------------
        std::unordered_map<int, double> longtail_distribution_gain;
        std::unordered_map<int, int> longtail_all_counts;

        for (const auto& kv : longtail_buckets_by_m) {
            const int m = kv.first;
            const auto& bucket = kv.second;
            longtail_all_counts[m] = static_cast<int>(bucket.size());

            double sum_gain = 0.0;
            for (const auto& c : bucket) sum_gain += static_cast<double>(c.oracle_nbv_gain);
            longtail_distribution_gain[m] = bucket.empty() ? 0.0 : sum_gain / static_cast<double>(bucket.size());
        }

        const double base_gain =
            (longtail_distribution_gain.count(1) && longtail_distribution_gain[1] > 0.0)
                ? longtail_distribution_gain[1]
                : 1.0;

        std::vector<CaseRef> longtail_selected;
        std::unordered_map<int, int> longtail_selected_counts;
        std::unordered_map<int, std::vector<CaseRef>> longtail_selected_by_bucket;

        for (auto& kv : longtail_buckets_by_m) {
            const int m = kv.first;
            auto bucket = kv.second;

            std::shuffle(bucket.begin(), bucket.end(), rng_longtail);

            const double g = longtail_distribution_gain.count(m) ? longtail_distribution_gain[m] : 0.0;
            int need = static_cast<int>(std::ceil(
                static_cast<double>(cfg.longtail_need_case_1) / base_gain * g));

            if (need < 0) need = 0;
            if (need > static_cast<int>(bucket.size())) need = static_cast<int>(bucket.size());

            longtail_selected_counts[m] = need;

            for (int i = 0; i < need; ++i) {
                longtail_selected.push_back(bucket[i]);
                longtail_selected_by_bucket[m].push_back(bucket[i]);
            }
        }

        std::sort(longtail_selected.begin(), longtail_selected.end(),
                  [](const CaseRef& a, const CaseRef& b) {
                      if (a.uid != b.uid) return a.uid < b.uid;
                      if (a.view_id != b.view_id) return a.view_id < b.view_id;
                      return a.step_id < b.step_id;
                  });

        const int target_nbv_case_num = static_cast<int>(longtail_selected.size());

        // -----------------------------
        // NBV protocol: sample start-view rollout groups
        // until total case num is close to long-tail size
        // -----------------------------
        std::vector<int> start_view_ids;
        std::vector<std::pair<int, int>> group_sizes;
        for (const auto& kv : nbv_groups_by_start_view) {
            start_view_ids.push_back(kv.first);
            group_sizes.push_back({kv.first, static_cast<int>(kv.second.size())});
        }

        std::shuffle(start_view_ids.begin(), start_view_ids.end(), rng_nbv);

        std::vector<int> nbv_selected_start_view_ids;
        std::vector<CaseRef> nbv_sampled_cases;
        int cur_nbv_case_num = 0;

        for (int vid : start_view_ids) {
            const auto& group = nbv_groups_by_start_view[vid];
            const int group_size = static_cast<int>(group.size());

            const int without_group_diff = std::abs(cur_nbv_case_num - target_nbv_case_num);
            const int with_group_diff = std::abs((cur_nbv_case_num + group_size) - target_nbv_case_num);

            if (cur_nbv_case_num < target_nbv_case_num) {
                if (with_group_diff <= without_group_diff || nbv_selected_start_view_ids.empty()) {
                    nbv_selected_start_view_ids.push_back(vid);
                    nbv_sampled_cases.insert(nbv_sampled_cases.end(), group.begin(), group.end());
                    cur_nbv_case_num += group_size;
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        std::sort(nbv_sampled_cases.begin(), nbv_sampled_cases.end(),
                  [](const CaseRef& a, const CaseRef& b) {
                      if (a.uid != b.uid) return a.uid < b.uid;
                      if (a.view_id != b.view_id) return a.view_id < b.view_id;
                      return a.step_id < b.step_id;
                  });

        // -----------------------------
        // SCVP protocol: sample step_000 start views
        // with uid-dependent seed
        // -----------------------------
        std::vector<int> scvp_candidate_start_view_ids;
        for (const auto& c : scvp_all_cases) {
            scvp_candidate_start_view_ids.push_back(c.view_id);
        }
        std::sort(scvp_candidate_start_view_ids.begin(), scvp_candidate_start_view_ids.end());

        std::vector<int> scvp_selected_start_view_ids = scvp_candidate_start_view_ids;
        std::shuffle(scvp_selected_start_view_ids.begin(), scvp_selected_start_view_ids.end(), rng_scvp);
        if (static_cast<int>(scvp_selected_start_view_ids.size()) > cfg.scvp_num_start_views) {
            scvp_selected_start_view_ids.resize(cfg.scvp_num_start_views);
        }
        std::sort(scvp_selected_start_view_ids.begin(), scvp_selected_start_view_ids.end());

        std::unordered_set<int> scvp_selected_set(
            scvp_selected_start_view_ids.begin(),
            scvp_selected_start_view_ids.end());

        std::vector<CaseRef> scvp_sampled_cases;
        for (const auto& c : scvp_all_cases) {
            if (scvp_selected_set.count(c.view_id)) {
                scvp_sampled_cases.push_back(c);
            }
        }
        std::sort(scvp_sampled_cases.begin(), scvp_sampled_cases.end(),
                  [](const CaseRef& a, const CaseRef& b) {
                      if (a.uid != b.uid) return a.uid < b.uid;
                      if (a.view_id != b.view_id) return a.view_id < b.view_id;
                      return a.step_id < b.step_id;
                  });

        const std::vector<CaseRef>& nbvnet_nbv_sampled_cases = nbv_sampled_cases;
        const std::vector<CaseRef>& pcnbv_nbv_sampled_cases = nbv_sampled_cases;
        const std::vector<CaseRef>& pcnbv_longtail_cases = longtail_selected;
        const std::vector<CaseRef>& mascvp_longtail_cases = longtail_selected;

        // -----------------------------
        // Build final physical-case union
        // -----------------------------
        std::unordered_set<StepKey, StepKeyHash> physical_case_union;

        auto addCasesToUnion = [&](const std::vector<CaseRef>& cases) {
            for (const auto& c : cases) {
                physical_case_union.insert({c.view_id, c.step_id});
            }
        };

        addCasesToUnion(nbvnet_nbv_sampled_cases);
        addCasesToUnion(pcnbv_nbv_sampled_cases);
        addCasesToUnion(scvp_sampled_cases);
        addCasesToUnion(pcnbv_longtail_cases);
        addCasesToUnion(mascvp_longtail_cases);

        // -----------------------------
        // Export only sampled/needed physical cases
        // -----------------------------
        for (const auto& rr : rollout.rollouts) {
            const double tree_res = (cfg.bbox_max - cfg.bbox_min) / static_cast<double>(cfg.grid_size);
            octomap::OcTree tree(tree_res);

            KeySet merged_keys;
            std::vector<Vec3> merged_points;

            if (rr.steps.empty()) continue;

            {
                const std::vector<int> init_views = rr.steps.front().visited_view_ids;
                for (int vid : init_views) {
                    if (vid < 0 || vid >= cfg.expected_num_views) continue;

                    insertViewCloudIntoTree(tree, cached_view_clouds[vid], view_origins[vid], cfg.bbox_min, cfg.bbox_max);

                    octomap::OcTree tmp(cfg.cache_resolution);
                    for (const auto& p : cached_view_clouds[vid]->points) {
                        octomap::OcTreeKey key;
                        if (!tmp.coordToKeyChecked(octomap::point3d(p.x, p.y, p.z), key)) continue;
                        if (merged_keys.insert(key).second) {
                            merged_points.emplace_back(p.x, p.y, p.z);
                        }
                    }
                }
            }

            for (size_t si = 0; si < rr.steps.size(); ++si) {
                const StepRecord& step = rr.steps[si];

                if (!isValidNonTerminalCase(step)) {
                    if (si + 1 < rr.steps.size() && step.oracle_nbv_gain > 0) {
                        const int next_vid = step.oracle_nbv_view_id;
                        if (next_vid >= 0 && next_vid < cfg.expected_num_views) {
                            insertViewCloudIntoTree(tree, cached_view_clouds[next_vid], view_origins[next_vid], cfg.bbox_min, cfg.bbox_max);

                            octomap::OcTree tmp(cfg.cache_resolution);
                            for (const auto& p : cached_view_clouds[next_vid]->points) {
                                octomap::OcTreeKey key;
                                if (!tmp.coordToKeyChecked(octomap::point3d(p.x, p.y, p.z), key)) continue;
                                if (merged_keys.insert(key).second) {
                                    merged_points.emplace_back(p.x, p.y, p.z);
                                }
                            }
                        }
                    }
                    continue;
                }

                StepKey sk{rr.start_view_id, step.step_id};
                if (physical_case_union.find(sk) != physical_case_union.end()) {
                    const fs::path step_dir = makeStepDir(object_root, rr.start_view_id, step.step_id);

                    const std::vector<int> state = makeStateVector(step, cfg.expected_num_views);
                    const std::vector<double> score = makePcnbvScoreVector(step, cfg.expected_num_views, rr.universe_voxel_count);

                    saveTreeOT(tree, step_dir / "grid.ot");
                    saveCloudPcd(merged_points, step_dir / "cloud.pcd");
                    writeIntListOnePerLine(step_dir / "state.txt", state);
                    writeFloatListOnePerLine(step_dir / "score.txt", score);
                    writeSingleInt(step_dir / "id.txt", step.oracle_nbv_view_id);
                    writeIntListOnePerLine(step_dir / "ids.txt", step.residual_set_cover_view_ids);
                }

                if (si + 1 < rr.steps.size() && step.oracle_nbv_gain > 0) {
                    const int next_vid = step.oracle_nbv_view_id;
                    if (next_vid >= 0 && next_vid < cfg.expected_num_views) {
                        insertViewCloudIntoTree(tree, cached_view_clouds[next_vid], view_origins[next_vid], cfg.bbox_min, cfg.bbox_max);

                        octomap::OcTree tmp(cfg.cache_resolution);
                        for (const auto& p : cached_view_clouds[next_vid]->points) {
                            octomap::OcTreeKey key;
                            if (!tmp.coordToKeyChecked(octomap::point3d(p.x, p.y, p.z), key)) continue;
                            if (merged_keys.insert(key).second) {
                                merged_points.emplace_back(p.x, p.y, p.z);
                            }
                        }
                    }
                }
            }
        }

        // -----------------------------
        // Save task protocol jsons
        // -----------------------------
        {
            Json::Value root(Json::arrayValue);
            for (const auto& c : nbvnet_nbv_sampled_cases) root.append(makeCaseJson(c));
            saveJsonFile(root, case_out_root / "nbvnet_nbv_sampled_cases.json");
        }
        {
            Json::Value root(Json::arrayValue);
            for (const auto& c : pcnbv_nbv_sampled_cases) root.append(makeCaseJson(c));
            saveJsonFile(root, case_out_root / "pcnbv_nbv_sampled_cases.json");
        }
        {
            Json::Value root(Json::arrayValue);
            for (const auto& c : scvp_sampled_cases) root.append(makeCaseJson(c));
            saveJsonFile(root, case_out_root / "scvp_cases.json");
        }
        {
            Json::Value root(Json::arrayValue);
            for (const auto& c : pcnbv_longtail_cases) root.append(makeCaseJson(c));
            saveJsonFile(root, case_out_root / "pcnbv_longtail_cases.json");
        }
        {
            Json::Value root(Json::arrayValue);
            for (const auto& c : mascvp_longtail_cases) root.append(makeCaseJson(c));
            saveJsonFile(root, case_out_root / "mascvp_longtail_cases.json");
        }

        saveNbvSamplingJson(
            case_out_root / "nbv_sampling.json",
            rollout.uid,
            cfg.random_seed,
            target_nbv_case_num,
            static_cast<int>(nbv_selected_start_view_ids.size()),
            static_cast<int>(nbv_sampled_cases.size()),
            group_sizes,
            nbv_selected_start_view_ids,
            nbv_sampled_cases);

        saveScvpSamplingJson(
            case_out_root / "scvp_sampling.json",
            rollout.uid,
            cfg.random_seed,
            cfg.scvp_num_start_views,
            static_cast<int>(scvp_selected_start_view_ids.size()),
            scvp_candidate_start_view_ids,
            scvp_selected_start_view_ids,
            scvp_sampled_cases);

        saveLongtailSamplingJson(
            case_out_root / "longtail_sampling.json",
            rollout.uid,
            cfg.random_seed,
            cfg.longtail_need_case_1,
            longtail_all_counts,
            longtail_distribution_gain,
            longtail_selected_counts,
            longtail_selected_by_bucket);

        std::cout << "Done uid = " << rollout.uid << "\n";
        std::cout << "NBV sampled cases: " << nbv_sampled_cases.size() << "\n";
        std::cout << "NBV sampled start views: " << nbv_selected_start_view_ids.size() << "\n";
        std::cout << "SCVP sampled cases: " << scvp_sampled_cases.size() << "\n";
        std::cout << "Long-tail selected: " << longtail_selected.size() << "\n";
        std::cout << "Physical case union size: " << physical_case_union.size() << "\n";
        std::cout << "Output root: " << cfg.output_root << "\n";

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
