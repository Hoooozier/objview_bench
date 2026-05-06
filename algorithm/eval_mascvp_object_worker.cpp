#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <json/json.h>

#include <octomap/octomap.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace fs = std::filesystem;

namespace {

using PointT = pcl::PointXYZRGB;
using CloudT = pcl::PointCloud<PointT>;

struct Config {
    std::string uid;
    std::string analysis_json;
    std::string rollout_json;
    std::string view_cache_dir;
    std::string raw_results_dir;
    std::string balanced_results_dir;
    std::string output_json;

    int expected_num_views = 128;
    double resolution = 0.02;
    double gamma = 0.5;
    int step_id = 0;
    int max_step_id = -1;
    int skip_existing = 0;
};

struct ObjectMeta {
    int c_selected_view_count = -1;
    std::string pool_type;
    std::string shape_type;
    std::string fill_bucket;
    double self_occlusion_attribute = std::numeric_limits<double>::quiet_NaN();
    int observation_saturation_view_num = -1;
};

struct StepRolloutRef {
    int start_view_id = -1;
    std::vector<int> visited_view_ids;
    int residual_set_cover_size = -1;
    std::vector<int> residual_set_cover_view_ids;
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

struct StaticCoverageData {
    std::vector<std::vector<int>> coverage_ids_per_view;
    int universe_size = 0;
};

struct EpisodeEval {
    int start_view_id = -1;
    int visited_view_count = 0;
    int gt_residual_size = -1;
    int gt_total_size = -1;

    int raw_pred_residual_size = -1;
    int raw_pred_total_size = -1;
    double raw_vc = std::numeric_limits<double>::quiet_NaN();

    int balanced_pred_residual_size = -1;
    int balanced_pred_total_size = -1;
    double balanced_vc = std::numeric_limits<double>::quiet_NaN();
};

void printUsage(const char* argv0) {
    std::cout
        << "Usage: " << argv0
        << " --uid UID --analysis_json PATH --rollout_json PATH --view_cache_dir DIR"
        << " --raw_results_dir DIR --balanced_results_dir DIR --output_json PATH [options]\n"
        << "Options:\n"
        << "  --expected_num_views 128\n"
        << "  --resolution 0.02\n"
        << "  --gamma 0.5             Decode selected indices from scores using score >= gamma\n"
        << "  --step_id 0\n"
        << "  --max_step_id -1         -1 means evaluate only step_id\n"
        << "  --skip_existing 0\n";
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
        else if (arg == "--analysis_json") cfg.analysis_json = needValue(arg);
        else if (arg == "--rollout_json") cfg.rollout_json = needValue(arg);
        else if (arg == "--view_cache_dir") cfg.view_cache_dir = needValue(arg);
        else if (arg == "--raw_results_dir") cfg.raw_results_dir = needValue(arg);
        else if (arg == "--balanced_results_dir") cfg.balanced_results_dir = needValue(arg);
        else if (arg == "--output_json") cfg.output_json = needValue(arg);
        else if (arg == "--expected_num_views") cfg.expected_num_views = std::stoi(needValue(arg));
        else if (arg == "--resolution") cfg.resolution = std::stod(needValue(arg));
        else if (arg == "--gamma") cfg.gamma = std::stod(needValue(arg));
        else if (arg == "--step_id") cfg.step_id = std::stoi(needValue(arg));
        else if (arg == "--max_step_id") cfg.max_step_id = std::stoi(needValue(arg));
        else if (arg == "--skip_existing") cfg.skip_existing = std::stoi(needValue(arg));
        else if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (cfg.uid.empty() || cfg.analysis_json.empty() || cfg.rollout_json.empty() ||
        cfg.view_cache_dir.empty() || cfg.raw_results_dir.empty() ||
        cfg.balanced_results_dir.empty() || cfg.output_json.empty()) {
        throw std::runtime_error("Missing required arguments.");
    }
    if (cfg.expected_num_views <= 0) throw std::runtime_error("--expected_num_views must be positive.");
    if (!std::isfinite(cfg.gamma)) throw std::runtime_error("--gamma must be finite.");
    if (cfg.step_id < 0) throw std::runtime_error("--step_id must be >= 0.");
    if (cfg.max_step_id != -1 && cfg.max_step_id < cfg.step_id) {
        throw std::runtime_error("--max_step_id must be -1 or >= --step_id.");
    }
    return cfg;
}

Json::Value loadJsonFile(const fs::path& path) {
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

std::string makeViewFilename(int idx) {
    std::ostringstream oss;
    oss << "view_" << std::setw(3) << std::setfill('0') << idx << ".pcd";
    return oss.str();
}

std::string makeCaseStem(const std::string& uid, int view_id, int step_id) {
    std::ostringstream oss;
    oss << uid << "__view_" << std::setw(3) << std::setfill('0') << view_id
        << "__step_" << std::setw(3) << std::setfill('0') << step_id;
    return oss.str();
}

ObjectMeta loadObjectMeta(const fs::path& analysis_json, const std::string& uid) {
    const Json::Value root = loadJsonFile(analysis_json);
    if (!root.isArray()) {
        throw std::runtime_error("Expected top-level array in analysis_json.");
    }

    for (const Json::Value& item : root) {
        if (!item.isObject()) continue;
        if (!item.isMember("uid") || !item["uid"].isString()) continue;
        if (item["uid"].asString() != uid) continue;

        ObjectMeta meta;
        if (item.isMember("selected_view_count") && item["selected_view_count"].isInt()) {
            meta.c_selected_view_count = item["selected_view_count"].asInt();
        }
        if (item.isMember("pool_type") && item["pool_type"].isString()) {
            meta.pool_type = item["pool_type"].asString();
        }
        if (item.isMember("shape_type") && item["shape_type"].isString()) {
            meta.shape_type = item["shape_type"].asString();
        }
        if (item.isMember("fill_bucket") && item["fill_bucket"].isString()) {
            meta.fill_bucket = item["fill_bucket"].asString();
        }
        if (item.isMember("self_occlusion_attribute") && item["self_occlusion_attribute"].isNumeric()) {
            meta.self_occlusion_attribute = item["self_occlusion_attribute"].asDouble();
        }
        if (item.isMember("observation_saturation_view_num") && item["observation_saturation_view_num"].isInt()) {
            meta.observation_saturation_view_num = item["observation_saturation_view_num"].asInt();
        }
        return meta;
    }

    throw std::runtime_error("UID not found in analysis_json: " + uid);
}

std::unordered_map<int, StepRolloutRef> loadRolloutRefs(const fs::path& rollout_json, int step_id) {
    const Json::Value root = loadJsonFile(rollout_json);
    if (!root.isObject() || !root.isMember("rollouts") || !root["rollouts"].isArray()) {
        throw std::runtime_error("Invalid rollout json structure.");
    }

    std::unordered_map<int, StepRolloutRef> refs;
    for (const Json::Value& rollout : root["rollouts"]) {
        if (!rollout.isObject()) continue;
        if (!rollout.isMember("start_view_id") || !rollout["start_view_id"].isInt()) continue;
        const int start_view_id = rollout["start_view_id"].asInt();
        if (!rollout.isMember("steps") || !rollout["steps"].isArray()) {
            throw std::runtime_error("Missing steps array for rollout start_view_id=" + std::to_string(start_view_id));
        }
        const Json::Value& steps = rollout["steps"];
        if (step_id >= static_cast<int>(steps.size())) {
            throw std::runtime_error("Requested step_id out of range for start_view_id=" + std::to_string(start_view_id));
        }
        const Json::Value& step = steps[step_id];
        StepRolloutRef ref;
        ref.start_view_id = start_view_id;
        if (step.isMember("visited_view_ids") && step["visited_view_ids"].isArray()) {
            for (const Json::Value& x : step["visited_view_ids"]) {
                if (x.isInt()) ref.visited_view_ids.push_back(x.asInt());
            }
            std::sort(ref.visited_view_ids.begin(), ref.visited_view_ids.end());
            ref.visited_view_ids.erase(
                std::unique(ref.visited_view_ids.begin(), ref.visited_view_ids.end()),
                ref.visited_view_ids.end());
        } else {
            throw std::runtime_error("Missing visited_view_ids in rollout step.");
        }
        if (step.isMember("residual_set_cover_size") && step["residual_set_cover_size"].isInt()) {
            ref.residual_set_cover_size = step["residual_set_cover_size"].asInt();
        } else {
            throw std::runtime_error("Missing residual_set_cover_size in rollout step.");
        }
        if (step.isMember("residual_set_cover_view_ids") && step["residual_set_cover_view_ids"].isArray()) {
            for (const Json::Value& x : step["residual_set_cover_view_ids"]) {
                if (x.isInt()) ref.residual_set_cover_view_ids.push_back(x.asInt());
            }
        } else {
            throw std::runtime_error("Missing residual_set_cover_view_ids in rollout step.");
        }
        refs.emplace(start_view_id, std::move(ref));
    }
    return refs;
}

std::vector<octomap::OcTreeKey> loadViewCoverageKeys(const fs::path& pcd_path, double resolution) {
    CloudT cloud;
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
    for (const auto& key : keyset) keys.push_back(key);
    return keys;
}

StaticCoverageData buildStaticCoverageData(const Config& cfg) {
    if (!fs::exists(cfg.view_cache_dir) || !fs::is_directory(cfg.view_cache_dir)) {
        throw std::runtime_error("Invalid view_cache_dir: " + cfg.view_cache_dir);
    }

    StaticCoverageData data;
    data.coverage_ids_per_view.resize(cfg.expected_num_views);

    KeyToIdMap voxel_to_id;
    voxel_to_id.reserve(500000);
    int next_id = 0;

    for (int vid = 0; vid < cfg.expected_num_views; ++vid) {
        const fs::path pcd_path = fs::path(cfg.view_cache_dir) / makeViewFilename(vid);
        if (!fs::exists(pcd_path)) {
            throw std::runtime_error("Missing cached view PCD: " + pcd_path.string());
        }

        const auto keys = loadViewCoverageKeys(pcd_path, cfg.resolution);
        std::vector<int> ids;
        ids.reserve(keys.size());
        for (const auto& key : keys) {
            auto it = voxel_to_id.find(key);
            if (it == voxel_to_id.end()) {
                voxel_to_id.emplace(key, next_id);
                ids.push_back(next_id);
                ++next_id;
            } else {
                ids.push_back(it->second);
            }
        }
        std::sort(ids.begin(), ids.end());
        ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
        data.coverage_ids_per_view[vid] = std::move(ids);
    }

    data.universe_size = next_id;
    return data;
}

std::vector<int> parseSelectedIndicesFromScores(const fs::path& result_json_path, double gamma) {
    const Json::Value root = loadJsonFile(result_json_path);
    if (!root.isObject() || !root.isMember("scores") || !root["scores"].isArray()) {
        throw std::runtime_error("Invalid result json structure: " + result_json_path.string());
    }

    const Json::Value& outer = root["scores"];
    if (outer.empty() || !outer[0].isArray()) {
        return {};
    }

    std::vector<int> out;
    const Json::Value& row = outer[0];
    out.reserve(row.size());
    for (Json::ArrayIndex i = 0; i < row.size(); ++i) {
        if (!row[i].isNumeric()) continue;
        if (row[i].asDouble() >= gamma) {
            out.push_back(static_cast<int>(i));
        }
    }
    return out;
}

std::vector<int> canonicalizePredictedResidual(
    const std::vector<int>& selected_indices,
    const std::vector<int>& visited_view_ids,
    int expected_num_views) {
    std::vector<int> out;
    out.reserve(selected_indices.size());
    std::vector<uint8_t> seen(expected_num_views, 0);
    std::vector<uint8_t> visited_mask(expected_num_views, 0);
    for (int visited_view_id : visited_view_ids) {
        if (visited_view_id >= 0 && visited_view_id < expected_num_views) {
            visited_mask[visited_view_id] = 1;
        }
    }

    for (int vid : selected_indices) {
        if (vid < 0 || vid >= expected_num_views) continue;
        if (visited_mask[vid]) continue;
        if (seen[vid]) continue;
        seen[vid] = 1;
        out.push_back(vid);
    }
    std::sort(out.begin(), out.end());
    return out;
}

double computeVoxelCoverage(
    const StaticCoverageData& data,
    const std::vector<int>& visited_view_ids,
    const std::vector<int>& predicted_residual) {
    if (data.universe_size <= 0) return 0.0;
    std::vector<uint8_t> covered(static_cast<std::size_t>(data.universe_size), 0);
    int covered_count = 0;

    auto addView = [&](int view_id) {
        for (int voxel_id : data.coverage_ids_per_view[view_id]) {
            if (!covered[voxel_id]) {
                covered[voxel_id] = 1;
                ++covered_count;
            }
        }
    };

    for (int visited_view_id : visited_view_ids) addView(visited_view_id);
    for (int vid : predicted_residual) addView(vid);
    return static_cast<double>(covered_count) / static_cast<double>(data.universe_size);
}

double meanOf(const std::vector<double>& xs) {
    if (xs.empty()) return std::numeric_limits<double>::quiet_NaN();
    const double sum = std::accumulate(xs.begin(), xs.end(), 0.0);
    return sum / static_cast<double>(xs.size());
}

double stddevOf(const std::vector<double>& xs, double mean) {
    if (xs.empty() || !std::isfinite(mean)) return std::numeric_limits<double>::quiet_NaN();
    double acc = 0.0;
    for (double x : xs) {
        const double d = x - mean;
        acc += d * d;
    }
    return std::sqrt(acc / static_cast<double>(xs.size()));
}

Json::Value toJsonArray(const std::vector<int>& xs) {
    Json::Value arr(Json::arrayValue);
    for (int x : xs) arr.append(x);
    return arr;
}

Json::Value toJson(const EpisodeEval& e) {
    Json::Value x(Json::objectValue);
    x["start_view_id"] = e.start_view_id;
    x["visited_view_count"] = e.visited_view_count;
    x["gt_residual_size"] = e.gt_residual_size;
    x["gt_total_size"] = e.gt_total_size;
    x["raw_pred_residual_size"] = e.raw_pred_residual_size;
    x["raw_pred_total_size"] = e.raw_pred_total_size;
    x["raw_vc"] = e.raw_vc;
    x["balanced_pred_residual_size"] = e.balanced_pred_residual_size;
    x["balanced_pred_total_size"] = e.balanced_pred_total_size;
    x["balanced_vc"] = e.balanced_vc;
    return x;
}

std::string makeOutputPathForStep(const std::string& base_output_json, int step_id) {
    fs::path p(base_output_json);
    const std::string stem = p.stem().string();
    std::ostringstream oss;
    oss << stem << "__step_" << std::setw(3) << std::setfill('0') << step_id << p.extension().string();
    return (p.parent_path() / oss.str()).string();
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Config cfg = parseArgs(argc, argv);
        const ObjectMeta meta = loadObjectMeta(cfg.analysis_json, cfg.uid);
        const StaticCoverageData coverage = buildStaticCoverageData(cfg);
        const int max_step_id = (cfg.max_step_id == -1) ? cfg.step_id : cfg.max_step_id;
        for (int step_id = cfg.step_id; step_id <= max_step_id; ++step_id) {
            const std::string output_json = makeOutputPathForStep(cfg.output_json, step_id);
            if (cfg.skip_existing && fs::exists(output_json)) {
                std::cout << "[skip] existing output: " << output_json << "\n";
                continue;
            }

            const auto rollout_refs = loadRolloutRefs(cfg.rollout_json, step_id);
            std::vector<EpisodeEval> episodes;
            episodes.reserve(cfg.expected_num_views);

            std::vector<double> gt_residual_sizes;
            std::vector<double> gt_total_sizes;
            std::vector<double> visited_view_counts;
            std::vector<double> raw_total_sizes;
            std::vector<double> raw_vcs;
            std::vector<double> balanced_total_sizes;
            std::vector<double> balanced_vcs;

            for (int start_view_id = 0; start_view_id < cfg.expected_num_views; ++start_view_id) {
                auto ref_it = rollout_refs.find(start_view_id);
                if (ref_it == rollout_refs.end()) {
                    throw std::runtime_error("Missing rollout for start_view_id=" + std::to_string(start_view_id));
                }
                const StepRolloutRef& ref = ref_it->second;

                const std::string stem = makeCaseStem(cfg.uid, start_view_id, step_id);
                const fs::path raw_json = fs::path(cfg.raw_results_dir) / cfg.uid / (stem + ".json");
                const fs::path balanced_json = fs::path(cfg.balanced_results_dir) / cfg.uid / (stem + ".json");
                if (!fs::exists(raw_json)) throw std::runtime_error("Missing raw result: " + raw_json.string());
                if (!fs::exists(balanced_json)) throw std::runtime_error("Missing balanced result: " + balanced_json.string());

                const std::vector<int> raw_sel = canonicalizePredictedResidual(
                    parseSelectedIndicesFromScores(raw_json, cfg.gamma),
                    ref.visited_view_ids,
                    cfg.expected_num_views);
                const std::vector<int> bal_sel = canonicalizePredictedResidual(
                    parseSelectedIndicesFromScores(balanced_json, cfg.gamma),
                    ref.visited_view_ids,
                    cfg.expected_num_views);

                EpisodeEval ep;
                ep.start_view_id = start_view_id;
                ep.visited_view_count = static_cast<int>(ref.visited_view_ids.size());
                ep.gt_residual_size = ref.residual_set_cover_size;
                ep.gt_total_size = ref.residual_set_cover_size + static_cast<int>(ref.visited_view_ids.size());
                ep.raw_pred_residual_size = static_cast<int>(raw_sel.size());
                ep.raw_pred_total_size = ep.raw_pred_residual_size + static_cast<int>(ref.visited_view_ids.size());
                ep.raw_vc = computeVoxelCoverage(coverage, ref.visited_view_ids, raw_sel);
                ep.balanced_pred_residual_size = static_cast<int>(bal_sel.size());
                ep.balanced_pred_total_size = ep.balanced_pred_residual_size + static_cast<int>(ref.visited_view_ids.size());
                ep.balanced_vc = computeVoxelCoverage(coverage, ref.visited_view_ids, bal_sel);
                episodes.push_back(ep);

                gt_residual_sizes.push_back(static_cast<double>(ep.gt_residual_size));
                gt_total_sizes.push_back(static_cast<double>(ep.gt_total_size));
                visited_view_counts.push_back(static_cast<double>(ep.visited_view_count));
                raw_total_sizes.push_back(static_cast<double>(ep.raw_pred_total_size));
                raw_vcs.push_back(ep.raw_vc);
                balanced_total_sizes.push_back(static_cast<double>(ep.balanced_pred_total_size));
                balanced_vcs.push_back(ep.balanced_vc);
            }

            const double gt_mean_residual = meanOf(gt_residual_sizes);
            const double gt_mean_total = meanOf(gt_total_sizes);
            const double visited_view_count_mean = meanOf(visited_view_counts);
            const double raw_mean_total = meanOf(raw_total_sizes);
            const double raw_mean_vc = meanOf(raw_vcs);
            const double balanced_mean_total = meanOf(balanced_total_sizes);
            const double balanced_mean_vc = meanOf(balanced_vcs);

            Json::Value root(Json::objectValue);
            root["uid"] = cfg.uid;
            root["step_id"] = step_id;
            root["gamma"] = cfg.gamma;
            root["expected_num_views"] = cfg.expected_num_views;
            root["resolution"] = cfg.resolution;
            root["universe_voxel_count"] = coverage.universe_size;

            Json::Value meta_json(Json::objectValue);
            meta_json["selected_view_count"] = meta.c_selected_view_count;
            meta_json["pool_type"] = meta.pool_type;
            meta_json["shape_type"] = meta.shape_type;
            meta_json["fill_bucket"] = meta.fill_bucket;
            if (std::isfinite(meta.self_occlusion_attribute)) {
                meta_json["self_occlusion_attribute"] = meta.self_occlusion_attribute;
            }
            meta_json["observation_saturation_view_num"] = meta.observation_saturation_view_num;
            root["object_meta"] = meta_json;

            Json::Value summary(Json::objectValue);
            summary["num_start_views"] = static_cast<int>(episodes.size());
            summary["gt_mean_residual_size"] = gt_mean_residual;
            summary["gt_mean_total_size"] = gt_mean_total;
            summary["visited_view_count_mean"] = visited_view_count_mean;
            summary["raw_mean_total_size"] = raw_mean_total;
            summary["raw_std_total_size"] = stddevOf(raw_total_sizes, raw_mean_total);
            summary["raw_mean_vc"] = raw_mean_vc;
            summary["raw_std_vc"] = stddevOf(raw_vcs, raw_mean_vc);
            summary["balanced_mean_total_size"] = balanced_mean_total;
            summary["balanced_std_total_size"] = stddevOf(balanced_total_sizes, balanced_mean_total);
            summary["balanced_mean_vc"] = balanced_mean_vc;
            summary["balanced_std_vc"] = stddevOf(balanced_vcs, balanced_mean_vc);
            summary["raw_size_bias"] = raw_mean_total - gt_mean_total;
            summary["balanced_size_bias"] = balanced_mean_total - gt_mean_total;
            summary["raw_vc_gap"] = 1.0 - raw_mean_vc;
            summary["balanced_vc_gap"] = 1.0 - balanced_mean_vc;
            root["summary"] = summary;

            Json::Value episodes_json(Json::arrayValue);
            for (const auto& ep : episodes) episodes_json.append(toJson(ep));
            root["episodes"] = episodes_json;

            fs::create_directories(fs::path(output_json).parent_path());
            Json::StreamWriterBuilder builder;
            builder["indentation"] = "  ";
            std::ofstream fout(output_json);
            if (!fout) throw std::runtime_error("Failed to open output_json for write: " + output_json);
            fout << Json::writeString(builder, root);

            std::cout
                << "[ok] uid=" << cfg.uid
                << " step=" << step_id
                << " starts=" << episodes.size()
                << " C=" << meta.c_selected_view_count
                << " gt_mean_total=" << gt_mean_total
                << " raw_mean_total=" << raw_mean_total
                << " balanced_mean_total=" << balanced_mean_total
                << " raw_mean_vc=" << raw_mean_vc
                << " balanced_mean_vc=" << balanced_mean_vc
                << " output=" << output_json
                << "\n";
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}

