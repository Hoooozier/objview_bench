#ifndef OBJVIEWBENCH_OBJVIEW_BENCHMARK_SUBMITTER_H_
#define OBJVIEWBENCH_OBJVIEW_BENCHMARK_SUBMITTER_H_

#include <algorithm>
#include <atomic>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <json/json.h>

#include "objview_algorithm.h"
#include "objview_shape_completion_client.h"

namespace objview {

namespace fs = std::filesystem;

struct SubmitterConfig {
    fs::path session_dir;
    fs::path cache_index_json = "render_cache/cache_index.json";
    bool query_feasibility = true;
    double algorithm_runtime_sec = 0.01;
    double wait_timeout_sec = 120.0;
    double poll_interval_sec = 0.01;
};

struct InteractionPaths {
    fs::path action;
    fs::path ready_algorithm;
    fs::path algorithm_started;
    fs::path current_step;
    fs::path ready_benchmark;
    fs::path episode_done;
    fs::path requests_dir;
    fs::path responses_dir;
};

class BenchmarkSubmitter {
public:
    explicit BenchmarkSubmitter(SubmitterConfig cfg) : cfg_(std::move(cfg)) {
        if (cfg_.session_dir.empty()) {
            throw std::runtime_error("SubmitterConfig.session_dir is required.");
        }
    }

    int run(Algorithm& algorithm) {
        const fs::path episode_config_path = cfg_.session_dir / "config" / "episode_config.json";
        if (!waitForFile(episode_config_path)) {
            throw std::runtime_error("Timed out waiting for episode_config.json");
        }

        const Json::Value episode_config = readJson(episode_config_path);
        const std::string episode_id = episode_config["episode_id"].asString();
        const InteractionPaths paths = interactionPaths(episode_config);
        const std::optional<ShapeCompletionCapability> shape_completion =
            parseShapeCompletionCapability(episode_config, cfg_.session_dir);
        algorithm.prepareEpisode(episode_config, cfg_.session_dir);
        const std::vector<ViewEntry> all_views = algorithm.candidateViewSpace();
        publishAlgorithmStartupReady(paths);

        int last_seen_step = -1;
        bool candidates_ready = false;
        std::vector<ViewEntry> feasible_views;
        std::vector<int> submitted_view_ids;
        while (true) {
            Json::Value current = waitForBenchmarkStep(paths, episode_id, last_seen_step);
            if (current.isNull()) return 0;

            if (!candidates_ready) {
                if (!all_views.empty()) {
                    feasible_views = cfg_.query_feasibility
                        ? queryFeasibleViews(paths, all_views, episode_id, current["step_index"].asInt())
                        : all_views;
                }
                candidates_ready = true;
                if (!all_views.empty() && feasible_views.empty()) {
                    submitAction(paths, makeStopAction(
                        episode_id,
                        current["step_index"].asInt(),
                        "candidate_exhausted",
                        0.0));
                    return 0;
                }
            }

            AlgorithmContext ctx;
            ctx.uid = episode_config["uid"].asString();
            ctx.episode_id = episode_id;
            ctx.step_index = current["step_index"].asInt();
            ctx.visited_view_num = current.get("visited_view_num", 0).asInt();
            ctx.current_pose = poseFromJsonArray(current["current_pose"]);
            ctx.session_dir = cfg_.session_dir;
            ctx.observation_manifest_path = current.get("observation_manifest_path", "").asString();
            ctx.observation_manifest_abs_path =
                resolveSessionPath(cfg_.session_dir, ctx.observation_manifest_path);
            if (!ctx.observation_manifest_path.empty() && fs::exists(ctx.observation_manifest_abs_path)) {
            ctx.observation_manifest = readJson(ctx.observation_manifest_abs_path);
            }
            ctx.episode_config = episode_config;
            ctx.current_step = current;
            ctx.shape_completion = shape_completion;
            ctx.interaction_requests_dir = paths.requests_dir;
            ctx.interaction_responses_dir = paths.responses_dir;
            ctx.interaction_wait_timeout_sec = cfg_.wait_timeout_sec;
            ctx.interaction_poll_interval_sec = cfg_.poll_interval_sec;
            ctx.candidate_views = filterRemainingViews(feasible_views, ctx.current_pose, submitted_view_ids);
            ctx.submitted_view_ids = submitted_view_ids;
            last_seen_step = ctx.step_index;

            const auto decision_start = std::chrono::steady_clock::now();
            const AlgorithmDecision decision = algorithm.decideNext(ctx);
            const double measured_runtime_sec =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - decision_start).count();
            const double algorithm_runtime_sec =
                decision.algorithm_runtime_sec.value_or(measured_runtime_sec);
            if (decision.type == AlgorithmDecision::Type::Stop) {
                submitAction(paths, makeStopAction(
                    episode_id,
                    ctx.step_index,
                    decision.stop_reason,
                    algorithm_runtime_sec));
                return 0;
            }

            Pose7d next_pose = decision.pose;
            if (decision.view_id >= 0) {
                const ViewEntry* next = findViewById(feasible_views, decision.view_id);
                if (next == nullptr) {
                    throw std::runtime_error("Algorithm selected an infeasible or missing view id: " +
                                             std::to_string(decision.view_id));
                }
                next_pose = next->pose;
                submitted_view_ids.push_back(decision.view_id);
            }
            submitAction(paths, makeMoveAction(
                episode_id,
                ctx.step_index,
                next_pose,
                algorithm_runtime_sec));
        }
    }

private:
    SubmitterConfig cfg_;

    static Json::Value readJson(const fs::path& path) {
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

    static fs::path resolveSessionPath(const fs::path& session_dir, const std::string& path) {
        if (path.empty()) return fs::path();
        fs::path p(path);
        if (p.is_absolute()) return p;
        return session_dir / p;
    }

    static void writeJsonAtomic(const fs::path& path, const Json::Value& root) {
        fs::create_directories(path.parent_path());
        const fs::path tmp_path = path.string() + ".tmp";

        Json::StreamWriterBuilder builder;
        builder["indentation"] = "  ";
        {
            std::ofstream fout(tmp_path, std::ios::binary);
            if (!fout) throw std::runtime_error("Failed to open output json: " + tmp_path.string());
            std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());
            writer->write(root, &fout);
            fout << "\n";
        }
        fs::rename(tmp_path, path);
    }

    static fs::path readyPathForJson(const fs::path& json_path) {
        return fs::path(json_path.string() + ".ready");
    }

    static Json::Value makeJsonRpcRequest(
        const std::string& request_id,
        const std::string& method,
        const Json::Value& params) {
        Json::Value request(Json::objectValue);
        request["jsonrpc"] = "2.0";
        request["id"] = request_id;
        request["method"] = method;
        request["params"] = params;
        return request;
    }

    static void touchFile(const fs::path& path) {
        fs::create_directories(path.parent_path());
        std::ofstream(path).close();
    }

    static void removeIfExists(const fs::path& path) {
        std::error_code ec;
        fs::remove(path, ec);
    }

    bool waitForFile(const fs::path& path) const {
        const auto start = std::chrono::steady_clock::now();
        while (!fs::exists(path)) {
            const auto now = std::chrono::steady_clock::now();
            const double elapsed = std::chrono::duration<double>(now - start).count();
            if (elapsed > cfg_.wait_timeout_sec) return false;
            std::this_thread::sleep_for(std::chrono::duration<double>(cfg_.poll_interval_sec));
        }
        return true;
    }

    static bool readDoneIfAvailable(const fs::path& path, const std::string& episode_id) {
        if (!fs::exists(path)) return false;
        try {
            Json::Value done = readJson(path);
            return done.get("episode_id", "").asString() == episode_id;
        }
        catch (...) {
            return false;
        }
    }

    InteractionPaths interactionPaths(const Json::Value& episode_config) const {
        const Json::Value paths = episode_config["interaction_paths"];
        auto rel = [&](const char* key, const char* fallback) -> fs::path {
            return cfg_.session_dir / paths.get(key, fallback).asString();
        };
        return {
            rel("action_path", "actions/action.json"),
            rel("ready_algorithm_path", "actions/ready_algorithm"),
            rel("algorithm_started_path", "actions/algorithm_started"),
            rel("current_step_path", "state/current_step.json"),
            rel("ready_benchmark_path", "state/ready_benchmark"),
            rel("episode_done_path", "state/episode_done.json"),
            rel("requests_dir", "requests"),
            rel("responses_dir", "responses"),
        };
    }

    static Pose7d poseFromJsonArray(const Json::Value& arr) {
        if (!arr.isArray() || arr.size() != 7) {
            throw std::runtime_error("Expected pose array with 7 values.");
        }
        Pose7d p;
        for (int i = 0; i < 7; ++i) p.v[i] = arr[i].asDouble();
        return p;
    }

    static const ViewEntry* findViewById(const std::vector<ViewEntry>& views, int view_idx) {
        for (const auto& v : views) {
            if (v.view_idx == view_idx) return &v;
        }
        return nullptr;
    }

    static std::vector<ViewEntry> filterRemainingViews(
        const std::vector<ViewEntry>& views,
        const Pose7d& current_pose,
        const std::vector<int>& submitted_view_ids) {
        std::set<int> submitted(submitted_view_ids.begin(), submitted_view_ids.end());
        std::vector<ViewEntry> remaining;
        remaining.reserve(views.size());
        for (const auto& view : views) {
            if (submitted.count(view.view_idx)) continue;
            if (samePose(view.pose, current_pose)) continue;
            remaining.push_back(view);
        }
        return remaining;
    }

    std::vector<ViewEntry> queryFeasibleViews(
        const InteractionPaths& paths,
        const std::vector<ViewEntry>& views,
        const std::string& episode_id,
        int step_index) const {
        if (views.empty()) return {};

        Json::Value poses(Json::arrayValue);
        for (const auto& view : views) {
            Json::Value p(Json::arrayValue);
            for (double x : view.pose.v) p.append(x);
            poses.append(p);
        }
        Json::Value params(Json::objectValue);
        params["poses"] = poses;
        const Json::Value response = callJsonRpc(paths, "is_feasible", params, episode_id, step_index);
        if (response.isMember("error")) {
            throw std::runtime_error("Feasibility RPC error: " +
                                     response["error"].get("message", "").asString());
        }

        const Json::Value results = response["result"]["results"];
        if (!results.isArray() || results.size() != views.size()) {
            throw std::runtime_error("Invalid feasibility RPC response size.");
        }

        std::vector<ViewEntry> feasible;
        feasible.reserve(views.size());
        for (Json::ArrayIndex i = 0; i < results.size(); ++i) {
            if (results[i].get("feasible", false).asBool()) {
                feasible.push_back(views[static_cast<size_t>(i)]);
            }
        }
        return feasible;
    }

    Json::Value callJsonRpc(
        const InteractionPaths& paths,
        const std::string& method,
        const Json::Value& params,
        const std::string& episode_id,
        int step_index) const {
        fs::create_directories(paths.requests_dir);
        fs::create_directories(paths.responses_dir);

        const std::string request_id = makeRequestId(method, episode_id, step_index);
        const fs::path request_path = paths.requests_dir / (request_id + ".json");
        const fs::path response_path = paths.responses_dir / (request_id + ".json");
        const fs::path request_ready_path = readyPathForJson(request_path);
        const fs::path response_ready_path = readyPathForJson(response_path);
        removeIfExists(request_ready_path);
        removeIfExists(response_ready_path);

        writeJsonAtomic(request_path, makeJsonRpcRequest(request_id, method, params));
        touchFile(request_ready_path);

        if (!waitForSpecificFile(response_ready_path)) {
            throw std::runtime_error("Timed out waiting for RPC response ready file: " +
                                     response_ready_path.string());
        }

        const Json::Value response = readJson(response_path);
        removeIfExists(response_ready_path);
        return response;
    }

    std::string makeRequestId(
        const std::string& method,
        const std::string& episode_id,
        int step_index) const {
        static std::atomic<unsigned long long> counter{0};
        const auto now = std::chrono::steady_clock::now().time_since_epoch();
        const auto us = std::chrono::duration_cast<std::chrono::microseconds>(now).count();
        return "cpp_" + sanitizeRequestIdPart(method) +
               "_" + sanitizeRequestIdPart(episode_id) +
               "_step" + std::to_string(step_index) +
               "_" + std::to_string(counter.fetch_add(1)) +
               "_" + std::to_string(static_cast<long long>(us));
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
            }
            else {
                out.push_back('_');
            }
        }
        if (out.empty()) return "unknown";
        return out;
    }

    bool waitForSpecificFile(const fs::path& path) const {
        const auto start = std::chrono::steady_clock::now();
        while (!fs::exists(path)) {
            const auto now = std::chrono::steady_clock::now();
            const double elapsed = std::chrono::duration<double>(now - start).count();
            if (elapsed > cfg_.wait_timeout_sec) return false;
            std::this_thread::sleep_for(std::chrono::duration<double>(cfg_.poll_interval_sec));
        }
        return true;
    }

    Json::Value waitForBenchmarkStep(
        const InteractionPaths& paths,
        const std::string& episode_id,
        int last_seen_step) const {
        const auto start = std::chrono::steady_clock::now();

        while (true) {
            if (readDoneIfAvailable(paths.episode_done, episode_id)) {
                return Json::Value();
            }
            if (fs::exists(paths.ready_benchmark)) {
                try {
                    Json::Value current = readJson(paths.current_step);
                    const int step = current.get("step_index", -1).asInt();
                    if (current.get("episode_id", "").asString() == episode_id && step != last_seen_step) {
                        std::error_code ec;
                        fs::remove(paths.ready_benchmark, ec);
                        return current;
                    }
                }
                catch (...) {
                }
            }

            const auto now = std::chrono::steady_clock::now();
            const double elapsed = std::chrono::duration<double>(now - start).count();
            if (elapsed > cfg_.wait_timeout_sec) {
                throw std::runtime_error("Timed out waiting for benchmark ready signal.");
            }
            std::this_thread::sleep_for(std::chrono::duration<double>(cfg_.poll_interval_sec));
        }
    }

    static Json::Value makeMoveAction(
        const std::string& episode_id,
        int step_index,
        const Pose7d& pose,
        double algorithm_runtime_sec) {
        Json::Value action(Json::objectValue);
        action["action"] = "move";
        action["episode_id"] = episode_id;
        action["step_index"] = step_index;

        Json::Value p(Json::arrayValue);
        for (double x : pose.v) p.append(x);
        action["pose"] = p;

        action["pose_format"] = "camera_lookat_roll";
        action["algorithm_runtime_sec"] = algorithm_runtime_sec;
        action["submitted_at"] = static_cast<Json::LargestInt>(std::time(nullptr));
        return action;
    }

    static Json::Value makeStopAction(
        const std::string& episode_id,
        int step_index,
        const std::string& stop_reason,
        double algorithm_runtime_sec) {
        Json::Value action(Json::objectValue);
        action["action"] = "stop";
        action["episode_id"] = episode_id;
        action["step_index"] = step_index;
        action["stop_reason"] = stop_reason;
        action["algorithm_runtime_sec"] = algorithm_runtime_sec;
        action["submitted_at"] = static_cast<Json::LargestInt>(std::time(nullptr));
        return action;
    }

    static void publishAlgorithmStartupReady(const InteractionPaths& paths) {
        fs::create_directories(paths.algorithm_started.parent_path());
        std::ofstream(paths.algorithm_started).close();
    }

    static void submitAction(const InteractionPaths& paths, const Json::Value& action) {
        writeJsonAtomic(paths.action, action);
        fs::create_directories(paths.ready_algorithm.parent_path());
        std::ofstream(paths.ready_algorithm).close();
    }
};

}  // namespace objview

#endif  // OBJVIEWBENCH_OBJVIEW_BENCHMARK_SUBMITTER_H_
