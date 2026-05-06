#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

#include <json/json.h>

#include "objview_benchmark_submitter.h"

namespace fs = std::filesystem;

namespace {

class MoveOnceThenStopAlgorithm : public objview::Algorithm {
public:
    std::vector<objview::ViewEntry> candidateViewSpace() const override {
        std::vector<objview::ViewEntry> views;
        for (int i = 0; i < 4; ++i) {
            objview::ViewEntry entry;
            entry.view_idx = i;
            entry.pose.v = {
                3.0 - static_cast<double>(i),
                static_cast<double>(i),
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            };
            views.push_back(entry);
        }
        return views;
    }

    objview::AlgorithmDecision decideNext(const objview::AlgorithmContext& ctx) override {
        if (ctx.episode_config["uid"].asString() != "fake_uid") {
            throw std::runtime_error("AlgorithmContext missing episode_config.");
        }
        if (ctx.current_step["observation_manifest_path"].asString().empty()) {
            throw std::runtime_error("AlgorithmContext missing current_step.");
        }
        if (ctx.observation_manifest_path != ctx.current_step["observation_manifest_path"].asString()) {
            throw std::runtime_error("AlgorithmContext observation_manifest_path mismatch.");
        }
        if (ctx.session_dir.empty()) {
            throw std::runtime_error("AlgorithmContext missing session_dir.");
        }
        if (ctx.observation_manifest_abs_path.empty() || !fs::exists(ctx.observation_manifest_abs_path)) {
            throw std::runtime_error("AlgorithmContext missing observation_manifest_abs_path.");
        }
        if (ctx.observation_manifest["source"].asString() != "unit_test") {
            throw std::runtime_error("AlgorithmContext missing observation_manifest json.");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        if (ctx.step_index == 0) {
            return objview::AlgorithmDecision::move(2);
        }
        if (ctx.step_index == 1) {
            objview::Pose7d pose;
            pose.v = {2.5, -1.25, 0.5, 0.0, 0.0, 0.0, 0.0};
            auto decision = objview::AlgorithmDecision::movePose(pose);
            decision.withRuntime(0.123);
            return decision;
        }
        return objview::AlgorithmDecision::stop("plan_end");
    }
};

Json::Value readJson(const fs::path& path) {
    std::ifstream fin(path, std::ios::binary);
    if (!fin) throw std::runtime_error("Failed to open json: " + path.string());

    Json::CharReaderBuilder builder;
    builder["collectComments"] = false;
    Json::Value root;
    std::string errs;
    if (!Json::parseFromStream(builder, fin, &root, &errs)) {
        throw std::runtime_error("Failed to parse json: " + errs);
    }
    return root;
}

void writeJson(const fs::path& path, const Json::Value& root) {
    fs::create_directories(path.parent_path());
    Json::StreamWriterBuilder builder;
    builder["indentation"] = "  ";
    std::ofstream fout(path, std::ios::binary);
    if (!fout) throw std::runtime_error("Failed to open output json: " + path.string());
    std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());
    writer->write(root, &fout);
    fout << "\n";
}

Json::Value poseArray(double x, double y, double z) {
    Json::Value p(Json::arrayValue);
    p.append(x);
    p.append(y);
    p.append(z);
    p.append(0.0);
    p.append(0.0);
    p.append(0.0);
    p.append(0.0);
    return p;
}

void writeManifest(const fs::path& session_dir, int step_index) {
    Json::Value manifest(Json::objectValue);
    manifest["source"] = "unit_test";
    manifest["step_index"] = step_index;
    const std::string filename = "step_" + std::string(3 - std::to_string(step_index).size(), '0') +
                                 std::to_string(step_index) + ".json";
    writeJson(session_dir / "observations" / filename, manifest);
}

bool waitForFile(const fs::path& path, double timeout_sec = 5.0) {
    const auto start = std::chrono::steady_clock::now();
    while (!fs::exists(path)) {
        const auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration<double>(now - start).count() > timeout_sec) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return true;
}

void setupSession(const fs::path& session_dir) {
    Json::Value episode(Json::objectValue);
    episode["uid"] = "fake_uid";
    episode["episode_id"] = "fake_episode";
    episode["interaction_paths"]["action_path"] = "actions/action.json";
    episode["interaction_paths"]["ready_algorithm_path"] = "actions/ready_algorithm";
    episode["interaction_paths"]["current_step_path"] = "state/current_step.json";
    episode["interaction_paths"]["ready_benchmark_path"] = "state/custom_ready_benchmark";
    episode["interaction_paths"]["episode_done_path"] = "state/episode_done.json";
    episode["interaction_paths"]["requests_dir"] = "requests";
    episode["interaction_paths"]["responses_dir"] = "responses";
    writeJson(session_dir / "config" / "episode_config.json", episode);

    Json::Value current(Json::objectValue);
    current["episode_id"] = "fake_episode";
    current["step_index"] = 0;
    current["visited_view_num"] = 1;
    current["current_pose"] = poseArray(3.0, 0.0, 0.0);
    current["observation_manifest_path"] = "observations/step_000.json";
    writeManifest(session_dir, 0);
    writeJson(session_dir / "state" / "current_step.json", current);
    std::ofstream(session_dir / "state" / "custom_ready_benchmark").close();

}

void handleFeasibilityRequest(const fs::path& session_dir) {
    const fs::path requests_dir = session_dir / "requests";
    const fs::path responses_dir = session_dir / "responses";

    fs::path request_ready_path;
    const auto start = std::chrono::steady_clock::now();
    while (request_ready_path.empty()) {
        if (fs::exists(requests_dir)) {
            for (const auto& entry : fs::directory_iterator(requests_dir)) {
                if (entry.path().filename().string().find(".json.ready") != std::string::npos) {
                    request_ready_path = entry.path();
                    break;
                }
            }
        }
        if (!request_ready_path.empty()) break;
        if (std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() > 5.0) {
            throw std::runtime_error("Timed out waiting for feasibility request.");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    const fs::path request_path = request_ready_path.string().substr(
        0,
        request_ready_path.string().size() - std::string(".ready").size());
    Json::Value request = readJson(request_path);
    if (request["method"].asString() != "is_feasible") {
        throw std::runtime_error("Expected is_feasible request.");
    }
    const Json::Value poses = request["params"]["poses"];
    if (!poses.isArray() || poses.size() != 4) {
        throw std::runtime_error("Expected 4 poses in feasibility request.");
    }

    Json::Value response(Json::objectValue);
    response["jsonrpc"] = "2.0";
    response["id"] = request["id"].asString();
    Json::Value results(Json::arrayValue);
    for (Json::ArrayIndex i = 0; i < poses.size(); ++i) {
        Json::Value r(Json::objectValue);
        r["feasible"] = (i == 1 || i == 2);
        r["reason"] = r["feasible"].asBool() ? "feasible" : "blocked_for_test";
        results.append(r);
    }
    response["result"]["results"] = results;

    const fs::path response_path = responses_dir / request_path.filename();
    writeJson(response_path, response);
    std::ofstream(response_path.string() + ".ready").close();
    fs::remove(request_ready_path);
}

void publishStep1(const fs::path& session_dir) {
    Json::Value current(Json::objectValue);
    current["episode_id"] = "fake_episode";
    current["step_index"] = 1;
    current["visited_view_num"] = 2;
    current["current_pose"] = poseArray(1.0, 2.0, 0.0);
    current["observation_manifest_path"] = "observations/step_001.json";
    writeManifest(session_dir, 1);
    writeJson(session_dir / "state" / "current_step.json", current);
    std::ofstream(session_dir / "state" / "custom_ready_benchmark").close();
}

void publishStep2(const fs::path& session_dir) {
    Json::Value current(Json::objectValue);
    current["episode_id"] = "fake_episode";
    current["step_index"] = 2;
    current["visited_view_num"] = 3;
    current["current_pose"] = poseArray(2.5, -1.25, 0.5);
    current["observation_manifest_path"] = "observations/step_002.json";
    writeManifest(session_dir, 2);
    writeJson(session_dir / "state" / "current_step.json", current);
    std::ofstream(session_dir / "state" / "custom_ready_benchmark").close();
}

void checkMoveAction(const fs::path& action_path) {
    Json::Value action = readJson(action_path);
    if (action["action"].asString() != "move") throw std::runtime_error("Expected move action.");
    if (action["episode_id"].asString() != "fake_episode") throw std::runtime_error("Wrong episode_id.");
    if (action["step_index"].asInt() != 0) throw std::runtime_error("Wrong move step_index.");
    if (action["algorithm_runtime_sec"].asDouble() < 0.015) {
        throw std::runtime_error("Move action did not report measured algorithm runtime.");
    }
    const Json::Value pose = action["pose"];
    if (!pose.isArray() || pose.size() != 7) throw std::runtime_error("Move pose must be length 7.");
    if (std::abs(pose[0].asDouble() - 1.0) > 1e-9 ||
        std::abs(pose[1].asDouble() - 2.0) > 1e-9 ||
        std::abs(pose[2].asDouble() - 0.0) > 1e-9) {
        throw std::runtime_error("Move action did not use view_id=2 pose.");
    }
}

void checkFreePoseMoveAction(const fs::path& action_path) {
    Json::Value action = readJson(action_path);
    if (action["action"].asString() != "move") throw std::runtime_error("Expected free-pose move action.");
    if (action["step_index"].asInt() != 1) throw std::runtime_error("Wrong free-pose move step_index.");
    const Json::Value pose = action["pose"];
    if (!pose.isArray() || pose.size() != 7) throw std::runtime_error("Free-pose move pose must be length 7.");
    if (std::abs(action["algorithm_runtime_sec"].asDouble() - 0.123) > 1e-9) {
        throw std::runtime_error("Free-pose move action did not use algorithm-reported runtime.");
    }
    if (std::abs(pose[0].asDouble() - 2.5) > 1e-9 ||
        std::abs(pose[1].asDouble() + 1.25) > 1e-9 ||
        std::abs(pose[2].asDouble() - 0.5) > 1e-9) {
        throw std::runtime_error("Free-pose move action did not use decision pose.");
    }
}

void checkStopAction(const fs::path& action_path) {
    Json::Value action = readJson(action_path);
    if (action["action"].asString() != "stop") throw std::runtime_error("Expected stop action.");
    if (action["step_index"].asInt() != 2) throw std::runtime_error("Wrong stop step_index.");
    if (action["stop_reason"].asString() != "plan_end") throw std::runtime_error("Wrong stop reason.");
    if (action["algorithm_runtime_sec"].asDouble() < 0.015) {
        throw std::runtime_error("Stop action did not report measured algorithm runtime.");
    }
}

}  // namespace

int main() {
    try {
        const fs::path root = fs::current_path() / "tmp_submitter_protocol";
        const fs::path session_dir = root / "session";
        const fs::path cache_index_json = root / "cache_index.json";
        fs::remove_all(root);
        fs::create_directories(root);
        setupSession(session_dir);

        objview::SubmitterConfig cfg;
        cfg.session_dir = session_dir;
        cfg.cache_index_json = cache_index_json;
        cfg.query_feasibility = true;
        cfg.wait_timeout_sec = 5.0;
        cfg.poll_interval_sec = 0.01;

        MoveOnceThenStopAlgorithm algorithm;
        objview::BenchmarkSubmitter submitter(cfg);

        int submitter_return = -999;
        std::thread worker([&]() {
            submitter_return = submitter.run(algorithm);
        });

        const fs::path action_path = session_dir / "actions" / "action.json";
        const fs::path ready_algorithm = session_dir / "actions" / "ready_algorithm";

        handleFeasibilityRequest(session_dir);

        if (!waitForFile(ready_algorithm)) throw std::runtime_error("Move ready_algorithm was not created.");
        checkMoveAction(action_path);
        fs::remove(ready_algorithm);
        fs::remove(action_path);

        publishStep1(session_dir);

        if (!waitForFile(ready_algorithm)) throw std::runtime_error("Free-pose move ready_algorithm was not created.");
        checkFreePoseMoveAction(action_path);
        fs::remove(ready_algorithm);
        fs::remove(action_path);

        publishStep2(session_dir);

        if (!waitForFile(ready_algorithm)) throw std::runtime_error("Stop ready_algorithm was not created.");
        checkStopAction(action_path);

        worker.join();
        if (submitter_return != 0) throw std::runtime_error("Submitter returned non-zero.");

        fs::remove_all(root);
        std::cout << "BenchmarkSubmitter protocol test passed.\n";
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
