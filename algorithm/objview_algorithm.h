#ifndef OBJVIEWBENCH_OBJVIEW_ALGORITHM_H_
#define OBJVIEWBENCH_OBJVIEW_ALGORITHM_H_

#include <array>
#include <cmath>
#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <json/json.h>

#include "global_path_planner.h"

namespace objview {

struct Pose7d {
    std::array<double, 7> v{};
};

struct ViewEntry {
    int view_idx = -1;
    Pose7d pose;
};

struct ShapeCompletionBackendInfo {
    int num_input_points_model = 0;
    int num_output_points = 0;
};

struct ShapeCompletionCapability {
    bool available = false;
    std::filesystem::path service_root;
    std::filesystem::path requests_dir;
    std::filesystem::path responses_dir;
    std::filesystem::path outputs_dir;
    std::filesystem::path ready_path;
    std::unordered_map<std::string, ShapeCompletionBackendInfo> backends;

    const ShapeCompletionBackendInfo* findBackend(const std::string& name) const {
        const auto it = backends.find(name);
        return it == backends.end() ? nullptr : &it->second;
    }
};

inline Vec3 cameraPosition(const Pose7d& pose) {
    return Vec3(pose.v[0], pose.v[1], pose.v[2]);
}

inline bool samePose(const Pose7d& a, const Pose7d& b, double atol = 1e-6) {
    for (int i = 0; i < 7; ++i) {
        if (std::abs(a.v[i] - b.v[i]) > atol) return false;
    }
    return true;
}

struct AlgorithmContext {
    std::string uid;
    std::string episode_id;
    int step_index = -1;
    int visited_view_num = 0;
    Pose7d current_pose;
    std::filesystem::path session_dir;
    std::string observation_manifest_path;
    std::filesystem::path observation_manifest_abs_path;
    Json::Value observation_manifest;
    Json::Value episode_config;
    Json::Value current_step;
    std::optional<ShapeCompletionCapability> shape_completion;
    std::filesystem::path interaction_requests_dir;
    std::filesystem::path interaction_responses_dir;
    double interaction_wait_timeout_sec = 120.0;
    double interaction_poll_interval_sec = 0.01;
    std::vector<ViewEntry> candidate_views;
    std::vector<int> submitted_view_ids;
};

struct AlgorithmDecision {
    enum class Type {
        Move,
        Stop,
    };

    Type type = Type::Stop;
    int view_id = -1;
    Pose7d pose;
    std::string stop_reason = "plan_end";
    std::optional<double> algorithm_runtime_sec;

    static AlgorithmDecision move(int next_view_id) {
        AlgorithmDecision d;
        d.type = Type::Move;
        d.view_id = next_view_id;
        return d;
    }

    static AlgorithmDecision movePose(const Pose7d& next_pose) {
        AlgorithmDecision d;
        d.type = Type::Move;
        d.view_id = -1;
        d.pose = next_pose;
        return d;
    }

    static AlgorithmDecision stop(const std::string& reason) {
        AlgorithmDecision d;
        d.type = Type::Stop;
        d.stop_reason = reason;
        return d;
    }

    AlgorithmDecision& withRuntime(double runtime_sec) {
        algorithm_runtime_sec = runtime_sec;
        return *this;
    }
};

class Algorithm {
public:
    virtual ~Algorithm() = default;
    virtual void prepareEpisode(
        const Json::Value& episode_config,
        const std::filesystem::path& session_dir) {
        (void)episode_config;
        (void)session_dir;
    }
    virtual std::vector<ViewEntry> candidateViewSpace() const = 0;
    virtual AlgorithmDecision decideNext(const AlgorithmContext& ctx) = 0;
};

}  // namespace objview

#endif  // OBJVIEWBENCH_OBJVIEW_ALGORITHM_H_
