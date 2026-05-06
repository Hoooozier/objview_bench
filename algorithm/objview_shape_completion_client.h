#ifndef OBJVIEWBENCH_OBJVIEW_SHAPE_COMPLETION_CLIENT_H_
#define OBJVIEWBENCH_OBJVIEW_SHAPE_COMPLETION_CLIENT_H_

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

#include <json/json.h>

#include "objview_algorithm.h"

namespace objview {

namespace fs = std::filesystem;

inline fs::path resolveShapeCompletionPath(const fs::path& session_dir,
                                           const Json::Value& capability_json,
                                           const char* key) {
    const std::string rel = capability_json.get(key, "").asString();
    if (rel.empty()) return fs::path();
    const fs::path path(rel);
    return path.is_absolute() ? path : (session_dir / path);
}

inline std::optional<ShapeCompletionCapability> parseShapeCompletionCapability(
    const Json::Value& episode_config,
    const fs::path& session_dir) {
    const Json::Value capability_json =
        episode_config["optional_capabilities"]["shape_completion"];
    if (!capability_json.isObject()) return std::nullopt;
    const Json::Value workspace_json =
        capability_json.isMember("session_workspace") &&
                capability_json["session_workspace"].isObject()
            ? capability_json["session_workspace"]
        : capability_json.isMember("workspace") && capability_json["workspace"].isObject()
            ? capability_json["workspace"]
            : capability_json;

    ShapeCompletionCapability capability;
    capability.available = true;
    capability.service_root = resolveShapeCompletionPath(session_dir, workspace_json, "service_root");
    capability.requests_dir = resolveShapeCompletionPath(session_dir, workspace_json, "requests_dir");
    capability.responses_dir = resolveShapeCompletionPath(session_dir, workspace_json, "responses_dir");
    capability.outputs_dir = resolveShapeCompletionPath(session_dir, workspace_json, "outputs_dir");
    capability.ready_path = resolveShapeCompletionPath(session_dir, workspace_json, "ready_path");

    const Json::Value backends = capability_json["backends"];
    if (backends.isObject()) {
        for (const auto& key : backends.getMemberNames()) {
            const Json::Value backend_json = backends[key];
            ShapeCompletionBackendInfo info;
            info.num_input_points_model =
                backend_json.get("num_input_points_model", 0).asInt();
            info.num_output_points =
                backend_json.get("num_output_points", 0).asInt();
            capability.backends.emplace(key, info);
        }
    }

    return capability;
}

struct ShapeCompletionRuntime {
    double preprocess_sec = 0.0;
    double inference_sec = 0.0;
    double postprocess_sec = 0.0;
    double total_sec = 0.0;
};

struct ShapeCompletionResult {
    fs::path completed_pointcloud_path;
    int num_input_points_raw = 0;
    int num_input_points_model = 0;
    int num_output_points = 0;
    ShapeCompletionRuntime runtime;
};

class ShapeCompletionClient {
public:
    ShapeCompletionClient(
        ShapeCompletionCapability capability,
        double wait_timeout_sec,
        double poll_interval_sec,
        bool verbose = false)
        : capability_(std::move(capability)),
          wait_timeout_sec_(wait_timeout_sec),
          poll_interval_sec_(poll_interval_sec),
          verbose_(verbose) {
        if (!capability_.available) {
            throw std::runtime_error("ShapeCompletionClient requires an available capability.");
        }
        if (capability_.requests_dir.empty() || capability_.responses_dir.empty()) {
            throw std::runtime_error("Shape completion capability is missing requests/responses paths.");
        }
        if (!capability_.ready_path.empty() && !fs::exists(capability_.ready_path)) {
            throw std::runtime_error(
                "Shape completion service is not ready: " + capability_.ready_path.string());
        }
    }

    const ShapeCompletionCapability& capability() const { return capability_; }

    ShapeCompletionResult completeShape(const fs::path& partial_pointcloud_path,
                                        const fs::path& output_pointcloud_path,
                                        const std::string& request_id) const {
        logPath("completeShape.partial_pointcloud_path", partial_pointcloud_path);
        logPath("completeShape.output_pointcloud_path", output_pointcloud_path);
        logPath("completeShape.requests_dir", capability_.requests_dir);
        logPath("completeShape.responses_dir", capability_.responses_dir);
        logPath("completeShape.outputs_dir", capability_.outputs_dir);

        if (capability_.requests_dir.empty()) {
            throw std::runtime_error("Shape completion requests_dir is empty.");
        }
        if (capability_.responses_dir.empty()) {
            throw std::runtime_error("Shape completion responses_dir is empty.");
        }
        if (verbose_) {
            std::cerr << "[shape_completion_client] create_directories requests_dir="
                      << capability_.requests_dir.string() << std::endl;
        }
        fs::create_directories(capability_.requests_dir);
        if (verbose_) {
            std::cerr << "[shape_completion_client] create_directories responses_dir="
                      << capability_.responses_dir.string() << std::endl;
        }
        fs::create_directories(capability_.responses_dir);
        if (!output_pointcloud_path.empty()) {
            if (output_pointcloud_path.parent_path().empty()) {
                throw std::runtime_error(
                    "Shape completion output_pointcloud_path has an empty parent path: " +
                    output_pointcloud_path.string());
            }
            if (verbose_) {
                std::cerr << "[shape_completion_client] create_directories output parent="
                          << output_pointcloud_path.parent_path().string() << std::endl;
            }
            fs::create_directories(output_pointcloud_path.parent_path());
        }

        const fs::path request_path = capability_.requests_dir / (request_id + ".json");
        const fs::path response_path = capability_.responses_dir / (request_id + ".json");
        const fs::path request_ready_path = readyPathForJson(request_path);
        const fs::path response_ready_path = readyPathForJson(response_path);
        removeIfExists(request_ready_path);
        removeIfExists(response_ready_path);

        Json::Value params(Json::objectValue);
        params["partial_pointcloud_path"] = partial_pointcloud_path.string();
        params["output_pointcloud_path"] = output_pointcloud_path.string();
        writeJsonAtomic(request_path, makeJsonRpcRequest(request_id, "complete_shape", params));
        touchFile(request_ready_path);

        if (!waitForSpecificFile(response_ready_path)) {
            throw std::runtime_error(
                "Timed out waiting for shape completion response ready file: " +
                response_ready_path.string());
        }

        const Json::Value response = readJson(response_path);
        removeIfExists(response_ready_path);
        if (response.isMember("error")) {
            throw std::runtime_error(
                "Shape completion RPC error: " +
                response["error"].get("message", "").asString());
        }
        return parseResult(response["result"]);
    }

private:
    ShapeCompletionCapability capability_;
    double wait_timeout_sec_ = 120.0;
    double poll_interval_sec_ = 0.01;
    bool verbose_ = false;

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

    static Json::Value makeJsonRpcRequest(const std::string& request_id,
                                          const std::string& method,
                                          const Json::Value& params) {
        Json::Value request(Json::objectValue);
        request["jsonrpc"] = "2.0";
        request["id"] = request_id;
        request["method"] = method;
        request["params"] = params;
        return request;
    }

    static fs::path readyPathForJson(const fs::path& json_path) {
        return fs::path(json_path.string() + ".ready");
    }

    static void touchFile(const fs::path& path) {
        fs::create_directories(path.parent_path());
        std::ofstream(path).close();
    }

    static void removeIfExists(const fs::path& path) {
        std::error_code ec;
        fs::remove(path, ec);
    }

    bool waitForSpecificFile(const fs::path& path) const {
        const auto start = std::chrono::steady_clock::now();
        while (!fs::exists(path)) {
            const auto now = std::chrono::steady_clock::now();
            const double elapsed = std::chrono::duration<double>(now - start).count();
            if (elapsed > wait_timeout_sec_) return false;
            std::this_thread::sleep_for(std::chrono::duration<double>(poll_interval_sec_));
        }
        return true;
    }

    static ShapeCompletionResult parseResult(const Json::Value& result_json) {
        if (!result_json.isObject()) {
            throw std::runtime_error("Shape completion RPC result must be an object.");
        }

        ShapeCompletionResult result;
        result.completed_pointcloud_path =
            result_json.get("completed_pointcloud_path", "").asString();
        result.num_input_points_raw =
            result_json.get("num_input_points_raw", 0).asInt();
        result.num_input_points_model =
            result_json.get("num_input_points_model", 0).asInt();
        result.num_output_points =
            result_json.get("num_output_points", 0).asInt();

        const Json::Value runtime_json = result_json["runtime"];
        if (runtime_json.isObject()) {
            result.runtime.preprocess_sec =
                runtime_json.get("preprocess_sec", 0.0).asDouble();
            result.runtime.inference_sec =
                runtime_json.get("inference_sec", 0.0).asDouble();
            result.runtime.postprocess_sec =
                runtime_json.get("postprocess_sec", 0.0).asDouble();
            result.runtime.total_sec =
                runtime_json.get("total_sec", 0.0).asDouble();
        }
        return result;
    }

    void logPath(const std::string& label, const fs::path& path) const {
        if (!verbose_) return;
        std::cerr << "[shape_completion_client] " << label << "=" << path.string() << std::endl;
    }
};

}  // namespace objview

#endif  // OBJVIEWBENCH_OBJVIEW_SHAPE_COMPLETION_CLIENT_H_
