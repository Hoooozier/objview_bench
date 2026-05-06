#ifndef OBJVIEWBENCH_OBJVIEW_INTERACTION_RPC_CLIENT_H_
#define OBJVIEWBENCH_OBJVIEW_INTERACTION_RPC_CLIENT_H_

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <json/json.h>

#include "objview_algorithm.h"

namespace objview {

struct InteractionFeasibilityResult {
    std::vector<bool> feasible;
    Json::Value response;
    std::string request_id;
    std::filesystem::path request_path;
    std::filesystem::path response_path;
    double rpc_elapsed_sec = 0.0;
};

class InteractionRpcClient {
public:
    InteractionRpcClient(
        std::filesystem::path requests_dir,
        std::filesystem::path responses_dir,
        double wait_timeout_sec,
        double poll_interval_sec)
        : requests_dir_(std::move(requests_dir)),
          responses_dir_(std::move(responses_dir)),
          wait_timeout_sec_(wait_timeout_sec),
          poll_interval_sec_(poll_interval_sec) {
        if (requests_dir_.empty() || responses_dir_.empty()) {
            throw std::runtime_error("InteractionRpcClient requires request/response dirs.");
        }
    }

    InteractionFeasibilityResult isFeasible(
        const std::string& request_id,
        const std::vector<Pose7d>& poses) const {
        Json::Value poses_json(Json::arrayValue);
        for (const auto& pose : poses) {
            Json::Value p(Json::arrayValue);
            for (double x : pose.v) p.append(x);
            poses_json.append(p);
        }

        Json::Value params(Json::objectValue);
        params["poses"] = poses_json;
        Json::Value response = call(request_id, "is_feasible", params);

        if (response.isMember("error")) {
            throw std::runtime_error(
                "Interaction feasibility RPC error: " +
                response["error"].get("message", "").asString());
        }
        const Json::Value results = response["result"]["results"];
        if (!results.isArray() || results.size() != poses.size()) {
            throw std::runtime_error("Interaction feasibility RPC returned invalid result size.");
        }

        InteractionFeasibilityResult out;
        out.response = response;
        out.request_id = request_id;
        out.request_path = requests_dir_ / (request_id + ".json");
        out.response_path = responses_dir_ / (request_id + ".json");
        out.feasible.reserve(results.size());
        for (const Json::Value& item : results) {
            out.feasible.push_back(item.get("feasible", false).asBool());
        }
        out.rpc_elapsed_sec = last_rpc_elapsed_sec_;
        return out;
    }

    static std::string makeRequestId(
        const std::string& prefix,
        const std::string& episode_id,
        int step_index) {
        static std::atomic<unsigned long long> counter{0};
        const auto now = std::chrono::steady_clock::now().time_since_epoch();
        const auto us = std::chrono::duration_cast<std::chrono::microseconds>(now).count();
        return sanitize(prefix) + "_" + sanitize(episode_id) +
               "_step" + std::to_string(step_index) +
               "_" + std::to_string(counter.fetch_add(1)) +
               "_" + std::to_string(static_cast<long long>(us));
    }

private:
    std::filesystem::path requests_dir_;
    std::filesystem::path responses_dir_;
    double wait_timeout_sec_ = 120.0;
    double poll_interval_sec_ = 0.01;
    mutable double last_rpc_elapsed_sec_ = 0.0;

    Json::Value call(
        const std::string& request_id,
        const std::string& method,
        const Json::Value& params) const {
        std::filesystem::create_directories(requests_dir_);
        std::filesystem::create_directories(responses_dir_);

        const std::filesystem::path request_path = requests_dir_ / (request_id + ".json");
        const std::filesystem::path response_path = responses_dir_ / (request_id + ".json");
        const std::filesystem::path request_ready_path = readyPathForJson(request_path);
        const std::filesystem::path response_ready_path = readyPathForJson(response_path);
        removeIfExists(request_ready_path);
        removeIfExists(response_ready_path);

        Json::Value request(Json::objectValue);
        request["jsonrpc"] = "2.0";
        request["id"] = request_id;
        request["method"] = method;
        request["params"] = params;

        const auto start = std::chrono::steady_clock::now();
        writeJsonAtomic(request_path, request);
        touchFile(request_ready_path);
        if (!waitForFile(response_ready_path)) {
            throw std::runtime_error(
                "Timed out waiting for interaction RPC response: " +
                response_ready_path.string());
        }
        Json::Value response = readJson(response_path);
        last_rpc_elapsed_sec_ =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        removeIfExists(response_ready_path);
        return response;
    }

    static std::filesystem::path readyPathForJson(const std::filesystem::path& json_path) {
        return std::filesystem::path(json_path.string() + ".ready");
    }

    static void removeIfExists(const std::filesystem::path& path) {
        std::error_code ec;
        std::filesystem::remove(path, ec);
    }

    static void touchFile(const std::filesystem::path& path) {
        std::filesystem::create_directories(path.parent_path());
        std::ofstream(path).close();
    }

    static Json::Value readJson(const std::filesystem::path& path) {
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

    static void writeJsonAtomic(const std::filesystem::path& path, const Json::Value& root) {
        std::filesystem::create_directories(path.parent_path());
        const std::filesystem::path tmp_path = path.string() + ".tmp";
        Json::StreamWriterBuilder builder;
        builder["indentation"] = "  ";
        {
            std::ofstream fout(tmp_path, std::ios::binary);
            if (!fout) {
                throw std::runtime_error("Failed to open output json: " + tmp_path.string());
            }
            std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());
            writer->write(root, &fout);
            fout << "\n";
        }
        std::filesystem::rename(tmp_path, path);
    }

    bool waitForFile(const std::filesystem::path& path) const {
        const auto start = std::chrono::steady_clock::now();
        while (!std::filesystem::exists(path)) {
            const double elapsed =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            if (elapsed > wait_timeout_sec_) return false;
            std::this_thread::sleep_for(std::chrono::duration<double>(poll_interval_sec_));
        }
        return true;
    }

    static std::string sanitize(const std::string& value) {
        std::string out;
        out.reserve(value.size());
        for (unsigned char c : value) {
            if ((c >= 'a' && c <= 'z') ||
                (c >= 'A' && c <= 'Z') ||
                (c >= '0' && c <= '9') ||
                c == '_' || c == '-') {
                out.push_back(static_cast<char>(c));
            } else {
                out.push_back('_');
            }
        }
        return out.empty() ? "unknown" : out;
    }
};

}  // namespace objview

#endif  // OBJVIEWBENCH_OBJVIEW_INTERACTION_RPC_CLIENT_H_
