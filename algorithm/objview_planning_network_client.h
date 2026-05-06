#ifndef OBJVIEWBENCH_OBJVIEW_PLANNING_NETWORK_CLIENT_H_
#define OBJVIEWBENCH_OBJVIEW_PLANNING_NETWORK_CLIENT_H_

#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

#include <json/json.h>

namespace objview {

struct PlanningNetworkInferResult {
    Json::Value response;
    Json::Value result;
    std::string request_id;
    std::filesystem::path request_path;
    std::filesystem::path response_path;
    double rpc_elapsed_sec = 0.0;
};

class PlanningNetworkRpcClient {
public:
    PlanningNetworkRpcClient(
        std::filesystem::path service_root,
        double wait_timeout_sec,
        double poll_interval_sec)
        : service_root_(std::move(service_root)),
          wait_timeout_sec_(wait_timeout_sec),
          poll_interval_sec_(poll_interval_sec) {
        if (service_root_.empty()) {
            throw std::runtime_error("PlanningNetworkRpcClient service_root is required.");
        }
        requests_dir_ = service_root_ / "requests";
        responses_dir_ = service_root_ / "responses";
    }

    bool ready() const {
        return std::filesystem::exists(service_root_ / "service_ready");
    }

    PlanningNetworkInferResult infer(
        const std::string& request_id,
        const std::filesystem::path& input_npz,
        int topk) const {
        if (!ready()) {
            throw std::runtime_error(
                "Planning network service is not ready: " +
                (service_root_ / "service_ready").string());
        }

        std::filesystem::create_directories(requests_dir_);
        std::filesystem::create_directories(responses_dir_);

        const std::filesystem::path request_path = requests_dir_ / (request_id + ".json");
        const std::filesystem::path response_path = responses_dir_ / (request_id + ".json");
        const std::filesystem::path request_ready_path = readyPathForJson(request_path);
        const std::filesystem::path response_ready_path = readyPathForJson(response_path);
        removeIfExists(request_ready_path);
        removeIfExists(response_ready_path);

        Json::Value params(Json::objectValue);
        params["input_npz"] = input_npz.string();
        params["topk"] = topk;

        Json::Value request(Json::objectValue);
        request["jsonrpc"] = "2.0";
        request["id"] = request_id;
        request["method"] = "infer";
        request["params"] = params;

        const auto start = std::chrono::steady_clock::now();
        writeJsonAtomic(request_path, request);
        touchFile(request_ready_path);

        if (!waitForFile(response_ready_path)) {
            throw std::runtime_error(
                "Timed out waiting for planning network response: " +
                response_ready_path.string());
        }

        PlanningNetworkInferResult out;
        out.response = readJson(response_path);
        out.request_id = request_id;
        out.request_path = request_path;
        out.response_path = response_path;
        out.rpc_elapsed_sec =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        removeIfExists(response_ready_path);

        if (out.response.isMember("error")) {
            const std::string message = out.response["error"].get("message", "").asString();
            throw std::runtime_error("Planning network RPC error: " + message);
        }
        if (!out.response.isMember("result")) {
            throw std::runtime_error("Planning network RPC response missing result.");
        }
        out.result = out.response["result"];
        return out;
    }

private:
    std::filesystem::path service_root_;
    std::filesystem::path requests_dir_;
    std::filesystem::path responses_dir_;
    double wait_timeout_sec_ = 120.0;
    double poll_interval_sec_ = 0.01;

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
};

}  // namespace objview

#endif  // OBJVIEWBENCH_OBJVIEW_PLANNING_NETWORK_CLIENT_H_
