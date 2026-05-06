#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <chrono>
#include <cmath>
#include <exception>
#include <cstdint>

#include <json/json.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "objview_pointcloud_io.h"
#include "objview_shape_completion_client.h"

namespace fs = std::filesystem;

namespace {

void expect(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

Json::Value makeEpisodeConfig() {
    Json::Value episode(Json::objectValue);
    Json::Value capability(Json::objectValue);
    capability["service_root"] = "shape_completion";
    capability["requests_dir"] = "shape_completion/requests";
    capability["responses_dir"] = "shape_completion/responses";
    capability["outputs_dir"] = "shape_completion/outputs";
    capability["ready_path"] = "shape_completion/service_ready";

    Json::Value pointr(Json::objectValue);
    pointr["num_input_points_model"] = 2048;
    pointr["num_output_points"] = 8192;
    capability["backends"]["PoinTr-C"] = pointr;

    episode["optional_capabilities"]["shape_completion"] = capability;
    return episode;
}

Json::Value readJson(const fs::path& path) {
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

bool waitForFile(const fs::path& path, double timeout_sec = 5.0) {
    const auto start = std::chrono::steady_clock::now();
    while (!fs::exists(path)) {
        const auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration<double>(now - start).count() > timeout_sec) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return true;
}

void testMissingCapabilityReturnsNullopt() {
    const Json::Value episode(Json::objectValue);
    const auto capability =
        objview::parseShapeCompletionCapability(episode, fs::path("/tmp/session"));
    expect(!capability.has_value(), "Missing capability block should return nullopt.");
}

void testShapeCompletionCapabilityParsing() {
    const fs::path session_dir("/tmp/fake_session");
    const auto capability =
        objview::parseShapeCompletionCapability(makeEpisodeConfig(), session_dir);
    expect(capability.has_value(), "Shape completion capability should parse.");
    expect(capability->available, "Parsed capability should be available.");
    expect(capability->service_root == session_dir / "shape_completion",
           "service_root should resolve relative to session_dir.");
    expect(capability->requests_dir == session_dir / "shape_completion/requests",
           "requests_dir should resolve relative to session_dir.");
    expect(capability->responses_dir == session_dir / "shape_completion/responses",
           "responses_dir should resolve relative to session_dir.");
    expect(capability->outputs_dir == session_dir / "shape_completion/outputs",
           "outputs_dir should resolve relative to session_dir.");
    expect(capability->ready_path == session_dir / "shape_completion/service_ready",
           "ready_path should resolve relative to session_dir.");

    const objview::ShapeCompletionBackendInfo* pointr =
        capability->findBackend("PoinTr-C");
    expect(pointr != nullptr, "PoinTr-C backend metadata should be present.");
    expect(pointr->num_input_points_model == 2048,
           "PoinTr-C num_input_points_model mismatch.");
    expect(pointr->num_output_points == 8192,
           "PoinTr-C num_output_points mismatch.");
}

void testAbsolutePathsStayAbsolute() {
    Json::Value episode = makeEpisodeConfig();
    episode["optional_capabilities"]["shape_completion"]["requests_dir"] =
        "/abs/requests";
    const auto capability =
        objview::parseShapeCompletionCapability(episode, fs::path("/tmp/fake_session"));
    expect(capability.has_value(), "Capability should parse with absolute paths.");
    expect(capability->requests_dir == fs::path("/abs/requests"),
           "Absolute requests_dir should not be rebased.");
}

void testCompleteShapeRpcRoundTrip() {
    const fs::path temp_root = fs::temp_directory_path() / "objview_shape_completion_client_test";
    std::error_code ec;
    fs::remove_all(temp_root, ec);
    fs::create_directories(temp_root);

    Json::Value episode = makeEpisodeConfig();
    const auto capability_opt =
        objview::parseShapeCompletionCapability(episode, temp_root);
    expect(capability_opt.has_value(), "Capability should parse for RPC round-trip test.");
    const objview::ShapeCompletionCapability capability = *capability_opt;
    fs::create_directories(capability.requests_dir);
    fs::create_directories(capability.responses_dir);
    fs::create_directories(capability.outputs_dir);
    std::ofstream(capability.ready_path).close();

    const fs::path partial_path = capability.outputs_dir / "partial_step_000.pcd";
    const fs::path completed_path = capability.outputs_dir / "completed_step_000.pcd";
    std::ofstream(partial_path).close();

    std::exception_ptr service_error;
    std::thread fake_service([&]() {
        try {
            fs::path request_ready_path;
            const auto start = std::chrono::steady_clock::now();
            while (request_ready_path.empty()) {
                for (const auto& entry : fs::directory_iterator(capability.requests_dir)) {
                    const std::string name = entry.path().filename().string();
                    if (name.size() > 11 && name.substr(name.size() - 11) == ".json.ready") {
                        request_ready_path = entry.path();
                        break;
                    }
                }
                if (!request_ready_path.empty()) break;
                if (std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() > 5.0) {
                    throw std::runtime_error("Timed out waiting for shape completion request.");
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }

            const fs::path request_path =
                fs::path(request_ready_path.string().substr(
                    0,
                    request_ready_path.string().size() - std::string(".ready").size()));
            const Json::Value request = readJson(request_path);
            expect(request["method"].asString() == "complete_shape",
                   "RPC method should be complete_shape.");
            expect(request["params"]["partial_pointcloud_path"].asString() == partial_path.string(),
                   "partial_pointcloud_path mismatch.");
            expect(request["params"]["output_pointcloud_path"].asString() == completed_path.string(),
                   "output_pointcloud_path mismatch.");

            Json::Value response(Json::objectValue);
            response["jsonrpc"] = "2.0";
            response["id"] = request["id"].asString();
            response["result"]["completed_pointcloud_path"] = completed_path.string();
            response["result"]["num_input_points_raw"] = 12345;
            response["result"]["num_input_points_model"] = 2048;
            response["result"]["num_output_points"] = 8192;
            response["result"]["runtime"]["preprocess_sec"] = 0.01;
            response["result"]["runtime"]["inference_sec"] = 0.02;
            response["result"]["runtime"]["postprocess_sec"] = 0.03;
            response["result"]["runtime"]["total_sec"] = 0.06;

            const fs::path response_path = capability.responses_dir / request_path.filename();
            writeJson(response_path, response);
            std::ofstream(response_path.string() + ".ready").close();
        }
        catch (...) {
            service_error = std::current_exception();
        }
    });

    objview::ShapeCompletionClient client(capability, 5.0, 0.01);
    const objview::ShapeCompletionResult result =
        client.completeShape(partial_path, completed_path, "unit_test_request");

    fake_service.join();
    if (service_error) std::rethrow_exception(service_error);

    expect(result.completed_pointcloud_path == completed_path,
           "completed_pointcloud_path mismatch.");
    expect(result.num_input_points_raw == 12345, "num_input_points_raw mismatch.");
    expect(result.num_input_points_model == 2048, "num_input_points_model mismatch.");
    expect(result.num_output_points == 8192, "num_output_points mismatch.");
    expect(std::abs(result.runtime.total_sec - 0.06) < 1e-12,
           "runtime.total_sec mismatch.");
    expect(waitForFile(capability.requests_dir / "unit_test_request.json"),
           "Request json should exist.");
}

void testSavePointCloudXYZRGBBinaryRoundTrip() {
    const fs::path temp_root = fs::temp_directory_path() / "objview_pointcloud_io_test";
    std::error_code ec;
    fs::remove_all(temp_root, ec);
    fs::create_directories(temp_root);

    pcl::PointCloud<pcl::PointXYZRGB> cloud;
    cloud.push_back(pcl::PointXYZRGB(
        static_cast<std::uint8_t>(255), static_cast<std::uint8_t>(0), static_cast<std::uint8_t>(0)));
    cloud.back().x = 1.0f;
    cloud.back().y = 2.0f;
    cloud.back().z = 3.0f;

    cloud.push_back(pcl::PointXYZRGB(
        static_cast<std::uint8_t>(0), static_cast<std::uint8_t>(255), static_cast<std::uint8_t>(0)));
    cloud.back().x = -1.5f;
    cloud.back().y = 0.25f;
    cloud.back().z = 4.5f;

    cloud.width = static_cast<uint32_t>(cloud.size());
    cloud.height = 1;
    cloud.is_dense = false;

    const fs::path pcd_path = temp_root / "partial_rgb.pcd";
    objview::savePointCloudXYZRGBBinary(pcd_path, cloud);
    expect(fs::exists(pcd_path), "Saved PointXYZRGB PCD should exist.");

    pcl::PointCloud<pcl::PointXYZRGB> loaded;
    if (pcl::io::loadPCDFile<pcl::PointXYZRGB>(pcd_path.string(), loaded) != 0) {
        throw std::runtime_error("Failed to load saved PointXYZRGB PCD.");
    }

    expect(loaded.size() == cloud.size(), "Loaded PointXYZRGB cloud size mismatch.");
    for (size_t i = 0; i < cloud.size(); ++i) {
        expect(std::abs(loaded[i].x - cloud[i].x) < 1e-6f, "Loaded x mismatch.");
        expect(std::abs(loaded[i].y - cloud[i].y) < 1e-6f, "Loaded y mismatch.");
        expect(std::abs(loaded[i].z - cloud[i].z) < 1e-6f, "Loaded z mismatch.");
        expect(loaded[i].r == cloud[i].r, "Loaded r mismatch.");
        expect(loaded[i].g == cloud[i].g, "Loaded g mismatch.");
        expect(loaded[i].b == cloud[i].b, "Loaded b mismatch.");
    }
}

}  // namespace

int main() {
    try {
        testMissingCapabilityReturnsNullopt();
        testShapeCompletionCapabilityParsing();
        testAbsolutePathsStayAbsolute();
        testCompleteShapeRpcRoundTrip();
        testSavePointCloudXYZRGBBinaryRoundTrip();
        std::cout << "test_shape_completion_client passed" << std::endl;
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "test_shape_completion_client failed: " << e.what() << std::endl;
        return 1;
    }
}
