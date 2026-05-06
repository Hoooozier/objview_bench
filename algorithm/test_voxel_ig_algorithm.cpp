#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include <cnpy.h>
#include <json/json.h>
#include <opencv2/imgcodecs.hpp>

#include "voxel_ig_algorithm.h"

namespace fs = std::filesystem;

namespace {

void writeText(const fs::path& path, const std::string& text) {
    fs::create_directories(path.parent_path());
    std::ofstream fout(path, std::ios::binary);
    if (!fout) throw std::runtime_error("Failed to write: " + path.string());
    fout << text;
}

void writeJson(const fs::path& path, const Json::Value& root) {
    fs::create_directories(path.parent_path());
    Json::StreamWriterBuilder builder;
    builder["indentation"] = "  ";
    std::ofstream fout(path, std::ios::binary);
    if (!fout) throw std::runtime_error("Failed to write json: " + path.string());
    std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());
    writer->write(root, &fout);
    fout << "\n";
}

objview::Pose7d pose(double x, double y, double z) {
    objview::Pose7d p;
    p.v = {x, y, z, 0.0, 0.0, 0.0, 0.0};
    return p;
}

Json::Value frameMeta(double x, double y, double z) {
    Json::Value root(Json::objectValue);
    root["intrinsics"]["image_width"] = 4;
    root["intrinsics"]["image_height"] = 4;
    root["intrinsics"]["fov_x_rad"] = 1.5707963267948966;
    root["intrinsics"]["fov_y_rad"] = 1.5707963267948966;
    root["intrinsics"]["principal_x"] = 2.0;
    root["intrinsics"]["principal_y"] = 2.0;
    root["pose"]["camera_xyz"].append(x);
    root["pose"]["camera_xyz"].append(y);
    root["pose"]["camera_xyz"].append(z);
    root["pose"]["lookat_xyz"].append(0.0);
    root["pose"]["lookat_xyz"].append(0.0);
    root["pose"]["lookat_xyz"].append(0.0);
    root["pose"]["roll_rad"] = 0.0;
    const Eigen::Vector3d camera(x, y, z);
    const Eigen::Vector3d lookat(0.0, 0.0, 0.0);
    Eigen::Vector3d z_forward = (lookat - camera).normalized();
    Eigen::Vector3d world_up(0.0, 0.0, 1.0);
    if (std::abs(z_forward.dot(world_up)) > 0.999) {
        world_up = Eigen::Vector3d(0.0, 1.0, 0.0);
    }
    Eigen::Vector3d x_right = z_forward.cross(world_up).normalized();
    Eigen::Vector3d y_down = z_forward.cross(x_right).normalized();

    Eigen::Matrix4d camera_to_world = Eigen::Matrix4d::Identity();
    camera_to_world.block<3, 1>(0, 0) = x_right;
    camera_to_world.block<3, 1>(0, 1) = y_down;
    camera_to_world.block<3, 1>(0, 2) = z_forward;
    camera_to_world(0, 3) = x;
    camera_to_world(1, 3) = y;
    camera_to_world(2, 3) = z;

    root["camera_to_world"] = Json::arrayValue;
    for (int r = 0; r < 4; ++r) {
        Json::Value row(Json::arrayValue);
        for (int c = 0; c < 4; ++c) {
            row.append(camera_to_world(r, c));
        }
        root["camera_to_world"].append(row);
    }
    return root;
}

void writeObservationStep(const fs::path& session_dir, int step_index, const objview::Pose7d& current_pose) {
    const fs::path frame_meta_path =
        session_dir / "observations" / ("step_" + std::to_string(step_index)) / "frame_meta.json";
    const fs::path depth_path =
        session_dir / "observations" / ("step_" + std::to_string(step_index)) / "depth.npz";
    const fs::path mask_path =
        session_dir / "observations" / ("step_" + std::to_string(step_index)) / "mask.png";

    writeJson(frame_meta_path, frameMeta(current_pose.v[0], current_pose.v[1], current_pose.v[2]));

    const float depth[] = {
        0.30f, 0.35f, 0.40f, 0.45f,
        0.35f, 0.50f, 0.55f, 0.45f,
        0.40f, 0.55f, 0.60f, 0.50f,
        0.45f, 0.45f, 0.50f, 0.55f,
    };
    cnpy::npz_save(depth_path.string(), "depth", depth, {4, 4}, "w");

    cv::Mat mask(4, 4, CV_8UC1, cv::Scalar(255));
    mask.at<uint8_t>(0, 0) = 0;
    mask.at<uint8_t>(3, 3) = 0;
    if (!cv::imwrite(mask_path.string(), mask)) {
        throw std::runtime_error("Failed to write mask image.");
    }
}

objview::AlgorithmContext makeContext(const fs::path& session_dir,
                                      int step_index,
                                      const objview::Pose7d& current_pose,
                                      const std::vector<objview::ViewEntry>& views,
                                      const std::vector<int>& submitted) {
    writeObservationStep(session_dir, step_index, current_pose);

    Json::Value manifest(Json::objectValue);
    manifest["source"] = "unit_test";
    manifest["frame_meta_path"] =
        (fs::path("observations") / ("step_" + std::to_string(step_index)) / "frame_meta.json").string();
    manifest["depth_path"] =
        (fs::path("observations") / ("step_" + std::to_string(step_index)) / "depth.npz").string();
    manifest["mask_path"] =
        (fs::path("observations") / ("step_" + std::to_string(step_index)) / "mask.png").string();

    objview::AlgorithmContext ctx;
    ctx.uid = "fake_uid";
    ctx.episode_id = "fake_episode";
    ctx.step_index = step_index;
    ctx.visited_view_num = step_index + 1;
    ctx.current_pose = current_pose;
    ctx.session_dir = session_dir;
    ctx.observation_manifest_path = "observations/step_" + std::to_string(step_index) + ".json";
    ctx.observation_manifest_abs_path = session_dir / ctx.observation_manifest_path;
    ctx.observation_manifest = manifest;
    ctx.candidate_views = views;
    ctx.submitted_view_ids = submitted;
    return ctx;
}

void runMethodSmoke(objview::VoxelIgMethod method,
                    const std::string& method_name,
                    const fs::path& root,
                    const fs::path& views_path) {
    const fs::path session_dir = root / method_name;
    fs::remove_all(session_dir);
    fs::create_directories(session_dir);

    objview::VoxelIgAlgorithmConfig cfg;
    cfg.views_path = views_path.string();
    cfg.view_radius = 3.0;
    cfg.octomap_resolution = 2.0 / 64.0;
    cfg.map_bbox_min = -1.0;
    cfg.map_bbox_max = 1.0;
    cfg.method = method;
    cfg.ray_stride = 2;
    cfg.debug_save_ot = true;
    cfg.silent = false;

    objview::VoxelIgAlgorithm algo(cfg);
    const auto views = algo.candidateViewSpace();
    if (views.size() != 4) throw std::runtime_error("Expected 4 candidate views.");
    std::cout << "[test] " << method_name
              << " initial_octree_nodes=" << algo.mapLeafCountForTest()
              << " leafs=" << algo.leafCountForTest()
              << " bbox_leafs=" << algo.bboxLeafCountForTest()
              << " bbox_nodes=" << algo.bboxNodeCountForTest() << std::endl;

    auto d0 = algo.decideNext(makeContext(session_dir, 0, pose(3.0, 0.0, 0.0), views, {}));
    if (d0.type != objview::AlgorithmDecision::Type::Move) {
        throw std::runtime_error(method_name + ": expected a move decision at step 0.");
    }
    if (d0.view_id < 0 || d0.view_id >= static_cast<int>(views.size()) || d0.view_id == 0) {
        throw std::runtime_error(method_name + ": returned invalid or current-pose view id.");
    }
    if (!d0.algorithm_runtime_sec || *d0.algorithm_runtime_sec < 0.0) {
        throw std::runtime_error(method_name + ": missing runtime accounting.");
    }

    auto d1 = algo.decideNext(makeContext(session_dir, 1, pose(0.0, 3.0, 0.0), views, {d0.view_id}));
    if (d1.type != objview::AlgorithmDecision::Type::Move) {
        throw std::runtime_error(method_name + ": expected a move decision at step 1.");
    }
    if (d1.view_id == d0.view_id) {
        throw std::runtime_error(method_name + ": repeated an already-submitted view id.");
    }

    if (algo.observedStepCountForTest() != 2) {
        throw std::runtime_error(method_name + ": algorithm did not consume both observations.");
    }
    if (algo.mapLeafCountForTest() == 0) {
        throw std::runtime_error(method_name + ": octomap did not update.");
    }
    if (!fs::exists(session_dir / "voxel_ig_debug" / "map_step_1__with_unknown.ot")) {
        throw std::runtime_error(
            method_name + ": debug octomap with unknown voxels was not written.");
    }
    if (!fs::exists(session_dir / "voxel_ig_debug" / "map_step_1__known_only.ot")) {
        throw std::runtime_error(
            method_name + ": debug octomap known-only map was not written.");
    }
}

}  // namespace

int main() {
    const fs::path tmp = fs::temp_directory_path() / "objview_voxel_ig_algorithm_test";
    fs::remove_all(tmp);
    fs::create_directories(tmp);

    const fs::path views_path = tmp / "views.txt";
    writeText(views_path, "1 0 0\n0 1 0\n0 0 1\n-1 0 0\n");

    std::cout << "[test] oa" << std::endl;
    runMethodSmoke(objview::VoxelIgMethod::OA, "oa", tmp, views_path);
    std::cout << "[test] uv" << std::endl;
    runMethodSmoke(objview::VoxelIgMethod::UV, "uv", tmp, views_path);
    std::cout << "[test] rse" << std::endl;
    runMethodSmoke(objview::VoxelIgMethod::RSE, "rse", tmp, views_path);
    std::cout << "[test] apora" << std::endl;
    runMethodSmoke(objview::VoxelIgMethod::APORA, "apora", tmp, views_path);
    std::cout << "[test] pcv" << std::endl;
    runMethodSmoke(objview::VoxelIgMethod::PCV, "pcv", tmp, views_path);
    std::cout << "[test] kr" << std::endl;
    runMethodSmoke(objview::VoxelIgMethod::Kr, "kr", tmp, views_path);

    fs::remove_all(tmp);
    std::cout << "Voxel-IG algorithm smoke tests passed." << std::endl;
    return 0;
}
