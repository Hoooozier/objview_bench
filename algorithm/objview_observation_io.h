#ifndef OBJVIEWBENCH_OBJVIEW_OBSERVATION_IO_H_
#define OBJVIEWBENCH_OBJVIEW_OBSERVATION_IO_H_

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <cnpy.h>
#include <Eigen/Dense>
#include <json/json.h>
#include <octomap/Pointcloud.h>
#include <opencv2/imgcodecs.hpp>

namespace objview {

struct DepthImage {
    int width = 0;
    int height = 0;
    std::vector<float> depth;

    float at(int x, int y) const {
        return depth.at(static_cast<size_t>(y) * static_cast<size_t>(width) +
                        static_cast<size_t>(x));
    }
};

struct MaskImage {
    int width = 0;
    int height = 0;
    std::vector<uint8_t> valid;

    bool at(int x, int y) const {
        return valid.at(static_cast<size_t>(y) * static_cast<size_t>(width) +
                        static_cast<size_t>(x)) != 0;
    }
};

struct CameraIntrinsics {
    int width = 0;
    int height = 0;
    double fov_x_rad = 0.0;
    double fov_y_rad = 0.0;
    double principal_x = 0.0;
    double principal_y = 0.0;
};

inline DepthImage loadDepthNpz(const std::filesystem::path& path,
                               const std::string& array_name = "depth") {
    cnpy::NpyArray array;
    try {
        array = cnpy::npz_load(path.string(), array_name);
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to load depth npz " + path.string() +
                                 " array '" + array_name + "': " + e.what());
    }

    if (array.shape.size() != 2) {
        throw std::runtime_error("Depth npz array must be 2D: " + path.string());
    }

    const size_t height = array.shape[0];
    const size_t width = array.shape[1];
    if (height == 0 || width == 0) {
        throw std::runtime_error("Depth npz array is empty: " + path.string());
    }

    DepthImage image;
    image.width = static_cast<int>(width);
    image.height = static_cast<int>(height);
    image.depth.resize(width * height);

    if (array.word_size == sizeof(float)) {
        const float* src = array.data<float>();
        image.depth.assign(src, src + image.depth.size());
    } else if (array.word_size == sizeof(double)) {
        const double* src = array.data<double>();
        for (size_t i = 0; i < image.depth.size(); ++i) {
            image.depth[i] = static_cast<float>(src[i]);
        }
    } else {
        throw std::runtime_error("Depth npz array must be float32 or float64: " +
                                 path.string());
    }

    return image;
}

inline MaskImage loadMaskImage(const std::filesystem::path& path) {
    const cv::Mat mask = cv::imread(path.string(), cv::IMREAD_GRAYSCALE);
    if (mask.empty()) {
        throw std::runtime_error("Failed to load mask image: " + path.string());
    }

    MaskImage image;
    image.width = mask.cols;
    image.height = mask.rows;
    image.valid.resize(static_cast<size_t>(image.width) * static_cast<size_t>(image.height));
    for (int y = 0; y < image.height; ++y) {
        const uint8_t* row = mask.ptr<uint8_t>(y);
        for (int x = 0; x < image.width; ++x) {
            image.valid[static_cast<size_t>(y) * static_cast<size_t>(image.width) +
                        static_cast<size_t>(x)] = row[x] > 0 ? 1 : 0;
        }
    }
    return image;
}

inline CameraIntrinsics parseCameraIntrinsics(const Json::Value& frame_meta) {
    const Json::Value intr = frame_meta["intrinsics"];
    CameraIntrinsics out;
    out.width = intr.get("image_width", 0).asInt();
    out.height = intr.get("image_height", 0).asInt();
    out.fov_x_rad = intr.get("fov_x_rad", 0.0).asDouble();
    out.fov_y_rad = intr.get("fov_y_rad", 0.0).asDouble();
    out.principal_x = intr.get("principal_x", 0.0).asDouble();
    out.principal_y = intr.get("principal_y", 0.0).asDouble();
    if (out.width <= 0 || out.height <= 0 ||
        out.fov_x_rad <= 0.0 || out.fov_y_rad <= 0.0) {
        throw std::runtime_error("Invalid camera intrinsics in frame_meta.");
    }
    return out;
}

inline Eigen::Matrix4d parseMatrix4d(const Json::Value& value, const std::string& name) {
    if (!value.isArray() || value.size() != 4) {
        throw std::runtime_error(name + " must be a 4x4 array.");
    }
    Eigen::Matrix4d out;
    for (int r = 0; r < 4; ++r) {
        if (!value[r].isArray() || value[r].size() != 4) {
            throw std::runtime_error(name + " must be a 4x4 array.");
        }
        for (int c = 0; c < 4; ++c) {
            out(r, c) = value[r][c].asDouble();
        }
    }
    return out;
}

inline octomap::Pointcloud backprojectDepthToWorldPointcloud(
    const DepthImage& depth,
    const MaskImage& mask,
    const CameraIntrinsics& intr,
    const Eigen::Matrix4d& camera_to_world,
    double bbox_min = -std::numeric_limits<double>::infinity(),
    double bbox_max = std::numeric_limits<double>::infinity()) {

    if (depth.width != mask.width || depth.height != mask.height) {
        throw std::runtime_error("Depth and mask shape mismatch.");
    }
    if (depth.width != intr.width || depth.height != intr.height) {
        throw std::runtime_error("Depth image shape does not match frame_meta intrinsics.");
    }

    const double fx = 0.5 * static_cast<double>(intr.width) / std::tan(0.5 * intr.fov_x_rad);
    const double fy = 0.5 * static_cast<double>(intr.height) / std::tan(0.5 * intr.fov_y_rad);

    octomap::Pointcloud cloud;
    cloud.reserve(depth.depth.size());
    for (int y = 0; y < depth.height; ++y) {
        for (int x = 0; x < depth.width; ++x) {
            if (!mask.at(x, y)) continue;
            const float z = depth.at(x, y);
            if (!std::isfinite(z) || z <= 0.0f) continue;

            const double x_cam = ((static_cast<double>(x) + 0.5) - intr.principal_x) *
                                 static_cast<double>(z) / fx;
            const double y_cam = ((static_cast<double>(y) + 0.5) - intr.principal_y) *
                                 static_cast<double>(z) / fy;
            const Eigen::Vector4d p_cam(x_cam, y_cam, static_cast<double>(z), 1.0);
            const Eigen::Vector4d p_world = camera_to_world * p_cam;
            if (!std::isfinite(p_world.x()) || !std::isfinite(p_world.y()) ||
                !std::isfinite(p_world.z())) {
                continue;
            }
            if (p_world.x() < bbox_min || p_world.x() > bbox_max ||
                p_world.y() < bbox_min || p_world.y() > bbox_max ||
                p_world.z() < bbox_min || p_world.z() > bbox_max) {
                continue;
            }
            cloud.push_back(static_cast<float>(p_world.x()),
                            static_cast<float>(p_world.y()),
                            static_cast<float>(p_world.z()));
        }
    }
    return cloud;
}

}  // namespace objview

#endif  // OBJVIEWBENCH_OBJVIEW_OBSERVATION_IO_H_
