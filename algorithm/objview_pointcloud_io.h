#ifndef OBJVIEWBENCH_OBJVIEW_POINTCLOUD_IO_H_
#define OBJVIEWBENCH_OBJVIEW_POINTCLOUD_IO_H_

#include <filesystem>
#include <iostream>
#include <stdexcept>

#include <octomap/Pointcloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace objview {

namespace fs = std::filesystem;

inline pcl::PointCloud<pcl::PointXYZ> toPclPointCloudXYZ(const octomap::Pointcloud& cloud) {
    pcl::PointCloud<pcl::PointXYZ> pcl_cloud;
    pcl_cloud.reserve(cloud.size());
    for (size_t i = 0; i < cloud.size(); ++i) {
        const auto& p = cloud.getPoint(static_cast<unsigned int>(i));
        pcl_cloud.push_back(pcl::PointXYZ(p.x(), p.y(), p.z()));
    }
    pcl_cloud.width = static_cast<uint32_t>(pcl_cloud.size());
    pcl_cloud.height = 1;
    pcl_cloud.is_dense = false;
    return pcl_cloud;
}

inline void savePointCloudXYZBinary(const fs::path& path,
                                    const pcl::PointCloud<pcl::PointXYZ>& cloud,
                                    bool verbose = false) {
    if (verbose) {
        std::cerr << "[pointcloud_io] savePointCloudXYZBinary path=" << path.string() << std::endl;
        std::cerr << "[pointcloud_io] create_directories parent="
                  << path.parent_path().string() << std::endl;
    }
    if (path.empty()) {
        throw std::runtime_error("savePointCloudXYZBinary received an empty output path.");
    }
    if (path.parent_path().empty()) {
        throw std::runtime_error(
            "savePointCloudXYZBinary output path has an empty parent path: " + path.string());
    }
    fs::create_directories(path.parent_path());
    if (pcl::io::savePCDFileBinary(path.string(), cloud) != 0) {
        throw std::runtime_error("Failed to save PCD file: " + path.string());
    }
}

inline void savePointCloudXYZBinary(const fs::path& path,
                                    const octomap::Pointcloud& cloud,
                                    bool verbose = false) {
    savePointCloudXYZBinary(path, toPclPointCloudXYZ(cloud), verbose);
}

inline void savePointCloudXYZRGBBinary(const fs::path& path,
                                       const pcl::PointCloud<pcl::PointXYZRGB>& cloud,
                                       bool verbose = false) {
    if (verbose) {
        std::cerr << "[pointcloud_io] savePointCloudXYZRGBBinary path=" << path.string() << std::endl;
        std::cerr << "[pointcloud_io] create_directories parent="
                  << path.parent_path().string() << std::endl;
    }
    if (path.empty()) {
        throw std::runtime_error("savePointCloudXYZRGBBinary received an empty output path.");
    }
    if (path.parent_path().empty()) {
        throw std::runtime_error(
            "savePointCloudXYZRGBBinary output path has an empty parent path: " + path.string());
    }
    fs::create_directories(path.parent_path());
    if (pcl::io::savePCDFileBinary(path.string(), cloud) != 0) {
        throw std::runtime_error("Failed to save PCD file: " + path.string());
    }
}

}  // namespace objview

#endif  // OBJVIEWBENCH_OBJVIEW_POINTCLOUD_IO_H_
