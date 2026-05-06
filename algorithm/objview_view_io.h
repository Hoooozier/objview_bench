#ifndef OBJVIEWBENCH_OBJVIEW_VIEW_IO_H_
#define OBJVIEWBENCH_OBJVIEW_VIEW_IO_H_

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "objview_geometry.h"

namespace objview {

inline std::vector<Vec3> loadViewPositions(const std::string& path, double radius) {
    std::ifstream fin(path);
    if (!fin) {
        throw std::runtime_error("Failed to open views file: " + path);
    }

    std::vector<Vec3> views;
    std::string line;
    while (std::getline(fin, line)) {
        if (line.empty()) continue;

        std::istringstream iss(line);
        Vec3 v;
        if (!(iss >> v.x() >> v.y() >> v.z())) {
            throw std::runtime_error("Failed to parse views line: " + line);
        }
        if (v.norm() < 1e-12) {
            throw std::runtime_error("Zero-norm view in views file.");
        }
        views.push_back(v.normalized() * radius);
    }

    if (views.empty()) {
        throw std::runtime_error("No views loaded from: " + path);
    }
    return views;
}

}  // namespace objview

#endif  // OBJVIEWBENCH_OBJVIEW_VIEW_IO_H_
