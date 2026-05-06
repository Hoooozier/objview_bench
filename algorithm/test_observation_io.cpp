#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>

#include <cnpy.h>

#include "objview_observation_io.h"

namespace fs = std::filesystem;

int main() {
    const fs::path tmp = fs::temp_directory_path() / "objview_depth_npz_test.npz";
    fs::remove(tmp);

    const float depth[] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
    };
    cnpy::npz_save(tmp.string(), "depth", depth, {2, 3}, "w");

    const objview::DepthImage image = objview::loadDepthNpz(tmp);
    if (image.width != 3 || image.height != 2) {
        throw std::runtime_error("Unexpected depth image shape.");
    }
    if (image.depth.size() != 6) {
        throw std::runtime_error("Unexpected depth image data size.");
    }
    if (std::abs(image.at(2, 1) - 6.0f) > 1e-6f) {
        throw std::runtime_error("Unexpected depth value.");
    }

    fs::remove(tmp);
    std::cout << "Observation IO depth npz test passed." << std::endl;
    return 0;
}
