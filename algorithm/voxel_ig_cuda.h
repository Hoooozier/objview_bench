#ifndef OBJVIEWBENCH_VOXEL_IG_CUDA_H_
#define OBJVIEWBENCH_VOXEL_IG_CUDA_H_

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>

#include "objview_algorithm.h"
#include "objview_observation_io.h"

namespace objview {

struct DenseGridSpec {
    float bbox_min[3] = {-1.0f, -1.0f, -1.0f};
    float bbox_max[3] = {1.0f, 1.0f, 1.0f};
    int grid_dims[3] = {64, 64, 64};
};

inline bool denseGridSpecEquals(const DenseGridSpec& a, const DenseGridSpec& b) {
    constexpr float kTol = 1e-6f;
    for (int i = 0; i < 3; ++i) {
        if (std::fabs(a.bbox_min[i] - b.bbox_min[i]) > kTol) return false;
        if (std::fabs(a.bbox_max[i] - b.bbox_max[i]) > kTol) return false;
        if (a.grid_dims[i] != b.grid_dims[i]) return false;
    }
    return true;
}

struct DenseGridVolume {
    DenseGridSpec spec;
    std::vector<float> occupancy_prob;
};

class VoxelIgCudaScorer {
public:
    explicit VoxelIgCudaScorer(const DenseGridSpec& spec,
                               float unknown_lower,
                               float unknown_upper);

    ~VoxelIgCudaScorer();

    void uploadVolume(const DenseGridVolume& volume);

    float scoreViewOA(const Pose7d& pose,
                      const CameraIntrinsics& intrinsics,
                      int ray_stride) const;

    float scoreViewUV(const Pose7d& pose,
                      const CameraIntrinsics& intrinsics,
                      int ray_stride) const;

    float scoreViewRSE(const Pose7d& pose,
                       const CameraIntrinsics& intrinsics,
                       int ray_stride) const;

    float scoreViewKr(const Pose7d& pose,
                      const CameraIntrinsics& intrinsics,
                      int ray_stride) const;

    const DenseGridSpec& spec() const { return spec_; }

private:
    enum class MethodTag : int {
        OA = 1,
        UV = 2,
        RSE = 3,
        KR = 4,
    };

    float scoreView(MethodTag method,
                    const Pose7d& pose,
                    const CameraIntrinsics& intrinsics,
                    int ray_stride) const;

    DenseGridSpec spec_;
    float unknown_lower_ = 0.45f;
    float unknown_upper_ = 0.65f;
    std::size_t voxel_count_ = 0;
    float* d_occupancy_prob_ = nullptr;
    mutable float* d_view_gain_ = nullptr;
};

}  // namespace objview

#endif  // OBJVIEWBENCH_VOXEL_IG_CUDA_H_
