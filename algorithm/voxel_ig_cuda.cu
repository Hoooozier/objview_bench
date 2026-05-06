#include "voxel_ig_cuda.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <stdexcept>

namespace objview {
namespace {

inline void cudaCheck(cudaError_t err, const char* expr, const char* file, int line) {
    if (err == cudaSuccess) return;
    std::ostringstream oss;
    oss << "CUDA error at " << file << ":" << line << " for " << expr
        << ": " << cudaGetErrorString(err);
    throw std::runtime_error(oss.str());
}

#define OBJVIEW_CUDA_CHECK(expr) cudaCheck((expr), #expr, __FILE__, __LINE__)

struct Float3 {
    float x;
    float y;
    float z;
};

struct CameraBasisDevice {
    Float3 x_right;
    Float3 y_down;
    Float3 z_forward;
    float fx;
    float fy;
    float cx;
    float cy;
};

__host__ __device__ inline Float3 makeFloat3(float x, float y, float z) {
    Float3 v{x, y, z};
    return v;
}

__host__ __device__ inline Float3 add3(const Float3& a, const Float3& b) {
    return makeFloat3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ inline Float3 sub3(const Float3& a, const Float3& b) {
    return makeFloat3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ inline Float3 mul3(const Float3& a, float s) {
    return makeFloat3(a.x * s, a.y * s, a.z * s);
}

__host__ __device__ inline float dot3(const Float3& a, const Float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ inline Float3 cross3(const Float3& a, const Float3& b) {
    return makeFloat3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}

__host__ __device__ inline float norm3(const Float3& a) {
    return sqrtf(dot3(a, a));
}

__host__ __device__ inline Float3 normalize3(const Float3& a) {
    const float n = norm3(a);
    if (n < 1e-12f) return makeFloat3(0.0f, 0.0f, 0.0f);
    return mul3(a, 1.0f / n);
}

inline CameraBasisDevice makeCameraBasisDevice(const Pose7d& pose,
                                               const CameraIntrinsics& intrinsics) {
    CameraBasisDevice basis{};
    basis.fx = static_cast<float>(
        0.5 * static_cast<double>(intrinsics.width) / std::tan(0.5 * intrinsics.fov_x_rad));
    basis.fy = static_cast<float>(
        0.5 * static_cast<double>(intrinsics.height) / std::tan(0.5 * intrinsics.fov_y_rad));
    basis.cx = static_cast<float>(intrinsics.principal_x);
    basis.cy = static_cast<float>(intrinsics.principal_y);

    Float3 camera = makeFloat3(
        static_cast<float>(pose.v[0]),
        static_cast<float>(pose.v[1]),
        static_cast<float>(pose.v[2]));
    Float3 lookat = makeFloat3(
        static_cast<float>(pose.v[3]),
        static_cast<float>(pose.v[4]),
        static_cast<float>(pose.v[5]));
    Float3 z_forward = sub3(lookat, camera);
    if (norm3(z_forward) < 1e-12f) {
        lookat = makeFloat3(0.0f, 0.0f, 0.0f);
        z_forward = sub3(lookat, camera);
    }
    z_forward = normalize3(z_forward);

    if (norm3(sub3(z_forward, makeFloat3(0.0f, 0.0f, -1.0f))) < 1e-6f) {
        z_forward = normalize3(makeFloat3(1e-8f, 1e-8f, -1.0f));
    }
    if (norm3(sub3(z_forward, makeFloat3(0.0f, 0.0f, 1.0f))) < 1e-6f) {
        z_forward = normalize3(makeFloat3(1e-8f, 1e-8f, 1.0f));
    }

    const Float3 world_up = makeFloat3(0.0f, 0.0f, 1.0f);
    Float3 x_left = normalize3(cross3(mul3(z_forward, -1.0f), world_up));
    Float3 y_up = normalize3(cross3(x_left, mul3(z_forward, -1.0f)));

    const float c = static_cast<float>(std::cos(pose.v[6]));
    const float s = static_cast<float>(std::sin(pose.v[6]));
    const Float3 x_left_0 = x_left;
    const Float3 y_up_0 = y_up;
    x_left = add3(mul3(x_left_0, c), mul3(y_up_0, s));
    y_up = add3(mul3(x_left_0, -s), mul3(y_up_0, c));

    basis.x_right = mul3(x_left, -1.0f);
    basis.y_down = mul3(y_up, -1.0f);
    basis.z_forward = z_forward;
    return basis;
}

__device__ inline Float3 pixelToWorldRayDevice(int x, int y, const CameraBasisDevice& basis) {
    const float x_cam = (static_cast<float>(x) + 0.5f - basis.cx) / basis.fx;
    const float y_cam = (static_cast<float>(y) + 0.5f - basis.cy) / basis.fy;
    return normalize3(add3(add3(mul3(basis.x_right, x_cam), mul3(basis.y_down, y_cam)), basis.z_forward));
}

__device__ inline bool intersectAabb(const Float3& origin,
                                     const Float3& dir,
                                     const DenseGridSpec& spec,
                                     float* t_enter,
                                     float* t_exit) {
    float tmin = -HUGE_VALF;
    float tmax = HUGE_VALF;
    const float bbox_min[3] = {spec.bbox_min[0], spec.bbox_min[1], spec.bbox_min[2]};
    const float bbox_max[3] = {spec.bbox_max[0], spec.bbox_max[1], spec.bbox_max[2]};
    const float o[3] = {origin.x, origin.y, origin.z};
    const float d[3] = {dir.x, dir.y, dir.z};
    for (int axis = 0; axis < 3; ++axis) {
        if (fabsf(d[axis]) < 1e-12f) {
            if (o[axis] < bbox_min[axis] || o[axis] > bbox_max[axis]) return false;
            continue;
        }
        const float inv_d = 1.0f / d[axis];
        float t1 = (bbox_min[axis] - o[axis]) * inv_d;
        float t2 = (bbox_max[axis] - o[axis]) * inv_d;
        if (t1 > t2) {
            const float tmp = t1;
            t1 = t2;
            t2 = tmp;
        }
        tmin = fmaxf(tmin, t1);
        tmax = fminf(tmax, t2);
        if (tmin > tmax) return false;
    }
    if (tmax < fmaxf(0.0f, tmin)) return false;
    *t_enter = tmin;
    *t_exit = tmax;
    return true;
}

__device__ inline float voxelEntropyDevice(float occupancy) {
    const float p = fminf(fmaxf(occupancy, 1e-9f), 1.0f - 1e-9f);
    return -p * logf(p) - (1.0f - p) * logf(1.0f - p);
}

enum class DeviceMethodTag : int {
    OA = 1,
    UV = 2,
    RSE = 3,
    KR = 4,
};

__device__ inline float informationFunctionDevice(DeviceMethodTag method,
                                                  float ray_information,
                                                  float voxel_information,
                                                  float visible,
                                                  bool is_unknown,
                                                  bool previous_voxel_unknown,
                                                  bool is_endpoint,
                                                  bool is_occupied) {
    switch (method) {
        case DeviceMethodTag::OA:
            return ray_information + visible * voxel_information;
        case DeviceMethodTag::UV:
            return is_unknown ? (ray_information + visible * voxel_information) : ray_information;
        case DeviceMethodTag::RSE:
            if (is_endpoint) {
                if (previous_voxel_unknown) {
                    return is_occupied ? (ray_information + visible * voxel_information) : 0.0f;
                }
                return 0.0f;
            }
            if (is_unknown) {
                return ray_information + visible * voxel_information;
            }
            return 0.0f;
        case DeviceMethodTag::KR:
            if (is_endpoint) {
                return is_occupied ? (ray_information + voxel_information) : 0.0f;
            }
            return ray_information + voxel_information;
    }
    return ray_information;
}

__global__ void scoreViewKernel(const float* occupancy_prob,
                                DenseGridSpec spec,
                                CameraBasisDevice basis,
                                Float3 origin,
                                int width,
                                int height,
                                int ray_stride,
                                float unknown_lower,
                                float unknown_upper,
                                DeviceMethodTag method,
                                float* out_gain) {
    const int sampled_width = (width + ray_stride - 1) / ray_stride;
    const int sampled_height = (height + ray_stride - 1) / ray_stride;
    const int num_rays = sampled_width * sampled_height;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_rays) return;

    const int su = tid % sampled_width;
    const int sv = tid / sampled_width;
    const int x = su * ray_stride;
    const int y = sv * ray_stride;
    if (x >= width || y >= height) return;

    const Float3 dir = pixelToWorldRayDevice(x, y, basis);

    float t_enter = 0.0f;
    float t_exit = 0.0f;
    if (!intersectAabb(origin, dir, spec, &t_enter, &t_exit)) {
        return;
    }

    const int nx = spec.grid_dims[0];
    const int ny = spec.grid_dims[1];
    const int nz = spec.grid_dims[2];
    const float voxel_size_x = (spec.bbox_max[0] - spec.bbox_min[0]) / static_cast<float>(nx);
    const float voxel_size_y = (spec.bbox_max[1] - spec.bbox_min[1]) / static_cast<float>(ny);
    const float voxel_size_z = (spec.bbox_max[2] - spec.bbox_min[2]) / static_cast<float>(nz);
    const float eps = 1e-4f * fminf(voxel_size_x, fminf(voxel_size_y, voxel_size_z));
    const float start_t = fmaxf(t_enter, 0.0f) + eps;
    const Float3 p = add3(origin, mul3(dir, start_t));

    int ix = static_cast<int>(floorf((p.x - spec.bbox_min[0]) / voxel_size_x));
    int iy = static_cast<int>(floorf((p.y - spec.bbox_min[1]) / voxel_size_y));
    int iz = static_cast<int>(floorf((p.z - spec.bbox_min[2]) / voxel_size_z));
    ix = ix < 0 ? 0 : (ix >= nx ? nx - 1 : ix);
    iy = iy < 0 ? 0 : (iy >= ny ? ny - 1 : iy);
    iz = iz < 0 ? 0 : (iz >= nz ? nz - 1 : iz);

    const float dir_arr[3] = {dir.x, dir.y, dir.z};
    const float origin_arr[3] = {origin.x, origin.y, origin.z};
    const float bbox_min[3] = {spec.bbox_min[0], spec.bbox_min[1], spec.bbox_min[2]};
    const float voxel_size[3] = {voxel_size_x, voxel_size_y, voxel_size_z};
    int idx_arr[3] = {ix, iy, iz};
    int step[3] = {0, 0, 0};
    float tMax[3] = {HUGE_VALF, HUGE_VALF, HUGE_VALF};
    float tDelta[3] = {HUGE_VALF, HUGE_VALF, HUGE_VALF};

    for (int axis = 0; axis < 3; ++axis) {
        if (dir_arr[axis] > 0.0f) {
            step[axis] = 1;
            const float next_boundary = bbox_min[axis] + static_cast<float>(idx_arr[axis] + 1) * voxel_size[axis];
            tMax[axis] = (next_boundary - origin_arr[axis]) / dir_arr[axis];
            tDelta[axis] = voxel_size[axis] / dir_arr[axis];
        } else if (dir_arr[axis] < 0.0f) {
            step[axis] = -1;
            const float next_boundary = bbox_min[axis] + static_cast<float>(idx_arr[axis]) * voxel_size[axis];
            tMax[axis] = (next_boundary - origin_arr[axis]) / dir_arr[axis];
            tDelta[axis] = -voxel_size[axis] / dir_arr[axis];
        }
    }

    float gain = 0.0f;
    float visible = 1.0f;
    bool previous_unknown = false;

    while (ix >= 0 && ix < nx && iy >= 0 && iy < ny && iz >= 0 && iz < nz) {
        const size_t flat_idx =
            static_cast<size_t>(ix) +
            static_cast<size_t>(nx) * (
                static_cast<size_t>(iy) +
                static_cast<size_t>(ny) * static_cast<size_t>(iz));
        const float occupancy = occupancy_prob[flat_idx];
        const float entropy = voxelEntropyDevice(occupancy);
        const bool is_unknown = (occupancy > unknown_lower && occupancy < unknown_upper);
        const bool is_occupied = (occupancy >= unknown_upper);

        int next_axis = 0;
        if (tMax[1] < tMax[next_axis]) next_axis = 1;
        if (tMax[2] < tMax[next_axis]) next_axis = 2;
        const float next_t = tMax[next_axis];
        const int next_ix = ix + (next_axis == 0 ? step[0] : 0);
        const int next_iy = iy + (next_axis == 1 ? step[1] : 0);
        const int next_iz = iz + (next_axis == 2 ? step[2] : 0);
        const bool next_outside =
            next_ix < 0 || next_ix >= nx ||
            next_iy < 0 || next_iy >= ny ||
            next_iz < 0 || next_iz >= nz;
        const bool is_endpoint = is_occupied || next_outside || (next_t > t_exit + eps);

        gain = informationFunctionDevice(
            method,
            gain,
            entropy,
            visible,
            is_unknown,
            previous_unknown,
            is_endpoint,
            is_occupied);

        visible *= (1.0f - occupancy);
        previous_unknown = is_unknown;

        if (is_endpoint) break;

        if (next_axis == 0) {
            ix += step[0];
            tMax[0] += tDelta[0];
        } else if (next_axis == 1) {
            iy += step[1];
            tMax[1] += tDelta[1];
        } else {
            iz += step[2];
            tMax[2] += tDelta[2];
        }
    }

    atomicAdd(out_gain, gain);
}

}  // namespace

VoxelIgCudaScorer::VoxelIgCudaScorer(const DenseGridSpec& spec,
                                     float unknown_lower,
                                     float unknown_upper)
    : spec_(spec),
      unknown_lower_(unknown_lower),
      unknown_upper_(unknown_upper) {
    voxel_count_ =
        static_cast<std::size_t>(spec_.grid_dims[0]) *
        static_cast<std::size_t>(spec_.grid_dims[1]) *
        static_cast<std::size_t>(spec_.grid_dims[2]);
    if (voxel_count_ == 0) {
        throw std::runtime_error("VoxelIgCudaScorer requires a positive grid size.");
    }
    OBJVIEW_CUDA_CHECK(cudaMalloc(&d_occupancy_prob_, voxel_count_ * sizeof(float)));
    OBJVIEW_CUDA_CHECK(cudaMalloc(&d_view_gain_, sizeof(float)));
}

VoxelIgCudaScorer::~VoxelIgCudaScorer() {
    if (d_occupancy_prob_ != nullptr) cudaFree(d_occupancy_prob_);
    if (d_view_gain_ != nullptr) cudaFree(d_view_gain_);
}

void VoxelIgCudaScorer::uploadVolume(const DenseGridVolume& volume) {
    if (!denseGridSpecEquals(volume.spec, spec_)) {
        throw std::runtime_error("VoxelIgCudaScorer volume spec mismatch.");
    }
    if (volume.occupancy_prob.size() != voxel_count_) {
        throw std::runtime_error("VoxelIgCudaScorer volume size mismatch.");
    }
    OBJVIEW_CUDA_CHECK(cudaMemcpy(
        d_occupancy_prob_,
        volume.occupancy_prob.data(),
        voxel_count_ * sizeof(float),
        cudaMemcpyHostToDevice));
}

float VoxelIgCudaScorer::scoreView(MethodTag method,
                                   const Pose7d& pose,
                                   const CameraIntrinsics& intrinsics,
                                   int ray_stride) const {
    if (ray_stride <= 0) {
        throw std::runtime_error("VoxelIgCudaScorer ray_stride must be positive.");
    }
    if (d_occupancy_prob_ == nullptr || d_view_gain_ == nullptr) {
        throw std::runtime_error("VoxelIgCudaScorer device buffers are not initialized.");
    }
    const CameraBasisDevice basis = makeCameraBasisDevice(pose, intrinsics);
    const Float3 origin = makeFloat3(
        static_cast<float>(pose.v[0]),
        static_cast<float>(pose.v[1]),
        static_cast<float>(pose.v[2]));
    const int sampled_width = (intrinsics.width + ray_stride - 1) / ray_stride;
    const int sampled_height = (intrinsics.height + ray_stride - 1) / ray_stride;
    const int num_rays = sampled_width * sampled_height;

    constexpr int kBlockSize = 256;
    const int grid_size = (num_rays + kBlockSize - 1) / kBlockSize;

    float zero = 0.0f;
    OBJVIEW_CUDA_CHECK(cudaMemcpy(d_view_gain_, &zero, sizeof(float), cudaMemcpyHostToDevice));
    scoreViewKernel<<<grid_size, kBlockSize>>>(
        d_occupancy_prob_,
        spec_,
        basis,
        origin,
        intrinsics.width,
        intrinsics.height,
        ray_stride,
        unknown_lower_,
        unknown_upper_,
        static_cast<DeviceMethodTag>(static_cast<int>(method)),
        d_view_gain_);
    OBJVIEW_CUDA_CHECK(cudaGetLastError());
    OBJVIEW_CUDA_CHECK(cudaDeviceSynchronize());

    float gain = 0.0f;
    OBJVIEW_CUDA_CHECK(cudaMemcpy(&gain, d_view_gain_, sizeof(float), cudaMemcpyDeviceToHost));
    return gain;
}

float VoxelIgCudaScorer::scoreViewOA(const Pose7d& pose,
                                     const CameraIntrinsics& intrinsics,
                                     int ray_stride) const {
    return scoreView(MethodTag::OA, pose, intrinsics, ray_stride);
}

float VoxelIgCudaScorer::scoreViewUV(const Pose7d& pose,
                                     const CameraIntrinsics& intrinsics,
                                     int ray_stride) const {
    return scoreView(MethodTag::UV, pose, intrinsics, ray_stride);
}

float VoxelIgCudaScorer::scoreViewRSE(const Pose7d& pose,
                                      const CameraIntrinsics& intrinsics,
                                      int ray_stride) const {
    return scoreView(MethodTag::RSE, pose, intrinsics, ray_stride);
}

float VoxelIgCudaScorer::scoreViewKr(const Pose7d& pose,
                                     const CameraIntrinsics& intrinsics,
                                     int ray_stride) const {
    return scoreView(MethodTag::KR, pose, intrinsics, ray_stride);
}

}  // namespace objview
