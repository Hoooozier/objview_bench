#include <vector>
#include <octomap/octomap.h>
#include <octomap/OcTree.h>
#include <octomap/ColorOcTree.h>
#include <cuda_runtime.h>

#ifndef OCTOMAP_CUDA_RAYCASTER_H_
#define OCTOMAP_CUDA_RAYCASTER_H_

typedef unsigned long long cu_uint64_t;

struct KeyValue {
    cu_uint64_t key;
    cu_uint64_t value;
};

class CudaHashTable {
public:
    CudaHashTable() { createHashTable(); }

    ~CudaHashTable();

    void insert(const KeyValue* kvs, cu_uint64_t num_kvs);

    void query(KeyValue* kvs, cu_uint64_t num_kvs);

    cu_uint64_t static gpuQuery(const KeyValue* hash_table,
        cu_uint64_t key);

    KeyValue* data() { return hash_table_; }
private:
    void createHashTable();

    KeyValue* hash_table_;
};

const cu_uint64_t kHashTableCapacity = 128 * 1024 * 1024;
const cu_uint64_t kNumKeyValues = kHashTableCapacity / 2;
const cu_uint64_t kEmpty = 0xffffffffffffffff;
const cu_uint64_t vEmpty = kEmpty;

namespace octomap {

    struct CudaOcTreeKey {
        uint16_t k[3];
    };

    class CudaPoint3d {
    public:
        __host__ __device__ CudaPoint3d() { data_[0] = data_[1] = data_[2] = 0.0; }

        __host__ __device__ CudaPoint3d(const CudaPoint3d& pt) {
            data_[0] = pt.data_[0];
            data_[1] = pt.data_[1];
            data_[2] = pt.data_[2];
        }

        __host__ __device__ float& operator()(unsigned int i) { return data_[i]; }

        __host__ __device__ CudaPoint3d normalized() {
            double norm = sqrt(x() * x() + y() * y() + z() * z());
            CudaPoint3d res(*this);
            res.x() /= norm;
            res.y() /= norm;
            res.z() /= norm;
            return res;
        }

        __host__ __device__ float& x() { return data_[0]; }
        __host__ __device__ float& y() { return data_[1]; }
        __host__ __device__ float& z() { return data_[2]; }

    private:
        float data_[3];
    };

    class CudaRayCaster {
    public:
        struct OcTreeData {
            std::vector<cu_uint64_t> keys;
            std::vector<cu_uint64_t> occupancy;
            double resolution;
        };

        enum OccStatus {
            Occupied = 1,
            Free = 2,
        };

        explicit CudaRayCaster(const OcTreeData& octree_data, bool print_info = true);
        explicit CudaRayCaster(const std::vector<octomap::OcTreeKey>& octree_keys, const std::vector<bool>& occupieds, double resolution, bool print_info = true);
        explicit CudaRayCaster(const octomap::OcTree& octree, bool print_info = true);
        explicit CudaRayCaster(const octomap::ColorOcTree& octree, bool print_info = true);

        bool* castRay(const std::vector<octomap::point3d>& origins,
            const std::vector<octomap::point3d>& dirs,
            std::vector<octomap::point3d>* end_pts, bool ignore_unknown,
            const std::vector<double>& max_range);

    private:
        CudaHashTable cu_hash_table_;
        double resolution_;
    };

}  // namespace octomap

#endif  // OCTOMAP_CUDA_RAYCASTER_H_
