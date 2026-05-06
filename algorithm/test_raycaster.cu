#include <random>
#include <chrono>
#include <limits>
#include <cuda_raycaster.h>

int main() {
    try {

        double resolution = 0.1;
        octomap::OcTree tree(resolution);

        const int numPointClouds = 10;  
        const int pointsPerCloud = 1000;
        std::mt19937 gen(42); 
        std::uniform_real_distribution<double> originDist(-10.0, 10.0); 
        std::uniform_real_distribution<double> directionDist(-1.0, 1.0); 
        std::uniform_real_distribution<double> rangeDist(0.5, 10.0); 

        std::cout << "Inserting " << numPointClouds << " random point clouds with number of points " << pointsPerCloud << " into OcTree..." << std::endl;

        for (int i = 0; i < numPointClouds; ++i) {
            octomap::point3d sensorOrigin(originDist(gen), originDist(gen), originDist(gen));

            octomap::Pointcloud pointCloud;

            // ������ɵ��Ƶ�
            for (int j = 0; j < pointsPerCloud; ++j) {
                octomap::point3d randomPoint(
                    sensorOrigin.x() + directionDist(gen) * rangeDist(gen),
                    sensorOrigin.y() + directionDist(gen) * rangeDist(gen),
                    sensorOrigin.z() + directionDist(gen) * rangeDist(gen)
                );
                pointCloud.push_back(randomPoint);
            }

            tree.insertPointCloud(pointCloud, sensorOrigin);
        }
        std::cout << "Point cloud insertion completed." << std::endl;

        std::uniform_real_distribution<double> queryOriginDist(-5.0, 5.0);
        std::uniform_real_distribution<double> queryDirectionDist(-1.0, 1.0);

        const int numQueriesFull = 100000000;
        std::cout << "Generating " << numQueriesFull << " raycasting queries..." << std::endl;

        std::vector<octomap::point3d> origins_full;
        std::vector<octomap::point3d> directions_full;

        for (int i = 0; i < numQueriesFull; ++i) {
            octomap::point3d origin(queryOriginDist(gen), queryOriginDist(gen), queryOriginDist(gen));
            octomap::point3d direction(queryDirectionDist(gen), queryDirectionDist(gen), queryDirectionDist(gen));
            direction.normalize();

            origins_full.push_back(origin);
            directions_full.push_back(direction);
        }

        std::vector<int> test_numQueries = {100000, 1000000, 10000000, 100000000};

        for (int numQueries : test_numQueries) {
            std::cout << "NumQueries: " << numQueries << std::endl;
            auto startTime = std::chrono::high_resolution_clock::now();
            auto endTime = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed;
            int hitCount = 0;

            std::cout << "Performing " << numQueries << " raycasting queries" << std::endl;
            if (numQueries <= 1000000) {
                std::cout << "CPU version" << std::endl;
                startTime = std::chrono::high_resolution_clock::now();

                hitCount = 0;
                for (int i = 0; i < numQueries; ++i) {
                    octomap::point3d origin = origins_full[i];
                    octomap::point3d direction = directions_full[i];

                    // ִ�� raycasting
                    octomap::point3d hitPoint;
                    if (tree.castRay(origin, direction, hitPoint, true, 10.0)) {
                        if (hitCount == 0) {
							std::cout << "First hit point: " << hitPoint << std::endl;
						}
                        ++hitCount;
                    }
                }

                endTime = std::chrono::high_resolution_clock::now();
                elapsed = endTime - startTime;
                std::cout << "Raycasting completed." << std::endl;
                std::cout << "Total hits: " << hitCount << std::endl;
                std::cout << "Elapsed time: " << elapsed.count() << " seconds." << std::endl;
                std::cout << "Average time per query: " << (elapsed.count() / numQueries) << " seconds." << std::endl;
            }

            // CUDA version
            std::cout << "CUDA version" << std::endl;
            std::vector<octomap::point3d> origins;
            std::vector<octomap::point3d> directions;
            for (int i = 0; i < numQueries; ++i) {
                origins.push_back(origins_full[i]);
                directions.push_back(directions_full[i]);
            }

            startTime = std::chrono::high_resolution_clock::now();

            octomap::CudaRayCaster cuda_ray_caster(tree);
            
            std::vector<octomap::point3d> end_pts;
            std::vector<double> max_ranges(numQueries, 10.0);
            bool* find_end_pts = cuda_ray_caster.castRay(origins, directions, &end_pts, true, max_ranges);

            hitCount = 0;
            for (int i = 0; i < numQueries; ++i) {
                if (find_end_pts[i]) {
                    if (hitCount == 0) {
                        std::cout << "First hit point: " << end_pts[i] << std::endl;
                    }
                    ++hitCount;
                }
            }

            endTime = std::chrono::high_resolution_clock::now();
            elapsed = endTime - startTime;
            std::cout << "Raycasting completed." << std::endl;
            std::cout << "Total hits: " << hitCount << std::endl;
            std::cout << "Elapsed time: " << elapsed.count() << " seconds." << std::endl;
            std::cout << "Average time per query: " << (elapsed.count() / numQueries) << " seconds." << std::endl;
            delete[] find_end_pts;
        }

    }
    catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
