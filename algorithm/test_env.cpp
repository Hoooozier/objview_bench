#include <iostream>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <opencv2/opencv.hpp>
#include <boost/version.hpp>
#include <octomap/octomap.h>
#include <json/json.h>
#include <gurobi_c++.h>

void runCudaTest();

int main() {
    try {
        // Test Boost
        std::cout << "Boost version: " << BOOST_LIB_VERSION << std::endl;

        // Test Eigen
        Eigen::Matrix3f mat;
        mat << 1, 2, 3,
               4, 5, 6,
               7, 8, 9;
        std::cout << "Eigen matrix:\n" << mat << std::endl;

        // Test PCL
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
        cloud->width = 5;
        cloud->height = 1;
        cloud->points.resize(cloud->width * cloud->height);
        for (auto& point : cloud->points) {
            point.x = rand() % 1024;
            point.y = rand() % 1024;
            point.z = rand() % 1024;
        }
        std::cout << "PCL cloud size: " << cloud->points.size() << std::endl;

        // Test OpenCV
        cv::Mat img = cv::Mat::zeros(3, 3, CV_8UC1);
        img.at<uchar>(1, 1) = 255;
        std::cout << "OpenCV matrix:\n" << img << std::endl;

        // Test OctoMap
        octomap::OcTree tree(0.1);  // Create an octree with 0.1m resolution
        tree.updateNode(octomap::point3d(1.0, 1.0, 1.0), true);  // Insert a node
        std::cout << "OctoMap tree size: " << tree.calcNumNodes() << std::endl;

        // Test JsonCpp
        Json::Value root;
        root["project"] = "ObjViewBench";
        root["version"] = "1.0";
        root["dependencies"] = Json::arrayValue;
        root["dependencies"].append("Boost");
        root["dependencies"].append("Eigen");
        root["dependencies"].append("PCL");
        root["dependencies"].append("OpenCV");
        root["dependencies"].append("OctoMap");
        root["dependencies"].append("Gurobi");
        root["dependencies"].append("JsonCpp");

        Json::StreamWriterBuilder writer;
        std::string jsonString = Json::writeString(writer, root);

        std::cout << "JsonCpp generated JSON:\n" << jsonString << std::endl;

        // Test CUDA
        runCudaTest();

        // Test Gurobi
        try {
            GRBEnv env = GRBEnv(true);  // Silent mode
            env.start();
            GRBModel model = GRBModel(env);

            // Add variables
            GRBVar x = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, "x");
            GRBVar y = model.addVar(0.0, 1.0, 0.0, GRB_BINARY, "y");

            // Set objective: maximize x + y
            model.setObjective(x + y, GRB_MAXIMIZE);

            // Add constraint: x + 2 * y <= 1
            model.addConstr(x + 2 * y <= 1, "c0");

            // Optimize the model
            model.optimize();

            // Output results
            std::cout << "Gurobi optimization result: x=" << x.get(GRB_DoubleAttr_X)
                      << ", y=" << y.get(GRB_DoubleAttr_X) << std::endl;

        } catch (GRBException& e) {
            std::cerr << "Gurobi error: Code " << e.getErrorCode() << " - " << e.getMessage() << std::endl;
        }

    } catch (std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
    }

    std::cout << "All tests completed successfully!" << std::endl;
    return 0;
}
