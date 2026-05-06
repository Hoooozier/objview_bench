cmake_minimum_required(VERSION 3.24)

# Project information
project(objview VERSION 0.1 LANGUAGES C CXX CUDA)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES native)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Compilation options
set(CMAKE_CXX_FLAGS "-g")
add_compile_options(-O3)

# Enable CUDA language and set the CUDA Toolkit root directory
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

# Find required third-party libraries
find_package(Boost REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(jsoncpp REQUIRED)
find_package(ZLIB REQUIRED)

# Set the C++ flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Include additional directories
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  /gurobi1301/linux64/include/
  /usr/local/include
  /usr/local/include/eigen3
  /usr/include/jsoncpp
  ${CUDAToolkit_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIR}
)

# Link additional directories
link_directories(
  /gurobi1301/linux64/lib/
  /usr/local/lib
  ${CUDAToolkit_LIBRARY_DIR}
  ${PCL_LIBRARY_DIRS}
  ${Boost_LIBRARY_DIRS}
  ${TORCH_LIBRARY_DIRS}
)

####################### Targets #######################

# executable target: TestEnv
add_executable(TestEnv
  test_env.cpp 
  test_cuda.cu
)
target_link_libraries(TestEnv PUBLIC
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
  ${Boost_LIBRARIES}
  ${jsoncpp_LIBRARIES}
  pthread
  gurobi_c++
  gurobi130
  octomath
  octomap
  jsoncpp
  cudart
)

# executable target: TestRaycaster
add_executable(TestRaycaster
  test_raycaster.cu
  cuda_raycaster.cu
)
target_link_libraries(TestRaycaster PUBLIC
  octomath
  octomap
  cudart
)

# executable target: TestGlobalPathPlanner
add_executable(TestGlobalPathPlanner
  test_global_path_planner.cpp
)
target_link_libraries(TestGlobalPathPlanner PUBLIC
  gurobi_c++
  gurobi130
)

# executable target: ObjViewAlgorithmRunner
add_executable(ObjViewAlgorithmRunner
  objview_algorithm_runner.cpp
  cuda_raycaster.cu
  voxel_ig_cuda.cu
)
target_link_libraries(ObjViewAlgorithmRunner PUBLIC
  jsoncpp
  gurobi_c++
  gurobi130
  octomath
  octomap
  cnpy
  ZLIB::ZLIB
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
  cudart
)

# executable target: TestRandomTspOrderAlgorithm
add_executable(TestRandomTspOrderAlgorithm
  test_random_tsp_order_algorithm.cpp
)
target_link_libraries(TestRandomTspOrderAlgorithm PUBLIC
  jsoncpp
  gurobi_c++
  gurobi130
)

# executable target: TestVoxelIgAlgorithm
add_executable(TestVoxelIgAlgorithm
  test_voxel_ig_algorithm.cpp
  voxel_ig_cuda.cu
)
target_link_libraries(TestVoxelIgAlgorithm PUBLIC
  jsoncpp
  octomath
  octomap
  cnpy
  ZLIB::ZLIB
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
  cudart
)

# executable target: TestObservationIO
add_executable(TestObservationIO
  test_observation_io.cpp
)
target_link_libraries(TestObservationIO PUBLIC
  cnpy
  ZLIB::ZLIB
  ${OpenCV_LIBS}
)

# executable target: TestBenchmarkSubmitterProtocol
add_executable(TestBenchmarkSubmitterProtocol
  test_benchmark_submitter_protocol.cpp
)
target_link_libraries(TestBenchmarkSubmitterProtocol PUBLIC
  jsoncpp
  pthread
)

# executable target: TestShapeCompletionClient
add_executable(TestShapeCompletionClient
  test_shape_completion_client.cpp
)
target_link_libraries(TestShapeCompletionClient PUBLIC
  jsoncpp
  pthread
  ${PCL_LIBRARIES}
)

# executable target: PcdSetCover
add_executable(PcdSetCover
  set_cover_from_pcd.cpp
  cuda_raycaster.cu
)
target_link_libraries(PcdSetCover PUBLIC
  ${PCL_LIBRARIES}
  ${Boost_LIBRARIES}
  pthread
  gurobi_c++
  gurobi130
  octomath
  octomap
  cudart
)
# executable target: BatchPcdSetCover
add_executable(BatchPcdSetCover
  batch_set_cover_from_pcd.cpp
)

# executable target: PcdSaturation
add_executable(PcdSaturation
  saturation_from_pcd.cpp
  cuda_raycaster.cu
)
target_link_libraries(PcdSaturation PUBLIC
  ${PCL_LIBRARIES}
  ${Boost_LIBRARIES}
  pthread
  gurobi_c++
  gurobi130
  octomath
  octomap
  cudart
)
# executable target: BatchPcdSaturation
add_executable(BatchPcdSaturation
  batch_saturation_from_pcd.cpp
)

# executable target: PcdCache
add_executable(PcdCache
  cache_from_pcd.cpp
  cuda_raycaster.cu
)
target_link_libraries(PcdCache PUBLIC
  ${PCL_LIBRARIES}
  ${Boost_LIBRARIES}
  pthread
  octomath
  octomap
  cudart
)
# executable target: BatchPcdCache
add_executable(BatchPcdCache
  batch_cache_from_pcd.cpp
)
target_link_libraries(BatchPcdCache PUBLIC
  jsoncpp
)

# executable target: RolloutCache
add_executable(RolloutCache
  rollout_from_cache.cpp
)
target_link_libraries(RolloutCache PUBLIC
  ${PCL_LIBRARIES}
  ${Boost_LIBRARIES}
  pthread
  gurobi_c++
  gurobi130
  octomath
  octomap
  jsoncpp
)
# executable target: BatchRolloutCache
add_executable(BatchRolloutCache
  batch_rollout_from_cache.cpp
)
target_link_libraries(BatchRolloutCache PUBLIC
  jsoncpp
)

# executable target: ExportMethodCase
add_executable(ExportMethodCase
  export_method_case.cpp
)
target_link_libraries(ExportMethodCase PUBLIC
  ${PCL_LIBRARIES}
  ${Boost_LIBRARIES}
  pthread
  gurobi_c++
  gurobi130
  octomath
  octomap
  jsoncpp
)
# executable target: BatchExportMethodCase
add_executable(BatchExportMethodCase
  batch_export_method_case.cpp
)
target_link_libraries(BatchExportMethodCase PUBLIC
  jsoncpp
)

# executable target: ExportMascvpOfflineCase
add_executable(ExportMascvpOfflineCase
  export_mascvp_offline_case.cpp
)
target_link_libraries(ExportMascvpOfflineCase PUBLIC
  ${PCL_LIBRARIES}
  ${Boost_LIBRARIES}
  pthread
  octomath
  octomap
  jsoncpp
  cnpy
  ZLIB::ZLIB
)

# executable target: BatchExportMascvpOfflineCase
add_executable(BatchExportMascvpOfflineCase
  batch_export_mascvp_offline_case.cpp
)
target_link_libraries(BatchExportMascvpOfflineCase PUBLIC
  jsoncpp
)

# executable target: EvalMascvpObjectWorker
add_executable(EvalMascvpObjectWorker
  eval_mascvp_object_worker.cpp
)
target_link_libraries(EvalMascvpObjectWorker PUBLIC
  ${PCL_LIBRARIES}
  ${Boost_LIBRARIES}
  pthread
  octomath
  octomap
  jsoncpp
)

# executable target: BatchEvalMascvpObjectWorker
add_executable(BatchEvalMascvpObjectWorker
  batch_eval_mascvp_object_worker.cpp
)
target_link_libraries(BatchEvalMascvpObjectWorker PUBLIC
  jsoncpp
)

# executable target: BenbvSingleWorker
add_executable(BenbvSingleWorker
  benbv_single_worker.cpp
  cuda_raycaster.cu
)
target_link_libraries(BenbvSingleWorker PUBLIC
  ${PCL_LIBRARIES}
  ${Boost_LIBRARIES}
  pthread
  octomath
  octomap
  jsoncpp
  cnpy
  ZLIB::ZLIB
  cudart
)

# executable target: BatchBenbvSingleWorker
add_executable(BatchBenbvSingleWorker
  batch_benbv_single_worker.cpp
)
target_link_libraries(BatchBenbvSingleWorker PUBLIC
  jsoncpp
)
