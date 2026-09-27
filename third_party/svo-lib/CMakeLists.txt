cmake_minimum_required(VERSION 3.0)

project(SVO-LIB)

# [rlvo] Find Eigen via CMake instead of the hardcoded /usr/include/eigen3 (Linux-only).
# On macOS pass -DEigen3_DIR=/opt/homebrew/opt/eigen@3/share/eigen3/cmake to use Eigen 3.4 (Eigen 5 breaks old code).
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
get_target_property(RLVO_EIGEN_INC Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
include_directories(${RLVO_EIGEN_INC} ${RLVO_EIGEN_INC}/unsupported)
add_subdirectory(svo_env)
add_subdirectory(svo)
add_subdirectory(svo_common)
add_subdirectory(svo_img_align)
add_subdirectory(svo_tracker)
add_subdirectory(svo_direct)
add_subdirectory(vikit/vikit_common)
add_subdirectory(vikit/vikit_cameras)
add_subdirectory(vikit/vikit_solver)
add_subdirectory(minkindr/minkindr)
add_subdirectory(fast_neon)
add_subdirectory(rpg_common)
add_subdirectory(opengv)
add_subdirectory(glog)
add_subdirectory(gflags)

