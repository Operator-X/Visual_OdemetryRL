#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "svo_env/svo_vec_env.h"
#include "svo_env/utils.h"

// [rlvo] gt_init_pose is an input, but binding it as a writeable Eigen::Ref makes pybind11 reject non-contiguous
// arrays. svo_wrapper.py passes a strided view (gt_poses[:, 0, :]) whenever an env initializes with GT poses.
// Accept any float array for it instead (forcecast + c_style copies to contiguous only when needed).
using InputMatrix = pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast>;

static Eigen::Map<MatrixRowMajor<>> as_matrix(InputMatrix &a) {
  return Eigen::Map<MatrixRowMajor<>>(a.mutable_data(), a.shape(0), a.shape(1));
}

PYBIND11_MODULE(svo_env, m) {
    pybind11::class_<svo::SvoVecEnv>(m, "SVOEnv")
        .def(pybind11::init<const std::string, const std::string, const int, const bool>())
        .def("step", [](svo::SvoVecEnv &self, pybind11::array_t<uint8_t> &input_images, Ref<Vector<>> time_nsec,
                        Ref<MatrixRowMajor<>> actions, Ref<Vector<>> use_RL_actions, Ref<MatrixRowMajor<>> out_pose,
                        Ref<MatrixRowMajor<>> observations, Ref<Vector<>> dones, Ref<Vector<>> stages,
                        Ref<Vector<>> runtime, Ref<Vector<>> use_gt_init_pose, InputMatrix gt_init_pose) {
            auto gt = as_matrix(gt_init_pose);
            self.step(input_images, time_nsec, actions, use_RL_actions, out_pose, observations, dones, stages,
                      runtime, use_gt_init_pose, gt);
        })
        .def("reset", &svo::SvoVecEnv::reset)
        .def("env_step", [](svo::SvoVecEnv &self, Ref<Vector<>> indices, pybind11::array_t<uint8_t> input_images,
                            Ref<Vector<>> time_nsec, Ref<MatrixRowMajor<>> actions, Ref<Vector<>> use_RL_actions,
                            Ref<MatrixRowMajor<>> out_pose, Ref<MatrixRowMajor<>> observations, Ref<Vector<>> dones,
                            Ref<Vector<>> stages, Ref<Vector<>> runtime, Ref<Vector<>> use_gt_init_pose,
                            InputMatrix gt_init_pose) {
            auto gt = as_matrix(gt_init_pose);
            self.env_step(indices, input_images, time_nsec, actions, use_RL_actions, out_pose, observations, dones,
                          stages, runtime, use_gt_init_pose, gt);
        })
        .def("env_visualize_features", &svo::SvoVecEnv::env_visualize_features)
        .def("setSeed", &svo::SvoVecEnv::setSeed);
    m.def("load_image_batch", &load_image_batch);
}
