#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <opencv2/opencv.hpp>

#include "svo_env/utils.h"

pybind11::array_t<uint8_t> load_image_batch(const std::vector<std::string>& image_paths,
                                            int num_images,
                                            int height,
                                            int width) {
    // Create a 4-dimensional pybind11 array
//    int channels = 3;
    int channels = 1;
    pybind11::array_t<uint8_t> result({num_images, height, width, channels});
    auto result_ptr = result.mutable_data();

    // [rlvo] no process-wide omp_set_num_threads (it also changed torch's thread count every step);
    // local clause instead, same RLVO_SVO_THREADS override as SvoVecEnv (default 8 = authors' value)
    int num_threads = 8;
    if (const char* t = std::getenv("RLVO_SVO_THREADS")) {
      num_threads = std::max(1, std::atoi(t));
    }

    #pragma omp parallel for schedule(dynamic) num_threads(num_threads)
    for (int i = 0; i < num_images; ++i) {
//        cv::Mat image = cv::imread(image_paths[i], cv::IMREAD_COLOR);
        cv::Mat image = cv::imread(image_paths[i], cv::IMREAD_GRAYSCALE);
        std::memcpy(result_ptr + i * height * width * channels, image.data, height * width * channels);
    }

    return result;
}