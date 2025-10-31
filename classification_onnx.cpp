// classification_onnx_grid_green.cpp
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <random>
#include <algorithm>

namespace fs = std::filesystem;

int main() {
    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "EfficientNetV2S");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        const wchar_t* model_path = L"efficientnetv2s_finetuned.onnx";
        Ort::Session session(env, model_path, session_options);
        Ort::AllocatorWithDefaultOptions allocator;

        // Class names
        std::vector<std::string> class_names = { "dandelion", "daisy", "tulips", "sunflowers", "roses" };

        // Collect images
        std::vector<std::string> all_images;
        for (const auto& entry : fs::directory_iterator("C:/Users/User/source/repos/classification_onnx/flowers/tulip")) {
            if (entry.is_regular_file()) all_images.push_back(entry.path().string());
        }
        if (all_images.empty()) { std::cerr << "No images found\n"; return -1; }

        std::shuffle(all_images.begin(), all_images.end(), std::mt19937{ std::random_device{}() });
        if (all_images.size() > 8) all_images.resize(8);

        const int size = 384; // model input
        const int grid_cols = 4;
        const int grid_rows = 2;
        const int display_w = 200;
        const int display_h = 200;

        cv::Mat grid_img = cv::Mat::zeros(display_h * grid_rows, display_w * grid_cols, CV_8UC3);

        for (size_t idx = 0; idx < all_images.size(); ++idx) {
            cv::Mat orig_img = cv::imread(all_images[idx]);
            if (orig_img.empty()) continue;

            // Preprocess
            cv::Mat img;
            cv::cvtColor(orig_img, img, cv::COLOR_BGR2RGB);
            cv::resize(img, img, cv::Size(size, size));
            img.convertTo(img, CV_32FC3); // keep 0..255

            std::vector<float> input_tensor_values;
            input_tensor_values.reserve(size * size * 3);
            for (int y = 0; y < size; ++y) {
                for (int x = 0; x < size; ++x) {
                    cv::Vec3f px = img.at<cv::Vec3f>(y, x);
                    input_tensor_values.push_back(px[0]);
                    input_tensor_values.push_back(px[1]);
                    input_tensor_values.push_back(px[2]);
                }
            }

            std::vector<int64_t> input_dims = { 1, size, size, 3 }; // NHWC
            Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                memory_info,
                input_tensor_values.data(),
                input_tensor_values.size(),
                input_dims.data(),
                input_dims.size()
            );

            Ort::AllocatedStringPtr input_name_alloc = session.GetInputNameAllocated(0, allocator);
            Ort::AllocatedStringPtr output_name_alloc = session.GetOutputNameAllocated(0, allocator);
            const char* input_names[] = { input_name_alloc.get() };
            const char* output_names[] = { output_name_alloc.get() };

            auto output_tensors = session.Run(Ort::RunOptions{ nullptr },
                input_names, &input_tensor, 1,
                output_names, 1);

            auto& out0 = output_tensors.front();
            float* out_data = out0.GetTensorMutableData<float>();
            auto out_shape = out0.GetTensorTypeAndShapeInfo().GetShape();

            size_t out_size = 1;
            for (auto d : out_shape) out_size *= (d < 0 ? 1 : d);

            std::vector<float> output_values(out_data, out_data + out_size);
            int top = std::distance(output_values.begin(),
                std::max_element(output_values.begin(), output_values.end()));
            float max_prob = output_values[top];

            // --- Console print ---
            std::cout << "\nImage: " << all_images[idx] << "\nClass probabilities:\n";
            for (size_t i = 0; i < output_values.size(); ++i)
                std::cout << "Class " << i << " (" << class_names[i] << "): " << output_values[i] << "\n";
            std::cout << "Predicted class: " << top << " (" << class_names[top] << ") prob=" << max_prob << "\n";

            // Resize original image for grid
            cv::Mat thumb;
            cv::resize(orig_img, thumb, cv::Size(display_w, display_h));

            // Overlay label (green)
            std::string label_text = class_names[top] + " : " + std::to_string(max_prob).substr(0, 5);
            cv::putText(thumb, label_text, cv::Point(5, 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);

            // Grid placement
            int r = idx / grid_cols;
            int c = idx % grid_cols;
            thumb.copyTo(grid_img(cv::Rect(c * display_w, r * display_h, display_w, display_h)));
        }

        cv::imshow("Predictions Grid", grid_img);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
    catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
