// Inference.cpp
// TODO Description

// User includes
#include <torch/script.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#include "stb/stb_image_resize.h"
#include "stb/stb_image_write.h"
// C header files
#include <stdlib.h>
#include <stdint.h>
// C++ header files
#include <iostream>
#include <memory>
#include <tuple>

int main(int argc, const char *argv[]) {
    // Check usage
    if (argc != 3) {
        std::cerr << "Usage: inference <path-to-exported-script-module> <path-to-original-image>\n";
        exit(EXIT_FAILURE);
    }

    // Load Model
    torch::jit::script::Module module;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load()
        module = torch::jit::load(argv[1]);
    }
    catch (const c10::Error& e) {
        std::cerr << "Error loading the model \n";
        exit(EXIT_FAILURE);
    }
    // TODO Move this to logging
    std::cout << "Loaded model successfully\n";

    // Create a Vector of inputs and add a single input
    // `torch::jit::Ivalue` (a type-erased value type `script::Module` methods accept and return)
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::ones({1, 3, 256, 256}));

    // Load image
    // See: https://stackoverflow.com/a/40812978
    int width, height, bpp;
    uint8_t* image = stb::stbi_load(argv[2], &width, &height, &bpp, 3);
    stb::stbi_image_free(image);

    // Resize image

    // Convert image to Tensor

    // Execute the model and turn its output into a tensor
    auto output = module.forward(inputs).toTuple()->elements()[1].toTensor();

    // Convert Tensor to image
    
    // Write image
    // See: https://stackoverflow.com/a/40812978
    // stbi_write_jpg("output.jpg", width, height, CHANNEL_NUM, image, width*CHANNEL_NUM);

    exit(EXIT_SUCCESS);
}

