/* Nightsight in C++

   Usage
   -----
   (from the root directory)
   ./app/build/nightsight     \
       <path-to-traced-model> \
       <path-to-input-image>  \
       <path-to-output-image>
*/

#include "config.h"
#include <iostream>
#include <memory>
#include <stdlib.h>
#include <torch/script.h>
#include <tuple>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image.h"
#include "stb/stb_image_resize.h"
#include "stb/stb_image_write.h"

int main(int argc, const char *argv[]) {
  // Check usage
  if (argc != 4) {
    // Report version
    std::cout << argv[0] << " Version " << Nightsight_VERSION_MAJOR << "."
              << Nightsight_VERSION_MINOR << std::endl;
    // Report usage
    std::cerr << "Usage: " << argv[0]
              << " <path-to-exported-script-module> <path-to-original-image> "
              << "<path-to-new-image>\n";
    exit(EXIT_FAILURE);
  }

  // Load Model
  std::cout << "Loading model";
  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load()
    module = torch::jit::load(argv[1]);
  } catch (const c10::Error &e) {
    std::cerr << " - failed\n";
    exit(EXIT_FAILURE);
  }
  std::cout << " - done\n";

  // Load image
  // https://stackoverflow.com/a/40812978
  int width, height, ch;
  unsigned char *image;
  std::cout << "Loading image";
  image = stbi_load(/*filename*/ argv[2],
                    /*input_width*/ &width,
                    /*input_height*/ &height,
                    /*8-bit components per-pixel*/ &ch,
                    /*desired_channels*/ 0);
  if (!image) {
    std::cerr << " - failed\n";
    exit(EXIT_FAILURE);
  }
  std::cout << " - done\n";

  // Resize image
  std::cout << "Allocating memory for resized image";
  unsigned char *resized;
  resized = (unsigned char *)malloc(256 * 256 * ch);
  if (!resized) {
    std::cout << " - failed\n";
    exit(EXIT_FAILURE);
  }
  std::cout << " - done\n";
  std::cout << "Resizing image";
  int resize_status = stbir_resize_uint8(/*input_data*/ image,
                                         /*input_width*/ width,
                                         /*input_height*/ height,
                                         /*input_stride_in_bytes*/ 0,
                                         /*ouput_data*/ resized,
                                         /*output_width*/ 256,
                                         /*output_height*/ 256,
                                         /*output_stride_in_bytes*/ 0,
                                         /*NUM_CHANNELS*/ ch);
  if (!resize_status) {
    std::cerr << " - failed\n";
    exit(EXIT_FAILURE);
  } else {
    width = 256;
    height = 256;
    ch = ch;
    std::cout << " - done\n";
  }

  // Convert image to Tensor
  // https://stackoverflow.com/a/63154900
  // https://github.com/pytorch/pytorch/issues/12506
  std::cout << "Converting input to tensor";
  torch::Tensor imageTensor = torch::from_blob(/*Pure c++ matrix data*/ resized,
                                               /*dims*/ {width, height, ch},
                                               /*dtype*/ torch::kUInt8)
                                  .clone()
                                  .to(torch::kFloat32)
                                  .clamp_(0.0, 255.0)
                                  .div_(255.0)
                                  .permute({2, 0, 1})
                                  .unsqueeze(0);
  std::cout << " - done\n";

  // Create a Vector of inputs
  std::cout << "Enhancing input";
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(imageTensor);
  // Execute the model and turn its output into a tensor
  auto outputTensor = module.forward(inputs)
                          .toTuple()
                          ->elements()[1]
                          .toTensor()
                          .permute({0, 2, 3, 1})
                          .mul_(255.0)
                          .add_(0.5)
                          .clamp_(0.0, 255.0)
                          .to(torch::kUInt8);
  std::cout << " - done\n";

  // Convert Tensor to image
  std::cout << "Converting output tensor to data_ptr";
  auto output = (unsigned char *)outputTensor.data_ptr();
  if (!output) {
    std::cout << " - failed\n";
    exit(EXIT_FAILURE);
  } else {
    std::cout << " - done\n";
  }

  // Write image
  std::cout << "Writing output to file";
  // See: https://stackoverflow.com/a/40812978
  int write_status = stbi_write_jpg(/*filename*/ argv[3],
                                    /*width*/ width,
                                    /*height*/ height,
                                    /*NUM_CHANNELS*/ ch,
                                    /*Matrix*/ output,
                                    /*Quality*/ width * 3);
  if (!write_status) {
    std::cerr << " - failed\n";
    exit(EXIT_FAILURE);
  } else {
    std::cout << " - done\n";
  }

  exit(EXIT_SUCCESS);
}
