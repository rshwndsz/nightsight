#include <torch/script.h>

#include <iostream>
#include <memory>
#include <tuple>
#include <stdlib.h>

int main(int argc, const char *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: inference <path-to-exported-script-module>\n";
        exit(EXIT_FAILURE);
    }

    torch::jit::script::Module module;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load()
        module = torch::jit::load(argv[1]);
    }
    catch (const c10::Error& e) {
        std::cerr << "Error loading the model \n";
        exit(EXIT_FAILURE);
    }

    std::cout << "OK\n";

    // Create a Vector of inputs and add a single input
    // `torch::jit::Ivalue` (a type-erased value type `script::Module` methods accept and return)
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::ones({1, 3, 256, 256}));

    // Execute the model and turn its output into a tensor
    auto output = module.forward(inputs).toTuple()->elements()[1].toTensor();

    //std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';

    exit(EXIT_SUCCESS);
}

