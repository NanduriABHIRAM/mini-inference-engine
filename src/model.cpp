#include "inference.hpp"

#include <cmath>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>

namespace mie {
namespace {
std::vector<float> random_values(std::size_t count, std::mt19937& rng,
                                 float low, float high) {
    std::uniform_real_distribution<float> distribution(low, high);
    std::vector<float> values(count);
    for (float& value : values) value = distribution(rng);
    return values;
}
}  // namespace

MLP load_mlp_file(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Cannot open model file: " + path);

    std::string token;
    if (!(input >> token) || token != "MIE_MLP_V1")
        throw std::runtime_error("Unsupported model file format in: " + path);

    std::string activation;
    if (!(input >> token >> activation) || token != "activation")
        throw std::runtime_error("Model file is missing its activation declaration.");

    MLP model;
    if (activation == "softmax") {
        model.output_activation = OutputActivation::Softmax;
    } else if (activation == "sigmoid") {
        model.output_activation = OutputActivation::Sigmoid;
    } else {
        throw std::runtime_error("Unsupported model output activation: " + activation);
    }

    int layer_count = 0;
    if (!(input >> token >> layer_count) || token != "layers" || layer_count <= 0 || layer_count > 128)
        throw std::runtime_error("Model file has an invalid layer count.");

    int expected_inputs = 0;
    for (int index = 0; index < layer_count; ++index) {
        DenseLayer layer;
        if (!(input >> token >> layer.in_features >> layer.out_features) || token != "layer" ||
            layer.in_features <= 0 || layer.out_features <= 0 ||
            layer.in_features > 1'000'000 || layer.out_features > 1'000'000) {
            throw std::runtime_error("Model file has an invalid layer declaration.");
        }
        if (index > 0 && layer.in_features != expected_inputs)
            throw std::runtime_error("Model layers do not have compatible feature dimensions.");

        const std::size_t weight_count =
            static_cast<std::size_t>(layer.in_features) * static_cast<std::size_t>(layer.out_features);
        if (weight_count > 100'000'000)
            throw std::runtime_error("Model layer is too large to load safely.");
        std::size_t declared_weights = 0;
        if (!(input >> token >> declared_weights) || token != "weights" || declared_weights != weight_count)
            throw std::runtime_error("Model file has an invalid weight count.");
        layer.W.resize(weight_count);
        for (float& value : layer.W) {
            if (!(input >> value) || !std::isfinite(value))
                throw std::runtime_error("Model file contains an invalid weight value.");
        }

        std::size_t declared_biases = 0;
        if (!(input >> token >> declared_biases) || token != "biases" ||
            declared_biases != static_cast<std::size_t>(layer.out_features)) {
            throw std::runtime_error("Model file has an invalid bias count.");
        }
        layer.b.resize(declared_biases);
        for (float& value : layer.b) {
            if (!(input >> value) || !std::isfinite(value))
                throw std::runtime_error("Model file contains an invalid bias value.");
        }

        expected_inputs = layer.out_features;
        model.layers.push_back(std::move(layer));
    }

    if (input >> token) throw std::runtime_error("Model file contains unexpected trailing data.");
    if (model.output_activation == OutputActivation::Sigmoid && model.output_dim() != 1)
        throw std::runtime_error("Sigmoid models must have exactly one output feature.");
    return model;
}

MLP make_reference_model() {
    std::mt19937 rng(12345);
    const int dims[] = {512, 256, 256, 10};
    MLP model;
    for (int i = 0; i < 3; ++i) {
        DenseLayer layer;
        layer.in_features = dims[i];
        layer.out_features = dims[i + 1];
        const float scale = std::sqrt(2.0f / layer.in_features);
        layer.W = random_values(static_cast<std::size_t>(layer.in_features) * layer.out_features,
                                rng, -scale, scale);
        layer.b = random_values(static_cast<std::size_t>(layer.out_features), rng, -0.05f, 0.05f);
        model.layers.push_back(std::move(layer));
    }
    return model;
}

std::vector<float> make_reference_input(int batch, int features) {
    std::mt19937 rng(67890 + batch);
    return random_values(static_cast<std::size_t>(batch) * features, rng, -1.0f, 1.0f);
}
}  // namespace mie
