#include "inference.hpp"

#include <cmath>
#include <random>

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
