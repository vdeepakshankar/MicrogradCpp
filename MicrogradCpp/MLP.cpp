#include "MLP.h"

// 1. Constructor
MLP::MLP(int nin, std::vector<int> nouts) {
    // The size of the inputs for the *current* layer being built.
    // Initially, this is the number of inputs to the whole network.
    int current_nin = nin;

    for (size_t i = 0; i < nouts.size(); ++i) {
        int current_nout = nouts[i];

        // Determine if this is the last layer
        // Andrej's Micrograd typically has ReLU on all layers EXCEPT the last one.
        bool is_last_layer = (i == nouts.size() - 1);
        bool nonlin = !is_last_layer;

        // Create the layer
        layers.push_back(std::make_shared<Layer>(current_nin, current_nout, nonlin));

        // The output of this layer becomes the input of the next
        current_nin = current_nout;
    }
}

// 2. Forward Pass
std::vector<std::shared_ptr<Value>> MLP::operator()(std::vector<std::shared_ptr<Value>> x) {
    // We create a temporary variable to hold the data as it flows through the network
    std::vector<std::shared_ptr<Value>> current_x = x;

    for (auto& layer : layers) {
        // Pass the data through the current layer
        // The output becomes the input for the next iteration
        current_x = (*layer)(current_x);
    }

    return current_x;
}

// 3. Parameters
std::vector<std::shared_ptr<Value>> MLP::parameters() {
    std::vector<std::shared_ptr<Value>> params;

    // Collect parameters from every layer
    for (auto& layer : layers) {
        auto layer_params = layer->parameters();
        params.insert(params.end(), layer_params.begin(), layer_params.end());
    }

    return params;
}

// 4. Print
std::ostream& operator<<(std::ostream& os, const MLP& m) {
    os << "MLP of [";
    for (size_t i = 0; i < m.layers.size(); ++i) {
        os << *m.layers[i];
        if (i < m.layers.size() - 1) os << ", ";
    }
    os << "]";
    return os;
}
