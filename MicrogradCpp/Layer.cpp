#include "Layer.h"
// 1. Constructor
Layer::Layer(int nin, int nout, bool nonlin) {
    for (int i = 0; i < nout; ++i) {
        // Create 'nout' neurons, each expecting 'nin' inputs
        neurons.push_back(std::make_shared<Neuron>(nin, nonlin));
    }
}

// 2. Forward Pass
std::vector<std::shared_ptr<Value>> Layer::operator()(const std::vector<std::shared_ptr<Value>>& x) {
    std::vector<std::shared_ptr<Value>> out;

    // Pass the input 'x' to every neuron in this layer
    for (auto& neuron : neurons) {
        out.push_back((*neuron)(x));
    }

    return out;
}

// 3. Parameters
std::vector<std::shared_ptr<Value>> Layer::parameters() {
    std::vector<std::shared_ptr<Value>> params;

    // Collect parameters from every neuron
    for (auto& neuron : neurons) {
        auto neuron_params = neuron->parameters();
        // Append neuron params to the layer list
        params.insert(params.end(), neuron_params.begin(), neuron_params.end());
    }

    return params;
}

// 4. Print
std::ostream& operator<<(std::ostream& os, const Layer& l) {
    os << "Layer of [";
    for (size_t i = 0; i < l.neurons.size(); ++i) {
        os << *l.neurons[i];
        if (i < l.neurons.size() - 1) os << ", ";
    }
    os << "]";
    return os;
}