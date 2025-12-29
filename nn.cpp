#include "nn.h"

// Random number generator for weight initialization
static std::random_device rd;
static std::mt19937 gen(rd());
static std::uniform_real_distribution<> dis(-1.0, 1.0);

// Neuron implementation
Neuron::Neuron(int nin, bool nonlin) : nonlin(nonlin) {
    for (int i = 0; i < nin; i++) {
        w.push_back(make_value(dis(gen)));
    }
    b = make_value(dis(gen));
}

std::shared_ptr<Value> Neuron::operator()(const std::vector<std::shared_ptr<Value>>& x) {
    // w * x + b
    auto act = b;
    for (size_t i = 0; i < w.size(); i++) {
        act = (*act) + ((*w[i]) * x[i]);
    }
    return nonlin ? act->tanh() : act;
}

std::vector<std::shared_ptr<Value>> Neuron::parameters() {
    std::vector<std::shared_ptr<Value>> params = w;
    params.push_back(b);
    return params;
}

// Layer implementation
Layer::Layer(int nin, int nout, bool nonlin) {
    for (int i = 0; i < nout; i++) {
        neurons.push_back(Neuron(nin, nonlin));
    }
}

std::vector<std::shared_ptr<Value>> Layer::operator()(const std::vector<std::shared_ptr<Value>>& x) {
    std::vector<std::shared_ptr<Value>> outs;
    for (auto& n : neurons) {
        outs.push_back(n(x));
    }
    return outs;
}

std::vector<std::shared_ptr<Value>> Layer::parameters() {
    std::vector<std::shared_ptr<Value>> params;
    for (auto& n : neurons) {
        auto p = n.parameters();
        params.insert(params.end(), p.begin(), p.end());
    }
    return params;
}

// MLP implementation
MLP::MLP(int nin, const std::vector<int>& nouts) {
    std::vector<int> sz = {nin};
    sz.insert(sz.end(), nouts.begin(), nouts.end());
    
    for (size_t i = 0; i < nouts.size(); i++) {
        bool is_last_layer = (i == nouts.size() - 1);
        layers.push_back(Layer(sz[i], sz[i + 1], !is_last_layer));
    }
}

std::vector<std::shared_ptr<Value>> MLP::operator()(const std::vector<std::shared_ptr<Value>>& x) {
    auto out = x;
    for (auto& layer : layers) {
        out = layer(out);
    }
    return out;
}

std::vector<std::shared_ptr<Value>> MLP::parameters() {
    std::vector<std::shared_ptr<Value>> params;
    for (auto& layer : layers) {
        auto p = layer.parameters();
        params.insert(params.end(), p.begin(), p.end());
    }
    return params;
}
