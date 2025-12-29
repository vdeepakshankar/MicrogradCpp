#include "Neuron.h"
#include <random>
#include <iostream>

// Helper to generate random numbers between -1.0 and 1.0
// Simulates: random.uniform(-1, 1)
double random_uniform() {
    static std::random_device rd;  // Non-deterministic seed
    static std::mt19937 gen(rd()); // Mersenne Twister engine
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    return dis(gen);
}

// 1. Constructor
Neuron::Neuron(int nin, bool nonlin) : nonlin(nonlin) {
    // Initialize weights with random values between -1 and 1
    for (int i = 0; i < nin; ++i) {
        w.push_back(std::make_shared<Value>(random_uniform()));
    }
    // Initialize bias with random value between -1 and 1
    b = std::make_shared<Value>(random_uniform());
}

// 2. Forward Pass (operator())
std::shared_ptr<Value> Neuron::operator()(const std::vector<std::shared_ptr<Value>>& x) {
    // act = sum(w * x) + b

    // Start with the bias
    std::shared_ptr<Value> act = b;

    // Accumulate the dot product
    for (size_t i = 0; i < w.size(); ++i) {
        // We assume x.size() matches w.size()
        // act = act + (w[i] * x[i])
        act = act + (w[i] * x[i]);
    }

    // Apply non-linearity if requested
    // Andrej's code typically uses ReLU in the modern version, 
    // but Tanh in the original video. Since you have 'bool nonlin',
    // we use ReLU (standard modern default). 
    // Change to .tanh() if you want the exact video demo.
    return nonlin ? act->relu() : act;
}

// 3. Parameters
std::vector<std::shared_ptr<Value>> Neuron::parameters() {
    // Return [w..., b]
    std::vector<std::shared_ptr<Value>> params = w;
    params.push_back(b);
    return params;
}

// 4. Print
std::ostream& operator<<(std::ostream& os, const Neuron& n) {
    os << (n.nonlin ? "ReLU" : "Linear") << "Neuron(" << n.w.size() << ")";
    return os;
}