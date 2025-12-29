#pragma once
#include <vector>
#include <memory>
#include "Value.h"
#include "Module.h"

struct Neuron : public Module {
    std::vector<std::shared_ptr<Value>> w; // Weights
    std::shared_ptr<Value> b;              // Bias
    bool nonlin;                           // Apply non-linearity?

    // Constructor: nin = number of inputs
    Neuron(int nin, bool nonlin = true);

    // Forward pass (calls the neuron)
    std::shared_ptr<Value> operator()(const std::vector<std::shared_ptr<Value>>& x);

    // Override from Module
    std::vector<std::shared_ptr<Value>> parameters() override;

    // Optional: Print representation
    friend std::ostream& operator<<(std::ostream& os, const Neuron& n);
};