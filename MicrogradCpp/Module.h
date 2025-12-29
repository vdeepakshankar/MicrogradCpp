#pragma once
#include <vector>
#include <memory>
#include "Value.h" // Assuming Value struct is defined here

struct Module {
    // 1. Virtual Destructor
    // Essential in C++ to ensure derived classes are cleaned up correctly
    virtual ~Module() = default;

    // 2. zero_grad
    // Resets gradients of all parameters to 0.0
    void zero_grad();

    // 3. parameters
    // Returns the list of trainable parameters (weights + biases)
    // defined as 'virtual' so derived classes (Neuron, Layer) can override it.
    virtual std::vector<std::shared_ptr<Value>> parameters();
};