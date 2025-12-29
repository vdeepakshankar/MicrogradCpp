#pragma once
#include <vector>
#include <memory>
#include <iostream>
#include "Module.h"
#include "Layer.h"

struct MLP : public Module {
    std::vector<std::shared_ptr<Layer>> layers;

    // Constructor
    // nin   = number of inputs to the network
    // nouts = vector defining the size of each layer 
    //         e.g., {4, 4, 1} means two hidden layers of 4, and one output of 1
    MLP(int nin, std::vector<int> nouts);

    // Forward Pass
    std::vector<std::shared_ptr<Value>> operator()(std::vector<std::shared_ptr<Value>> x);

    // Get parameters from all layers
    std::vector<std::shared_ptr<Value>> parameters() override;

    // Print
    friend std::ostream& operator<<(std::ostream& os, const MLP& m);
};