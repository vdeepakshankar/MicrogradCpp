#include "Module.h"

// 1. Zero Grad Implementation
void Module::zero_grad() {
    // Calls the VIRTUAL parameters() method
    // This will hit the child's implementation (e.g., Layer::parameters)
    for (auto& p : parameters()) {
        p->grad = 0.0;
    }
}

// 2. Parameters Implementation (Base Case)
std::vector<std::shared_ptr<Value>> Module::parameters() {
    return {}; // Returns empty vector
}