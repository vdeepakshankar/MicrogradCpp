#include "Value.h"
#include "nn.h"
#include <iostream>
#include <iomanip>
#include <vector>

int main() {
    std::cout << "MicrogradCpp - Hello World Example" << std::endl;
    std::cout << "===================================" << std::endl << std::endl;
    
    // Create a simple dataset for binary classification
    // XOR-like problem
    std::vector<std::vector<double>> xs = {
        {2.0, 3.0, -1.0},
        {3.0, -1.0, 0.5},
        {0.5, 1.0, 1.0},
        {1.0, 1.0, -1.0}
    };
    
    std::vector<double> ys = {1.0, -1.0, -1.0, 1.0}; // Desired targets
    
    // Create MLP: 3 inputs -> 4 hidden -> 4 hidden -> 1 output
    MLP model(3, {4, 4, 1});
    
    std::cout << "Model created with " << model.parameters().size() << " parameters" << std::endl;
    std::cout << "Training on " << xs.size() << " examples..." << std::endl << std::endl;
    
    // Training loop
    int epochs = 100;
    double learning_rate = 0.01;
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        // Forward pass
        std::vector<std::shared_ptr<Value>> ypred;
        for (auto& x : xs) {
            std::vector<std::shared_ptr<Value>> input;
            for (auto val : x) {
                input.push_back(make_value(val));
            }
            auto out = model(input);
            ypred.push_back(out[0]);
        }
        
        // Calculate loss (MSE)
        auto loss = make_value(0.0);
        for (size_t i = 0; i < ys.size(); i++) {
            auto diff = (*ypred[i]) - make_value(ys[i]);
            loss = (*loss) + ((*diff) * diff);
        }
        
        // Zero gradients
        for (auto p : model.parameters()) {
            p->grad = 0.0;
        }
        
        // Backward pass
        loss->backward();
        
        // Update parameters
        for (auto p : model.parameters()) {
            p->data -= learning_rate * p->grad;
        }
        
        // Print progress
        if (epoch % 10 == 0 || epoch == epochs - 1) {
            std::cout << "Epoch " << std::setw(3) << epoch 
                      << " | Loss: " << std::fixed << std::setprecision(6) << loss->data 
                      << std::endl;
        }
    }
    
    std::cout << std::endl << "Training complete!" << std::endl << std::endl;
    
    // Test the trained model
    std::cout << "Final predictions:" << std::endl;
    for (size_t i = 0; i < xs.size(); i++) {
        std::vector<std::shared_ptr<Value>> input;
        for (auto val : xs[i]) {
            input.push_back(make_value(val));
        }
        auto out = model(input);
        std::cout << "Input: [";
        for (size_t j = 0; j < xs[i].size(); j++) {
            std::cout << std::setw(5) << std::fixed << std::setprecision(1) << xs[i][j];
            if (j < xs[i].size() - 1) std::cout << ", ";
        }
        std::cout << "] | Predicted: " << std::setw(7) << std::fixed << std::setprecision(4) 
                  << out[0]->data << " | Target: " << std::setw(5) << ys[i] << std::endl;
    }
    
    return 0;
}
