#include "test.h"
#include "Value.h"
#include "MLP.h"

// Helper function to create Value objects
std::shared_ptr<Value> v(double val) {
    return std::make_shared<Value>(val);
}

Test::Test()
{
    // -----------------------------------------------------------------------
       // 1. SETUP THE DATASET
       // -----------------------------------------------------------------------
       // The inputs (xs)
    std::vector<std::vector<std::shared_ptr<Value>>> xs = {
        {v(2.0), v(3.0), v(-1.0)},
        {v(3.0), v(-1.0), v(0.5)},
        {v(0.5), v(1.0), v(1.0)},
        {v(1.0), v(1.0), v(-1.0)}
    };

    // The desired targets (ys)
    std::vector<std::shared_ptr<Value>> ys = {
        v(1.0),
        v(-1.0),
        v(-1.0),
        v(1.0)
    };

    // -----------------------------------------------------------------------
    // 2. INITIALIZE THE NEURAL NETWORK
    // -----------------------------------------------------------------------
    // MLP with:
    // - 3 Inputs (matching the dataset)
    // - Two hidden layers of 4 neurons each
    // - 1 Output neuron
    MLP model(3, { 4, 4, 1 });

    std::cout << "Model Architecture: " << model << "\n\n";

    // -----------------------------------------------------------------------
    // 3. TRAINING LOOP
    // -----------------------------------------------------------------------
    // We will run Gradient Descent for 20 iterations (steps)
    int steps = 20;
    double learning_rate = 0.05; // Slightly conservative rate

    for (int k = 0; k < steps; ++k) {

        // A. FORWARD PASS
        std::vector<std::shared_ptr<Value>> ypred;
        std::shared_ptr<Value> total_loss = v(0.0);

        for (size_t i = 0; i < xs.size(); ++i) {
            // Run the model on input x[i]
            // The model returns a vector, but since output layer is size 1, 
            // we take the first element [0].
            auto output_vector = model(xs[i]);
            auto prediction = output_vector[0];
            ypred.push_back(prediction);

            // Calculate Mean Squared Error Loss for this single example
            // Loss = (prediction - target)^2
            auto diff = prediction - ys[i];
            auto loss = diff->pow(2);

            // Accumulate total loss
            total_loss = total_loss + loss;
        }

        // B. ZERO GRADIENTS
        // Reset old gradients before calculating new ones!
        model.zero_grad();

        // C. BACKWARD PASS
        // Calculate gradients for every weight in the network
        total_loss->backward();

        // D. UPDATE PARAMETERS (Gradient Descent)
        // data = data - learning_rate * gradient
        for (auto& p : model.parameters()) {
            p->data -= learning_rate * p->grad;
        }

        // E. LOGGING
        std::cout << "Step " << k << " | Loss: " << total_loss->data << "\n";
    }

    // -----------------------------------------------------------------------
    // 4. FINAL RESULTS
    // -----------------------------------------------------------------------
    std::cout << "\nFinal Predictions:\n";
    for (size_t i = 0; i < xs.size(); ++i) {
        // Run one last forward pass to see the result
        auto pred = model(xs[i])[0];
        std::cout << "Input " << i << " -> Target: " << ys[i]->data
            << " | Prediction: " << pred->data << "\n";
    }
}