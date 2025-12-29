#ifndef NN_H
#define NN_H

#include "Value.h"
#include <vector>
#include <random>
#include <memory>

// Neuron class
class Neuron {
private:
    std::vector<std::shared_ptr<Value>> w;
    std::shared_ptr<Value> b;
    bool nonlin;

public:
    Neuron(int nin, bool nonlin = true);
    std::shared_ptr<Value> operator()(const std::vector<std::shared_ptr<Value>>& x);
    std::vector<std::shared_ptr<Value>> parameters();
};

// Layer class
class Layer {
private:
    std::vector<Neuron> neurons;

public:
    Layer(int nin, int nout, bool nonlin = true);
    std::vector<std::shared_ptr<Value>> operator()(const std::vector<std::shared_ptr<Value>>& x);
    std::vector<std::shared_ptr<Value>> parameters();
};

// MLP (Multi-Layer Perceptron) class
class MLP {
private:
    std::vector<Layer> layers;

public:
    MLP(int nin, const std::vector<int>& nouts);
    std::vector<std::shared_ptr<Value>> operator()(const std::vector<std::shared_ptr<Value>>& x);
    std::vector<std::shared_ptr<Value>> parameters();
};

#endif // NN_H
