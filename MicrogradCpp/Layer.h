#pragma once
#include "Neuron.h"
#include "Module.h"
#include <vector>
#include <memory>
#include <iostream>

class Layer : public Module {
  public:
    std::vector<std::shared_ptr<Neuron>> neurons;

    Layer(int nin, int nout, bool nonlin = true);
    std::vector<std::shared_ptr<Value>> operator()(const std::vector<std::shared_ptr<Value>>& x);
    std::vector<std::shared_ptr<Value>> parameters() override;

    friend std::ostream& operator<<(std::ostream& os, const Layer& l);
};