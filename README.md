# MicrogradCpp

A C++ implementation of [Andrej Karpathy's micrograd](https://github.com/karpathy/micrograd) - a tiny autograd engine for building and training neural networks.

## Overview

This project implements a minimal automatic differentiation engine in C++ that can be used to build and train neural networks. It includes:

- **Value**: A class that wraps scalars and supports automatic differentiation
- **Neuron, Layer, MLP**: Building blocks for creating multi-layer perceptrons
- **Hello World Example**: A simple training demo showing how to use the library

## Features

- Automatic differentiation (autograd) engine
- Support for basic arithmetic operations (+, -, *, /)
- Activation functions (tanh, ReLU)
- Multi-layer perceptron (MLP) implementation
- Backpropagation through computation graph
- Simple and educational code structure

## Building

```bash
mkdir build
cd build
cmake ..
make
```

## Running

After building, run the demo:

```bash
./micrograd_demo
```

This will train a simple neural network on a toy dataset and show the training progress.

## Example Usage

```cpp
#include "Value.h"
#include "nn.h"

// Create some values
auto a = make_value(2.0);
auto b = make_value(-3.0);

// Perform operations
auto c = a + b;
auto d = c * make_value(2.0);
auto e = d->tanh();

// Backpropagate
e->backward();

// Access gradients
std::cout << "Gradient of a: " << a->grad << std::endl;
```

## Learning Resources

This implementation is inspired by Andrej Karpathy's excellent tutorial on building a neural network library from scratch. It's designed to be educational and help understand:

- How automatic differentiation works
- How neural networks compute gradients
- The fundamentals of backpropagation
- Building ML frameworks from scratch

## License

This project is created for educational purposes.
