#include "Value.h"
#include <algorithm>

// Addition
std::shared_ptr<Value> Value::operator+(std::shared_ptr<Value> other) {
    auto out = std::make_shared<Value>(this->data + other->data, 
                                        std::vector<std::shared_ptr<Value>>{shared_from_this(), other}, "+");
    
    out->_backward = [this_ptr = shared_from_this(), other, out]() {
        this_ptr->grad += out->grad;
        other->grad += out->grad;
    };
    
    return out;
}

std::shared_ptr<Value> Value::operator+(double other) {
    return *this + make_value(other);
}

// Multiplication
std::shared_ptr<Value> Value::operator*(std::shared_ptr<Value> other) {
    auto out = std::make_shared<Value>(this->data * other->data,
                                        std::vector<std::shared_ptr<Value>>{shared_from_this(), other}, "*");
    
    out->_backward = [this_ptr = shared_from_this(), other, out]() {
        this_ptr->grad += other->data * out->grad;
        other->grad += this_ptr->data * out->grad;
    };
    
    return out;
}

std::shared_ptr<Value> Value::operator*(double other) {
    return *this * make_value(other);
}

// Power
std::shared_ptr<Value> Value::pow(double n) {
    auto out = std::make_shared<Value>(std::pow(this->data, n),
                                        std::vector<std::shared_ptr<Value>>{shared_from_this()}, "pow");
    
    out->_backward = [this_ptr = shared_from_this(), n, out]() {
        this_ptr->grad += n * std::pow(this_ptr->data, n - 1) * out->grad;
    };
    
    return out;
}

// Negation
std::shared_ptr<Value> Value::operator-() {
    return *this * -1.0;
}

// Subtraction
std::shared_ptr<Value> Value::operator-(std::shared_ptr<Value> other) {
    return *this + (-(*other));
}

std::shared_ptr<Value> Value::operator-(double other) {
    return *this + (-other);
}

// Division
std::shared_ptr<Value> Value::operator/(std::shared_ptr<Value> other) {
    return *this * other->pow(-1.0);
}

std::shared_ptr<Value> Value::operator/(double other) {
    return *this * (1.0 / other);
}

// Tanh activation
std::shared_ptr<Value> Value::tanh() {
    double x = this->data;
    double t = (std::exp(2 * x) - 1) / (std::exp(2 * x) + 1);
    auto out = std::make_shared<Value>(t, std::vector<std::shared_ptr<Value>>{shared_from_this()}, "tanh");
    
    out->_backward = [this_ptr = shared_from_this(), t, out]() {
        this_ptr->grad += (1 - t * t) * out->grad;
    };
    
    return out;
}

// ReLU activation
std::shared_ptr<Value> Value::relu() {
    double out_data = this->data < 0 ? 0.0 : this->data;
    auto out = std::make_shared<Value>(out_data, std::vector<std::shared_ptr<Value>>{shared_from_this()}, "ReLU");
    
    out->_backward = [this_ptr = shared_from_this(), out]() {
        this_ptr->grad += (out->data > 0 ? 1.0 : 0.0) * out->grad;
    };
    
    return out;
}

// Build topological order
void Value::build_topo(std::shared_ptr<Value> v, std::vector<std::shared_ptr<Value>>& topo, 
                       std::set<std::shared_ptr<Value>>& visited) {
    if (visited.find(v) == visited.end()) {
        visited.insert(v);
        for (auto child : v->_prev) {
            build_topo(child, topo, visited);
        }
        topo.push_back(v);
    }
}

// Backpropagation
void Value::backward() {
    std::vector<std::shared_ptr<Value>> topo;
    std::set<std::shared_ptr<Value>> visited;
    build_topo(shared_from_this(), topo, visited);
    
    this->grad = 1.0;
    std::reverse(topo.begin(), topo.end());
    for (auto v : topo) {
        v->_backward();
    }
}
