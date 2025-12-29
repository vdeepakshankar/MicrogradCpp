#include "Value.h"

// Constructor
Value::Value(double data, std::vector<std::shared_ptr<Value>> children, std::string _op)
    : data(data), grad(0.0), op(_op), _prev(children.begin(), children.end()), _backward([]() {})
{
}

// --------------------------------------------------------------------------
// CORE MATH
// --------------------------------------------------------------------------

std::shared_ptr<Value> Value::add(std::shared_ptr<Value> rhs) {
    auto lhs = shared_from_this();
    auto out = std::make_shared<Value>(lhs->data + rhs->data, std::vector<std::shared_ptr<Value>>{ lhs, rhs }, "+");

    std::weak_ptr<Value> weak_out = out;
    out->_backward = [lhs, rhs, weak_out]() {
        auto out_ptr = weak_out.lock();
        if (out_ptr) {
            lhs->grad += 1.0 * out_ptr->grad;
            rhs->grad += 1.0 * out_ptr->grad;
        }
        };
    return out;
}

std::shared_ptr<Value> Value::mul(std::shared_ptr<Value> rhs) {
    auto lhs = shared_from_this();
    auto out = std::make_shared<Value>(lhs->data * rhs->data, std::vector<std::shared_ptr<Value>>{ lhs, rhs }, "*");

    std::weak_ptr<Value> weak_out = out;
    out->_backward = [lhs, rhs, weak_out]() {
        auto out_ptr = weak_out.lock();
        if (out_ptr) {
            lhs->grad += (rhs->data * out_ptr->grad);
            rhs->grad += (lhs->data * out_ptr->grad);
        }
        };
    return out;
}

std::shared_ptr<Value> Value::pow(double exponent) {
    auto self = shared_from_this();
    auto out = std::make_shared<Value>(std::pow(self->data, exponent), std::vector<std::shared_ptr<Value>>{self}, "**" + std::to_string(exponent));

    std::weak_ptr<Value> weak_out = out;
    out->_backward = [self, exponent, weak_out]() {
        auto out_ptr = weak_out.lock();
        if (out_ptr) {
            self->grad += (exponent * std::pow(self->data, exponent - 1.0)) * out_ptr->grad;
        }
        };
    return out;
}

// --------------------------------------------------------------------------
// ACTIVATIONS
// --------------------------------------------------------------------------

std::shared_ptr<Value> Value::relu() {
    auto self = shared_from_this();
    auto out = std::make_shared<Value>(self->data < 0 ? 0.0 : self->data, std::vector<std::shared_ptr<Value>>{ self }, "ReLU");

    std::weak_ptr<Value> weak_out = out;
    out->_backward = [self, weak_out]() {
        auto out_ptr = weak_out.lock();
        if (out_ptr) {
            self->grad += (out_ptr->data > 0 ? 1.0 : 0.0) * out_ptr->grad;
        }
        };
    return out;
}

// Added Tanh to match Micrograd
std::shared_ptr<Value> Value::tanh() {
    auto self = shared_from_this();
    double x = self->data;
    double t = (std::exp(2 * x) - 1) / (std::exp(2 * x) + 1); // tanh formula

    auto out = std::make_shared<Value>(t, std::vector<std::shared_ptr<Value>>{self}, "tanh");

    std::weak_ptr<Value> weak_out = out;
    out->_backward = [self, weak_out]() {
        auto out_ptr = weak_out.lock();
        if (out_ptr) {
            // d/dx tanh(x) = 1 - tanh(x)^2
            double t = out_ptr->data;
            self->grad += (1.0 - t * t) * out_ptr->grad;
        }
        };
    return out;
}

// Added Exp to match Micrograd
std::shared_ptr<Value> Value::exp() {
    auto self = shared_from_this();
    auto out = std::make_shared<Value>(std::exp(self->data), std::vector<std::shared_ptr<Value>>{self}, "exp");

    std::weak_ptr<Value> weak_out = out;
    out->_backward = [self, weak_out]() {
        auto out_ptr = weak_out.lock();
        if (out_ptr) {
            self->grad += out_ptr->data * out_ptr->grad; // d/dx e^x = e^x
        }
        };
    return out;
}

// --------------------------------------------------------------------------
// CONVENIENCE WRAPPERS
// --------------------------------------------------------------------------

std::shared_ptr<Value> Value::add(double rhs) {
    return add(std::make_shared<Value>(rhs));
}

std::shared_ptr<Value> Value::mul(double rhs) {
    return mul(std::make_shared<Value>(rhs));
}

// --------------------------------------------------------------------------
// OPERATORS (FRIENDS)
// --------------------------------------------------------------------------

// Addition
std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs) { return lhs->add(rhs); }
std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& lhs, double rhs) { return lhs->add(rhs); }
std::shared_ptr<Value> operator+(double lhs, const std::shared_ptr<Value>& rhs) { return rhs->add(lhs); }

// Multiplication
std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs) { return lhs->mul(rhs); }
std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& lhs, double rhs) { return lhs->mul(rhs); }
std::shared_ptr<Value> operator*(double lhs, const std::shared_ptr<Value>& rhs) { return rhs->mul(lhs); }

// Negation & Subtraction
std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& rhs) { return rhs->mul(-1.0); }
std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs) { return lhs->add(rhs->mul(-1.0)); }

// Division
std::shared_ptr<Value> operator/(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs) { return lhs->mul(rhs->pow(-1.0)); }

// Print
std::ostream& operator<<(std::ostream& os, const std::shared_ptr<Value>& v) {
    os << "Value(data=" << v->data << ", grad=" << v->grad << ")";
    return os;
}

// --------------------------------------------------------------------------
// ENGINE
// --------------------------------------------------------------------------

void Value::backward() {
    std::vector<std::shared_ptr<Value>> topo;
    std::unordered_set<std::shared_ptr<Value>> visited;

    std::function<void(std::shared_ptr<Value>)> build_topo = [&](std::shared_ptr<Value> v) {
        if (visited.find(v) == visited.end()) {
            visited.insert(v);
            for (const auto& child : v->_prev) {
                build_topo(child);
            }
            topo.push_back(v);
        }
        };

    build_topo(shared_from_this());
    this->grad = 1.0;

    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        (*it)->_backward();
    }
}
void Value::print()
{
    std::cout << "Value(data=" << data << ", grad=" << grad << ", op=\"" << op << "\")" << std::endl;
}