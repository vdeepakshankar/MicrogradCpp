#ifndef VALUE_H
#define VALUE_H

#include <memory>
#include <vector>
#include <functional>
#include <cmath>
#include <string>
#include <set>

class Value : public std::enable_shared_from_this<Value> {
public:
    double data;
    double grad;
    std::function<void()> _backward;
    std::vector<std::shared_ptr<Value>> _prev;
    std::string _op;
    std::string label;

    Value(double data, const std::vector<std::shared_ptr<Value>>& children = {}, const std::string& op = "")
        : data(data), grad(0.0), _prev(children), _op(op), label("") {
        _backward = []() {};
    }

    // Addition
    std::shared_ptr<Value> operator+(std::shared_ptr<Value> other);
    std::shared_ptr<Value> operator+(double other);
    
    // Multiplication
    std::shared_ptr<Value> operator*(std::shared_ptr<Value> other);
    std::shared_ptr<Value> operator*(double other);
    
    // Power
    std::shared_ptr<Value> pow(double n);
    
    // Negation
    std::shared_ptr<Value> operator-();
    
    // Subtraction
    std::shared_ptr<Value> operator-(std::shared_ptr<Value> other);
    std::shared_ptr<Value> operator-(double other);
    
    // Division
    std::shared_ptr<Value> operator/(std::shared_ptr<Value> other);
    std::shared_ptr<Value> operator/(double other);
    
    // Activation functions
    std::shared_ptr<Value> tanh();
    std::shared_ptr<Value> relu();
    
    // Backpropagation
    void backward();
    
private:
    void build_topo(std::shared_ptr<Value> v, std::vector<std::shared_ptr<Value>>& topo, std::set<std::shared_ptr<Value>>& visited);
};

// Helper function to create shared_ptr<Value>
inline std::shared_ptr<Value> make_value(double data) {
    return std::make_shared<Value>(data);
}

#endif // VALUE_H
