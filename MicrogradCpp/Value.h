#pragma once // vital for headers
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <unordered_set>
#include <cmath> 

struct Value : public std::enable_shared_from_this<Value> {
    double data;
    double grad;
    std::unordered_set<std::shared_ptr<Value>> _prev;
    std::string op;
    std::function<void()> _backward;

    Value(double data, std::vector<std::shared_ptr<Value>> children = {}, std::string _op = "");

    // Core Operations
    std::shared_ptr<Value> add(std::shared_ptr<Value> rhs);
    std::shared_ptr<Value> mul(std::shared_ptr<Value> rhs);
    std::shared_ptr<Value> pow(double exponent);

    // Activations & Non-linearities
    std::shared_ptr<Value> relu();
    std::shared_ptr<Value> tanh(); // Added to match Micrograd
    std::shared_ptr<Value> exp();  // Added to match Micrograd

    // Convenience Wrappers
    std::shared_ptr<Value> add(double rhs);
    std::shared_ptr<Value> mul(double rhs);

    // Engine
    void backward();

    // Friend Operators
    friend std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs);
    friend std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& lhs, double rhs);
    friend std::shared_ptr<Value> operator+(double lhs, const std::shared_ptr<Value>& rhs);

    friend std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs);
    friend std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& lhs, double rhs);
    friend std::shared_ptr<Value> operator*(double lhs, const std::shared_ptr<Value>& rhs);

    friend std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& rhs);
    friend std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs);
    friend std::shared_ptr<Value> operator/(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs);

    friend std::ostream& operator<<(std::ostream& os, const std::shared_ptr<Value>& v);
    void print();
};