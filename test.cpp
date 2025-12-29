#include "Value.h"
#include <iostream>
#include <cassert>
#include <cmath>

void test_basic_ops() {
    std::cout << "Testing basic operations..." << std::endl;
    
    auto a = make_value(2.0);
    auto b = make_value(3.0);
    
    // Test addition
    auto c = (*a) + b;
    assert(std::abs(c->data - 5.0) < 1e-6);
    
    // Test multiplication
    auto d = (*a) * b;
    assert(std::abs(d->data - 6.0) < 1e-6);
    
    // Test subtraction
    auto e = (*a) - b;
    assert(std::abs(e->data - (-1.0)) < 1e-6);
    
    // Test division
    auto f = (*b) / a;
    assert(std::abs(f->data - 1.5) < 1e-6);
    
    std::cout << "  Addition: " << c->data << " ✓" << std::endl;
    std::cout << "  Multiplication: " << d->data << " ✓" << std::endl;
    std::cout << "  Subtraction: " << e->data << " ✓" << std::endl;
    std::cout << "  Division: " << f->data << " ✓" << std::endl;
}

void test_gradients() {
    std::cout << "\nTesting gradients..." << std::endl;
    
    auto x = make_value(2.0);
    auto y = make_value(3.0);
    
    // z = x * y + x
    auto xy = (*x) * y;
    auto z = (*xy) + x;
    
    z->backward();
    
    // dz/dx = y + 1 = 3 + 1 = 4
    // dz/dy = x = 2
    assert(std::abs(x->grad - 4.0) < 1e-6);
    assert(std::abs(y->grad - 2.0) < 1e-6);
    
    std::cout << "  dz/dx = " << x->grad << " (expected 4.0) ✓" << std::endl;
    std::cout << "  dz/dy = " << y->grad << " (expected 2.0) ✓" << std::endl;
}

void test_tanh() {
    std::cout << "\nTesting tanh activation..." << std::endl;
    
    auto x = make_value(0.5);
    auto y = x->tanh();
    
    double expected = std::tanh(0.5);
    assert(std::abs(y->data - expected) < 1e-6);
    
    y->backward();
    double expected_grad = 1 - expected * expected;
    assert(std::abs(x->grad - expected_grad) < 1e-6);
    
    std::cout << "  tanh(0.5) = " << y->data << " ✓" << std::endl;
    std::cout << "  gradient = " << x->grad << " ✓" << std::endl;
}

void test_relu() {
    std::cout << "\nTesting ReLU activation..." << std::endl;
    
    auto x1 = make_value(2.0);
    auto y1 = x1->relu();
    assert(std::abs(y1->data - 2.0) < 1e-6);
    
    auto x2 = make_value(-2.0);
    auto y2 = x2->relu();
    assert(std::abs(y2->data - 0.0) < 1e-6);
    
    std::cout << "  ReLU(2.0) = " << y1->data << " ✓" << std::endl;
    std::cout << "  ReLU(-2.0) = " << y2->data << " ✓" << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "   MicrogradCpp - Unit Tests" << std::endl;
    std::cout << "========================================" << std::endl << std::endl;
    
    test_basic_ops();
    test_gradients();
    test_tanh();
    test_relu();
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "   All tests passed! ✓" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}
