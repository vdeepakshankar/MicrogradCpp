cmake_minimum_required(VERSION 3.10)
project(MicrogradCpp)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add executable for demo
add_executable(micrograd_demo
    main.cpp
    Value.cpp
    nn.cpp
)

# Add executable for tests
add_executable(micrograd_test
    test.cpp
    Value.cpp
)

# Include directories
target_include_directories(micrograd_demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(micrograd_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
