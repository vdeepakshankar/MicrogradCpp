cmake_minimum_required(VERSION 3.10)
project(MicrogradCpp)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add executable
add_executable(micrograd_demo
    main.cpp
    Value.cpp
    nn.cpp
)

# Include directories
target_include_directories(micrograd_demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
