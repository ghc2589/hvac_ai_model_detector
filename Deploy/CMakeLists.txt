cmake_minimum_required(VERSION 3.15)
project(hvac_server LANGUAGES CXX)

# Include Conan configuration
include(${CMAKE_BINARY_DIR}/conan_toolchain.cmake)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_executable(hvac 
    main.cpp 
    controllers/health_controller.cpp
    controllers/prediction_controller.cpp
    controllers/controller_manager.cpp
    services/ai_service.cpp
    services/phi3_service.cpp
)

find_package(Crow REQUIRED)
find_package(RapidJSON REQUIRED)
find_package(onnxruntime REQUIRED CONFIG)
find_package(OpenCV REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -mtune=native")

target_include_directories(hvac PRIVATE ${rapidjson_INCLUDE_DIRS})
target_link_libraries(hvac PRIVATE
        Crow::Crow
        rapidjson
        onnxruntime::onnxruntime
        opencv::opencv
)
