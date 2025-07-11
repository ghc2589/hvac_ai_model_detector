#include <stdio.h>
#include <iostream>
#include <crow.h>
#include <onnxruntime_cxx_api.h>
#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>
#include "controllers/controller_manager.h"
#include "services/phi3_service.h"

int main(){
    // Inicializar AI services y controladores
    if (!ControllerManager::initializeAll(
        "../AI-Models/hvac_fullspec.onnx",
        "../AI-Models/det_ppocrv3_en.onnx",
        "../AI-Models/rec_ppocrv4_en.onnx",
        "../AI-Models/Phi-3-mini-4k-instruct-q4.gguf"
    )) {
        std::cerr << "Failed to initialize AI services. Exiting." << std::endl;
        return 1;
    }
    
    crow::SimpleApp app;

    CROW_ROUTE(app, "/")([](){
        return "Hello, HVAC AI Model Detector!";
    });
    
    // Setup controllers
    ControllerManager::setupAllRoutes(app);

    std::cout << "HVAC AI Model Detector server starting on port 18080..." << std::endl;
    app.port(18080).multithreaded().run();
    
    return 0;
}