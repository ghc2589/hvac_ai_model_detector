#include "controller_manager.h"
#include "health_controller.h"
#include "prediction_controller.h"
#include <iostream>

// Static member definition
bool ControllerManager::services_initialized_ = false;

bool ControllerManager::initializeAll(const std::string& hvac_model_path, const std::string& det_model_path, const std::string& rec_model_path, const std::string& llama_model_path) {
    std::cout << "Initializing all controllers..." << std::endl;
    
    // Initialize prediction controller (which requires AI service)
    if (!PredictionController::initialize(hvac_model_path, det_model_path, rec_model_path, llama_model_path)) {
        std::cerr << "Failed to initialize prediction controller" << std::endl;
        return false;
    }
    
    services_initialized_ = true;
    std::cout << "All controllers initialized successfully" << std::endl;
    return true;
}

void ControllerManager::setupAllRoutes(crow::SimpleApp& app) {
    std::cout << "Setting up all routes..." << std::endl;
    
    // Setup routes for all controllers
    HealthController::setupRoutes(app);
    
    if (services_initialized_) {
        PredictionController::setupRoutes(app);
    }
    
    std::cout << "All routes configured" << std::endl;
}

void ControllerManager::cleanupAll() {
    std::cout << "Cleaning up all controllers..." << std::endl;
    
    if (services_initialized_) {
        PredictionController::cleanup();
        services_initialized_ = false;
    }
    
    std::cout << "Cleanup completed" << std::endl;
}
