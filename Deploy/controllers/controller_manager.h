#pragma once

#include <crow.h>
#include <string>

class ControllerManager {
public:
    static bool initializeAll(const std::string& hvac_model_path, const std::string& det_model_path, const std::string& rec_model_path, const std::string& phi3_service);
    static void setupAllRoutes(crow::SimpleApp& app);
    static void cleanupAll();
    
private:
    static bool services_initialized_;
};
