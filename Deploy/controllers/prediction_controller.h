#pragma once

#include <crow.h>
#include <memory>
#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>
#include "../services/ai_service.h"

class PredictionController {
public:
    static bool initialize(const std::string& hvac_model_path, const std::string& det_model_path, const std::string& rec_model_path, const std::string& llama_model_path);
    static void setupRoutes(crow::SimpleApp& app);
    static void cleanup();
    
private:
    static std::shared_ptr<AIService> ai_service_;
    static crow::response predict(const crow::request& req);
    static crow::response getModelInfo();
    static std::string stringify(const rapidjson::Document& doc);
};
