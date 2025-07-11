#pragma once

#include <crow.h>
#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>
#include "../Services/ai_service.h"
#include <memory>

class Controllers {
public:
    Controllers();
    static void setupRoutes(crow::SimpleApp& app);
    static bool initializeServices();
    
private:
    static std::shared_ptr<AIService> ai_service_;
    
    static crow::response health();
    static crow::response predict(const crow::request& req);
};
