#pragma once

#include <crow.h>
#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>

class HealthController {
public:
    static void setupRoutes(crow::SimpleApp& app);
    
private:
    static crow::response health();
};
