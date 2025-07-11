#include "health_controller.h"
#include <ctime>

void HealthController::setupRoutes(crow::SimpleApp& app) {
    CROW_ROUTE(app, "/health")([](){
        return health();
    });
}

crow::response HealthController::health() {
    rapidjson::Document doc;
    doc.SetObject();
    rapidjson::Document::AllocatorType& allocator = doc.GetAllocator();
    
    doc.AddMember("status", "healthy", allocator);
    doc.AddMember("service", "HVAC AI Model Detector", allocator);
    doc.AddMember("timestamp", static_cast<int64_t>(std::time(nullptr)), allocator);
    
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    doc.Accept(writer);
    
    crow::response res(200, buffer.GetString());
    res.add_header("Content-Type", "application/json");
    return res;
}
