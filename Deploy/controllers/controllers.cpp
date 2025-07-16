// controllers.cpp  – /predict
#include <crow.h>
#include "ai_service.h"
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

crow::response Controllers::predict(const crow::request& req)
{
    rapidjson::Document resp;
    resp.SetObject();
    auto& alloc = resp.GetAllocator();

    if (!ai_service_) {
        resp.AddMember("error", "AI service not initialized", alloc);
        rapidjson::StringBuffer buf;
        rapidjson::Writer<rapidjson::StringBuffer> w(buf);
        resp.Accept(w);
        return crow::response(500, buf.GetString())
               .set_header("Content-Type", "application/json");
    }

    try {
        // ─── 1. multipart -------------------------------------------------
        crow::multipart::message msg(req);
        if (msg.parts.empty()) {
            resp.AddMember("error", "No image file provided", alloc);
            rapidjson::StringBuffer buf;
            rapidjson::Writer<rapidjson::StringBuffer> w(buf);
            resp.Accept(w);
            return crow::response(400, buf.GetString())
                   .set_header("Content-Type", "application/json");
        }

        const auto& part = msg.parts[0];
        if (part.body.empty()) {
            resp.AddMember("error", "Empty image file", alloc);
            rapidjson::StringBuffer buf;
            rapidjson::Writer<rapidjson::StringBuffer> w(buf);
            resp.Accept(w);
            return crow::response(400, buf.GetString())
                   .set_header("Content-Type", "application/json");
        }

        rapidjson::Document result =
            ai_service_->processImage(
                reinterpret_cast<const unsigned char*>(part.body.data()),
                part.body.size());

        rapidjson::Value spec;
        spec.CopyFrom(result, alloc);
        resp.AddMember("specifications", spec, alloc);

        rapidjson::StringBuffer buf;
        rapidjson::Writer<rapidjson::StringBuffer> w(buf);
        resp.Accept(w);

        return crow::response(200, buf.GetString())
               .set_header("Content-Type", "application/json");

    } catch (const std::exception& ex) {
        resp.AddMember("error", rapidjson::Value(ex.what(), alloc), alloc);
        rapidjson::StringBuffer buf;
        rapidjson::Writer<rapidjson::StringBuffer> w(buf);
        resp.Accept(w);
        return crow::response(500, buf.GetString())
               .set_header("Content-Type", "application/json");
    }
}
