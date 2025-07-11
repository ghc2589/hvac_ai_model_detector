#include "prediction_controller.h"
#include <crow/multipart.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

std::shared_ptr<AIService> PredictionController::ai_service_ = nullptr;

/* ────────────────────────────────────────────────────────── */
/*  INIT / CLEANUP                                           */
/* ────────────────────────────────────────────────────────── */
bool PredictionController::initialize(const std::string& hvac,
                                      const std::string& det,
                                      const std::string& rec,
                                      const std::string& phi3)
{
    ai_service_ = std::make_shared<AIService>();
    if (!ai_service_->initialize(hvac, det, rec, phi3)) {
        CROW_LOG_ERROR << "AI service init failed";
        return false;
    }
    CROW_LOG_INFO << "PredictionController ready";
    return true;
}

void PredictionController::cleanup() { ai_service_.reset(); }

/* ────────────────────────────────────────────────────────── */
/*  ROUTES                                                   */
/* ────────────────────────────────────────────────────────── */
void PredictionController::setupRoutes(crow::SimpleApp& app)
{
    CROW_ROUTE(app, "/predict").methods("POST"_method)([](const crow::request& r){
        return predict(r);
    });
    CROW_ROUTE(app, "/model/info").methods("GET"_method)([](){
        return getModelInfo();
    });
}

crow::response PredictionController::predict(const crow::request& req)
{
    rapidjson::Document doc; doc.SetObject();
    auto& a = doc.GetAllocator();

    /* sanity check */
    if (!ai_service_) {
        doc.AddMember("error", "AI service not initialised", a);
        crow::response res(500, stringify(doc));
        res.add_header("Content-Type", "application/json");
        return res;
    }

    /* multipart parsing */
    crow::multipart::message msg(req);
    if (msg.parts.empty() || msg.parts[0].body.empty()) {
        doc.AddMember("error", "image not provided", a);
        crow::response res(400, stringify(doc));
        res.add_header("Content-Type", "application/json");
        return res;
    }

    const auto& img = msg.parts[0].body;
    rapidjson::Document pr = ai_service_->processImage(
        reinterpret_cast<const unsigned char*>(img.data()), img.size());

    /* genera el JSON de respuesta en 'doc' … (omito por brevedad) */
    /* ... */

    crow::response res(200, stringify(pr));
    res.add_header("Content-Type", "application/json");
    return res;
}


/* ────────────────────────────────────────────────────────── */
/*  /model/info (unchanged)                                   */
/* ────────────────────────────────────────────────────────── */
crow::response PredictionController::getModelInfo()
{
    rapidjson::Document d; d.SetObject();
    auto& a = d.GetAllocator();

    if (!ai_service_) {
        d.AddMember("error", "AI service not initialised", a);
        return crow::response(500, stringify(d));
    }

    d.AddMember("model_loaded", true, a);
    d.AddMember("model_type",   "HVAC specification extractor", a);

    rapidjson::Value arr(rapidjson::kArrayType);
    arr.PushBack("image/jpeg", a).PushBack("image/png", a);
    d.AddMember("supported_formats", arr, a);

    return crow::response(200, stringify(d));
}

/* ────────────────────────────────────────────────────────── */
/*  helper: stringify rapidjson::Document                    */
/* ────────────────────────────────────────────────────────── */
std::string PredictionController::stringify(const rapidjson::Document& d)
{
    rapidjson::StringBuffer buf;
    rapidjson::Writer<rapidjson::StringBuffer> wr(buf);
    d.Accept(wr);
    return { buf.GetString(), buf.GetSize() };
}
