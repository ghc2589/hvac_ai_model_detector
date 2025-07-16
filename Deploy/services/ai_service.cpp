#include "ai_service.h"

#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <regex>
#include <string>
#include <chrono>
#include <ctime>
#include <thread>
#include <future>
#include <mutex>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <opencv2/opencv.hpp>
#include <rapidjson/document.h>
#include <rapidjson/error/en.h>
#include <stdexcept>

AIService::AIService() {
}

AIService::~AIService() = default;

bool AIService::initialize(const std::string& hvac_model_path,
                           const std::string& det_model_path,
                           const std::string& rec_model_path,
                           const std::string& phi3_model_path) {
    try {
        ort_env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "HVAC_AI_Service");
        session_options_ = std::make_unique<Ort::SessionOptions>();
        session_options_->SetIntraOpNumThreads(10);
        session_options_->SetExecutionMode(ORT_PARALLEL);
        session_options_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);


        ort_session_ = std::make_unique<Ort::Session>(*ort_env_, hvac_model_path.c_str(), *session_options_);
        memory_info_ = std::make_unique<Ort::MemoryInfo>(
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));

        if (!initializePaddleOCRModels(det_model_path, rec_model_path)) {
            std::cerr << "Failed to initialize PaddleOCR models" << std::endl;
            return false;
        }

        if (!loadModelMetadata()) {
            std::cerr << "Failed to load model metadata" << std::endl;
            return false;
        }

        std::cout << "AI Service initialized successfully" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[AIService][ERROR] Exception: " << e.what() << std::endl;
        return false;
    }
}


bool AIService::loadModelMetadata() {
    try {
        // Get model metadata
        Ort::ModelMetadata metadata = ort_session_->GetModelMetadata();
        
        // Try to get custom metadata (vocab, idx2model, spec_lookup)
        try {
            Ort::AllocatorWithDefaultOptions allocator;
            
            // Load vocab
            auto vocab_value = metadata.LookupCustomMetadataMapAllocated("vocab_json", allocator);
            if (vocab_value) {
                rapidjson::Document vocab_doc;
                vocab_doc.Parse(vocab_value.get());
                for (auto& member : vocab_doc.GetObject()) {
                    vocab_[member.name.GetString()] = member.value.GetInt();
                }
                std::cout << "Loaded vocab with " << vocab_.size() << " entries" << std::endl;
            }
            
            // Load idx2model mapping
            auto idx2model_value = metadata.LookupCustomMetadataMapAllocated("idx2model_json", allocator);
            if (idx2model_value) {
                rapidjson::Document idx2model_doc;
                idx2model_doc.Parse(idx2model_value.get());
                for (auto& member : idx2model_doc.GetObject()) {
                    idx2model_[std::stoi(member.name.GetString())] = member.value.GetString();
                }
                std::cout << "Loaded idx2model with " << idx2model_.size() << " entries" << std::endl;
            }
            
            // Load spec lookup
            auto spec_value = metadata.LookupCustomMetadataMapAllocated("spec_lookup_json", allocator);
            if (spec_value) {
                spec_lookup_.Parse(spec_value.get());
                std::cout << "Loaded spec lookup" << std::endl;
            }
            
        } catch (const std::exception& e) {
            std::cerr << "Could not load metadata from model, using defaults: " << e.what() << std::endl;
        }
        
        // If metadata loading failed, use basic defaults
        if (vocab_.empty()) {
            // Basic character set
            for (char c = 'A'; c <= 'Z'; ++c) {
                vocab_[std::string(1, c)] = c - 'A' + 1;
            }
            for (char c = '0'; c <= '9'; ++c) {
                vocab_[std::string(1, c)] = c - '0' + 27;
            }
            vocab_["-"] = 37;
            vocab_[" "] = 38;
            vocab_["<PAD>"] = 0;
            
            std::cout << "Using default vocab with " << vocab_.size() << " entries" << std::endl;
        }
        
        if (idx2model_.empty()) {
            idx2model_[0] = "GPG1461080M41";
            idx2model_[1] = "GPG1361080M41";
            idx2model_[2] = "GPG1261080M41";
            
            std::cout << "Using default idx2model with " << idx2model_.size() << " entries" << std::endl;
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error loading model metadata: " << e.what() << std::endl;
        return false;
    }
}

bool AIService::initializePaddleOCRModels(const std::string& det_model_path, const std::string& rec_model_path) {
    try {
        // Initialize detection model
        det_session_ = std::make_unique<Ort::Session>(*ort_env_, det_model_path.c_str(), *session_options_);
        std::cout << "Detection model loaded successfully: " << det_model_path << std::endl;
        
        // Initialize recognition model
        rec_session_ = std::make_unique<Ort::Session>(*ort_env_, rec_model_path.c_str(), *session_options_);
        std::cout << "Recognition model loaded successfully: " << rec_model_path << std::endl;
        Ort::AllocatorWithDefaultOptions allocator;
        auto rec_input_name_ptr = rec_session_->GetInputNameAllocated(0, allocator);
        auto rec_output_name_ptr = rec_session_->GetOutputNameAllocated(0, allocator);

        rec_input_name_ = rec_input_name_ptr.get();
        rec_output_name_ = rec_output_name_ptr.get();
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error initializing PaddleOCR models: " << e.what() << std::endl;
        return false;
    }
}

rapidjson::Document AIService::processImage(const unsigned char* image_data, size_t data_size) {
    rapidjson::Document errorDoc;
    errorDoc.SetObject();
    rapidjson::Document::AllocatorType& allocator = errorDoc.GetAllocator();

    std::cout << "Processing image of size: " << data_size << " bytes" << std::endl;

    if (!image_data || data_size == 0) {
        errorDoc.AddMember("status", "error", allocator);
        errorDoc.AddMember("error", "Invalid image data", allocator);
        return errorDoc;
    }
    try {
        // Decode image for OCR
        std::vector<uchar> image_buffer(image_data, image_data + data_size);
        cv::Mat original_image = cv::imdecode(image_buffer, cv::IMREAD_COLOR);
        if (original_image.empty()) {
            errorDoc.AddMember("status", "error", allocator);
            errorDoc.AddMember("error", "Failed to decode image", allocator);
            return errorDoc;
        }
        // Extract text and structured data using OCR
        auto result = extractTextAndStructuredData(image_data, data_size);
        return result;
    } catch (const std::exception& e) {
        std::cerr << "Exception in processImage: " << e.what() << std::endl;
        errorDoc.AddMember("status", "error", allocator);
        errorDoc.AddMember("error", rapidjson::Value(e.what(), allocator), allocator);
        return errorDoc;
    }
}

std::vector<std::string> AIService::recognizeTextBatch(
    const cv::Mat& image,
    const std::vector<std::vector<cv::Point2f>>& all_corners
) {
    std::vector<cv::Mat> processed_crops;

    int target_height = 48;
    int max_width = 0;

    // 1. Preprocess all crops
    for (const auto& corners : all_corners) {
        cv::Mat crop = straightenTextRegion(image, corners);
        if (crop.empty() || crop.cols < 5 || crop.rows < 5) {
            processed_crops.push_back(cv::Mat());
            continue;
        }

        int target_width = static_cast<int>(crop.cols * target_height / static_cast<float>(crop.rows));
        target_width = std::max(target_width, 10);
        max_width = std::max(max_width, target_width);

        // Preprocesamiento
        cv::Mat resized;
        cv::resize(crop, resized, cv::Size(target_width, target_height));
        processed_crops.push_back(resized);
    }

    // 2. Create input batch Tensor for recognition
    int batch_size = processed_crops.size();
    int channels = 3;
    std::vector<int64_t> input_shape = {batch_size, channels, target_height, max_width};
    std::vector<float> input_tensor_values(batch_size * channels * target_height * max_width, 0.0f);  // padding

    for (int b = 0; b < batch_size; ++b) {
        const cv::Mat& img = processed_crops[b];
        if (img.empty()) continue;

        cv::Mat padded;
        cv::Mat processed = preprocessImageForRecognition(img);

        if (processed.cols < max_width) {
            int pad = max_width - processed.cols;
            cv::copyMakeBorder(processed, padded, 0, 0, 0, pad, cv::BORDER_CONSTANT, cv::Scalar(0));
        } else {
            padded = processed;
        }

        for (int c = 0; c < channels; ++c) {
            for (int h = 0; h < target_height; ++h) {
                for (int w = 0; w < max_width; ++w) {
                    input_tensor_values[b * channels * target_height * max_width +
                                        c * target_height * max_width +
                                        h * max_width + w] = padded.at<cv::Vec3f>(h, w)[c];
                }
            }
        }
    }

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        *memory_info_,
        input_tensor_values.data(),
        input_tensor_values.size(),
        input_shape.data(),
        input_shape.size()
    );

    const char* input_names[] = {rec_input_name_.c_str()};
    const char* output_names[] = {rec_output_name_.c_str()};

    auto output_tensors = rec_session_->Run(
        Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1
    );

    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

    int time_steps = static_cast<int>(output_shape[1]);
    int num_classes = static_cast<int>(output_shape[2]);

    std::vector<std::string> results(batch_size);

    // === Paralel decoding ===
    int decode_threads = std::min<int>(std::thread::hardware_concurrency(), std::thread::hardware_concurrency());
    int step = (batch_size + decode_threads - 1) / decode_threads;

    std::vector<std::future<void>> futures;
    for (int t = 0; t < decode_threads; ++t) {
        int start = t * step;
        int end = std::min(start + step, batch_size);
        if (start >= end) break;

        futures.emplace_back(std::async(std::launch::async, [&, start, end]() {
            for (int b = start; b < end; ++b) {
                float* single_data = output_data + b * time_steps * num_classes;
                std::vector<int64_t> shape = {1, time_steps, num_classes};
                results[b] = decodeRecognitionOutput(single_data, shape);
            }
        }));
    }

    for (auto& fut : futures) fut.get();

    return results;
}



rapidjson::Document AIService::extractTextAndStructuredData(const unsigned char* image_data,
                                                            size_t data_size) {
    std::unordered_map<std::string, std::string> out_map;

    try {
        auto t0 = std::chrono::high_resolution_clock::now();

        // 1. Decode Image
        std::vector<uchar> image_buffer(image_data, image_data + data_size);
        cv::Mat img = cv::imdecode(image_buffer, cv::IMREAD_COLOR);
        if (img.empty()) {
            rapidjson::Document errorDoc;
            errorDoc.SetObject();
            errorDoc.AddMember("status", "error", errorDoc.GetAllocator());
            errorDoc.AddMember("error", "Failed to decode image", errorDoc.GetAllocator());
            return errorDoc;
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double preprocess_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "[TIMER] Preprocesamiento: " << preprocess_ms << " ms" << std::endl;

        // 2. OCR pipeline
        auto t2 = std::chrono::high_resolution_clock::now();
        auto boxes = detectTextBoxes(img);
        auto t3 = std::chrono::high_resolution_clock::now();
        double detection_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
        std::cout << "[TIMER] Detección OCR: " << detection_ms << " ms" << std::endl;

        auto t4 = std::chrono::high_resolution_clock::now();

        std::vector<std::vector<cv::Point2f>> all_corners;
        all_corners.reserve(boxes.size());
        for (const auto& box : boxes) {
            all_corners.push_back(box.corners);
        }

        std::vector<std::future<std::vector<std::string>>> futures;
        int batch_size = all_corners.size();
        int num_threads = std::thread::hardware_concurrency();  // auto-ajustable
        num_threads = std::min(num_threads, 8);  // evita oversaturar
        int step = (batch_size + num_threads - 1) / num_threads;

        for (int i = 0; i < batch_size; i += step) {
            int end = std::min(i + step, batch_size);
            futures.emplace_back(std::async(std::launch::async, [&, i, end]() {
                std::vector<std::vector<cv::Point2f>> sub_corners(all_corners.begin() + i, all_corners.begin() + end);
                return recognizeTextBatch(img, sub_corners);
            }));
        }

        std::vector<std::string> all_texts;
        for (auto& fut : futures) {
            auto part = fut.get();
            all_texts.insert(all_texts.end(), part.begin(), part.end());
        }

        for (size_t i = 0; i < boxes.size(); ++i) {
            boxes[i].text = all_texts[i];
        }

        auto t5 = std::chrono::high_resolution_clock::now();
        double recognition_ms = std::chrono::duration<double, std::milli>(t5 - t4).count();
        std::cout << "[TIMER] Reconocimiento OCR (batch paralelo): " << recognition_ms << " ms" << std::endl;



        // Extract sorted data from OCR results
        std::string ocr_text = extractSortedByXYCoordinatesData(boxes, img.cols, img.rows);
        std::cout << "ocr text: " << ocr_text << std::endl;

        // 4. Create Output JSON
        rapidjson::Document json;
        json.SetObject();
        rapidjson::Document::AllocatorType& allocator = json.GetAllocator();

        std::vector<std::string> schema_fields = {
            "brand", "model_code", "serial_number", "refrigerant_type", "factory_charge_lb", "refrigerant_charge_lb",
            "voltage", "phase", "frequency_hz", "rla_a", "fla_a", "high_side_psi", "low_side_psi", "test_pressure_psi",
            "heating_input_btu", "heating_output_btu", "heating_efficiency_pct", "gas_type", "gas_supply_min_inwc",
            "gas_supply_max_inwc", "manifold_pressure_inwc", "gas_input_min_btu", "gas_input_max_btu", "gas_output_cap_btu",
            "gas_supply_inwc", "air_temp_rise_f", "max_ext_static_inwc", "cooling_capacity_btu", "ieer", "eer",
            "compressor_quantity", "compressor_type", "compressor_hz", "min_ambient_f", "max_ambient_f", "max_air_temp_f",
            "installation_type"
        };

        std::string model_code = extractModelCode(ocr_text);
        bool use_model = false;
        float best_sim = 0.0f;
        float p_hat = 0.0f;
        int best_idx = -1;
        rapidjson::Value status;

        if (!model_code.empty()) {
            std::vector<int64_t> input_vec = encodeText(model_code);
            std::vector<int64_t> input_shape{1, static_cast<int64_t>(input_vec.size())};
            Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(
                *memory_info_, input_vec.data(), input_vec.size(), input_shape.data(), input_shape.size());
            const char* input_names[] = {"input"};
            const char* output_names[] = {"logits"};

            auto output_tensors = ort_session_->Run(
                Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
            float* logits = output_tensors[0].GetTensorMutableData<float>();
            auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
            int num_classes = static_cast<int>(output_shape[1]);

            // Softmax
            std::vector<float> probs(num_classes);
            float max_logit = logits[0];
            for (int i = 1; i < num_classes; ++i) max_logit = std::max(max_logit, logits[i]);
            float sum = 0.0f;
            for (int i = 0; i < num_classes; ++i) {
                probs[i] = std::exp(logits[i] - max_logit);
                sum += probs[i];
            }
            for (int i = 0; i < num_classes; ++i) probs[i] /= sum;

            // Top-K
            int TOP_K = 10;
            std::vector<int> top_indices(num_classes);
            std::iota(top_indices.begin(), top_indices.end(), 0);
            std::sort(top_indices.begin(), top_indices.end(), [&](int a, int b) {
                return probs[a] > probs[b];
            });

            std::string norm = normalizeText(model_code);
            for (int k = 0; k < std::min(TOP_K, num_classes); ++k) {
                int idx = top_indices[k];
                std::string cand = idx2model_[idx];
                float sim = static_cast<float>(calculateSimilarity(cand, norm));
                if (sim > best_sim) {
                    best_sim = sim;
                    best_idx = idx;
                }
            }

            float SIM_TH = 0.75f;
            float PROB_TH = 0.5f;
            p_hat = (best_idx >= 0) ? probs[best_idx] : 0.0f;

            if (best_sim >= SIM_TH && p_hat >= PROB_TH) {
                use_model = true;
                status.SetString("ok", allocator);
            } else {
                status.SetString("low_confidence", allocator);
            }
        } else {
            status.SetString("no_model_code", allocator);
        }

        if (use_model && best_idx >= 0 && spec_lookup_.IsObject() && spec_lookup_.HasMember(idx2model_[best_idx].c_str())) {
            json.SetObject();
            const rapidjson::Value& spec = spec_lookup_[idx2model_[best_idx].c_str()];
            for (auto it = spec.MemberBegin(); it != spec.MemberEnd(); ++it) {
                rapidjson::Value key(it->name, allocator);
                rapidjson::Value val(it->value, allocator);
                json.AddMember(key, val, allocator);
            }
            json.AddMember("similarity", best_sim, allocator);
            json.AddMember("confidence", p_hat, allocator);
            json.AddMember("status", status, allocator);
        } else {
            json.SetObject();
            auto extracted = extractHVACFieldsFromOCR(ocr_text);
            for (const auto& field : schema_fields) {
                if (extracted.count(field)) {
                    json.AddMember(rapidjson::Value(field.c_str(), allocator), rapidjson::Value(extracted[field].c_str(), allocator), allocator);
                } else {
                    json.AddMember(rapidjson::Value(field.c_str(), allocator), rapidjson::Value().SetNull(), allocator);
                }
            }
            json.AddMember("similarity", best_sim, allocator);
            json.AddMember("confidence", p_hat, allocator);
            json.AddMember("status", status, allocator);
        }

        return json;

    } catch (const std::exception& e) {
        std::cerr << "extractTextAndStructuredData: " << e.what() << '\n';
        rapidjson::Document errorDoc;
        errorDoc.SetObject();
        errorDoc.AddMember("status", "error", errorDoc.GetAllocator());
        errorDoc.AddMember("error", rapidjson::Value(e.what(), errorDoc.GetAllocator()), errorDoc.GetAllocator());
        return errorDoc;
    }
}

std::string AIService::extractModelCode(const std::string& text) {
    std::smatch match;
    std::string best_candidate;

    std::regex model_tag_rgx(
        R"((?:MODEL(?:\s*NAME)?)[\s:!\/\-]+([A-Z]{2,}[A-Z0-9\-\/]{4,}))",
        std::regex::icase
    );
    if (std::regex_search(text, match, model_tag_rgx)) {
        return match.str(1); 
    }

    std::regex embedded_rgx(R"(MODEL(?:\s*NAME)?([A-Z]{2,}[A-Z0-9\-\/]{4,}))", std::regex::icase);
    if (std::regex_search(text, match, embedded_rgx)) {
        std::string full = match.str(0);
        std::regex strip_rgx(R"(MODEL(?:\s*NAME)?)", std::regex::icase);
        return std::regex_replace(full, strip_rgx, "");
    }

    std::regex alt_rgx(R"([A-Z]{2,}[A-Z0-9\-\/]{4,})", std::regex::icase);
    auto words_begin = std::sregex_iterator(text.begin(), text.end(), alt_rgx);
    auto words_end = std::sregex_iterator();

    size_t max_score = 0;
    for (auto it = words_begin; it != words_end; ++it) {
        std::string candidate = it->str();
        std::string lower = candidate;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

        if (lower.find("sn") == 0 || lower.find("s/n") == 0 || 
            lower.find("ipx") == 0 || lower.find("kg") != std::string::npos ||
            lower.find("made") != std::string::npos) continue;

        bool has_letter = false, has_digit = false;
        for (char c : candidate) {
            if (std::isalpha(c)) has_letter = true;
            if (std::isdigit(c)) has_digit = true;
        }

        if (has_letter && has_digit && candidate.length() > max_score) {
            best_candidate = candidate;
            max_score = candidate.length();
        }
    }

    return best_candidate;
}




std::unordered_map<std::string, std::string> AIService::extractHVACFieldsFromOCR(const std::string& text) {
    std::unordered_map<std::string, std::string> fields;

    fields["model_code"] = extractModelCode(text);

    std::regex sn_rgx(R"((S[\./]?N[\s:]*([A-Z0-9\-]{6,})))", std::regex::icase);
    std::smatch sn_match;
    if (std::regex_search(text, sn_match, sn_rgx)) {
        fields["serial_number"] = sn_match.str(2);
    }

    std::regex ref_rgx(R"((R[\- ]?\d{2,3}[A-Z]?))", std::regex::icase);
    std::smatch ref_match;
    if (std::regex_search(text, ref_match, ref_rgx)) {
        fields["refrigerant_type"] = ref_match.str(1);
    }

    std::regex volt_rgx(R"((\d{3}/\d{3}|\d{3}|\d{2,3})\s*V)", std::regex::icase);
    std::smatch volt_match;
    if (std::regex_search(text, volt_match, volt_rgx)) {
        fields["voltage"] = volt_match.str(0);
    }

    std::regex phase_rgx(R"((\d)\s*PH)", std::regex::icase);
    std::smatch phase_match;
    if (std::regex_search(text, phase_match, phase_rgx)) {
        fields["phase"] = phase_match.str(1);
    }

    std::regex freq_rgx(R"((\d{2})\s*HZ)", std::regex::icase);
    std::smatch freq_match;
    if (std::regex_search(text, freq_match, freq_rgx)) {
        fields["frequency_hz"] = freq_match.str(1);
    }

    std::regex charge_rgx(R"((\d{1,2}\.\d)\s*LB)", std::regex::icase);
    std::smatch charge_match;
    if (std::regex_search(text, charge_match, charge_rgx)) {
        fields["factory_charge_lb"] = charge_match.str(1);
    }

    std::regex rla_rgx(R"((RLA[\s:]*([\d\.]+)\s*A))", std::regex::icase);
    std::smatch rla_match;
    if (std::regex_search(text, rla_match, rla_rgx)) {
        fields["rla_a"] = rla_match.str(2);
    }

    std::regex fla_rgx(R"((FLA[\s:]*([\d\.]+)\s*A))", std::regex::icase);
    std::smatch fla_match;
    if (std::regex_search(text, fla_match, fla_rgx)) {
        fields["fla_a"] = fla_match.str(2);
    }

    std::regex brand_rgx(R"((Trane|Carrier|Lennox|York|Rheem|Goodman))", std::regex::icase);
    std::smatch brand_match;
    if (std::regex_search(text, brand_match, brand_rgx)) {
        fields["brand"] = brand_match.str(1);
    }

    return fields;
}


rapidjson::Document AIService::cleanJson(const std::string& raw) const
{
    std::size_t first = raw.find('{');
    if (first == std::string::npos)
        throw std::runtime_error("no '{'");

    // si hay fence ``` al final, cortar antes
    std::size_t fence = raw.rfind("```");
    std::size_t last  = (fence == std::string::npos)
                      ? raw.rfind('}')
                      : raw.rfind('}', fence);

    if (last == std::string::npos || last <= first)
        throw std::runtime_error("no matching '}'");

    std::string j = raw.substr(first, last - first + 1);

    // quitar tabs / CR si molestan
    j.erase(std::remove_if(j.begin(), j.end(),
           [](unsigned char c){ return c == '\t' || c == '\r'; }), j.end());

    rapidjson::Document d;
    d.Parse(j.c_str());
    std:: cout << "JSON cleaned: " << j << std::endl;
    if (d.HasParseError())
        throw std::runtime_error(
            std::string("RapidJSON error: ")
            + rapidjson::GetParseError_En(d.GetParseError()));

    if (!d.IsObject())
        throw std::runtime_error("JSON is not an object");

    return d;
}




std::vector<int64_t> AIService::encodeText(const std::string& text) {
    std::vector<int64_t> encoded(MAX_LEN, 0);
    
    int len = std::min(static_cast<int>(text.length()), MAX_LEN);
    for (int i = 0; i < len; ++i) {
        std::string char_str(1, text[i]);
        auto it = vocab_.find(char_str);
        encoded[i] = (it != vocab_.end()) ? it->second : 0;
    }
    
    return encoded;
}

std::string AIService::normalizeText(const std::string& text) {
    std::string normalized;
    for (char c : text) {
        if (c != ' ' && c != '-') {
            normalized += std::toupper(c);
        }
    }
    return normalized;
}

double AIService::calculateSimilarity(const std::string& a, const std::string& b) {
    // Simple Levenshtein distance-based similarity
    int len_a = a.length();
    int len_b = b.length();
    
    if (len_a == 0 && len_b == 0) return 1.0;
    if (len_a == 0 || len_b == 0) return 0.0;
    
    std::vector<std::vector<int>> dp(len_a + 1, std::vector<int>(len_b + 1, 0));
    
    for (int i = 0; i <= len_a; ++i) dp[i][0] = i;
    for (int j = 0; j <= len_b; ++j) dp[0][j] = j;
    
    for (int i = 1; i <= len_a; ++i) {
        for (int j = 1; j <= len_b; ++j) {
            if (a[i-1] == b[j-1]) {
                dp[i][j] = dp[i-1][j-1];
            } else {
                dp[i][j] = 1 + std::min({dp[i-1][j], dp[i][j-1], dp[i-1][j-1]});
            }
        }
    }
    
    int max_len = std::max(len_a, len_b);
    return 1.0 - static_cast<double>(dp[len_a][len_b]) / max_len;
}



std::vector<TextBox> AIService::detectTextBoxes(const cv::Mat& image) {
    std::vector<TextBox> text_boxes;
    
    try {
        std::cout << "Input image: " << image.cols << "x" << image.rows << " channels: " << image.channels() << std::endl;
        
        // Save original image for debugging
        // std::string original_debug_filename = "debug_original_input.png";
        // cv::imwrite(original_debug_filename, image);
        
        // Preprocess image for detection
        cv::Mat preprocessed = preprocessImageForDetection(image);
        
        // Save preprocessed image for debugging (correct denormalization)
        cv::Mat preprocessed_debug;
        preprocessed.convertTo(preprocessed_debug, CV_8U, 255.0, 0.0); // Convert [0,1] back to [0,255]
        // std::string preprocessed_debug_filename = "debug_preprocessed_input.png";
        // cv::imwrite(preprocessed_debug_filename, preprocessed_debug);
        
        // Create input tensor with actual processed image dimensions
        int actual_height = preprocessed.rows;
        int actual_width = preprocessed.cols;
        std::vector<int64_t> input_shape{1, 3, actual_height, actual_width};
        size_t input_tensor_size = 1 * 3 * actual_height * actual_width;
        
        std::vector<float> input_tensor_values(input_tensor_size);
        
        // Fill input tensor (HWC to CHW conversion)
        for (int c = 0; c < 3; ++c) {
            for (int h = 0; h < actual_height; ++h) {
                for (int w = 0; w < actual_width; ++w) {
                    input_tensor_values[c * actual_height * actual_width + h * actual_width + w] = 
                        preprocessed.at<cv::Vec3f>(h, w)[c];
                }
            }
        }
        
        // Create input tensor
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            *memory_info_, input_tensor_values.data(), input_tensor_size, 
            input_shape.data(), input_shape.size());
        
        // Run inference
        Ort::AllocatorWithDefaultOptions allocator;
        auto input_name = det_session_->GetInputNameAllocated(0, allocator);
        auto output_name = det_session_->GetOutputNameAllocated(0, allocator);
        
        const char* input_names[] = {input_name.get()};
        const char* output_names[] = {output_name.get()};
        
        auto output_tensors = det_session_->Run(
            Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
        
        // Process output
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        
        // Calculate total tensor size
        size_t total_output_size = 1;
        for (auto dim : output_shape) {
            total_output_size *= dim;
        }
        
        // Analyze output data statistics
        float min_val = *std::min_element(output_data, output_data + total_output_size);
        float max_val = *std::max_element(output_data, output_data + total_output_size);
        float sum = std::accumulate(output_data, output_data + total_output_size, 0.0f);
        float mean = sum / total_output_size;
        
        // Calculate standard deviation
        float variance = 0.0f;
        for (size_t i = 0; i < total_output_size; ++i) {
            float diff = output_data[i] - mean;
            variance += diff * diff;
        }
        variance /= total_output_size;
        float std_dev = std::sqrt(variance);
        
        // Calculate adaptive threshold based on output statistics optimized for English text
        // Use English-optimized thresholds for better detection
        float adaptive_threshold = mean + 1.0f * std_dev; // Reduced for better English text detection
        
        // Use DET_THRESHOLD as minimum threshold for English text
        float final_threshold = std::max(adaptive_threshold, DET_THRESHOLD);
        
        // Count values above adaptive threshold
        int above_adaptive_count = 0;
        for (size_t i = 0; i < total_output_size; ++i) {
            if (output_data[i] > final_threshold) {
                above_adaptive_count++;
            }
        }
        
        // Extract text boxes from output
        // PaddleOCR detection output format: [batch, channel, height, width] = [1, 1, 640, 640]
        int output_height = static_cast<int>(output_shape[2]);
        int output_width = static_cast<int>(output_shape[3]);
        
        // Calculate scale factors correctly
        // We need to map from detection_map coordinates to original image coordinates
        // detection_map has dimensions: output_width x output_height
        // preprocessed image has dimensions: actual_width x actual_height
        // original image has dimensions: image.cols x image.rows
        
        // First, scale from detection map to preprocessed image
        float det_to_prep_x = static_cast<float>(actual_width) / output_width;
        float det_to_prep_y = static_cast<float>(actual_height) / output_height;
        
        // Then, scale from preprocessed image to original image
        float prep_to_orig_x = static_cast<float>(image.cols) / actual_width;
        float prep_to_orig_y = static_cast<float>(image.rows) / actual_height;
        
        // Combined scale factors from detection map to original image
        float scale_x = det_to_prep_x * prep_to_orig_x;
        float scale_y = det_to_prep_y * prep_to_orig_y;
        
        // Create detection map properly - need to handle the tensor layout
        // The output tensor is [1, 1, height, width], so we need to skip the batch and channel dimensions
        cv::Mat detection_map(output_height, output_width, CV_32F);
        
        // Copy data from the 4D tensor to the 2D matrix
        // For tensor shape [1, 1, height, width], data is laid out as:
        // output_data[batch * channels * height * width + channel * height * width + h * width + w]
        for (int h = 0; h < output_height; ++h) {
            for (int w = 0; w < output_width; ++w) {
                int tensor_idx = h * output_width + w; // Skip batch=0 and channel=0 dimensions
                detection_map.at<float>(h, w) = output_data[tensor_idx];
            }
        }
        
        // Save detection map for debugging
        cv::Mat detection_map_vis;
        cv::normalize(detection_map, detection_map_vis, 0, 255, cv::NORM_MINMAX, CV_8U);
        // std::string detection_map_filename = "debug_detection_map.png";
        // cv::imwrite(detection_map_filename, detection_map_vis);
        
        cv::Mat binary_map;
        cv::threshold(detection_map, binary_map, final_threshold, 1.0, cv::THRESH_BINARY);
        
        // Count non-zero pixels in binary map
        int non_zero_pixels = cv::countNonZero(binary_map);
        
        // Save binary map for debugging
        cv::Mat binary_map_vis;
        binary_map.convertTo(binary_map_vis, CV_8U, 255);
        // std::string binary_map_filename = "debug_binary_map.png";
        // cv::imwrite(binary_map_filename, binary_map_vis);
        
        // Convert to 8-bit for contour detection
        cv::Mat binary_8u;
        binary_map.convertTo(binary_8u, CV_8U, 255);
        
        // Apply morphological operations to connect text characters
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::Mat morphed;
        cv::morphologyEx(binary_8u, morphed, cv::MORPH_CLOSE, kernel);
        
        // Apply dilation to expand text regions
        cv::Mat dilated;
        cv::dilate(morphed, dilated, kernel, cv::Point(-1, -1), 2);
        
        // Save morphed binary map for debugging
        // std::string morphed_filename = "debug_morphed_map.png";
        // cv::imwrite(morphed_filename, dilated);
        
        // Find contours on the morphed image
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(dilated, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        
        std::cout << "Found " << contours.size() << " contours" << std::endl;
        
        // Convert contours to text boxes with improved accuracy
        for (size_t i = 0; i < contours.size(); ++i) {
            const auto& contour = contours[i];
            
            if (contour.size() >= 4) {
                // Use polygon approximation for better box fitting
                std::vector<cv::Point> approx;
                double epsilon = 0.01 * cv::arcLength(contour, true);
                cv::approxPolyDP(contour, approx, epsilon, true);
                
                // Get bounding rectangle
                cv::Rect rect = cv::boundingRect(contour);
                
                // Skip very small detections - reduced threshold for small text
                if (rect.width < 3 || rect.height < 2) {
                    continue;
                }
                
                TextBox box;
                box.confidence = final_threshold; // Use actual threshold as confidence
                
                // Scale back to original image coordinates with improved precision
                // Add padding based on box size (40% on each side) for better text capture
                float padding_x = rect.width * 0.1f; // 40% of box width
                float padding_y = rect.height * 0.1f; // 40% of box height
                
                // Ensure minimum padding
                padding_x = std::max(padding_x, 5.0f);
                padding_y = std::max(padding_y, 5.0f);
                
                box.corners = {
                    cv::Point2f((rect.x - padding_x) * scale_x, (rect.y - padding_y) * scale_y),
                    cv::Point2f((rect.x + rect.width + padding_x) * scale_x, (rect.y - padding_y) * scale_y),
                    cv::Point2f((rect.x + rect.width + padding_x) * scale_x, (rect.y + rect.height + padding_y) * scale_y),
                    cv::Point2f((rect.x - padding_x) * scale_x, (rect.y + rect.height + padding_y) * scale_y)
                };
                
                // Ensure coordinates are within image bounds
                for (auto& corner : box.corners) {
                    corner.x = std::max(0.0f, std::min(corner.x, static_cast<float>(image.cols - 1)));
                    corner.y = std::max(0.0f, std::min(corner.y, static_cast<float>(image.rows - 1)));
                }
                
                text_boxes.push_back(box);
            }
        }
        
        std::cout << "Detected " << text_boxes.size() << " text boxes" << std::endl;
        
        // Apply histogram equalization to original image for better visualization
        cv::Mat equalized_for_viz;
        if (image.channels() == 3) {
            // For color images, convert to YUV, equalize Y channel, then convert back
            cv::Mat yuv;
            cv::cvtColor(image, yuv, cv::COLOR_BGR2YUV);
            std::vector<cv::Mat> yuv_channels;
            cv::split(yuv, yuv_channels);
            // cv::equalizeHist(yuv_channels[0], yuv_channels[0]); // Equalize Y (luminance) channel
            cv::merge(yuv_channels, yuv);
            cv::cvtColor(yuv, equalized_for_viz, cv::COLOR_YUV2BGR);
        } else {
            // For grayscale images, directly equalize
            cv::equalizeHist(image, equalized_for_viz);
        }
        
        // Visualize detected boxes on the equalized image
        std::string debug_filename = "debug_detected_boxes.png";
        visualizeTextBoxes(equalized_for_viz, text_boxes, debug_filename);
        
    } catch (const std::exception& e) {
        std::cerr << "Error in text detection: " << e.what() << std::endl;
    }
    
    return text_boxes;
}

std::string AIService::recognizeText(const cv::Mat& image, const std::vector<cv::Point2f>& corners) {
    try {
        // First, extract and straighten the text region using perspective transform
        cv::Mat straightened = straightenTextRegion(image, corners);
        if (straightened.empty()) {
            return "";
        }
        
        // Skip very small regions that are likely noise
        if (straightened.cols < 10 || straightened.rows < 5) {
            return "";
        }
        
        // Preprocess for recognition
        cv::Mat preprocessed = preprocessImageForRecognition(straightened);
        
        // Create input tensor for recognition model
        // PaddleOCR recognition expects: [batch, channels, height, width]
        int rec_height = preprocessed.rows;
        int rec_width = preprocessed.cols;
        int channels = preprocessed.channels();
        
        std::vector<int64_t> input_shape{1, channels, rec_height, rec_width};
        size_t input_tensor_size = 1 * channels * rec_height * rec_width;
        
        std::vector<float> input_tensor_values(input_tensor_size);
        
        // Fill input tensor - convert from HWC to CHW format
        if (channels == 1) {
            // Grayscale image (shouldn't happen with current preprocessing)
            for (int h = 0; h < rec_height; ++h) {
                for (int w = 0; w < rec_width; ++w) {
                    input_tensor_values[h * rec_width + w] = preprocessed.at<float>(h, w);
                }
            }
        } else if (channels == 3) {
            // Color image (RGB) - expected format for recognition model
            for (int c = 0; c < channels; ++c) {
                for (int h = 0; h < rec_height; ++h) {
                    for (int w = 0; w < rec_width; ++w) {
                        input_tensor_values[c * rec_height * rec_width + h * rec_width + w] = 
                            preprocessed.at<cv::Vec3f>(h, w)[c];
                    }
                }
            }
        } else {
            std::cerr << "Unsupported number of channels: " << channels << std::endl;
            return "";
        }
        
        // Create input tensor
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            *memory_info_, input_tensor_values.data(), input_tensor_size, 
            input_shape.data(), input_shape.size());
        
        // Run recognition inference
        Ort::AllocatorWithDefaultOptions allocator;
        auto input_name = rec_session_->GetInputNameAllocated(0, allocator);
        auto output_name = rec_session_->GetOutputNameAllocated(0, allocator);
        
        const char* input_names[] = {input_name.get()};
        const char* output_names[] = {output_name.get()};
        
        auto output_tensors = rec_session_->Run(
            Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
        
        // Process recognition output
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        
        // Decode the recognition output to text
        std::string recognized_text = decodeRecognitionOutput(output_data, output_shape);
        
        return recognized_text;
        
    } catch (const std::exception& e) {
        std::cerr << "Error in text recognition: " << e.what() << std::endl;
        return "";
    }
}

cv::Mat AIService::preprocessImageForDetection(const cv::Mat& image) {
    cv::Mat processed;
    
    // Define maximum dimensions for nameplate text quality
    // Increased values to preserve text clarity in detailed nameplates
    const int MAX_WIDTH = 1400;   // Increased for better text resolution
    const int MAX_HEIGHT = 1600;  // Increased for better text resolution  
    const int MIN_DIMENSION = 1200; // Higher minimum to maintain text clarity
    
    // Calculate current aspect ratio
    float aspect_ratio = static_cast<float>(image.cols) / static_cast<float>(image.rows);
    
    // Determine target dimensions
    int target_width, target_height;
    
    // If image is already small enough, don't scale down too much
    if (image.cols <= MAX_WIDTH && image.rows <= MAX_HEIGHT) {
        // Image is already within limits, just round to multiple of 32
        target_width = ((image.cols + 16) / 32) * 32;
        target_height = ((image.rows + 16) / 32) * 32;
    } else {
        // Image is too large, scale it down while preserving aspect ratio
        if (aspect_ratio > 1.0f) {
            // Landscape: limit by width
            target_width = MAX_WIDTH;
            target_height = static_cast<int>(MAX_WIDTH / aspect_ratio);
        } else {
            // Portrait: limit by height
            target_height = MAX_HEIGHT;
            target_width = static_cast<int>(MAX_HEIGHT * aspect_ratio);
        }
        
        // Ensure minimum dimensions for quality
        if (target_width < MIN_DIMENSION) {
            target_width = MIN_DIMENSION;
            target_height = static_cast<int>(MIN_DIMENSION / aspect_ratio);
        }
        if (target_height < MIN_DIMENSION) {
            target_height = MIN_DIMENSION;
            target_width = static_cast<int>(MIN_DIMENSION * aspect_ratio);
        }
        
        // Round to nearest multiple of 32
        target_width = ((target_width + 16) / 32) * 32;
        target_height = ((target_height + 16) / 32) * 32;
    }
    
    // Resize image to target size using high-quality interpolation
    cv::Mat resized;
    if (target_width == image.cols && target_height == image.rows) {
        // No resize needed, just copy
        resized = image.clone();
    } else {
        // Choose interpolation method based on whether we're scaling up or down
        int interpolation = (target_width < image.cols) ? cv::INTER_AREA : cv::INTER_CUBIC;
        cv::resize(image, resized, cv::Size(target_width, target_height), 0, 0, interpolation);
    }
    
    processed = resized;
    
    cv::Mat equalized;
    if (processed.channels() == 3) {
        // For color images, convert to YUV, equalize Y channel, then convert back
        cv::Mat yuv;
        cv::cvtColor(processed, yuv, cv::COLOR_BGR2YUV);
        std::vector<cv::Mat> yuv_channels;
        cv::split(yuv, yuv_channels);
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(/*clipLimit=*/2.0, /*tileGridSize=*/cv::Size(8,8));
        clahe->apply(yuv_channels[0], yuv_channels[0]);
        cv::merge(yuv_channels, yuv);
        cv::cvtColor(yuv, equalized, cv::COLOR_YUV2BGR);
    } else {
        // For grayscale images, directly equalize
        cv::equalizeHist(processed, equalized);
    }
    cv::Mat bilat;
    cv::bilateralFilter(equalized, bilat, /*d=*/9, /*sigmaColor=*/75, /*sigmaSpace=*/75);
    // 2. Unsharp mask para realzar bordes del texto
    cv::Mat gauss, sharp;
    cv::GaussianBlur(bilat, gauss, cv::Size(0,0), /*sigma=*/3);
    cv::addWeighted(bilat, 1.5, gauss, -0.5, 0, sharp);
    cv::Mat tophat;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15,15));
    cv::morphologyEx(sharp, tophat, cv::MORPH_TOPHAT, kernel);

    cv::Mat f;
    tophat.convertTo(f, CV_32F, 1.0/255);
    cv::pow(f, /*gamma=*/0.6, f);
    f.convertTo(tophat, CV_8U, 255);

    cv::Mat binInput;
    if (tophat.channels() == 3) {
        // pasa a gris
        cv::cvtColor(tophat, binInput, cv::COLOR_BGR2GRAY);
    } else {
        binInput = tophat;  // ya es mono
    }
    //  8-bits
    if (binInput.type() != CV_8UC1) {
        binInput.convertTo(binInput, CV_8U, 255.0);
    }

    // 5. Adative thresholding
    cv::Mat bin;
    cv::adaptiveThreshold(
        binInput, bin, 255,
        cv::ADAPTIVE_THRESH_GAUSSIAN_C,
        cv::THRESH_BINARY_INV,
        /*blockSize=*/15, /*C=*/2
    );

    // 6. Morphological operations to clean up noise
    cv::Mat m;
    cv::morphologyEx(bin, m, cv::MORPH_CLOSE,
                    cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3)));
    cv::morphologyEx(m, m, cv::MORPH_OPEN,
                    cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3)));

    // Save equalized image for debugging
    // std::string equalized_debug_filename = "debug_equalized_resized.png";
    // cv::imwrite(equalized_debug_filename, equalized);
    
    processed = equalized;
    
    // Convert to float and normalize to [0, 1]
    processed.convertTo(processed, CV_32F, 1.0/255.0);
    
    return processed;
}

cv::Mat AIService::preprocessImageForRecognition(const cv::Mat& cropped_image) {
    cv::Mat processed;
    
    // Resize to fixed height expected by the recognition model (48), maintaining aspect ratio
    int target_height = 48;
    int target_width = static_cast<int>(cropped_image.cols * target_height / cropped_image.rows);
    
    // Ensure minimum width to avoid too narrow images
    target_width = std::max(target_width, 10);
    
    cv::resize(cropped_image, processed, cv::Size(target_width, target_height), 0, 0, cv::INTER_CUBIC);
    
    // Ensure we have 3 channels (BGR) as expected by the recognition model
    if (processed.channels() == 1) {
        cv::cvtColor(processed, processed, cv::COLOR_GRAY2BGR);
    } else if (processed.channels() == 4) {
        cv::cvtColor(processed, processed, cv::COLOR_BGRA2BGR);
    }
    
    // Apply standard PaddleOCR normalization: [0, 255] -> [0, 1] -> [-1, 1]
    processed.convertTo(processed, CV_32F, 1.0/255.0); // Convert to [0, 1]
    processed = (processed - 0.5) / 0.5; // Convert to [-1, 1]: (x - 0.5) / 0.5
    
    return processed;
}

void AIService::visualizeTextBoxes(const cv::Mat& image, const std::vector<TextBox>& boxes, const std::string& filename) {
    cv::Mat debug_image = image.clone();
    
    // Simple green color for all boxes
    cv::Scalar green_color(0, 255, 0);
    
    // Calculate appropriate line thickness based on image size
    int thickness = std::max(2, static_cast<int>(std::sqrt(image.cols * image.rows) / 500));
    
    std::cout << "Visualizing " << boxes.size() << " text boxes with green rectangles" << std::endl;
    
    // Draw each detected text box as a simple green rectangle
    for (size_t i = 0; i < boxes.size(); ++i) {
        const auto& box = boxes[i];
        
        // Convert corners to cv::Point
        std::vector<cv::Point> points;
        for (const auto& corner : box.corners) {
            points.push_back(cv::Point(static_cast<int>(corner.x), static_cast<int>(corner.y)));
        }
        
        // Draw simple green rectangle outline
        cv::polylines(debug_image, points, true, green_color, thickness);
        
        // Draw simple box number in the center
        cv::Point center((points[0].x + points[2].x) / 2, (points[0].y + points[2].y) / 2);
        std::string box_num = std::to_string(i + 1);
        
        // Simple white text on small background
        cv::Size text_size = cv::getTextSize(box_num, cv::FONT_HERSHEY_SIMPLEX, 0.8, 2, nullptr);
        cv::Point text_pos(center.x - text_size.width / 2, center.y + text_size.height / 2);
        
        // Small white background for number
        cv::rectangle(debug_image, 
                     cv::Point(text_pos.x - 5, text_pos.y - text_size.height - 5),
                     cv::Point(text_pos.x + text_size.width + 5, text_pos.y + 5),
                     cv::Scalar(255, 255, 255), -1);
        
        // Draw box number in green
        cv::putText(debug_image, box_num, text_pos, cv::FONT_HERSHEY_SIMPLEX, 0.8, green_color, 2);
    }
    
    // Save the simple debug image
    try {
        bool saved = cv::imwrite(filename, debug_image);
        if (saved) {
            std::cout << "Simple debug image with " << boxes.size() << " green text boxes saved as: " << filename << std::endl;
        } else {
            std::cerr << "Failed to save debug image: " << filename << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error saving debug image: " << e.what() << std::endl;
    }
}

std::string AIService::decodeRecognitionOutput(float* output_data, const std::vector<int64_t>& output_shape) {
    try {
        // PaddleOCR recognition output format: [batch, time_steps, num_classes]
        // where num_classes includes all possible characters + blank
        
        if (output_shape.size() != 3) {
            std::cerr << "Unexpected output shape dimensions: " << output_shape.size() << std::endl;
            return "";
        }
        
        int batch_size = static_cast<int>(output_shape[0]);
        int time_steps = static_cast<int>(output_shape[1]);
        int num_classes = static_cast<int>(output_shape[2]);
        
        // Create PaddleOCR character dictionary (common English + numbers + symbols)
        std::vector<std::string> character_dict = createPaddleOCRCharDict();
        
        // Adjust dictionary size to match model output
        if (character_dict.size() != num_classes) {
            if (character_dict.size() < num_classes) {
                // Add unknown characters if dictionary is too small
                while (character_dict.size() < num_classes) {
                    character_dict.push_back("?");
                }
            } else {
                // Truncate if dictionary is too large
                character_dict.resize(num_classes);
            }
        }
        
        // Standard CTC decoding following PaddleOCR implementation
        // Based on BaseRecLabelDecode.decode() from PaddleOCR source:
        // 1. Get argmax predictions
        // 2. Remove consecutive duplicates (is_remove_duplicate=True for CTC)  
        // 3. Filter out blank tokens (index 0)
        // 4. Simply concatenate characters - NO special logic for spaces
        
        int blank_index = 0; // Blank is always at index 0 for PaddleOCR CTC
        std::vector<int> raw_predictions;
        
        // Step 1: Get argmax for each time step
        for (int t = 0; t < time_steps; ++t) {
            int max_idx = 0;
            float max_prob = output_data[t * num_classes + 0];
            
            for (int c = 1; c < num_classes; ++c) {
                float prob = output_data[t * num_classes + c];
                if (prob > max_prob) {
                    max_prob = prob;
                    max_idx = c;
                }
            }
            raw_predictions.push_back(max_idx);
        }
        
        // Step 2: Remove consecutive duplicates (CTC standard)
        std::vector<int> deduplicated;
        for (int t = 0; t < time_steps; ++t) {
            if (t == 0 || raw_predictions[t] != raw_predictions[t-1]) {
                deduplicated.push_back(raw_predictions[t]);
            }
        }
        
        // Step 3: Remove blank tokens and convert to text
        std::string decoded_text = "";
        
        for (size_t i = 0; i < deduplicated.size(); ++i) {
            int idx = deduplicated[i];
            
            if (idx == blank_index) {  // Skip blank tokens
                continue;
            }
            
            // Valid character - add to text
            if (idx >= 0 && idx < character_dict.size()) {
                std::string current_char = character_dict[idx];
                decoded_text += current_char;
            } else {
                decoded_text += "?";  // fallback
            }
        }
        
        // Clean up the decoded text
        std::string cleaned_text = cleanRecognizedText(decoded_text);
        
        return cleaned_text;
        
    } catch (const std::exception& e) {
        std::cerr << "Error in decodeRecognitionOutput: " << e.what() << std::endl;
        return "";
    }
}

std::vector<std::string> AIService::createPaddleOCRCharDict() {
    // PaddleOCR character dictionary for rec_ppocrv4_en.onnx model
    // Standard CTC dictionary structure: ["blank"] + characters + [" "] 
    // The model produces index 96 for spaces when it detects them
    std::vector<std::string> char_dict;
    
    // Add blank character at the beginning (index 0) for CTC
    char_dict.push_back("blank");
    
    // Official en_dict.txt character set (indices 1-95)
    std::vector<std::string> characters = {
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        ":", ";", "<", "=", ">","?", "@",
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
        "[", "\\", "]", "^", "_", "`",
        "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
        "{", "|", "}", "~",
        "!", "\"", "#", "$", "%", "&", "'", "(", ")", "*", "+", ",", "-", ".", "/", " ", " "
    };
    
    // Add each character to the dictionary (indices 1-95)
    for (const auto& c : characters) {
        char_dict.push_back(c);
    }
    
    // Add space character at index 96 (where the model predicts spaces)
    char_dict.push_back(" ");
    
    return char_dict;
}

std::string AIService::cleanRecognizedText(const std::string& raw_text) {
    if (raw_text.empty()) {
        return "";
    }
    
    std::string cleaned = raw_text;
    
    // Remove leading and trailing whitespace
    size_t start = cleaned.find_first_not_of(" \t\n\r");
    if (start == std::string::npos) {
        return "";
    }
    size_t end = cleaned.find_last_not_of(" \t\n\r");
    cleaned = cleaned.substr(start, end - start + 1);
    
    // Replace multiple consecutive spaces with single space
    std::regex multiple_spaces(R"(\s+)");
    cleaned = std::regex_replace(cleaned, multiple_spaces, " ");
    
    // Remove obviously wrong single characters that are likely OCR errors
    if (cleaned.length() == 1 && (cleaned[0] == '.' || cleaned[0] == ',' || cleaned[0] == ':' || cleaned[0] == ';')) {
        return "";
    }
    
    // Filter out very short meaningless strings
    if (cleaned.length() < 2) {
        return "";
    }
    
    return cleaned;
}

cv::Mat AIService::straightenTextRegion(const cv::Mat& image, const std::vector<cv::Point2f>& corners) {
    if (corners.size() != 4) {
        std::cout << "    Error: Need exactly 4 corners for perspective transform" << std::endl;
        return cv::Mat();
    }
    
    // Order corners: top-left, top-right, bottom-right, bottom-left
    std::vector<cv::Point2f> ordered_corners = orderCorners(corners);
    
    // Calculate output dimensions
    float width1 = cv::norm(ordered_corners[0] - ordered_corners[1]);
    float width2 = cv::norm(ordered_corners[2] - ordered_corners[3]);
    float height1 = cv::norm(ordered_corners[0] - ordered_corners[3]);
    float height2 = cv::norm(ordered_corners[1] - ordered_corners[2]);
    
    int output_width = static_cast<int>(std::max(width1, width2));
    int output_height = static_cast<int>(std::max(height1, height2));
    
    // Ensure minimum size
    output_width = std::max(output_width, 10);
    output_height = std::max(output_height, 10);
    
    // std::cout << "    Straightening text region: " << output_width << "x" << output_height << std::endl;
    
    // Define destination rectangle (straightened)
    std::vector<cv::Point2f> dst_corners = {
        cv::Point2f(0, 0),                                    // top-left
        cv::Point2f(output_width - 1, 0),                    // top-right
        cv::Point2f(output_width - 1, output_height - 1),    // bottom-right
        cv::Point2f(0, output_height - 1)                    // bottom-left
    };
    
    // Get perspective transform matrix
    cv::Mat transform_matrix = cv::getPerspectiveTransform(ordered_corners, dst_corners);
    
    // Apply perspective transform
    cv::Mat straightened;
    cv::warpPerspective(image, straightened, transform_matrix, cv::Size(output_width, output_height));
    
    return straightened;
}

std::vector<cv::Point2f> AIService::orderCorners(const std::vector<cv::Point2f>& corners) {
    // Order corners as: top-left, top-right, bottom-right, bottom-left
    std::vector<cv::Point2f> ordered(4);
    
    // Find center point
    cv::Point2f center(0, 0);
    for (const auto& corner : corners) {
        center += corner;
    }
    center *= (1.0f / corners.size());
    
    // Sort corners by angle from center
    std::vector<std::pair<float, cv::Point2f>> corner_angles;
    for (const auto& corner : corners) {
        float angle = std::atan2(corner.y - center.y, corner.x - center.x);
        corner_angles.push_back({angle, corner});
    }
    
    // Sort by angle using custom comparator
    std::sort(corner_angles.begin(), corner_angles.end(), 
              [](const std::pair<float, cv::Point2f>& a, const std::pair<float, cv::Point2f>& b) {
                  return a.first < b.first;
              });
    
    // Assign corners: start from top-left and go clockwise
    // Find the corner with smallest x+y (top-left)
    int top_left_idx = 0;
    float min_sum = corner_angles[0].second.x + corner_angles[0].second.y;
    for (int i = 1; i < 4; ++i) {
        float sum = corner_angles[i].second.x + corner_angles[i].second.y;
        if (sum < min_sum) {
            min_sum = sum;
            top_left_idx = i;
        }
    }
    
    // Assign corners starting from top-left going clockwise
    for (int i = 0; i < 4; ++i) {
        ordered[i] = corner_angles[(top_left_idx + i) % 4].second;
    }
    
    return ordered;
}

// Structured data extraction from OCR text boxes
std::string AIService::extractSortedByXYCoordinatesData(const std::vector<TextBox>& boxes, int image_width, int image_height) {
    if (boxes.empty()) return "";

    std::vector<TextBox> sorted_boxes = sortBoxesByPosition(boxes);

    std::vector<std::vector<const TextBox*>> lines;
    const float line_threshold = 30.0f;

    for (const auto& box : sorted_boxes) {
        cv::Point2f center = getBoxCenter(box.corners);
        bool added = false;
        for (auto& line : lines) {
            cv::Point2f line_center = getBoxCenter(line.front()->corners);
            if (std::abs(center.y - line_center.y) < line_threshold) {
                line.push_back(&box);
                added = true;
                break;
            }
        }
        if (!added) {
            lines.push_back({&box});
        }
    }

    std::string result;
    for (const auto& line : lines) {
        std::vector<const TextBox*> line_sorted = line;
        std::sort(line_sorted.begin(), line_sorted.end(), [this](const TextBox* a, const TextBox* b) {
            return getBoxCenter(a->corners).x < getBoxCenter(b->corners).x;
        });
        for (size_t i = 0; i < line_sorted.size(); ++i) {
            result += line_sorted[i]->text;
            if (i + 1 < line_sorted.size()) result += " ";
        }
        result += "\n";
    }
    return result;
}

bool AIService::isSerialNumber(const std::string& text) {
    if (text.empty() || text.length() < 4) return false;
    
    // Serial numbers are typically longer and more complex
    // Look for patterns with mixed alphanumeric characters
    bool has_letter = false;
    bool has_number = false;
    
    for (char c : text) {
        if (std::isalpha(c)) has_letter = true;
        if (std::isdigit(c)) has_number = true;
    }
    
    if (!has_letter || !has_number) return false;
    
    // Serial numbers are usually longer than model numbers
    if (text.length() >= 8) {
        return true;
    }
    
    // Check for common serial number keywords
    std::string lower_text = text;
    std::transform(lower_text.begin(), lower_text.end(), lower_text.begin(), ::tolower);
    
    if (lower_text.find("serial") != std::string::npos ||
        lower_text.find("s/n") != std::string::npos ||
        lower_text.find("ser") != std::string::npos) {
        return true;
    }
    
    return false;
}

cv::Point2f AIService::getBoxCenter(const std::vector<cv::Point2f>& corners) {
    cv::Point2f center(0, 0);
    for (const auto& corner : corners) {
        center += corner;
    }
    center.x /= corners.size();
    center.y /= corners.size();
    return center;
}

std::vector<TextBox> AIService::sortBoxesByPosition(const std::vector<TextBox>& boxes) {
    std::vector<TextBox> sorted_boxes = boxes;
    
    // Sort by Y coordinate first (top to bottom), then by X coordinate (left to right)
    std::sort(sorted_boxes.begin(), sorted_boxes.end(), [this](const TextBox& a, const TextBox& b) {
        cv::Point2f center_a = getBoxCenter(a.corners);
        cv::Point2f center_b = getBoxCenter(b.corners);
        
        // If Y coordinates are close (within 20 pixels), sort by X
        if (std::abs(center_a.y - center_b.y) < 20) {
            return center_a.x < center_b.x;
        }
        return center_a.y < center_b.y;
    });
    
    return sorted_boxes;
}