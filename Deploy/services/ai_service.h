#pragma once

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>
#include <string> // Ensure this header is included for std::string
#include <set>
#include <unordered_map>
#include <rapidjson/document.h>
#include "../models/hvac_model.h"
#include "../services/phi3_service.h"
#include <optional>

struct TextBox {
    std::vector<cv::Point2f> corners;
    float confidence;
    std::string text;
};

struct PredictionResult {
    std::string status;
    double confidence;
    double similarity;
    std::unordered_map<std::string, std::string> specifications;
    std::string extracted_text;
    HVACModel model_spec; // Structured specification data
};

class AIService {
public:
    AIService();
    ~AIService();
    
    bool initialize(const std::string& hvac_model_path, const std::string& det_model_path, const std::string& rec_model_path, const std::string& phi3_model_path);
    rapidjson::Document processImage(const unsigned char* image_data, size_t data_size);
    
private:
    // Constants from Python model
    static constexpr int MAX_LEN = 20;
    static constexpr float TEMP = 0.55f;
    static constexpr float PROB_TH = 0.70f;
    static constexpr float SIM_TH = 0.40f;
    static constexpr int TOP_K = 10;
    
    // PaddleOCR constants - optimized for English text detection
    static constexpr int DET_INPUT_SIZE = 640;
    static constexpr float DET_THRESHOLD = 0.3f;      // Detection pixel threshold (default for English)
    static constexpr float DET_BOX_THRESHOLD = 0.6f;   // Box threshold for text regions  
    static constexpr float DET_UNCLIP_RATIO = 1.5f;    // Expansion ratio for text regions
    static constexpr float REC_THRESHOLD = 0.7f;       // Higher threshold for confident English recognition
    
    bool loadModelMetadata();
    bool initializePaddleOCRModels(const std::string& det_model_path, const std::string& rec_model_path);
    
    // Image processing and text extraction
    std::string extractTextFromImage(const unsigned char* image_data, size_t data_size);
    rapidjson::Document extractTextAndStructuredData(const unsigned char* image_data, size_t data_size);
    std::vector<TextBox> detectTextBoxes(const cv::Mat& image);
    std::string recognizeText(const cv::Mat& image, const std::vector<cv::Point2f>& corners);
    cv::Mat preprocessImageForDetection(const cv::Mat& image);
    cv::Mat preprocessImageForRecognition(const cv::Mat& cropped_image);
    cv::Mat cropTextRegion(const cv::Mat& image, const std::vector<cv::Point2f>& corners);
    void visualizeTextBoxes(const cv::Mat& image, const std::vector<TextBox>& boxes, const std::string& filename);
    
    // Text recognition helpers
    std::string decodeRecognitionOutput(float* output_data, const std::vector<int64_t>& output_shape);
    std::vector<std::string> createPaddleOCRCharDict();
    std::string cleanRecognizedText(const std::string& raw_text);
    cv::Mat straightenTextRegion(const cv::Mat& image, const std::vector<cv::Point2f>& corners);
    std::vector<cv::Point2f> orderCorners(const std::vector<cv::Point2f>& corners);
    // Structured data extraction from OCR text boxes
    std::string extractSortedByXYCoordinatesData(const std::vector<TextBox>& boxes, int image_width, int image_height);
    std::string extractModelNumber(const std::vector<TextBox>& boxes);
    std::string extractSerialNumber(const std::vector<TextBox>& boxes);
    std::string extractVoltage(const std::vector<TextBox>& boxes);
    std::string extractFrequency(const std::vector<TextBox>& boxes);
    std::string extractBTU(const std::vector<TextBox>& boxes);
    std::string extractRefrigerant(const std::vector<TextBox>& boxes);
    std::string extractBrand(const std::vector<TextBox>& boxes);
    bool isModelNumber(const std::string& text);
    bool isSerialNumber(const std::string& text);
    bool isVoltage(const std::string& text);
    bool isFrequency(const std::string& text);
    bool isBTU(const std::string& text);
    bool isRefrigerant(const std::string& text);
    bool isBrand(const std::string& text);
    cv::Point2f getBoxCenter(const std::vector<cv::Point2f>& corners);
    std::vector<TextBox> sortBoxesByPosition(const std::vector<TextBox>& boxes);
    
    std::string extractModelFromOCRText(const std::string& ocr_text);
    std::vector<int64_t> encodeText(const std::string& text);
    std::string normalizeText(const std::string& text);
    double calculateSimilarity(const std::string& a, const std::string& b);
    std::vector<float> softmax(const std::vector<float>& logits, float temperature);
    rapidjson::Document predict(const std::string& model_text, const std::string& full_ocr_text);
    HVACModel jsonToHVACModel(const rapidjson::Value& json_obj, const std::string& model_code) const;
    HVACModel createModelFromOCR(const std::string& ocr_text, const std::string& model_code) const;
    
    // ONNX Runtime components for HVAC classification
    std::unique_ptr<Ort::Env> ort_env_;
    std::unique_ptr<Ort::SessionOptions> session_options_;
    std::unique_ptr<Ort::Session> ort_session_;
    std::unique_ptr<Ort::MemoryInfo> memory_info_;
    // PaddleOCR ONNX models
    std::unique_ptr<Ort::Session> det_session_;
    std::unique_ptr<Ort::Session> rec_session_;
    
    // Model metadata loaded from ONNX
    std::unordered_map<std::string, int> vocab_;
    std::unordered_map<int, std::string> idx2model_;
    rapidjson::Document spec_lookup_;
    rapidjson::Document cleanJson(const std::string& raw) const;

    /// Extrae por regex el resto de campos HVAC (serial_number,
    /// refrigerant_type, voltage, phase, frequency_hz, etc.)
    /// para usar como fallback cuando la inferencia ONNX no sea confiable.
    std::string extractModelCode(const std::string& text);
    std::unordered_map<std::string, std::string> extractHVACFieldsFromOCR(const std::string& text);
    std::vector<std::string> recognizeTextBatch(
        const cv::Mat& image,
        const std::vector<std::vector<cv::Point2f>>& all_corners
    );
    // Nombres de entrada/salida del modelo de reconocimiento OCR
    std::string rec_input_name_;
    std::string rec_output_name_;

};