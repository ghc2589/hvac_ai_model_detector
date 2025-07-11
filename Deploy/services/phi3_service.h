#pragma once
#include <string>
#include <llama.h>
#include "../models/hvac_model.h"

class PHI3Service {
public:
    PHI3Service();
    ~PHI3Service();
    bool initializeLlama(const std::string& phi3_model_path);
    std::string infer(const std::string& prompt);
private:
    llama_model* model_ = nullptr;
    llama_context* ctx_ = nullptr;
    std::string model_path_;
};
