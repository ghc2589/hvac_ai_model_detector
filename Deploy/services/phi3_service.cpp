// // ─────────────────────────────────────────────────────────────
// //  phi3_service.cpp  — Servicio ligero sobre llama.cpp b4570
// // ─────────────────────────────────────────────────────────────
// #include "phi3_service.h"
// #include <iostream>
// #include <vector>
// #include <string>
// #include "llama.h"

// static const char * JSON_GRAMMAR = R"GRAMMAR(
// root        ::= object
// object      ::= "{" members? "}"
// members     ::= pair ("," pair)*
// pair        ::= string ":" value
// value       ::= string | number | "null" | "true" | "false"

// string      ::= "\"" chars "\""
// chars       ::= char*
// char        ::= [^"\\]          # cualquier byte excepto " y \

// number      ::= "-"? int frac? exp?
// int         ::= "0" | [1-9][0-9]*
// frac        ::= "." [0-9]+
// exp         ::= [eE] [+-]? [0-9]+
// )GRAMMAR";



// // ─── Parámetros globales ─────────────────────────────────────
// static constexpr int32_t N_GPU_LAYERS = 99;     // Metal / CUDA
// static constexpr int32_t N_THREADS    = 10;
// static constexpr int32_t N_PREDICT    = 1048;    // antes 1048
// static constexpr int32_t EXTRA_CTX    = 4;      // margen

// PHI3Service::PHI3Service() = default;

// PHI3Service::~PHI3Service() {
//     if (ctx_)   llama_free(ctx_);
//     if (model_) llama_model_free(model_);
// }

// /* ----------------------------------------------------------- */
// bool PHI3Service::initializeLlama(const std::string& model_path) {

//     // 1. cargar modelo
//     llama_model_params mp = llama_model_default_params();
//     mp.n_gpu_layers       = N_GPU_LAYERS;

//     model_ = llama_model_load_from_file(model_path.c_str(), mp);
//     if (!model_) {
//         std::cerr << "ERROR: cannot load model: " << model_path << '\n';
//         return false;
//     }

//     // 2. crear contexto
//     llama_context_params cp = llama_context_default_params();
//     cp.n_ctx     = 2048;     // prompt + 128 ≤ 2k
//     cp.n_batch   = 2048;
//     cp.n_threads = N_THREADS;

//     ctx_ = llama_init_from_model(model_, cp);
//     if (!ctx_) {
//         std::cerr << "ERROR: cannot create llama_context\n";
//         llama_model_free(model_);
//         model_ = nullptr;
//         return false;
//     }

//     std::cout << "[PHI3] initialised  n_ctx=" << cp.n_ctx
//               << "  batch=" << cp.n_batch
//               << "  gpu_layers=" << mp.n_gpu_layers << '\n';
//     return true;
// }

// /* ----------------------------------------------------------- */
// std::string PHI3Service::infer(const std::string& prompt) {

//     if (!ctx_) { std::cerr << "ctx null\n"; return {}; }

//     const llama_vocab * vocab = llama_model_get_vocab(model_);

//     /* 1. tokenizar prompt (con BOS) */
//     int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(),
//                                    nullptr, 0, /*add_bos=*/true, /*special=*/true);
//     if (n_prompt <= 0) { std::cerr << "tokenize error\n"; return {}; }

//     std::vector<llama_token> tok(n_prompt);
//     llama_tokenize(vocab, prompt.c_str(), prompt.size(),
//                    tok.data(), tok.size(), true, true);

//     llama_batch batch = llama_batch_get_one(tok.data(), tok.size());

//         /* 2. stop-token = '}'  */
//         /* 2. stop-token = '}' */
//     llama_token stop_buf[2];
//     int n_stop = llama_tokenize(
//                     vocab,
//                     "}",            // texto
//                     1,              // longitud
//                     stop_buf,
//                     2,              // buffer máx
//                     /*add_special=*/true,
//                     /*parse_special=*/true);

//     if (n_stop <= 0) {
//         std::cerr << "could not tokenize stop token '}'\n";
//         return {};
//     }
//     const llama_token STOP_ID = stop_buf[0];

//     /* 3. bucle de decodificación */
//     int n_pos = 0, n_decode = 0;
//     std::string out;

//     while (n_pos + batch.n_tokens < n_prompt + N_PREDICT) {

//         if (llama_decode(ctx_, batch)) {
//             std::cerr << "llama_decode failed\n";
//             break;
//         }
//         n_pos += batch.n_tokens;

//         /* Cadena de samplers */
//         llama_sampler_chain_params sp = llama_sampler_chain_default_params();
//         sp.no_perf = true;
//         llama_sampler * smpl = llama_sampler_chain_init(sp);

//         // grammar (API antigua: vocab + string + root)
//         llama_sampler_chain_add(
//             smpl,
//             llama_sampler_init_grammar(
//                 vocab,           // 1º
//                 JSON_GRAMMAR,    // 2º
//                 "root"));        // 3º regla inicial

//         // ajustes de temperatura / top-p
//         llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.20f));
//         llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.80f, 1));

//         // opcional: dist sampling con seed por defecto
//         llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

//         llama_token next_id = llama_sampler_sample(smpl, ctx_, -1);
//         llama_sampler_free(smpl);

//         if (next_id == STOP_ID) break;          // cerró llave → fin

//         /* convertir token a texto */
//         char buf[128];
//         int n = llama_token_to_piece(vocab, next_id, buf, sizeof(buf), 0, true);
//         out.append(buf, n);

//         /* re-inyectar token como nuevo batch */
//         batch = llama_batch_get_one(&next_id, 1);
//         n_decode++;
//     }

//     std::cout << "\n[PHI3] decoded " << n_decode
//               << " tokens, got " << out.size() << " chars\n";

//     return out;
// }
