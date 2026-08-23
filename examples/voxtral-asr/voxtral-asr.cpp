// Real-time ASR (speech-to-text) transcription for
// Voxtral-Mini-4B-Realtime-2602 (arch `voxtral_rt_asr`).
//
// Pipeline (Delayed Stream Modeling, DSM):
//   1. load 16 kHz mono PCM16 WAV, pad as in the reference (left 32 tokens x
//      1280 samples, right align + 17 tokens x 1280 samples)
//   2. compute log-mel spectrogram (CPU, matches voxtral.cpp + torch)
//   3. causal audio encoder (32 blocks, RoPE, sliding window 750) + temporal
//      adapter (downsample 4x + GELU MLP) -> per-frame audio embeddings
//   5. decoder: prompt [BOS] + [STREAMING_PAD]*38, then one token per frame;
//      input at position i = token embedding + audio embedding (sum), via
//      llama_batch.embd; greedy argmax; stop at EOS
//
// The causal audio encoder + adapter run through mtmd (mmproj GGUF, clip_ctx);
// the llama.cpp decoder path is reused via llama_batch.embd. See
// docs-ports/voxtral-mini-4b-realtime.md for the architecture analysis.
//
// Usage: llama-voxtral-asr -m voxtral-realtime-asr-2602-bf16.gguf -mv mmproj-voxtral-realtime-asr-2602-bf16.gguf -f input.wav [options]

#include "common.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <cassert>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// constants (must match the GGUF metadata / voxtral.cpp)
// ---------------------------------------------------------------------------

static constexpr int32_t VOXTRAL_SAMPLE_RATE       = 16000;
static constexpr int32_t VOXTRAL_N_LEFT_PAD_TOKENS = 32;
static constexpr int32_t VOXTRAL_N_DELAY_TOKENS    = 6;

// ---------------------------------------------------------------------------
// logging (same style as the rest of llama.cpp tools)
// ---------------------------------------------------------------------------

static bool verbose_logging = false;

static void log_info(const char * fmt, ...) {
    if (!verbose_logging) {
        return;
    }
    va_list args;
    va_start(args, fmt);
    fprintf(stdout, "llama-voxtral-asr: ");
    vfprintf(stdout, fmt, args);
    fprintf(stdout, "\n");
    va_end(args);
}

static void log_error(const char * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "llama-voxtral-asr: error: ");
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
    va_end(args);
}

// ---------------------------------------------------------------------------
// WAV loading (16 kHz mono PCM16 or IEEE float32)
// ---------------------------------------------------------------------------

static bool load_wav_file(const std::string & path, std::vector<float> & audio_out) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        log_error("cannot open WAV file: %s", path.c_str());
        return false;
    }

    char hdr[12];
    if (fread(hdr, 1, 12, f) != 12 || memcmp(hdr, "RIFF", 4) != 0 || memcmp(hdr + 8, "WAVE", 4) != 0) {
        log_error("not a RIFF/WAVE file: %s", path.c_str());
        fclose(f);
        return false;
    }

    uint16_t             audio_format = 0, num_channels = 0, bits_per_sample = 0;
    uint32_t             sample_rate = 0;
    bool                 have_fmt    = false;
    bool                 have_data   = false;
    std::vector<int16_t> pcm;
    std::vector<float>   f32;

    while (true) {
        char chunk[8];
        if (fread(chunk, 1, 8, f) != 8) {
            break;
        }
        uint32_t chunk_size = 0;
        memcpy(&chunk_size, chunk + 4, 4);

        if (memcmp(chunk, "fmt ", 4) == 0) {
            // chunk layout: wFormatTag(2) nChannels(2) nSamplesPerSec(4)
            //               nAvgBytesPerSec(4) nBlockAlign(2) wBitsPerSample(2)
            uint32_t dummy = 0;
            size_t   nfmt  = fread(&audio_format, 2, 1, f);
            nfmt += fread(&num_channels, 2, 1, f);
            nfmt += fread(&sample_rate, 4, 1, f);
            nfmt += fread(&dummy, 4, 1, f);  // nAvgBytesPerSec (unused)
            nfmt += fread(&dummy, 2, 1, f);  // nBlockAlign (unused)
            nfmt += fread(&bits_per_sample, 2, 1, f);
            (void) nfmt;
            if (chunk_size > 16) {
                fseek(f, chunk_size - 16, SEEK_CUR);
            }
            have_fmt = true;
        } else if (memcmp(chunk, "data", 4) == 0) {
            if (audio_format == 1 && bits_per_sample == 16) {
                pcm.resize(chunk_size / 2);
                if (fread(pcm.data(), 2, pcm.size(), f) != pcm.size()) {
                    log_error("truncated WAV data chunk");
                    fclose(f);
                    return false;
                }
            } else if (audio_format == 3 && bits_per_sample == 32) {
                f32.resize(chunk_size / 4);
                if (fread(f32.data(), 4, f32.size(), f) != f32.size()) {
                    log_error("truncated WAV data chunk");
                    fclose(f);
                    return false;
                }
            } else {
                log_error("unsupported WAV format: fmt=%u ch=%u bits=%u (need PCM16 or float32)", audio_format,
                          num_channels, bits_per_sample);
                fclose(f);
                return false;
            }
            have_data = true;
            break;
        } else {
            fseek(f, chunk_size, SEEK_CUR);
        }
    }
    fclose(f);

    if (!have_fmt || !have_data) {
        log_error("WAV missing fmt/data chunks: %s", path.c_str());
        return false;
    }
    if (sample_rate != VOXTRAL_SAMPLE_RATE) {
        log_error("WAV sample rate %u != 16000 (resample first with ffmpeg)", sample_rate);
        return false;
    }
    if (num_channels != 1) {
        log_error("WAV has %u channels (need mono; mix with ffmpeg -ac 1)", num_channels);
        return false;
    }

    if (!pcm.empty()) {
        audio_out.resize(pcm.size());
        for (size_t i = 0; i < pcm.size(); ++i) {
            audio_out[i] = (float) pcm[i] / 32768.0f;
        }
    } else {
        audio_out = f32;
    }

    log_info("loaded WAV: %s, %zu samples (%.2f s)", path.c_str(), audio_out.size(),
             (double) audio_out.size() / VOXTRAL_SAMPLE_RATE);
    return true;
}

// ---------------------------------------------------------------------------
// mel spectrogram (exact replica of voxtral.cpp / torch.stft)
// ---------------------------------------------------------------------------

// PyTorch reflect padding (mirror without repeating the edge sample)
struct gguf_tensor {
    enum ggml_type       type = GGML_TYPE_F32;
    std::vector<int64_t> ne;
    std::vector<float>   f32;
};

static bool load_gguf_tensor_f32(const char * path, const char * name, gguf_tensor & out) {
    gguf_init_params iparams = { /*.no_alloc =*/true, /*.ctx =*/nullptr };
    gguf_context *   ctx     = gguf_init_from_file(path, iparams);
    if (!ctx) {
        log_error("failed to open GGUF: %s", path);
        return false;
    }

    const int n_tensors = gguf_get_n_tensors(ctx);
    int       idx       = -1;
    for (int i = 0; i < n_tensors; ++i) {
        if (strcmp(gguf_get_tensor_name(ctx, i), name) == 0) {
            idx = i;
            break;
        }
    }
    if (idx < 0) {
        log_error("tensor '%s' not found in %s", name, path);
        gguf_free(ctx);
        return false;
    }

    out.type               = gguf_get_tensor_type(ctx, idx);
    const int64_t * ne     = gguf_get_tensor_ne(ctx, idx);
    int             n_dims = 0;
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        if (ne[d] > 1) {
            n_dims = d + 1;
        }
    }
    out.ne.assign(ne, ne + n_dims);
    int64_t n_elements = 1;
    for (int d = 0; d < n_dims; ++d) {
        n_elements *= out.ne[d];
    }
    out.f32.resize(n_elements);

    // read raw bytes at data_offset + tensor_offset
    FILE * f = fopen(path, "rb");
    if (!f) {
        log_error("failed to open GGUF file: %s", path);
        gguf_free(ctx);
        return false;
    }
    const size_t offset = gguf_get_data_offset(ctx) + gguf_get_tensor_offset(ctx, idx);
    fseek(f, (long) offset, SEEK_SET);

    const size_t         row_size = ggml_row_size(out.type, out.ne[0]);
    const size_t         nbytes   = row_size * n_elements / out.ne[0];
    std::vector<uint8_t> raw(nbytes);
    if (fread(raw.data(), 1, nbytes, f) != nbytes) {
        log_error("failed to read tensor data for '%s'", name);
        fclose(f);
        gguf_free(ctx);
        return false;
    }
    fclose(f);
    gguf_free(ctx);

    // convert to F32
    switch (out.type) {
        case GGML_TYPE_F32:
            memcpy(out.f32.data(), raw.data(), nbytes);
            break;
        case GGML_TYPE_BF16:
            ggml_bf16_to_fp32_row((const ggml_bf16_t *) raw.data(), out.f32.data(), out.f32.size());
            break;
        case GGML_TYPE_F16:
            {
                const ggml_fp16_t * src = (const ggml_fp16_t *) raw.data();
                for (size_t i = 0; i < out.f32.size(); ++i) {
                    out.f32[i] = ggml_fp16_to_fp32(src[i]);
                }
            }
            break;
        case GGML_TYPE_Q8_0:
            {
                // self-contained Q8_0 dequant: per block of 32, an f16 scale + 32 int8 quants
                const uint8_t * src    = raw.data();
                const int64_t   ne0    = out.ne[0];
                const int64_t   n_rows = n_elements / ne0;
                for (int64_t r = 0; r < n_rows; ++r) {
                    const uint8_t * row = src + (size_t) r * row_size;
                    for (int64_t i = 0; i < ne0; i += 32) {
                        const uint16_t d_bits = row[0] | (row[1] << 8);
                        const float    d      = ggml_fp16_to_fp32(ggml_fp16_t{ d_bits });
                        const int8_t * qs     = (const int8_t *) (row + 2);
                        for (int k = 0; k < 32 && i + k < ne0; ++k) {
                            out.f32[(size_t) r * ne0 + i + k] = d * qs[k];
                        }
                        row += 34;
                    }
                }
            }
            break;
        case GGML_TYPE_Q4_0:
            {
                // Q4_0: per block of 32, f16 scale + 16 packed 4-bit quants (nibble order: qs[0]=q0,q1)
                const uint8_t * src    = raw.data();
                const int64_t   ne0    = out.ne[0];
                const int64_t   n_rows = n_elements / ne0;
                for (int64_t r = 0; r < n_rows; ++r) {
                    const uint8_t * row = src + (size_t) r * row_size;
                    for (int64_t i = 0; i < ne0; i += 32) {
                        const uint16_t  d_bits = row[0] | (row[1] << 8);
                        const float     d      = ggml_fp16_to_fp32(ggml_fp16_t{ d_bits });
                        const uint8_t * qs     = row + 2;
                        for (int k = 0; k < 32; ++k) {
                            const int8_t q                    = (qs[k / 2] >> (4 * (k & 1))) & 0xF;
                            out.f32[(size_t) r * ne0 + i + k] = d * ((float) q - 8.0f);
                        }
                        row += 18;
                    }
                }
            }
            break;
        default:
            {
                log_error("tensor '%s': type %s not supported by this tool (use bf16/f16/f32/Q8_0/Q4_0)", name,
                          ggml_type_name(out.type));
                return false;
            }
    }
    return true;
}

// ---------------------------------------------------------------------------
// hann window
// ---------------------------------------------------------------------------

static uint32_t load_gguf_uint32(const char * path, const char * key, uint32_t fallback) {
    gguf_init_params iparams = { /*.no_alloc =*/true, /*.ctx =*/nullptr };
    gguf_context *   ctx     = gguf_init_from_file(path, iparams);
    if (!ctx) {
        return fallback;
    }
    uint32_t val = fallback;
    const int idx = gguf_find_key(ctx, key);
    if (idx >= 0) {
        val = (uint32_t) gguf_get_val_u32(ctx, idx);
    }
    gguf_free(ctx);
    return val;
}

static void usage(const char * argv0) {
    fprintf(stderr, "usage: %s -m model.gguf -mv mmproj.gguf -f input.wav [options]\n", argv0);
    fprintf(stderr, "  -m, --model PATH        main GGUF (text decoder, arch voxtral_rt_asr)\n");
    fprintf(stderr, "  -mv, --mmproj PATH      mmproj GGUF (causal audio encoder, arch clip)\n");
    fprintf(stderr, "  -f, --file PATH         16 kHz mono PCM16 WAV\n");
    fprintf(stderr, "  -n, --max-tokens N      max decoder tokens (0 = to end of audio)\n");
    fprintf(stderr, "  -t, --threads N         CPU threads for the encoder/decoder\n");
    fprintf(stderr, "  -ngl, --gpu-layers N    decoder layers on the GPU/iGPU (0 = CPU only; also offloads the audio encoder mmproj when > 0)\n");
    fprintf(stderr, "  -v, --verbose           verbose logging\n");
    fprintf(stderr, "  -h, --help              show this help\n");
}

int main(int argc, char ** argv) {
    ggml_time_init();

    std::string model_path;
    std::string mmproj_path;
    std::string audio_path;
    std::string dump_logits_bin;
    std::string dump_embd;
    std::string dump_tensor;
    int32_t     max_tokens   = 0;
    int         n_threads    = 4;
    int32_t     n_gpu_layers = 0;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-m" || arg == "--model") {
            if (++i >= argc) {
                log_error("missing value for %s", arg.c_str());
                return 1;
            }
            model_path = argv[i];
        } else if (arg == "-mv" || arg == "--mmproj") {
            if (++i >= argc) {
                log_error("missing value for %s", arg.c_str());
                return 1;
            }
            mmproj_path = argv[i];
        } else if (arg == "-f" || arg == "--file" || arg == "--audio") {
            if (++i >= argc) {
                log_error("missing value for %s", arg.c_str());
                return 1;
            }
            audio_path = argv[i];
        } else if (arg == "-n" || arg == "--max-tokens") {
            if (++i >= argc) {
                log_error("missing value for %s", arg.c_str());
                return 1;
            }
            max_tokens = std::atoi(argv[i]);
        } else if (arg == "-ngl" || arg == "--gpu-layers") {
            if (++i >= argc) {
                log_error("missing value for %s", arg.c_str());
                return 1;
            }
            n_gpu_layers = std::atoi(argv[i]);
        } else if (arg == "--dump-logits-bin") {
            if (++i >= argc) {
                log_error("missing value for %s", arg.c_str());
                return 1;
            }
            dump_logits_bin = argv[i];
        } else if (arg == "--dump-embd") {
            if (++i >= argc) {
                log_error("missing value for %s", arg.c_str());
                return 1;
            }
            dump_embd = argv[i];
        } else if (arg == "--dump-tensor") {
            if (++i >= argc) {
                log_error("missing value for %s", arg.c_str());
                return 1;
            }
            dump_tensor = argv[i];
        } else if (arg == "-t" || arg == "--threads") {
            if (++i >= argc) {
                log_error("missing value for %s", arg.c_str());
                return 1;
            }
            n_threads = std::atoi(argv[i]);
        } else if (arg == "-v" || arg == "--verbose") {
            verbose_logging = true;
        } else if (arg == "-h" || arg == "--help") {
            usage(argv[0]);
            return 0;
        } else {
            log_error("unknown argument: %s", arg.c_str());
            return 1;
        }
    }

    if (model_path.empty() || mmproj_path.empty() || audio_path.empty()) {
        log_error("-m (main GGUF), -mv (mmproj GGUF) and -f are required (see --help)");
        return 1;
    }

    // debug helper: dump one tensor as f32 raw (for cross-tool weight comparison)
    if (!dump_tensor.empty()) {
        gguf_tensor t;
        if (!load_gguf_tensor_f32(model_path.c_str(), dump_tensor.c_str(), t)) {
            log_error("failed to load tensor '%s'", dump_tensor.c_str());
            return 1;
        }
        FILE * fd = fopen("/tmp/kilo/dumped_tensor.bin", "wb");
        if (fd) {
            const int32_t n  = (int32_t) t.f32.size();
            const int32_t nd = (int32_t) t.ne.size();
            fwrite(&n, sizeof(int32_t), 1, fd);
            fwrite(&nd, sizeof(int32_t), 1, fd);
            for (int i = 0; i < nd; ++i) {
                int64_t d = t.ne[i];
                fwrite(&d, sizeof(int64_t), 1, fd);
            }
            fwrite(t.f32.data(), sizeof(float), t.f32.size(), fd);
            fclose(fd);
        }
        log_info("dumped tensor '%s' to /tmp/kilo/dumped_tensor.bin", dump_tensor.c_str());
        return 0;
    }

    // ---------------- load the decoder via llama.cpp ----------------
    llama_backend_init();
    llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers       = n_gpu_layers;
    llama_model * model        = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
        log_error("failed to load model: %s", model_path.c_str());
        llama_backend_free();
        return 1;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx                = 16384;
    cparams.n_threads            = n_threads;
    cparams.n_threads_batch      = n_threads;
    cparams.no_perf              = true;
    llama_context * ctx_dec      = llama_init_from_model(model, cparams);
    if (!ctx_dec) {
        log_error("failed to create decoder context");
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int32_t       n_vocab = llama_vocab_n_tokens(vocab);

    // ---------------- load audio ----------------
    std::vector<float> audio;
    if (!load_wav_file(audio_path, audio)) {
        return 1;
    }

    // ---------------- realtime ASR via mtmd (dual-stream) ----------------
    mtmd_context_params mctx_params = mtmd_context_params_default();
    mctx_params.use_gpu   = n_gpu_layers > 0;  // offload the audio encoder (mmproj) to the GPU when the decoder is offloaded (-ngl > 0)
    mctx_params.n_threads = n_threads;
    mtmd_context * mctx   = mtmd_init_from_file(mmproj_path.c_str(), model, mctx_params);
    if (!mctx) {
        log_error("failed to load mmproj %s", mmproj_path.c_str());
        return 1;
    }

    // streaming prompt prefix: [BOS] + [STREAMING_PAD] x (N_LEFT + N_DELAY)
    const llama_token streaming_pad = (llama_token) load_gguf_uint32(
        model_path.c_str(), "voxtral_rt_asr.streaming.streaming_pad_token_id", 32);
    const llama_token bos = llama_vocab_bos(vocab);
    std::vector<llama_token> prompt_ids;
    prompt_ids.push_back(bos);
    for (int32_t i = 0; i < VOXTRAL_N_LEFT_PAD_TOKENS + VOXTRAL_N_DELAY_TOKENS; ++i) {
        prompt_ids.push_back(streaming_pad);
    }

    // audio bitmap with the per-position dual-stream prompt prefix attached
    mtmd_bitmap * bitmap = mtmd_bitmap_init_from_audio(audio.size(), audio.data());
    if (!bitmap) {
        log_error("failed to create audio bitmap");
        mtmd_free(mctx);
        return 1;
    }
    mtmd_bitmap_add_prefix_tokens(bitmap, prompt_ids.data(), prompt_ids.size());

    // tokenize: media marker + audio bitmap -> one audio chunk
    const char * marker = mtmd_default_marker();
    mtmd_input_text input_text = { marker, strlen(marker), true, true };
    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    const mtmd_bitmap * bitmaps[] = { bitmap };
    if (mtmd_tokenize(mctx, chunks, &input_text, bitmaps, 1) != 0) {
        log_error("mtmd_tokenize failed");
        mtmd_bitmap_free(bitmap);
        mtmd_input_chunks_free(chunks);
        mtmd_free(mctx);
        return 1;
    }
    mtmd_bitmap_free(bitmap);

    // find the audio chunk
    const mtmd_input_chunk * chunk = nullptr;
    for (size_t i = 0; i < mtmd_input_chunks_size(chunks); ++i) {
        const mtmd_input_chunk * c = mtmd_input_chunks_get(chunks, i);
        if (mtmd_input_chunk_get_type(c) == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
            chunk = c;
            break;
        }
    }
    if (!chunk) {
        log_error("no audio chunk produced");
        mtmd_input_chunks_free(chunks);
        mtmd_free(mctx);
        return 1;
    }
    log_info("audio chunk: %zu positions, %zu prefix tokens",
             mtmd_input_chunk_get_n_tokens(chunk), prompt_ids.size());

    if (!dump_embd.empty()) {
        if (mtmd_encode_chunk(mctx, chunk) != 0) {
            log_error("mtmd_encode_chunk failed");
        } else {
            float * embd = mtmd_get_output_embd(mctx);
            const size_t n_embd_v = (size_t) mtmd_input_chunk_get_n_tokens(chunk) * (size_t) llama_model_n_embd_inp(model);
            FILE * fd = fopen(dump_embd.c_str(), "wb");
            if (fd) {
                const int32_t n = (int32_t) n_embd_v;
                fwrite(&n, sizeof(int32_t), 1, fd);
                fwrite(embd, sizeof(float), n_embd_v, fd);
                fclose(fd);
                log_info("wrote encoded embeddings to %s", dump_embd.c_str());
            }
        }
    }

    // run the realtime ASR decode (encode + prefill + streaming loop)
    std::vector<llama_token> out_tokens(mtmd_input_chunk_get_n_tokens(chunk));
    size_t n_out = out_tokens.size();
    std::vector<float> step0_logits;
    mtmd_helper_voxtral_realtime_params hparams = {};
    hparams.max_tokens       = max_tokens;
    hparams.seq_id           = 0;
    if (!dump_logits_bin.empty()) {
        step0_logits.resize((size_t) n_vocab);
        hparams.out_step0_logits = step0_logits.data();
    }
    const int32_t res = mtmd_helper_eval_voxtral_realtime(
        mctx, ctx_dec, chunk, &hparams, out_tokens.data(), &n_out);
    if (res != 0) {
        log_error("realtime ASR decode failed (res=%d)", res);
        mtmd_input_chunks_free(chunks);
        mtmd_free(mctx);
        return 1;
    }
    out_tokens.resize(n_out);

    if (!dump_logits_bin.empty()) {
        FILE * fd = fopen(dump_logits_bin.c_str(), "wb");
        if (fd) {
            fwrite(step0_logits.data(), sizeof(float), n_vocab, fd);
            fclose(fd);
            log_info("wrote step-0 logits to %s", dump_logits_bin.c_str());
        }
    }

    // decode tokens to text (Tekken)
    std::string text;
    for (llama_token tok : out_tokens) {
        char      buf[16];
        const int n = llama_token_to_piece(vocab, tok, buf, sizeof(buf), 0, false);
        if (n > 0) {
            text.append(buf, n);
        }
    }
    fprintf(stdout, "\n%s\n", text.c_str());

    // diagnostics: dump the raw token ids to stderr
    fprintf(stderr, "tokens (%zu):", out_tokens.size());
    for (llama_token tok : out_tokens) {
        fprintf(stderr, " %d", tok);
    }
    fprintf(stderr, "\n");

    // cleanup
    mtmd_input_chunks_free(chunks);
    mtmd_free(mctx);
    llama_free(ctx_dec);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
