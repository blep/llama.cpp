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

#include "clip.h"
#include "common.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"
#include "llama.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// constants (must match the GGUF metadata / voxtral.cpp)
// ---------------------------------------------------------------------------

static constexpr int32_t VOXTRAL_SAMPLE_RATE        = 16000;
static constexpr int32_t VOXTRAL_NUM_MEL_BINS       = 128;
static constexpr int32_t VOXTRAL_HOP_LENGTH         = 160;
static constexpr int32_t VOXTRAL_WINDOW_SIZE        = 400;
static constexpr int32_t VOXTRAL_N_FFT              = 400;
static constexpr int32_t VOXTRAL_N_FREQ             = VOXTRAL_N_FFT / 2 + 1;  // 201
static constexpr float   VOXTRAL_GLOBAL_LOG_MEL_MAX = 1.5f;
static constexpr int32_t VOXTRAL_DEC_DIM           = 3072;
static constexpr int32_t VOXTRAL_DOWNSAMPLE_FACTOR = 4;

static constexpr int32_t VOXTRAL_N_LEFT_PAD_TOKENS        = 32;
static constexpr int32_t VOXTRAL_N_DELAY_TOKENS           = 6;
static constexpr int32_t VOXTRAL_N_RIGHT_PAD_TOKENS       = 17;
static constexpr int32_t VOXTRAL_RAW_AUDIO_LENGTH_PER_TOK = 1280;

static constexpr int32_t VOXTRAL_TOKEN_BOS           = 1;
static constexpr int32_t VOXTRAL_TOKEN_EOS           = 2;

static constexpr double VOXTRAL_PI = 3.14159265358979323846;

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
static inline int32_t reflect_idx(int32_t i, int32_t n) {
    if (i >= 0 && i < n) {
        return i;
    }
    if (n <= 1) {
        return 0;
    }
    if (i < 0) {
        int32_t x      = -i;
        int32_t period = 2 * (n - 1);
        x %= period;
        return x < n ? x : period - x;
    }
    int32_t x      = i - (n - 1);
    int32_t period = 2 * (n - 1);
    x %= period;
    return x < n ? n - 1 - x : x - (n - 1);
}

static void compute_mel_spectrogram(const float * audio,
                                    int32_t       n_samples,
                                    const float * mel_filters,  // [n_freq, n_mel]
                                    const float * hann_window,  // [window_size]
                                    float *       mel_out,      // [n_mel, n_frames]
                                    int32_t *     out_n_frames) {
    const int32_t n_stft_frames = n_samples / VOXTRAL_HOP_LENGTH + 1;
    const int32_t n_frames      = n_stft_frames - 1;  // drop last frame, matching Python [:-1]
    *out_n_frames               = n_frames;

    const int32_t pad = VOXTRAL_N_FFT / 2;

    if (n_frames <= 0) {
        return;
    }

    // reflect padding once (center=True, pad_mode="reflect")
    const int32_t      centered_len = n_samples + 2 * pad;
    std::vector<float> centered((size_t) centered_len, 0.0f);
    for (int32_t i = 0; i < centered_len; ++i) {
        const int32_t src    = i - pad;
        centered[(size_t) i] = audio[(size_t) reflect_idx(src, n_samples)];
    }

    std::vector<float> windowed((size_t) VOXTRAL_N_FFT);
    std::vector<float> power((size_t) VOXTRAL_N_FREQ);
    std::vector<float> mel_accum((size_t) VOXTRAL_NUM_MEL_BINS);

    for (int32_t frame = 0; frame < n_frames; ++frame) {
        const int32_t start     = frame * VOXTRAL_HOP_LENGTH;
        const float * frame_ptr = centered.data() + (size_t) start;

        for (int32_t i = 0; i < VOXTRAL_N_FFT; ++i) {
            windowed[(size_t) i] = frame_ptr[(size_t) i] * hann_window[(size_t) i];
        }

        // DFT
        for (int32_t k = 0; k < VOXTRAL_N_FREQ; ++k) {
            const float angle_k = 2.0f * (float) VOXTRAL_PI * (float) k;
            float       re = 0.0f, im = 0.0f;
            for (int32_t i = 0; i < VOXTRAL_N_FFT; ++i) {
                const float angle = angle_k * (float) i / (float) VOXTRAL_N_FFT;
                re += windowed[(size_t) i] * cosf(angle);
                im -= windowed[(size_t) i] * sinf(angle);
            }
            power[(size_t) k] = re * re + im * im;
        }

        // mel filterbank
        std::fill(mel_accum.begin(), mel_accum.end(), 0.0f);
        for (int32_t k = 0; k < VOXTRAL_N_FREQ; ++k) {
            const float * w  = mel_filters + (size_t) k * (size_t) VOXTRAL_NUM_MEL_BINS;
            const float   pk = power[(size_t) k];
            for (int32_t m = 0; m < VOXTRAL_NUM_MEL_BINS; ++m) {
                mel_accum[(size_t) m] += w[m] * pk;
            }
        }

        for (int32_t m = 0; m < VOXTRAL_NUM_MEL_BINS; ++m) {
            float val                                                = mel_accum[(size_t) m];
            val                                                      = std::max(val, 1e-10f);
            val                                                      = log10f(val);
            val                                                      = std::max(val, VOXTRAL_GLOBAL_LOG_MEL_MAX - 8.0f);
            val                                                      = (val + 4.0f) / 4.0f;
            mel_out[(size_t) m * (size_t) n_frames + (size_t) frame] = val;
        }
    }
}

static void compute_mel_even(const float *              samples,
                             int32_t                    n_samples,
                             const std::vector<float> & mel_filters,
                             const std::vector<float> & hann_window,
                             std::vector<float> &       mel_data,
                             int32_t &                  n_frames) {
    const int32_t max_frames = n_samples / VOXTRAL_HOP_LENGTH + 1;
    // NOTE: compute_mel_spectrogram writes with row stride n_frames, so the
    // buffer must be allocated with n_frames columns per row, NOT max_frames.
    // (allocating max_frames and dumping n_mel*n_frames reads misaligned rows)
    mel_data.assign((size_t) VOXTRAL_NUM_MEL_BINS * (max_frames - 1), 0.0f);
    n_frames = 0;
    compute_mel_spectrogram(samples, n_samples, mel_filters.data(), hann_window.data(), mel_data.data(), &n_frames);
    if (n_frames % 2 != 0) {
        for (int32_t m = 0; m < VOXTRAL_NUM_MEL_BINS; ++m) {
            memmove(mel_data.data() + (size_t) m * (n_frames - 1), mel_data.data() + (size_t) m * n_frames + 1,
                    (size_t) (n_frames - 1) * sizeof(float));
        }
        n_frames -= 1;
    }
}

// ---------------------------------------------------------------------------
// GGUF tensor reading (encoder + adapter + mel filters), dequantized to F32
// ---------------------------------------------------------------------------

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

static void compute_hann_window(std::vector<float> & hann) {
    hann.resize(VOXTRAL_WINDOW_SIZE);
    // match torch.hann_window(W, periodic=True): divide by W, not W-1
    for (int32_t i = 0; i < VOXTRAL_WINDOW_SIZE; ++i) {
        hann[(size_t) i] = 0.5f * (1.0f - cosf(2.0f * (float) VOXTRAL_PI * (float) i / (float) VOXTRAL_WINDOW_SIZE));
    }
}

// ---------------------------------------------------------------------------
// causal conv1d dims + graph (matches voxtral.cpp)
// ---------------------------------------------------------------------------

struct conv1d_dims {
    int32_t pad_left   = 0;
    int32_t pad_right  = 0;
    int32_t padded_len = 0;
    int32_t out_len    = 0;
};

static conv1d_dims compute_causal_conv1d_dims(int32_t in_len, int32_t kernel_size, int32_t stride) {
    conv1d_dims   d;
    const int32_t padding_total = kernel_size - stride;
    const float   n_frames      = (float) (in_len - kernel_size + padding_total) / (float) stride + 1.0f;
    const int32_t target_length = ((int32_t) ceilf(n_frames) - 1) * stride + (kernel_size - padding_total);
    d.pad_left                  = padding_total;
    d.pad_right                 = std::max(0, target_length - in_len);
    d.padded_len                = in_len + d.pad_left + d.pad_right;
    d.out_len                   = (d.padded_len - kernel_size) / stride + 1;
    return d;
}

static int32_t mel_frames_to_enc_tokens(int32_t n_frames) {
    const conv1d_dims d0    = compute_causal_conv1d_dims(n_frames, 3, 1);
    const conv1d_dims d1    = compute_causal_conv1d_dims(d0.out_len, 3, 2);
    const int32_t     trunc = d1.out_len % VOXTRAL_DOWNSAMPLE_FACTOR;
    return d1.out_len - trunc;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

struct voxtral_model_weights {
    ggml_context * ctx      = nullptr;  // holds the token embedding tensor
    ggml_tensor *  tok_embd = nullptr;

    int32_t token_streaming_pad  = 32;  // [STREAMING_PAD]
    int32_t token_streaming_word = 33;  // [STREAMING_WORD]

    std::vector<float> mel_filters_cpu;  // [n_freq, n_mel] F32
    std::vector<float> hann_window;
};

// mel filterbank + hann window for the STFT (the filterbank lives in the mmproj)
static bool load_mel_filters(const char * mmproj_path, voxtral_model_weights & W) {
    gguf_tensor t;
    if (!load_gguf_tensor_f32(mmproj_path, "a.mel_filters", t)) {
        log_error("a.mel_filters tensor not found in %s", mmproj_path);
        return false;
    }
    W.mel_filters_cpu = t.f32;
    compute_hann_window(W.hann_window);
    return true;
}

// token embedding lives in the main (text) GGUF; load it as F32
// read a uint32 metadata key from the main GGUF (missing -> default)
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

static bool load_token_embd(const char * main_path, voxtral_model_weights & W) {
    gguf_tensor t;
    if (!load_gguf_tensor_f32(main_path, "token_embd.weight", t)) {
        log_error("token_embd.weight not found in %s", main_path);
        return false;
    }
    W.token_streaming_pad  = (int32_t) load_gguf_uint32(main_path, "voxtral_rt_asr.streaming.streaming_pad_token_id",  32);
    W.token_streaming_word = (int32_t) load_gguf_uint32(main_path, "voxtral_rt_asr.streaming.streaming_word_token_id", 33);
    ggml_init_params p = { 4ull << 30, nullptr, false };
    W.ctx              = ggml_init(p);
    if (!W.ctx) {
        log_error("failed to init token-embd context");
        return false;
    }
    W.tok_embd = ggml_new_tensor(W.ctx, GGML_TYPE_F32, (int) t.ne.size(), t.ne.data());
    if (!W.tok_embd) {
        log_error("failed to create token_embd tensor");
        return false;
    }
    memcpy(W.tok_embd->data, t.f32.data(), t.f32.size() * sizeof(float));
    return true;
}

static void usage(const char * argv0) {
    fprintf(stderr, "usage: %s -m model.gguf -mv mmproj.gguf -f input.wav [options]\n", argv0);
    fprintf(stderr, "  -m, --model PATH        main GGUF (text decoder, arch voxtral_rt_asr)\n");
    fprintf(stderr, "  -mv, --mmproj PATH      mmproj GGUF (causal audio encoder, arch clip)\n");
    fprintf(stderr, "  -f, --file PATH         16 kHz mono PCM16 WAV\n");
    fprintf(stderr, "  -n, --max-tokens N      max decoder tokens (0 = to end of audio)\n");
    fprintf(stderr, "  -t, --threads N         CPU threads for the encoder/decoder\n");
    fprintf(stderr, "  -ngl, --gpu-layers N    decoder layers on the GPU/iGPU (0 = CPU only)\n");
    fprintf(stderr, "  -v, --verbose           verbose logging\n");
    fprintf(stderr, "  -h, --help              show this help\n");
}

int main(int argc, char ** argv) {
    ggml_time_init();

    std::string model_path;
    std::string mmproj_path;
    std::string audio_path;
    std::string dump_logits_bin;
    std::string dump_mel;
    std::string dump_adapter;
    std::string adapter_in;
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
        } else if (arg == "--dump-mel") {
            if (++i >= argc) {
                log_error("missing value for %s", arg.c_str());
                return 1;
            }
            dump_mel = argv[i];
        } else if (arg == "--dump-adapter") {
            if (++i >= argc) {
                log_error("missing value for %s", arg.c_str());
                return 1;
            }
            dump_adapter = argv[i];
        } else if (arg == "--adapter-in") {
            if (++i >= argc) {
                log_error("missing value for %s", arg.c_str());
                return 1;
            }
            adapter_in = argv[i];
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

    const int32_t       n_embd  = llama_model_n_embd(model);
    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int32_t       n_vocab = llama_vocab_n_tokens(vocab);

    if (n_embd != VOXTRAL_DEC_DIM) {
        log_error("model n_embd %d != expected %d", n_embd, VOXTRAL_DEC_DIM);
        return 1;
    }

    // ---------------- load audio ----------------
    std::vector<float> audio;
    if (!load_wav_file(audio_path, audio)) {
        return 1;
    }

    // ---------------- load the audio tower via mtmd (mmproj GGUF) ----------------
    clip_context_params clip_params = {};
    clip_params.use_gpu             = false;  // encoder on CPU
    clip_params.flash_attn_type     = CLIP_FLASH_ATTN_TYPE_DISABLED;
    clip_params.warmup              = false;
    clip_params.no_alloc            = false;
    const auto clip_res             = clip_init(mmproj_path.c_str(), clip_params);
    clip_ctx * ctx_a                = clip_res.ctx_a;
    if (!ctx_a) {
        log_error("failed to load mmproj %s", mmproj_path.c_str());
        return 1;
    }

    // token embedding lives in the main (text) GGUF
    voxtral_model_weights W;
    if (!load_token_embd(model_path.c_str(), W)) {
        return 1;
    }
    // mel filterbank + hann window come from the mmproj
    if (!load_mel_filters(mmproj_path.c_str(), W)) {
        return 1;
    }

    // streaming padding (matching Python pad_audio_streaming)
    const int32_t n_raw     = (int32_t) audio.size();
    const int32_t align_pad = (VOXTRAL_RAW_AUDIO_LENGTH_PER_TOK - (n_raw % VOXTRAL_RAW_AUDIO_LENGTH_PER_TOK)) %
                              VOXTRAL_RAW_AUDIO_LENGTH_PER_TOK;
    const int32_t right_pad = align_pad + VOXTRAL_N_RIGHT_PAD_TOKENS * VOXTRAL_RAW_AUDIO_LENGTH_PER_TOK;
    const int32_t left_pad  = VOXTRAL_N_LEFT_PAD_TOKENS * VOXTRAL_RAW_AUDIO_LENGTH_PER_TOK;

    std::vector<float> padded((size_t) left_pad + n_raw + right_pad, 0.0f);
    memcpy(padded.data() + left_pad, audio.data(), n_raw * sizeof(float));
    log_info("padded audio: %d samples (left=%d, right=%d)", (int) padded.size(), left_pad, right_pad);

    // mel spectrogram
    int32_t            n_frames = 0;
    std::vector<float> mel_data;
    compute_mel_even(padded.data(), (int32_t) padded.size(), W.mel_filters_cpu, W.hann_window, mel_data, n_frames);
    log_info("mel spectrogram: %d frames", n_frames);
    if (!dump_mel.empty()) {
        FILE * fd = fopen(dump_mel.c_str(), "wb");
        if (fd) {
            const int32_t n = VOXTRAL_NUM_MEL_BINS * n_frames;
            fwrite(&n, sizeof(int32_t), 1, fd);
            fwrite(mel_data.data(), sizeof(float), n, fd);
            fclose(fd);
            log_info("wrote mel to %s", dump_mel.c_str());
        }
    }

    const int32_t total_enc_tokens = mel_frames_to_enc_tokens(n_frames);
    if (total_enc_tokens <= 0) {
        log_error("audio too short to produce encoder tokens");
        return 1;
    }
    const int32_t dec_seq = total_enc_tokens / VOXTRAL_DOWNSAMPLE_FACTOR;
    log_info("encoder tokens: %d, decoder audio positions: %d", total_enc_tokens, dec_seq);

    // ---- audio tower via mtmd: mel -> causal encoder -> temporal adapter ----
    std::vector<float> audio_emb((size_t) VOXTRAL_DEC_DIM * dec_seq, 0.0f);
    {
        // mel_data is [n_mel, n_frames] (frame contiguous within each bin)
        std::vector<float> out;
        if (!clip_audio_encode(ctx_a, n_threads, mel_data.data(), VOXTRAL_NUM_MEL_BINS, n_frames, out)) {
            log_error("mtmd audio encode failed");
            return 1;
        }
        if ((int32_t) out.size() != VOXTRAL_DEC_DIM * dec_seq) {
            log_error("mtmd output size %zu != expected %d", out.size(), VOXTRAL_DEC_DIM * dec_seq);
            return 1;
        }
        // mtmd output is the adapter result [dec_dim, dec_seq] (ne0 = dec_dim,
        // flat [d + s*dec_dim]); transpose to [dec_seq, dec_dim] (token-major)
        for (int32_t s = 0; s < dec_seq; ++s) {
            for (int32_t d = 0; d < VOXTRAL_DEC_DIM; ++d) {
                audio_emb[(size_t) s * VOXTRAL_DEC_DIM + d] = out[(size_t) d + (size_t) s * VOXTRAL_DEC_DIM];
            }
        }
    }
    log_info("adapter done: %d audio embeddings", dec_seq);

    // debug: optionally override the adapter output with an external file
    if (!adapter_in.empty()) {
        FILE * fd = fopen(adapter_in.c_str(), "rb");
        if (!fd) {
            log_error("cannot open adapter input: %s", adapter_in.c_str());
            return 1;
        }
        int32_t n = 0;
        if (fread(&n, sizeof(int32_t), 1, fd) != 1 || n != (int32_t) audio_emb.size()) {
            log_error("adapter input size mismatch: expected %d, got %d", (int) audio_emb.size(), n);
            fclose(fd);
            return 1;
        }
        if (fread(audio_emb.data(), sizeof(float), n, fd) != (size_t) n) {
            log_error("failed to read adapter input");
            fclose(fd);
            return 1;
        }
        fclose(fd);
        log_info("overrode adapter output from %s", adapter_in.c_str());
    }

    if (!dump_adapter.empty()) {
        FILE * fd = fopen(dump_adapter.c_str(), "wb");
        if (fd) {
            const int32_t n = dec_seq * VOXTRAL_DEC_DIM;
            fwrite(&n, sizeof(int32_t), 1, fd);
            fwrite(audio_emb.data(), sizeof(float), n, fd);
            fclose(fd);
            log_info("wrote adapter output to %s", dump_adapter.c_str());
        }
    }

    // ---- DSM decode loop via llama.cpp ----
    // prompt: [BOS] + [STREAMING_PAD] * (N_LEFT + N_DELAY)
    std::vector<llama_token> prompt_ids;
    prompt_ids.push_back(VOXTRAL_TOKEN_BOS);
    for (int32_t i = 0; i < VOXTRAL_N_LEFT_PAD_TOKENS + VOXTRAL_N_DELAY_TOKENS; ++i) {
        prompt_ids.push_back(W.token_streaming_pad);
    }
    const int32_t L = (int32_t) prompt_ids.size();  // 39

    if (L > dec_seq) {
        log_error("prompt length %d exceeds audio tokens %d", L, dec_seq);
        return 1;
    }

    auto decode_batch = [&](const std::vector<float> & emb, const std::vector<int64_t> & pos) -> bool {
        const int   n     = (int) emb.size() / VOXTRAL_DEC_DIM;
        llama_batch batch = llama_batch_init(n, VOXTRAL_DEC_DIM, 1);
        batch.n_tokens    = n;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < VOXTRAL_DEC_DIM; ++j) {
                batch.embd[(size_t) i * VOXTRAL_DEC_DIM + j] = emb[(size_t) i * VOXTRAL_DEC_DIM + j];
            }
            batch.pos[i]       = pos[i];
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]    = (i == n - 1);  // logits only for the last token
        }
        bool ok = llama_decode(ctx_dec, batch) == 0;
        llama_batch_free(batch);
        if (verbose_logging) {
            llama_memory_t mem = llama_get_memory(ctx_dec);
            fprintf(stderr, "DBG decode n=%d kv pos=[%d..%d]\n", n, (int) llama_memory_seq_pos_min(mem, 0),
                    (int) llama_memory_seq_pos_max(mem, 0));
        }
        return ok;
    };

    // token embedding lookup (CPU): [n_vocab, n_embd]
    const float * tok_embd_w = (const float *) W.tok_embd->data;

    // helper: build input embedding for position i (token + audio)
    auto build_input_emb = [&](llama_token tok, int32_t pos, float * out) -> void {
        const float * te = tok_embd_w + (int64_t) tok * VOXTRAL_DEC_DIM;
        const float * ae = audio_emb.data() + (int64_t) pos * VOXTRAL_DEC_DIM;
        for (int j = 0; j < VOXTRAL_DEC_DIM; ++j) {
            out[j] = te[j] + ae[j];
        }
    };

    // prefill: positions 0..L-2 (L-1 tokens), then one step with the last prefix token
    {
        std::vector<float>   emb((size_t) (L - 1) * VOXTRAL_DEC_DIM);
        std::vector<int64_t> pos(L - 1);
        for (int32_t i = 0; i < L - 1; ++i) {
            build_input_emb(prompt_ids[i], i, emb.data() + (size_t) i * VOXTRAL_DEC_DIM);
            pos[i] = i;
        }
        if (!decode_batch(emb, pos)) {
            log_error("decoder prefill failed");
            return 1;
        }
    }

    std::vector<float>       logits((size_t) n_vocab);
    std::vector<llama_token> out_tokens;
    llama_token              token = 0;

    {
        // one step with the last prefix token at position L-1
        std::vector<float> emb(VOXTRAL_DEC_DIM);
        build_input_emb(prompt_ids[L - 1], L - 1, emb.data());
        std::vector<int64_t> pos = { L - 1 };
        if (!decode_batch(emb, pos)) {
            log_error("decoder step failed");
            return 1;
        }
        const float * lg = llama_get_logits_ith(ctx_dec, 0);
        memcpy(logits.data(), lg, n_vocab * sizeof(float));
        if (!dump_logits_bin.empty()) {
            FILE * fd = fopen(dump_logits_bin.c_str(), "wb");
            if (fd) {
                fwrite(logits.data(), sizeof(float), n_vocab, fd);
                fclose(fd);
                log_info("wrote step-0 logits to %s", dump_logits_bin.c_str());
            }
        }
        const size_t imax = std::max_element(logits.begin(), logits.end()) - logits.begin();
        token             = (llama_token) imax;
        out_tokens.push_back(token);
        log_info("first token: %d", token);
    }

    const bool unlimited = (max_tokens <= 0);
    const auto t_decode  = std::chrono::steady_clock::now();

    for (int32_t pos = L; pos < dec_seq && (unlimited || (int32_t) out_tokens.size() < max_tokens); ++pos) {
        if (token == VOXTRAL_TOKEN_EOS) {
            break;
        }

        std::vector<float> emb(VOXTRAL_DEC_DIM);
        build_input_emb(token, pos, emb.data());
        std::vector<int64_t> posv = { pos };
        if (!decode_batch(emb, posv)) {
            log_error("decoder step %d failed", pos);
            return 1;
        }
        const float * lg = llama_get_logits_ith(ctx_dec, 0);
        memcpy(logits.data(), lg, n_vocab * sizeof(float));
        const size_t imax = std::max_element(logits.begin(), logits.end()) - logits.begin();
        token             = (llama_token) imax;
        out_tokens.push_back(token);
        if (verbose_logging && (pos < 45 || pos % 10 == 0)) {
            fprintf(stderr, "step pos=%d tok=%d p_pad=%.3f p_word=%.3f p_eos=%.3f\n", pos, (int) token,
                    logits[W.token_streaming_pad], logits[W.token_streaming_word], logits[VOXTRAL_TOKEN_EOS]);
        }
    }

    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_decode).count();
    log_info("decode: %d steps, %.1f ms (%.1f ms/step)", (int) out_tokens.size() - 1, ms,
             out_tokens.size() > 1 ? ms / (out_tokens.size() - 1) : 0.0);

    // remove trailing EOS
    if (!out_tokens.empty() && out_tokens.back() == VOXTRAL_TOKEN_EOS) {
        out_tokens.pop_back();
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
    ggml_free(W.ctx);
    llama_free(ctx_dec);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
