#include "models.h"

#include <cmath>
#include <vector>

// Voxtral-Mini-4B-Realtime-2602 (realtime streaming ASR/STT) - text decoder only.
//
// The main GGUF carries ONLY the text backbone (the causal audio encoder +
// temporal adapter live in the mmproj GGUF, loaded by mtmd). The ASR loop feeds
// per-frame audio embeddings (summed with the token embeddings) via
// llama_batch.embd; see docs-ports/voxtral-mini-4b-realtime.md.
//
// The text decoder is Ministral-like: RMSNorm, no biases, GQA (32/8), RoPE 1e6,
// sliding window 8192, SwiGLU, tied output, plus a per-layer adaptive RMS-norm
// time conditioning MLP (ada_rms_norm_t_cond: h_norm *= (1 + ada_mlp(time_emb))).

void llama_model_voxtral_rt_asr::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    // The decoder uses full causal attention over the KV cache. The 8192
    // "sliding window" in the reference is only the KV cache capacity (old
    // tokens are shifted out), NOT an attention-mask restriction, so we do not
    // set swa_type / is_swa_impl here.

    // decoder-side streaming decode config
    ml.get_key(LLM_KV_ADA_NORM_DIM, hparams.n_ada_norm_dim, false);
    ml.get_key(LLM_KV_DELAY_TOKENS, hparams.n_delay_tokens, false);

    // precompute the sinusoidal time embedding for the ada t-cond
    // (t = delay_tokens, constant across the whole decode session)
    time_emb_cpu.resize(hparams.n_embd);
    const int32_t half = hparams.n_embd / 2;
    const float   t    = (float) (hparams.n_delay_tokens != 0 ? hparams.n_delay_tokens : 6);
    for (int32_t i = 0; i < half; i++) {
        const float inv_freq   = expf(-logf(10000.0f) * (float) i / (float) half);
        const float angle      = t * inv_freq;
        time_emb_cpu[i]        = cosf(angle);
        time_emb_cpu[i + half] = sinf(angle);
    }
}

void llama_model_voxtral_rt_asr::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const int64_t n_dec_layer    = hparams.n_layer();  // decoder blocks (26)
    const int64_t n_ada_norm_dim = hparams.n_ada_norm_dim;

    tok_embd    = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, TENSOR_DUPLICATED);
    }

    // text decoder blocks (Ministral-like, no biases)
    for (int i = 0; i < n_dec_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), { n_embd }, 0);
        create_tensor_qkv(layer, i, n_embd, n_embd_head_k * n_head, n_embd_k_gqa, n_embd_v_gqa, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd_head_k * n_head, n_embd }, 0);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), { n_embd }, 0);
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), { n_embd, n_ff }, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff, n_embd }, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP, "weight", i), { n_embd, n_ff }, 0);

        // adaptive RMS-norm time conditioning MLP (missing in the dummy test GGUF)
        if (n_ada_norm_dim > 0) {
            layer.ada_norm_t_cond_0 =
                create_tensor(tn(LLM_TENSOR_ADA_NORM_T_COND_0, "weight", i), { n_embd, n_ada_norm_dim }, 0);
            layer.ada_norm_t_cond_2 =
                create_tensor(tn(LLM_TENSOR_ADA_NORM_T_COND_2, "weight", i), { n_ada_norm_dim, n_embd }, 0);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_voxtral_rt_asr::build_arch_graph(
    const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_voxtral_rt_asr::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(n_embd_head == n_rot);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    // note: the STT streaming loop feeds precomputed embeddings (sum of text
    // token embedding + adapter output at the frame position) via llama_batch.embd
    inpL = build_inp_embd(model.tok_embd);

    ggml_tensor * inp_pos = build_inp_pos();

    // the decoder attends over the full KV cache (causal); the reference's
    // 8192 "sliding window" is only the KV cache capacity, not a mask
    auto * inp_attn = build_attn_inp_kv();

    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // time embedding for the ada RMS-norm t-cond: constant per decode session
    // (t = n_delay_tokens), computed at load time and copied in as graph input.
    // only build it when the ada MLP weights are present (the test dummy GGUF has none)
    const llama_model_voxtral_rt_asr * vx = static_cast<const llama_model_voxtral_rt_asr *>(&model);
    const bool has_ada = model.layers[0].ada_norm_t_cond_0 && model.layers[0].ada_norm_t_cond_2;
    ggml_tensor * time_emb = has_ada ? build_inp_time_emb(vx->time_emb_cpu.data()) : nullptr;

    const int64_t n_dec_layer = hparams.n_layer();
    for (int il = 0; il < n_dec_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        // norm
        cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur, n_embd_head, n_head, n_head_kv, il);

            Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                 ext_factor, attn_factor, beta_fast, beta_slow);

            Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                 ext_factor, attn_factor, beta_fast, beta_slow);

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(inp_attn, model.layers[il].wo, NULL, NULL, Qcur, Kcur, Vcur, nullptr, nullptr, nullptr,
                             kq_scale, il);
            cb(cur, "attn_out", il);
        }

        if (il == n_dec_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // feed-forward with adaptive RMS-norm time conditioning
        cur = build_norm(ffn_inp, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        if (time_emb && model.layers[il].ada_norm_t_cond_0 && model.layers[il].ada_norm_t_cond_2) {
            // h_norm *= (1 + ada_mlp(time_emb)) == h_norm + h_norm * ada_scale
            // (folded to avoid creating a constant tensor in the no_alloc graph)
            ggml_tensor * ada_hidden = ggml_mul_mat(ctx0, model.layers[il].ada_norm_t_cond_0, time_emb);    // [ada_dim]
            ada_hidden               = ggml_gelu_erf(ctx0, ada_hidden);
            ggml_tensor * ada_scale  = ggml_mul_mat(ctx0, model.layers[il].ada_norm_t_cond_2, ada_hidden);  // [n_embd]
            ada_scale                = ggml_cast(ctx0, ada_scale, GGML_TYPE_F32);  // ada weights are bf16; cur is f32
            cur                      = ggml_add(ctx0, cur, ggml_mul(ctx0, cur, ada_scale));
            cb(cur, "ada_cond", il);
        }

        cur = build_ffn(cur, model.layers[il].ffn_up, NULL, NULL, model.layers[il].ffn_gate, NULL, NULL,
                        model.layers[il].ffn_down, NULL, NULL, NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);
        cb(cur, "ffn_out", il);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    cur = inpL;

    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur, model.output_s);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
