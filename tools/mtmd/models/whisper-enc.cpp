#include "models.h"

ggml_cgraph * clip_graph_whisper_enc::build() {
    const int n_frames = img.nx();
    const int n_pos    = n_frames / 2;
    GGML_ASSERT(model.position_embeddings->ne[1] >= n_pos);

    ggml_tensor * inp = build_inp_raw(1);

    // conv1d block
    {
        // convolution + gelu
        ggml_tensor * cur = ggml_conv_1d_ph(ctx0, model.conv1d_1_w, inp, 1, 1);
        cur = ggml_add(ctx0, cur, model.conv1d_1_b);

        cur = ggml_gelu_erf(ctx0, cur);

        cur = ggml_conv_1d_ph(ctx0, model.conv1d_2_w, cur, 2, 1);
        cur = ggml_add(ctx0, cur, model.conv1d_2_b);

        cur = ggml_gelu_erf(ctx0, cur);
        // transpose
        inp = ggml_cont(ctx0, ggml_transpose(ctx0, cur));
        cb(inp, "after_conv1d", -1);
    }

    // sanity check (only check one layer, but it should be the same for all)
    GGML_ASSERT(model.layers[0].ln_1_w && model.layers[0].ln_1_b);
    GGML_ASSERT(model.layers[0].ln_2_w && model.layers[0].ln_2_b);
    GGML_ASSERT(model.layers[0].q_b);
    GGML_ASSERT(model.layers[0].v_b);
    GGML_ASSERT(!model.layers[0].k_b); // no bias for k

    ggml_tensor * pos_embd_selected = ggml_view_2d(
        ctx0, model.position_embeddings,
        model.position_embeddings->ne[0], n_pos,
        model.position_embeddings->nb[1], 0
    );
    ggml_tensor * cur = build_vit(
                            inp, n_pos,
                            NORM_TYPE_NORMAL,
                            hparams.ffn_op,
                            pos_embd_selected,
                            nullptr);

    cb(cur, "after_transformer", -1);

    if (model.audio_has_stack_frames()) {
        // StackAudioFrames
        // https://huggingface.co/fixie-ai/ultravox-v0_5-llama-3_2-1b/blob/main/ultravox_model.py
        cur = build_stack(cur, hparams.proj_stack_factor, n_embd);
        cb(cur, "after_stacked", -1);
    }

    if (proj_type == PROJECTOR_TYPE_ULTRAVOX) {
        // UltravoxProjector
        // pre-norm
        cur = ggml_rms_norm(ctx0, cur, 1e-6);
        cur = ggml_mul(ctx0, cur, model.mm_norm_pre_w);

        // ffn in
        cur = build_mm(model.mm_1_w, cur);

        // swiglu
        // see SwiGLU in ultravox_model.py, the second half passed through is silu, not the first half
        cur = ggml_swiglu_swapped(ctx0, cur);

        // mid-norm
        cur = ggml_rms_norm(ctx0, cur, 1e-6);
        cur = ggml_mul(ctx0, cur, model.mm_norm_mid_w);

        // ffn out
        cur = build_mm(model.mm_2_w, cur);

    } else if (proj_type == PROJECTOR_TYPE_QWEN2A) {
        // projector
        cur = build_mm(model.mm_fc_w, cur);
        cur = ggml_add(ctx0, cur, model.mm_fc_b);

    } else if (proj_type == PROJECTOR_TYPE_VOXTRAL) {
        // projector
        cur = build_ffn(cur,
            model.mm_1_w, model.mm_1_b,
            nullptr, nullptr,
            model.mm_2_w, model.mm_2_b,
            FFN_GELU_ERF,
            -1);

    } else if (proj_type == PROJECTOR_TYPE_MUSIC_FLAMINGO) {
        // projector
        cur = build_ffn(cur,
            model.mm_1_w, model.mm_1_b,
            nullptr, nullptr,
            model.mm_2_w, model.mm_2_b,
            FFN_GELU_ERF,
            -1);

    } else if (proj_type == PROJECTOR_TYPE_MERALION) {
        // stack (above) -> ln -> linear0+silu -> GLU -> out
        cur = ggml_norm(ctx0, cur, hparams.eps);
        cur = ggml_mul(ctx0, cur, model.mm_norm_pre_w);
        cur = ggml_add(ctx0, cur, model.mm_norm_pre_b);

        cur = ggml_mul_mat(ctx0, model.mm_0_w, cur);
        cur = ggml_add(ctx0, cur, model.mm_0_b);
        cur = ggml_silu(ctx0, cur);

        ggml_tensor * gate = ggml_mul_mat(ctx0, model.mm_1_w, cur);
        gate = ggml_add(ctx0, gate, model.mm_1_b);
        gate = ggml_silu(ctx0, gate);

        ggml_tensor * pool = ggml_mul_mat(ctx0, model.mm_2_w, cur);
        pool = ggml_add(ctx0, pool, model.mm_2_b);

        cur = ggml_mul(ctx0, gate, pool);

        cur = ggml_mul_mat(ctx0, model.mm_3_w, cur);
        cur = ggml_add(ctx0, cur, model.mm_3_b);

    } else if (proj_type == PROJECTOR_TYPE_GLMA) {
            cur = ggml_norm(ctx0, cur, hparams.eps);
            cur = ggml_mul(ctx0, cur, model.mm_norm_pre_w);
            cur = ggml_add(ctx0, cur, model.mm_norm_pre_b);
            cur = build_stack(cur, hparams.proj_stack_factor, n_embd);
            cur = build_ffn(cur, model.mm_1_w, model.mm_1_b, nullptr, nullptr, model.mm_2_w, model.mm_2_b, hparams.ffn_op, 0);
            cur = ggml_concat(ctx0, model.mm_boi, cur, 1);
            cur = ggml_concat(ctx0, cur, model.mm_eoi, 1);
    } else {
        GGML_ABORT("%s: unknown projector type", __func__);
    }

    cb(cur, "projected", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}

// Voxtral-Mini-4B-Realtime-2602 causal audio encoder.
// Causal conv stem (left-only pad) + RoPE transformer (NEOX, sliding-window
// causal mask) + temporal adapter (Linear + GELU + Linear, downsample 4x).
// The conv, the 32 encoder blocks and the adapter weights all live in the
// mmproj GGUF under the mtmd `a.*` naming.
ggml_cgraph * clip_graph_whisper_enc_causal::build() {
    const int64_t n_frames = img.nx();
    const int64_t n_mel    = img.ny();
    GGML_ASSERT(n_mel == model.conv1d_1_w->ne[1]);

    // causal conv stem: left pad (kernel - stride), stride 1 then stride 2
    ggml_tensor * inp = build_inp_raw(1); // [n_frames, n_mel, 1, 1]

    ggml_tensor * cur = ggml_pad_ext(ctx0, inp, 2, 0, 0, 0, 0, 0, 0, 0);
    cur = ggml_conv_1d(ctx0, model.conv1d_1_w, cur, 1, 0, 1); // [OL, OC, N]
    cur = ggml_add(ctx0, cur, ggml_reshape_3d(ctx0, model.conv1d_1_b, 1, model.conv1d_1_w->ne[2], 1));
    cur = ggml_gelu_erf(ctx0, cur);
    cb(cur, "conv1", -1);

    cur = ggml_pad_ext(ctx0, cur, 1, 0, 0, 0, 0, 0, 0, 0);
    cur = ggml_conv_1d(ctx0, model.conv1d_2_w, cur, 2, 0, 1); // [OL, OC, N]
    cur = ggml_add(ctx0, cur, ggml_reshape_3d(ctx0, model.conv1d_2_b, 1, model.conv1d_2_w->ne[2], 1));
    cur = ggml_gelu_erf(ctx0, cur);
    cb(cur, "conv2", -1);

    const int64_t n_pos = cur->ne[0];
    GGML_ASSERT(n_pos == n_frames / 2);

    // transpose to [n_embd, n_pos]
    inp = ggml_cont(ctx0, ggml_transpose(ctx0, cur));
    cb(inp, "after_conv1d", -1);

    // sanity check (only check one layer, but it should be the same for all)
    GGML_ASSERT(model.layers[0].ln_1_w && model.layers[0].ln_2_w);
    GGML_ASSERT(model.layers[0].q_b);
    GGML_ASSERT(model.layers[0].v_b);
    GGML_ASSERT(!model.layers[0].k_b); // no bias for k

    // RoPE positions + causal sliding-window mask (set by clip_encode)
    ggml_tensor * inp_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_pos);
    ggml_set_name(inp_pos, "inp_pos");
    ggml_set_input(inp_pos);

    ggml_tensor * kq_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_pos, n_pos);
    ggml_set_name(kq_mask, "kq_mask");
    ggml_set_input(kq_mask);

    // RoPE (rotate_half / NeoX), applied per-head on the reshaped Q/K
    auto add_pos_rope = [&](ggml_tensor * t, const clip_layer &) -> ggml_tensor * {
        return ggml_rope_ext(ctx0, t, inp_pos, nullptr,
            d_head, GGML_ROPE_TYPE_NEOX, 0, hparams.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    };

    build_vit_opts opts;
    opts.attn_mask = kq_mask;

    cur = build_vit(inp, n_pos, NORM_TYPE_RMS, hparams.ffn_op,
                    nullptr, add_pos_rope, opts);
    cb(cur, "after_transformer", -1);

    // temporal adapter: [n_embd, n_pos] -> [n_embd*down, n_pos/down] -> mm2(gelu(mm1(x)))
    const int64_t downsample = hparams.audio_proj_downsample_rate > 0
                               ? hparams.audio_proj_downsample_rate : 4;
    GGML_ASSERT(n_pos % downsample == 0);
    cur = ggml_reshape_2d(ctx0, cur, n_embd * downsample, n_pos / downsample);
    cur = build_ffn(cur,
        model.mm_1_w, nullptr,
        nullptr, nullptr,
        model.mm_2_w, nullptr,
        FFN_GELU_ERF, -1);
    cb(cur, "projected", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}
