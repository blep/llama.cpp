# MOSS-TTS-Nano-100M port to llama.cpp (analysis / reference)

Source: `/home/blep/prj/mallomar-server/docs/model-llama-cpp-ports.md` (candidate: MOSS-TTS-Nano-100M)
Status: **port target - selected over Moondream 0.5B (2026-08-14)**. Phase 0 (GGUF), Phase 1 (text
decoder forward) and Phase 2 (audio generation loop) DONE 2026-08-14; next: Phase 3 (codec + gen_audio pipeline).
Date: 2026-08-13 (updated 2026-08-14)

## License (verified 2026-08-13)

- **Decoder `MOSS-TTS-Nano-100M`: Apache-2.0** - HF card (`OpenMOSS-Team/MOSS-TTS-Nano-100M`,
  tag `license:apache-2.0`) and GitHub `OpenMOSS/MOSS-TTS-Nano` LICENSE file (standard Apache-2.0
  text, confirmed via GitHub API). The repo README caveat "not yet licensed for redistribution"
  is stale: the `LICENSE` file is present in the release repo.
- **Codec `MOSS-Audio-Tokenizer-Nano`: Apache-2.0** - HF card (`license:apache-2.0`) and GitHub
  `OpenMOSS/MOSS-Audio-Tokenizer` LICENSE (Apache-2.0). The local HF snapshot has no LICENSE file,
  but both HF metadata and GitHub confirm Apache-2.0.
- **Compatibility**: Apache-2.0 is permissive (commercial use, modification, redistribution
  allowed); compatible with llama.cpp's MIT license for the ported runtime. No license barrier
  to porting or redistributing converted GGUF weights, provided the Apache-2.0 license text is
  kept (e.g. NOTICE/license attribution in the conversion output).

## Model identity

- HF repo: `OpenMOSS-Team/MOSS-TTS-Nano-100M` (public, not gated) - **TTS**
- Params: 100M, weights stored **bfloat16** in `pytorch_model.bin` (verified 2026-08-14; the
  config's `"dtype": "float32"` is misleading)
- Arch: `moss_tts_nano`, `model_architecture` = `global_local_transformer`
  - GPT-2 core (12 layers, 768 hidden, 12 heads, RoPE, 32 768 ctx, 16 384 vocab, gelu_new)
  - `local_transformer_layers: 1` (local codebook refinement)
  - 16 audio codebooks x 1024 (`n_vq` 16 / `audio_codebook_sizes`)
  - Special token ids: im_start 4, im_end 5, audio_start 6, audio_end 7, audio_user_slot 8, audio_assistant_slot 9, audio_pad 1024
- Audio: 48 kHz, 2-channel (stereo) output
- Tokenizer: SentencePiece (`tokenizer.model`), `MossTTSNanoSentencePieceTokenizer`, 16 384 vocab
- Config: `attn_implementation` flash_attention_2 (CPU fallback exists in code), compute fp32
- Prompt format: `<user_inst> Reference(s): ... Instruction/Tokens/Quality/... Language: ... Text: <text> </user_inst><im_end>...<im_start>assistant` - see `prompting.py`

## Audio codec (second GGUF)

- HF repo: `OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano` (public) - **codec**
- Params: ~22M (`MossAudioTokenizerModel`, Cat = Causal Audio Tokenizer with Transformer, CNN-free)
- Quantizer: RLFQ, 16 codebooks x 1024, bitrate 0.125-4 kbps
- Audio: 48 kHz stereo in/out, 12.5 Hz token stream (`downsample_rate` 3840)
- Causal Transformer stacks: encoder 4+2+2+4 layers, decoder mirrored
- HF integration: `AutoModel.from_pretrained(..., trust_remote_code=True)`

## Local materials (downloaded 2026-08)

- TTS decoder: `/home/blep/ai-models/llm-ports/OpenMOSS-Team/MOSS-TTS-Nano-100M/`
  - `pytorch_model.bin` - 234 693 095 B, sha256 `24003f2f11ac8a2cbf70514db2d8f1c02fb451aa6b3c0bffc9da09f31cd7caa5`
  - `modeling_moss_tts_nano.py` (111 KB, `MossTTSNanoForCausalLM`, has generate/generate_stream/inference/inference_stream)
  - `gpt2_decoder.py` (MossTTSNanoGPT2Block / MossTTSNanoGPT2Model)
  - `configuration_moss_tts_nano.py`, `prompting.py`, `tokenization_moss_tts_nano.py`
  - `config.json`, `tokenizer.model`, `special_tokens_map.json`, `tokenizer_config.json`
- Codec: `/home/blep/ai-models/llm-ports/OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano/`
  - `model-00001-of-00001.safetensors` - 87 922 568 B, sha256 `34d9880d805eecb21bde975202b1c256dbd0eb98c8680b9d3aeffd2bc6ac2f67`
  - `modeling_moss_audio_tokenizer.py`, `configuration_moss_audio_tokenizer.py`, `config.json`, index
- Official ONNX CPU variants also exist: `MOSS-TTS-Nano-100M-ONNX`, `MOSS-Audio-Tokenizer-Nano-ONNX` (dual-run reference)

## Supported languages (20, from README)

zh, en, de, es, fr, ja, it, he, ko, ru, fa, ar, pl, pt, cs, da, sv, hu, el, tr

Language is inferred from the input text (no explicit language parameter).

## Official run recipe (reference to reproduce)

- GitHub: `https://github.com/OpenMOSS/MOSS-TTS-Nano.git` (pip install -e ., CLI `moss-tts-nano`)
- Entrypoint: `python infer.py --prompt-audio-path assets/audio/zh_1.wav --text "..."` (voice clone mode, default out `generated_audio/infer_output.wav`)
- CLI: `moss-tts-nano generate --prompt-speech assets/audio/zh_1.wav --text "..."` (default out `generated_audio/moss_tts_nano_output.wav`), `--text-file` for long text
- Web demo: `python app.py` / `moss-tts-nano serve` (port 18083)
- Requires reference audio (voice clone). Official repo carries `assets/audio/*.wav` samples; the HF snapshot here does not.
- Dependencies: torch, transformers (4.57.x), torchaudio, sentencepiece, WeTextProcessing (needs pynini via conda-forge if pip fails)
- GPU: no flash-attn needed on CPU; code has attention fallback and fp32 default

## Reference GGUF (existing conversion)

- `cstr/moss-tts-v1.5-GGUF` - LM + separate codec GGUF (OuteTTS `-mv` pattern). v1.5 is 17 GB f16, codec 3.5 GB - too big for a test model; Nano needs its own conversion.

## Port work items (from source doc)

- (a) Nano decoder arch - `LLM_ARCH_GPT2` may fit if tensor names/dims map, else custom
- (b) new `MTMD_GEN_AUDIO_TYPE_MOSSTTS` + gen_audio wiring (copy in-fork `pockettts` pattern)
- (c) Nano codec as second GGUF (via `-mv` / `-mmproj` mechanism)
- (d) Nano convert script (convert_hf_to_gguf.py)
- Effort: medium-high

## Port decision (2026-08-14): MOSS-TTS-Nano-100M first, Moondream 0.5B deferred

Rationale from verified hard facts (not estimates):

1. **Conversion source format** - MOSS-TTS ships a plain PyTorch `pytorch_model.bin` (234 MB fp32)
   with **exact GPT-2 tensor names** (`transformer.wte.weight`, `transformer.h.N.attn.c_attn.weight`,
   `transformer.h.N.ln_1`, `transformer.h.N.mlp`, `transformer.ln_f` - verified 2026-08-14). llama.cpp's
   existing GPT-2 GGUF loader maps that layout already. Custom tensors (`audio_embeddings.{0-15}.weight`,
   `audio_lm_heads.{0-15}.weight`, `local_transformer.*`, `text_lm_head.weight`, tied) are plain
   Embedding/Linear weights. Codec is safetensors. Moondream 0.5B, by contrast, ships weights only
   inside the `.mf` ONNX protobufs as **int8 DynamicQuantizeMatMul** (u8 + f32 scales, f16 activations)
   with graph-internal names; no raw safetensors exists and no converter handles it.
2. **Arch reuse** - MOSS-TTS text core IS GPT-2 (12 layers, 768 hidden, 12 heads, RoPE, tied,
   gelu_new) so a GPT-2-style layer stack applies; note the fork's `LLM_ARCH_GPT2` graph expects
   learned `pos_embd` while MOSS uses RoPE, so the runtime needs a RoPE variant (confirmed in
   Phase 0). Moondream is a custom dense
   decoder (dim 1024, 24 layers, 16 heads, vocab 51200) with zero support in mainline or this fork.
3. **Runtime precedent in this fork** - MOSS-TTS follows the validated in-fork `pockettts` pattern
   (MTMD gen_audio type + codec as second GGUF via `-mv`). Moondream has zero mtmd vision support
   (dropped in the refactor) and needs a SigLIP-style tower + custom projector + multi-crop tile
   reconstruction - no precedent.
4. **Numerical fidelity** - MOSS-TTS reference is fp32 PyTorch (lossless GGUF target). Moondream's
   reference numerics are int8-ONNX, so a GGUF dequant path drifts from the official artifact.

See `docs-ports/moondream-0.5b.md` for the corrected Moondream facts (the source doc's "shared
vision tower" and "smallest surface / 2-4 days" estimates are disproved by the extracted dims).

## Testing & development strategy (byte-for-byte parity)

### Goal

Match the official Python reference **byte for byte** on a meaningful TTS test set, at two levels:
1. the generated **audio token id sequence** (16 codebooks x frames) must match exactly;
2. the final **WAV file bytes** must match exactly (codec decode path included).

### Test set design

Covered axes (in `refs/moss-tts-nano-100m/test_cases.json`, one entry per case); generated by
`refs/moss-tts-nano-100m/run_test_set.py` (model loaded once) into
`tmp/moss-tts-nano-100m/test-set/` (+ `results.json` with per-case frames/duration). DONE 2026-08-14:
26 cases, all wavs valid (48 kHz stereo, non-silent).
- **Languages**: all 20 documented (zh, en, de, es, fr, ja, it, he, ko, ru, fa, ar, pl, pt, cs, da,
  sv, hu, el, tr) - at least the 11 already validated (en x3, zh x2, ja, de, fr, es, it, pt, ru, ar,
  ko) plus the remaining scripts. Language is inferred from text, no language parameter.
- **Text length / output window** (the main risk area; TTS has two windows):
  - short single sentence - no chunking;
  - medium multi-sentence - chunk boundaries, inter-chunk pauses
    (`voice_clone_max_text_tokens`: infer.py default 75, model method default 50 - triggers
    pocket-tts style sentence chunking);
  - long text - several chunks, chunk count recorded;
  - text long enough to hit the audio frame window `max_new_frames` (default 375 frames = 30 s at
    12.5 Hz) - captures truncation/stop-at-window behavior.
- **Numbers/dates/punctuation** (en-03 style) - text normalization is disabled in the harness
  (WeTextProcessing/pynini), only the robust `normalize_tts_text` cleanup runs; keep parity on that.
- **Prompt variety**: language-matched reference audio where available (`en_*.wav`, `zh_*.wav`,
  `jp_2.wav`), different reference lengths.

### Test harness (Python)

Lives in `refs/moss-tts-nano-100m/` (evolves from `run_samples.py`). Requirements:
- **Load the model + tokenizer + codec ONCE**, run all cases in-process via repeated
  `model.inference(...)` calls (already supported; this is the whole point - no per-case reload).
- **Deterministic**: fixed seed (`torch.manual_seed`). Two modes:
  - greedy (`do_sample=False`) - primary byte-for-byte target;
  - seeded sampling (`do_sample=True`) - validates RNG parity later.
- Per case, dump **golden artifacts** to `refs/moss-tts-nano-100m/golden/<case>/`:
  - normalized input text + case params (JSON manifest);
  - text token ids + prompt/reference audio code ids;
  - generated audio token ids (frames x 16);
  - final WAV bytes.
- CLI: `harness.py gen <test_cases.json>` (golden) and `harness.py compare <llama-output-dir>`.

### Tracing / delta localization

When outputs differ, locate the **first diverging stage** (tokenizer -> embeddings -> logits ->
sampling -> codec -> wav), not just the final audio:
- Python side trace mode (`--trace`): dump per-step intermediates - text tokenization ids, prompt
  code ids (codec encode), per-decode-step logits (pre/post sampling), sampled ids, local-transformer
  per-codebook outputs, and the codec decode waveform (pre-wav float values).
- llama.cpp side mirror: dump the same stages (ggml graph output dump / logits + sampling trace).
  Compare stage by stage; the first mismatch pinpoints the faulty op (e.g. RoPE base, gelu_new vs
  gelu, c_attn vs split QKV, local-transformer, codec RQ layers).
- **Numeric parity rule**: the parity tests must run llama.cpp at f16/f32 with the same compute
  order as the fp32 torch reference; quantized GGUF (Q4 etc.) is for quality testing only and
  cannot be byte-for-byte.

### Phased development

- **Phase 0 - Conversion: DONE (2026-08-14)**. The decoder converts to TWO GGUFs (Qwen3-TTS
  pattern, see `docs-ports/llama-cpp-integration-practices.md`):
  - main `refs/moss-tts-nano-100m-port/moss-tts-nano-100m-f32.gguf` (441 243 616 B f32, 148
    text tensors: global transformer);
  - mmproj `refs/moss-tts-nano-100m-port/mmproj-moss-tts-nano-100m-f32.gguf` (129 023 392 B f32,
    20 gen-code tensors: local transformer + 16 codebooks, `a.gen.code.*` mtmd names).
  Verified with `refs/moss-tts-nano-100m-port/verify_conversion.py`: all 168 tensors
  byte-identical to the source state dict. Layout, prerequisites and reproduction in
  `refs/moss-tts-nano-100m-port/CONVERSION.md`; converter source:
  `conversion/mosstts.py` (two classes: `MossTTSNanoModel(TextModel)`
  + `MossTTSNanoGenModel(MmprojModel)`).

  **How to run the conversion** (from the fork root, uv venv with torch/transformers/
  sentencepiece/numpy - the project `.venv` works):
  ```sh
  .venv/bin/python convert_hf_to_gguf.py \
      /home/blep/ai-models/llm-ports/OpenMOSS-Team/MOSS-TTS-Nano-100M \
      --outfile refs/moss-tts-nano-100m-port/moss-tts-nano-100m-f32.gguf \
      --outtype f32
  .venv/bin/python convert_hf_to_gguf.py \
      /home/blep/ai-models/llm-ports/OpenMOSS-Team/MOSS-TTS-Nano-100M \
      --outfile refs/moss-tts-nano-100m-port/mmproj-moss-tts-nano-100m-f32.gguf \
      --outtype f32 --mmproj
  ```
  The converter is **fully integrated into the conversion framework** (in-tree, nothing to patch):
  1. `gguf-py/gguf/constants.py`: `MODEL_ARCH.MOSSTTS` enum + `"mosstts"` name + `MODEL_TENSORS`
     (text tensors only); `VisionProjectorType.MOSSTTS_GEN = "mosstts_gen"`.
  2. `conversion/__init__.py`: `"MossTTSNanoForCausalLM": "mosstts"` in BOTH `TEXT_MODEL_MAP` and
     `MMPROJ_MODEL_MAP`.
  3. `conversion/mosstts.py` - the converter itself (registers both classes under the same HF
     arch name; the `MmprojModel` audio-config requirement is satisfied implicitly via the
     auto-inserted `hparams["audio_config"] = {}`).
  Verification: `.venv/bin/python refs/moss-tts-nano-100m-port/verify_conversion.py` ->
  `ALL TENSORS MATCH THE SOURCE STATE DICT`; both GGUFs reproduce byte-identical from the
  in-tree converter (verified 2026-08-14).

  Notes:
  - source weights are bf16 (config says fp32 - misleading); f32 GGUFs preserve them losslessly;
  - the global GPT-2 core uses RoPE, no `pos_embd` - `LLM_ARCH_GPT2`'s graph (learned pos embd)
    does NOT fit; the runtime needs a RoPE-variant (see Phase 1);
  - the 16 codebook embeddings feed the global input as a summed grid; stored 3D in
    `a.gen.code.embd` (mmproj) so the Phase 3 gen-audio graph can build the summed input.
- **Phase 1 - Text decoder forward: DONE (2026-08-14)**. New `LLM_ARCH_MOSSTTS` in C++:
  - `src/llama-arch.{h,cpp}`: arch `mosstts` only - no model-specific tensor entries (the
    gen-code tensors belong to the mtmd subsystem, `a.gen.code.*` in `tools/mtmd/clip-impl.h`);
  - `src/models/mosstts.cpp`: `llama_model_mosstts` - RoPE GPT-2 core (merged QKV + bias,
    ffn biases, gelu_new = ggml GELU, tied output) + per-layer input capture (`t_layer_inp`);
  - `src/models/models.h` + `src/llama-model.cpp`: model class, dispatch, `LLAMA_ROPE_TYPE_NORM`.
  Dual-run logits comparison on the **iGPU (Vulkan)** (`refs/moss-tts-nano-100m-port/logits_dump.cpp`
  C harness, `trace_forward.py` torch reference, `compare_forward.py`):
  - embeddings (token ids): identical (5e-11);
  - per-token logits: max abs diff ~0.13 over 16 384 vocab, argmax identical 13/13 on
    "The quick brown fox jumps over the lazy dog";
  - result_norm: max abs diff ~3.6e-2.
  The residual delta is **fp32 matmul accumulation** (torch/MKL vs ggml FMA+ordering), not a
  semantic bug: all ops verified (RoPE interleaved-pairs convention and freq formula, attention
  scale 1/sqrt(64), LayerNorm eps 1e-5, gelu_new), and torch fp32 sits at 1e-5 from an fp64
  reference while ggml sits at ~1e-2 (different summation order, amplified by the logits
  projection). Greedy/argmax decoding is token-identical; seeded sampling parity is the Phase 2
   question. Exact byte-parity would require matching the reference accumulation order.
   Harness (in refs): build via `g++ ... logits_dump.cpp -lllama -lggml-cpu -lggml-vulkan`; run
   `compare_forward.py --prompt "..."` (needs `refs/moss-tts-nano-100m-port/moss-tts-nano-100m-f32.gguf`
   + the C++ arch in the tree). The harness offloads to the iGPU (`n_gpu_layers = 99`, Vulkan).
- **Phase 2 - Audio generation loop: DONE (2026-08-14)**. Full generation loop validated by
  dual-run (`refs/moss-tts-nano-100m-port/gen_audio_dump.cpp` + `gen_golden.py` + `compare_frames.py`):
  - The global transformer runs through llama.cpp on the iGPU (Vulkan) with **precomputed grid
    embeddings** (`llama_batch.embd`, token-major layout, explicit positions); the 1-layer local
    transformer + 16 codebook heads are executed as a **GGML graph on the same Vulkan backend**
    (1-layer GPT-2 block: LayerNorms, QKV, RoPE, causal attention via `ggml_mul_mat(K,Q)` +
    `ggml_soft_max_ext` with a causal mask, gelu MLP), matching how Phase 3's mtmd component will
    run it. The harness reads its weights from the two GGUFs via a small embedded GGUF reader
    (main + mmproj), matching the split layout. Earlier scalar/OpenMP math in the harness was
    replaced by the GGML graph after review (the graph is ~6x faster).
  - Generation loop: per frame, decode the new row -> raw global hidden -> `ln_f` (output_norm)
    -> local transformer over `[ln_f(hidden), wte(slot), emb(c0)..emb(c_{ch-1})]` -> text logits
    (argmax of end=7 / slot=9) -> 16 channel logits (tied head) with repetition penalty
    (**once per distinct previous token, matching torch.unique - a bug in an early version**)
    -> argmax codes -> build the next row `[slot, c0..c15]`.
  - Verification numbers on "Hello" (continuation, no reference audio; greedy, rep penalty 1.2),
    run on the **iGPU (Vulkan)**:
    - **teacher-forced per-step parity: 111/120 exact frames, 1906/1920 channels (99.3%)**;
      all intermediate states match torch within fp32 accumulation (global hidden 0.005-0.02,
      local hidden ~0.009);
    - free-run (trajectory cascade): 1/120 exact frames, 55.8% channels - the first flip
      (frame 1) changes the next row, diverging the trajectory.
  - The ~1% per-step flips are **fp32 accumulation boundary cases**: the codebook head
    (1024x768) amplifies small hidden diffs into logit deltas that flip argmax (measured case:
    0.009 hidden diff -> ~1.0 logit diff for the winning code). Same root cause as Phase 1.
  - **Conclusion**: the generation algorithm is correct; byte-for-byte audio is blocked by fp32
    matmul accumulation between torch/MKL and the C++ implementation. Greedy token parity is
    ~99% per step and cascades on free run; exact byte-parity would require matching the
    reference accumulation order (not feasible cross-backend). Seeded-sampling parity is the
    remaining question. The iGPU (Vulkan fp32) numbers are essentially identical to the earlier
    CPU runs (Phase 1 logits diff 0.125 vs 0.126; Phase 2 parity 99.3% vs 98.9%) - the
    accumulation-level divergence is platform-independent.
  - Harness: build `g++ -O2 -std=c++17 -fopenmp -I include -I ggml/include -I common
    refs/moss-tts-nano-100m-port/gen_audio_dump.cpp -L build/bin -Wl,-rpath,$PWD/build/bin
    -lllama -lggml -lggml-base -lggml-cpu -lggml-vulkan -o build/bin/gen_audio_dump`; run
    `compare_frames.py --prompt "..." [--teacher]`; `gen_audio_dump ... --sample --seed 42`
    for seeded torch-compatible sampling.
  - **iGPU listening comparison** (2026-08-14, `gen_igpu_audio.py`):
    `tmp/moss-tts-nano-100m/test-set-igpu/<name>-{ref,port}.wav` for all 26 test-set texts
    (port = global + GGML-graph local transformers on the iGPU, tokens decoded by the reference
    codec since the codec is Phase 3). Long texts are chunked exactly like the reference
    `inference()` path (<=75-token chunks via `_split_text_into_best_sentences`, one generate per
    chunk, waveforms concatenated with the same inter-chunk pauses); single-pass generation over a
    very long sequence degenerates in the model itself, so chunking is required to keep both ref and
    port audible to the end (en-long 3 chunks -> 35 s, en-frame-window 5 chunks -> 70-75 s).
    The harness replays torch's CPU sampling stream bit-for-bit from the same seed per chunk
    (`at::mt19937` + `torch.multinomial` Gumbel fast path: `argmax(probs / Exp(1))`,
    temperature/top-k/top-p in torch order). Per-case parity in `results.json` ranges from
    near-identical short prompts (en-short 39/39 frames, cs-short 31/38, pl-short 24/41) to
    divergent cascades on long texts (en-long 68/425, en-frame-window 58/864), exactly the
    fp32-accumulation picture: logit flips get amplified because a single flipped sampled token
    changes the next frame's inputs. ALL 26 `<name>-port.wav` are valid, non-silent speech (rms
    ~0.6-6k, durations 3-75 s, no silent/garbage endings) - the earlier greedy-only run collapsed
    long cases to near-silence (greedy + repetition penalty degenerates into a repeated frame the
    codec renders as ~silence; the sampled reference confirms this is not a port bug). A/B the
    ref/port pairs by ear; short prompts track closely, long prompts diverge audibly but stay
    intelligible.
    Harness flags: `--sample [--seed N] [--text-temp/top-p/top-k] [--audio-temp/top-p/top-k]`.

### Running the tests (CLI)

Everything runs from the fork root with the project `.venv`; the harness needs the C++ arch
in the tree and the build dir. Build the Phase 2 harness once:

```sh
g++ -O2 -std=c++17 -fopenmp \
    -I include -I ggml/include -I common \
    refs/moss-tts-nano-100m-port/gen_audio_dump.cpp \
    -L build/bin -Wl,-rpath,$PWD/build/bin \
    -lllama -lggml -lggml-base -lggml-cpu -lggml-vulkan \
    -o build/bin/gen_audio_dump
```

Full 26-case iGPU listening comparison (reference torch on CPU + port on the iGPU/Vulkan,
seeded sampling, chunked long texts) -> `tmp/moss-tts-nano-100m/test-set-igpu/`:

```sh
.venv/bin/python refs/moss-tts-nano-100m-port/gen_igpu_audio.py
```

`gen_igpu_audio.py` options:
- `--only en-long,en-frame-window` - run a subset of cases (comma-separated names);
- `--seed N` - seed for both ref and port sampling (default 42);
- `--max-frames N` - per-chunk audio frame cap (default 375, the reference default);
- `--voice-clone-max-text-tokens N` - chunk budget (default 75, the reference default);
- `--no-sample` - greedy instead of sampled (not recommended: collapses to silence on long texts);
- `--output-dir DIR` / `--cases FILE` / `--prompt-audio-dir DIR`.

Outputs per case: `<name>-ref.wav` (torch), `<name>-port.wav` (llama.cpp iGPU),
`<name>-grid-<chunk>.txt` (prompt grids), and `results.json` (per-chunk + per-case parity).

Reference-only test set (torch `model.inference`, 26 cases) -> `tmp/moss-tts-nano-100m/test-set/`:

```sh
PYTHONPATH=refs/moss-tts-nano .venv/bin/python refs/moss-tts-nano-100m/run_test_set.py
```

(`run_test_set.py` imports the `moss_tts_nano` package from the official repo clone under
`refs/moss-tts-nano/`; that path must be on `PYTHONPATH`. `gen_igpu_audio.py` and the other
port scripts use `transformers` remote code directly and do not need it.)

Single-prompt golden / parity checks (Phase 1/2 debugging):

```sh
g++ -O2 -std=c++17 -fopenmp \
    -I include -I ggml/include -I common \
    refs/moss-tts-nano-100m-port/logits_dump.cpp \
    -L build/bin -Wl,-rpath,$PWD/build/bin \
    -lllama -lggml -lggml-base -lggml-cpu -lggml-vulkan \
    -o build/bin/logits_dump
.venv/bin/python refs/moss-tts-nano-100m-port/compare_forward.py --prompt "The quick brown fox jumps over the lazy dog"
.venv/bin/python refs/moss-tts-nano-100m-port/gen_golden.py --prompt "Hello" --max-frames 120   # greedy golden
.venv/bin/python refs/moss-tts-nano-100m-port/compare_frames.py --prompt "Hello" [--teacher]
```
- **Phase 3 - Codec + pipeline** (next): codec GGUF (Cat arch, ~22M), `-mv` wiring, new
  `MTMD_GEN_AUDIO_TYPE_MOSSTTS` gen_audio pipeline (copy `pockettts`): encode prompt, autoregressive
  decode, decode codes -> WAV; compare WAV bytes.
- **Phase 4 - Full suite + server**: byte-for-byte test across the whole test set; in-process server
  wiring via the `mtmd_helper_gen_audio_*` bindgen shims (no shell-out).

## HOWTO-add-model compliance audit (2026-08-14)

Checklist against `docs/development/HOWTO-add-model.md` (also referenced from
`docs-ports/llama-cpp-integration-practices.md`):

1. **Convert to GGUF** - DONE. `ModelBase.register` on a `TextModel` (`MossTTSNanoModel`) and a
   `MmprojModel` (`MossTTSNanoGenModel`) subclass; `MODEL_ARCH.MOSSTTS` + name + text-only
   `MODEL_TENSORS`; `VisionProjectorType.MOSSTTS_GEN`. Tensor mapping reuses the existing GPT-2
   entries in `tensor_mapping.py` (no new entries added); gen-code tensors are mapped manually in
   `modify_tensors` (same bypass as Qwen3-TTS). All tensors end in `.weight`/`.bias`.
   Converter fully integrated in-tree: `conversion/mosstts.py`, registered in both
   `TEXT_MODEL_MAP`/`MMPROJ_MODEL_MAP`, arch constants in `gguf-py/gguf/constants.py`.
   Both GGUFs reproduce byte-identical from the in-tree converter (verified 2026-08-14).
2. **Define the arch in llama.cpp** - DONE. `LLM_ARCH_MOSSTTS` in `llama-arch.h`; name in
   `LLM_ARCH_NAMES`; no new `LLM_KV_NAMES`/`LLM_TENSOR_NAMES`/`LLM_TENSOR_INFOS` (all tensor roles
   already exist); `load_arch_hparams` in the model class (standard pattern); `LLAMA_ROPE_TYPE_NORM`
   in `llama_model_rope_type`; `llama_model_mapping` dispatch.
3. **Other `llm_arch` switch sites** (HOWTO warns these are common CI failures) - AUDITED:
   - `llama-model-saver.cpp` `llama_model_saver_supports_arch`: mosstts uses the default `true`
     (quantize/resave supported; verified: `llama-quantize` Q8_0 works and the Q8_0 model keeps
     argmax parity 13/13 vs f32 and vs torch).
   - `test-llama-archs`: passes for `mosstts` on CPU and the iGPU backend.
   - all other `if (arch == X)` conditionals in `llama-context.cpp`, `llama-graph.cpp`,
     `llama-kv-cache.cpp`, `llama-quant.cpp`, `llama.cpp`, `llama-adapter.cpp` are model-specific
     special cases with generic defaults; mosstts takes the defaults. No MoE mandatory-hparam list
     applies (dense model).
4. **GGML graph** - DONE. `llama_model_mosstts` inherits `llama_model_base`; `build_arch_graph`
   returns a `llm_graph_context`; graph uses `ggml_rope_ext` (NORM) per the HOWTO tip.
5. **Backend / tool checks** (HOWTO: cli, completion, imatrix, quantize, server; CUDA/METAL/CPU) -
   PARTIAL:
   - the runtime inference runs on the **iGPU (Vulkan)** (`n_gpu_layers = 99`, offloaded 13/13) -
     the intended platform; cli: OK; quantize: OK (Q8_0, argmax parity); imatrix: OK; completion:
     uses the same decode path as cli (not separately exercised); server: not tested (needs the
     Phase 3 gen-audio pipeline); CUDA/METAL: not verifiable on this machine.
6. **Multimodal encoder (mtmd)** - PENDING (Phase 3). The converter side (`MmprojModel`) is done;
   the `libmtmd` encoder definition (`clip.cpp`), preprocessor (`mtmd.cpp`) and the gen-audio GGML
   graph (`tools/mtmd/models`, per `tools/mtmd/README-dev.md` for audio generation models) remain.
   The `a.gen.code.*` tensor names already match the mtmd `TN_A_GEN_CODE_*` convention.

## Validation results (2026-08-13, reference runs; iGPU for the port, uv venv)

- Official HF PyTorch run validated. Wav outputs in `tmp/moss-tts-nano-100m/` (disposable, listen here).
- Reference implementation preserved in `refs/moss-tts-nano/` (official GitHub repo clone: infer.py,
  moss_tts_nano package, assets/audio reference clips) + reproducer `refs/moss-tts-nano-100m/run_samples.py`.
- Recipe: run with the local HF snapshots as
  `--checkpoint` / `--audio-tokenizer-pretrained-name-or-path`, `--disable-wetext-processing`
  (WeTextProcessing/pynini not needed; robust `normalize_tts_text` kept on).
  Driver: `refs/moss-tts-nano-100m/run_samples.py` (loads model once, 14 samples, seed 42,
  voice_clone mode, do_sample, audio temp 0.8 / top_p 0.95 / rep penalty 1.2).
- Samples: 14 wavs, 11 languages (en x3 incl. numbers/date, zh x2, ja, de, fr, es, it, pt, ru, ar, ko).
  Language-matched reference prompts used where available (`en_*.wav`, `zh_*.wav`, `jp_2.wav`), else en.
- All wavs verified valid: 48 kHz, stereo, 2.9-8.9 s, RMS 0.06-0.13 (non-silent), peak <= 1.0.
  Same (text, seed) reproduces bit-identical wav (smoke-en.wav == en-01-fox.wav).
- Reference (torch) performance: ~4 s per short sample after load (~30 s model load, fp32); the
  torch reference is CPU-only (venv build), so the reference runs on CPU while the port runs on
  the iGPU.
- Reproducibility: seed 42 via `torch.manual_seed`.
