# MOSS-TTS-Nano-100M port to llama.cpp (analysis / reference)

Source: `/home/blep/prj/mallomar-server/docs/model-llama-cpp-ports.md` (candidate: MOSS-TTS-Nano-100M)
Status: **port target - selected over Moondream 0.5B (2026-08-14)**. Phase 0 (GGUF conversion) DONE and
verified 2026-08-14; next: Phase 1 (text decoder forward).
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

Covered axes (in `refs/moss-tts-nano-100m/test_cases.json`, one entry per case):
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

- **Phase 0 - Conversion: DONE (2026-08-14)**. Decoder GGUF produced and verified:
  `refs/moss-tts-nano-100m-port/moss-tts-nano-100m-f32.gguf` (570 265 824 B f32, 168 tensors).
  Verified with `refs/moss-tts-nano-100m-port/verify_conversion.py`: all 168 tensors
  byte-identical to the source state dict. Layout, prerequisites and reproduction in
  `refs/moss-tts-nano-100m-port/CONVERSION.md`; converter source:
  `refs/moss-tts-nano-100m-port/conversion-mosstts.py`.

  **How to run the conversion** (from the fork root, uv venv with torch/transformers/
  sentencepiece/numpy - the project `.venv` works):
  ```sh
  .venv/bin/python convert_hf_to_gguf.py \
      /home/blep/ai-models/llm-ports/OpenMOSS-Team/MOSS-TTS-Nano-100M \
      --outfile refs/moss-tts-nano-100m-port/moss-tts-nano-100m-f32.gguf \
      --outtype f32
  ```
  **Prerequisites** (not in the repo - the framework edits were reverted to keep the
  llama.cpp tree clean; see `refs/moss-tts-nano-100m-port/CONVERSION.md` for the exact steps):
  1. `gguf-py/gguf/constants.py`: register arch `mosstts` (enum `MODEL_ARCH.MOSSTTS`, name
     `"mosstts"`, `MODEL_TENSORS` = GPT-2 core tensors + `A_GEN_CODE_*`, mirror the `POCKETTTS`
     entry).
  2. `conversion/__init__.py`: add `"MossTTSNanoForCausalLM": "mosstts"` to `TEXT_MODEL_MAP`.
  3. Put `conversion-mosstts.py` (from refs/) at `conversion/mosstts.py` in the fork.
  Verification: `.venv/bin/python refs/moss-tts-nano-100m-port/verify_conversion.py` ->
  `ALL TENSORS MATCH THE SOURCE STATE DICT`.

  Notes:
  - source weights are bf16 (config says fp32 - misleading); f32 GGUF preserves them losslessly;
  - the global GPT-2 core uses RoPE, no `pos_embd` - `LLM_ARCH_GPT2`'s graph (learned pos embd)
    does NOT fit; the runtime needs a RoPE-variant (see Phase 1);
  - the 16 codebook embeddings feed the global input as a summed grid; stored 3D in
    `a.gen.code.embd` so Phase 2 can build the summed input embedding;
  - `TextModel.filter_tensors` drops `audio_*` names - the converter allowlists them.
- **Phase 1 - Text decoder forward** (next): new C++ `mosstts` arch + model class (RoPE GPT-2
  core, no pos_embd); run the text core in llama.cpp, compare logits against torch for a fixed
  prompt (tokenizer -> embeddings -> 12 layers -> lm_head).
- **Phase 2 - Audio generation loop**: codebook embeddings + local transformer (1 layer) + 16-head
  sampling (temperature 0.8 / top-p 0.95 / top-k 25 / repetition penalty 1.2 defaults); compare
  generated audio token ids (greedy) vs golden.
- **Phase 3 - Codec + pipeline**: codec GGUF (Cat arch, ~22M), `-mv` wiring, new
  `MTMD_GEN_AUDIO_TYPE_MOSSTTS` gen_audio pipeline (copy `pockettts`): encode prompt, autoregressive
  decode, decode codes -> WAV; compare WAV bytes.
- **Phase 4 - Full suite + server**: byte-for-byte test across the whole test set; in-process server
  wiring via the `mtmd_helper_gen_audio_*` bindgen shims (no shell-out).

## Validation results (2026-08-13, CPU-only, uv venv)

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
- CPU performance: ~4 s per short sample after load (~30 s model load, fp32).
- Reproducibility: seed 42 via `torch.manual_seed`.
