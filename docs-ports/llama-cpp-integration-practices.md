# llama.cpp integration practices (for ports)

Knowledge captured during the MOSS-TTS-Nano-100M and Moondream 0.5B ports, so future ports
integrate with llama.cpp the way its maintainers expect. Verified against the tree and its
history on 2026-08-14.

The authoritative, step-by-step procedure for adding a model is
[`docs/development/HOWTO-add-model.md`](../docs/development/HOWTO-add-model.md) (conversion,
arch definition, GGML graph, optional multimodal encoder, plus backend/tool checks). This doc
complements it with the tensor-registry and GGUF-structuring rules learned during the ports;
follow the HOWTO for the concrete steps, and use the rules below when it leaves choices open.

## Scope

Two tensor-name registries exist and serve different purposes:

- `gguf-py/gguf/constants.py` `MODEL_TENSOR` - the GGUF authoring side (converter). One entry
  per tensor ROLE, organized in sections.
- `src/llama-arch.h` `enum llm_tensor` + `LLM_TENSOR_NAMES`/`LLM_TENSOR_INFOS` (llama-arch.cpp) -
  the C++ runtime side. Also one entry per tensor ROLE for the TEXT model graph only.

## `LLM_TENSOR` semantics (C++)

- The enum enumerates tensor ROLES for the text-model graph: `TOKEN_EMBD`, `ATTN_NORM`,
  `ATTN_Q/K/V`, `FFN_UP/DOWN`, `OUTPUT`, `OUTPUT_NORM`, ... `LLM_TN_IMPL::str()` builds the GGUF
  tensor name from the enum + suffix + block id, via `LLM_TENSOR_NAMES`. `LLM_TENSOR_INFOS`
  classifies each tensor (layer scope + expected op) for buffer/offload probing.
- There is NO raw-name create path in `llama_model_loader::create_tensor`; a tensor can only be
  claimed through the enum.
- It is deliberately GENERIC and SHARED across models. New models must reuse existing entries.
  Model-specific text-side entries exist only as rare exceptions (e.g. `DSPARK_MARKOV_W1`,
  `MASKED_EMBD_CENTROIDS`, `HC_HEAD_FN`) and are named by the tensor's ROLE (optionally with an
  arch-scoped prefix like `DSPARK_`), never by the model name.

## The mtmd subsystem (multimodal and audio tensors)

- Vision (`V_*` / `mm.*`), audio-generator (`A_*` / `a.gen.*`, `a.gen.wav.*`) and other
  multimodal tensors are NOT part of the C++ `LLM_TENSOR` enum. The C++ enum has zero `V_*`/`MM_*`
  entries.
- They are loaded by the mtmd subsystem in `tools/mtmd/`, which has its OWN tensor-name macros in
  `tools/mtmd/clip-impl.h` (e.g. `TN_A_GEN_CODE_EMBD "a.gen.code.embd.%s"`,
  `TN_A_GEN_WAV_*`, `TN_MM_*`, `TN_ATTN_*`). The clip loader consumes those names directly.
- On the gguf-py side, `MODEL_TENSOR` mirrors this: a `# audio (mtmd)` section holds
  `A_ENC_*`, `A_GEN_CODE_*`, `A_GEN_WAV_*`; a vision section holds `V_*`. These are written by
  mmproj converters and read by mtmd.

## GGUF structuring: main model + mmproj

Models with an audio/video generator follow the two-GGUF pattern (see Qwen3-TTS):

- Main GGUF: the TEXT backbone (arch-specific, only text tensors).
- Second GGUF (the "mmproj"): the generator / codec tensors (`a.gen.*`), loaded by mtmd
  (CLIP modality `GEN_AUDIO`), wired via `-mv` / `--mmproj`.

The converter expresses this as TWO model classes: a `TextModel` for the main GGUF and an
`MmprojModel` for the second GGUF (e.g. `Qwen3TTSTalkerModel` + `Qwen3TTSSpeakerEncoderModel` in
`conversion/qwen3tts.py`; the latter calls `add_clip_has_gen_audio_encoder(...)` and
`add_clip_gen_audio_projector_type(...)`). MOSS-TTS-Nano-100M follows the same split:
`MossTTSNanoModel(TextModel)` + `MossTTSNanoGenModel(MmprojModel)` in `conversion/mosstts.py`,
with a dedicated `VisionProjectorType.MOSSTTS_GEN = "mosstts_gen"`.

### Converter-side wiring (what "just copying the file" misses)

The converter module alone is not enough; the arch must be registered in the gguf-py registry
or `format_tensor_name()` (conversion/base.py) raises `Missing {key!r} for MODEL_TENSORS` and
`TensorNameMap` populates no mappings for it. The required, in-tree edits are:

1. `gguf-py/gguf/constants.py`:
   - `MODEL_ARCH.MOSSTTS` enum entry + `"mosstts"` in `MODEL_ARCH_NAMES`;
   - a `MODEL_TENSORS` list for the TEXT arch (text tensors only);
   - a `VisionProjectorType.MOSSTTS_GEN` entry when the mmproj writes gen-audio tensors.
2. `conversion/__init__.py`: the HF arch name in BOTH `TEXT_MODEL_MAP` and `MMPROJ_MODEL_MAP`
   (both converter classes register under the same name, one per map) - this is what
   `get_model_class(name, mmproj=...)` and `load_all_models()` dispatch on.

Note: `MmprojModel.__init__` normally requires a `vision_config`/`audio_config` in the checkpoint
hparams, but auto-inserts `hparams["audio_config"] = {}` when absent, so a generator-only mmproj
(no encoder) loads fine without one.

## Historical precedent

- `mtmd: support Qwen3-TTS (#26254)` and `mtmd: support pocket-tts (#26871)` added ZERO
  `LLM_TENSOR` entries; their code predictor / codec / flow-decoder tensors live in mtmd.
- `model: Muse Glimmer Support (#26841)` (an audio model) added `LLM_ARCH_MUSE_GLIMMER` and ZERO
  new tensor entries (reuses generic `ATTN_*`/`FFN_*`).
- Therefore: adding a new `LLM_ARCH` is normal; adding new `LLM_TENSOR_*` entries is exceptional
  and only justified for text-side roles that no existing entry covers.

## Practical rules for a new port

1. Map the model's weights onto existing `LLM_TENSOR_*` roles; reuse them.
2. If the model has multimodal/audio-generator weights (vision, codec, code predictor, flow
   decoder), put them in the SECOND GGUF and load them through mtmd - do not add `LLM_TENSOR_*`
   entries and do not put them in the main text GGUF.
3. Keep the main GGUF to the text backbone only, so the C++ text loader claims exactly the
   tensors it uses.
4. For the dual-run harness (dev tool, not committed): read generator weights from the second
   GGUF directly (small GGUF reader) or via a dev-merged file; never force the main model to
   claim them.
5. Name new `MODEL_TENSOR`/`LLM_TENSOR` entries by ROLE, not by model, and reuse the mtmd
   `TN_*` name strings when the tensor is a generator tensor.

## What MOSS-TTS did wrong (and the fix)

- Phase 0 put ALL tensors (global transformer + local transformer + 16 codebook tables/heads)
  into ONE GGUF (arch `mosstts`), and the C++ text loader claimed the `a.gen.code.*` tensors via
  new `LLM_TENSOR_A_GEN_CODE_*` enum entries.
- That duplicated the mtmd `TN_A_GEN_CODE_*` mechanism and polluted the text-model enum.
- Correct layout (applied 2026-08-14): main GGUF `mosstts` = global transformer only;
  second gen-audio GGUF = the `a.gen.code.*` component, to be loaded by mtmd in Phase 3
  (`MTMD_GEN_AUDIO_TYPE_MOSSTTS`). `LLM_TENSOR_A_GEN_CODE_*` removed. The converter is now
  fully integrated in-tree (`conversion/mosstts.py` + constants + both model maps; both GGUFs
  regenerate byte-identical from it).
