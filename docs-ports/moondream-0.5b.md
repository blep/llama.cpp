# Moondream 0.5B port to llama.cpp (analysis / reference)

Source: `/home/blep/prj/mallomar-server/docs/model-llama-cpp-ports.md` (candidate: Moondream 0.5B, vision)
Status: **deferred (2026-08-14)** - MOSS-TTS-Nano-100M selected as the first port target.
Date: 2026-08-13 (updated 2026-08-14)

Note: Moondream is a **vision-language model** (image caption / VQA), not TTS. "Validation" = run the official
runtime on images. No audio output.

## Port decision (2026-08-14): deferred behind MOSS-TTS-Nano-100M

- Chosen order based on hard facts (see `docs-ports/moss-tts-nano-100m.md` "Port decision"):
  Moondream's conversion source is **int8-quantized ONNX inside the `.mf`** (no raw safetensors,
  no converter), its text arch is custom (no support), and the mtmd vision pipeline (tower +
  multi-crop reconstruction + projector) has zero precedent in this fork.
- **Correction to the source port doc**: the claim "the vision tower is shared with the 0.5B"
  is disproved by the extracted dims - 0.5B vision is enc_dim 720, MLP ff 2690, projection
  1440->8192->1024, while Moondream 2/3 use enc_dim 1152, ff 4304, projection 2048. The 0.5B
  projector also needs the crop reconstruction concat (global + avg-pooled local tiles).
- The source doc's "smallest surface / ~2-4 focused days" estimate predates discovering that the
  `.mf` embeds only int8 DynamicQuantizeMatMul weights; the estimate is not actionable as written.

## License (verified 2026-08-13)

- **Moondream 0.5B weights + code: Apache-2.0** - release blog (2024-12-05) states Apache License;
  the official repo `vikhyatk/moondream` LICENSE is the standard Apache-2.0 text; the hosting HF
  repo `vikhyatk/moondream2` (which carries the `.mf` artifacts) card is `license:apache-2.0`.
- **Compatibility**: Apache-2.0 is permissive (commercial use, modification, redistribution
  allowed); compatible with llama.cpp's MIT license for the ported runtime. No license barrier to
  porting or redistributing converted GGUF weights, provided the Apache-2.0 license text is kept.

## Model identity

- **Officially documented**: blog "Introducing Moondream 0.5B" (2024-12-05)
  https://moondream.ai/blog/introducing-moondream-0-5b — release day announcement, Apache-2.0,
  "world's smallest VLM", distillation target for edge devices building on Moondream 2B.
- Standalone HF repo `vikhyatk/moondream-0.5B` does **not exist** (404, verified 2026-08 HF API).
  The official artifacts are hosted inside the `vikhyatk/moondream2` HF repo at pinned commit
  `9dddae84d54db4ac56fe37817aeaeb502ed083e2`:
  - `moondream-0_5b-int8.mf.gz` - 621 619 051 B (593 MiB); int8 runtime memory 996 MiB
  - `moondream-0_5b-int4.mf.gz` - 442 376 060 B (422 MiB); int4 runtime memory 816 MiB
  - (blog figures "479 MiB / 375 MiB compressed" are understated vs measured .gz sizes)
- Official 0.5B release artifact: the `.mf` binary format (header magic `MOON`, NOT GGUF)
  - `moondream-0_5b-int8.mf` (692 802 047 B uncompressed, int8)
  - Loaded by the official `moondream` pip package: `md.vl(model=...)`
- **Canonical download location** (per release-day README, commit `d12435c3e7`):
  `https://huggingface.co/vikhyatk/moondream2/resolve/9dddae84d54db4ac56fe37817aeaeb502ed083e2/moondream-0_5b-int8.mf.gz`
  (blog "download here" -> moondream GitHub README)
- Arch: 0.5B custom dense decoder (NOT phi) + SigLIP-style vision tower (verify)
- No public raw safetensors; closest raw checkpoints are `vikhyatk/mystery-ckpts` (.pt, 2B-class ~4.16 GB)
- The HF transformers hub version tracks only the 2B model and does not support 0.5B (per
  release-day README) - hence no `vikhyatk/moondream-0.5B` HF repo.
- Community mirrors verified 2026-08:
  - `grifttown/moondream-0.5b` = int8 `.mf` mirror (etag-identical to `andito/moondream05`)
  - `whistleroosh/moondream-0.5B` = ONNX exports

## Local materials (downloaded 2026-08)

- `/home/blep/ai-models/llm-ports/grifttown/moondream-0.5b/moondream-0_5b-int8.mf`
  - sha256 `53793b60e3a0fe091bcd3e9e1853908a150367ed2d8d45e083eb79a5fbddbcdb`
  - **verified byte-identical (sha256) to the canonical official artifact** decompressed from
    `vikhyatk/moondream2@9dddae8.../moondream-0_5b-int8.mf.gz` (2026-08-13)
- Convert source options: the `.mf` embedded tensors, or raw weights requested from the team
- Dual-run reference: the official `moondream` pip package (md.vl)

## Official run recipe (reference to reproduce)

- Release-day README pins `pip install moondream==0.0.5`; the 0.0.6 client (extracted to
  `refs/moondream-0.5b/onnx-client/`) is functionally identical for `.mf` loading.
- Usage: `import moondream as md; model = md.vl(model="<path-to>.mf"); model.caption(images=..., length="short")`
- Offline: pass local `.mf` path; no HF download needed
- Current PyPI `moondream` (2.0.1) is kestrel/Photon and GPU-only; see Validation results below.

## Reference GGUF

- `ggml-org/moondream2-20250414-GGUF` (text f16 2.8 GB + mmproj f16 910 MB) - llama.cpp ran Moondream 2 via upstream PR #13745 (phi text arch + vicuna template + projector)
- Vision tower is shared with the 0.5B -> projector half ports as a variant

## Caveat

- Current mainline AND this fork have **zero moondream** (arch + mtmd greps = 0); old support was clip-era, dropped in the mtmd refactor. This is a fresh port with historical code as reference, not a bump.

## Port work items (from source doc)

- (a) new text arch (tensor map + graph + convert script; input: `.mf` embedded tensors or raw weights)
- (b) projector variant (SigLIP tower)
- (c) chat template (vicuna-style from Moondream 2)
- Effort: ~2-4 focused days

## Validation results (2026-08-13, CPU-only, uv venv)

- Official runtime validated. Reference material preserved in `refs/`:
  - `refs/moondream-0.5b/onnx-client/` - official `md.vl` ONNX runtime (moondream 0.0.6, verbatim)
  - `refs/moondream-0.5b/extracted-mf/` - `.mf` contents (config.json, tokenizer.json,
    initial_kv_cache.npy, 8 ONNX subgraphs) = the port's convert source
  - `refs/moondream/` - official `vikhyat/moondream` torch repo (Moondream 2/3 reference)
  - `refs/moondream-0.5b/validate.py` / `run_caption.py` - reproducers; `results.json` - outputs
- The current `moondream` PyPI package (2.0.1) is kestrel/Photon based and requires CUDA or MPS;
  it will NOT run on CPU. The CPU-capable official client is the old `moondream` 0.0.6 package
  (`md.vl(model="...mf")`), which runs the embedded ONNX subgraphs with onnxruntime (CPU).
- **`.mf` format (fully decoded, key port input)**: magic `MOON` + 1-byte version + records
  `u32 BE name-len, name, u64 BE content-len, content`. Contains: `config.json` (templates +
  special_tokens), `initial_kv_cache.npy` (f16, shape (24,2,1,16,1,64) = 1-token KV seed),
  `tokenizer.json`, and 8 ONNX subgraphs:
  - `vision_encoder` (patch_count,3,378,378)f16 -> (patch_count,729,720)  [SigLIP-style tower]
  - `vision_projection` (1,729,1440) -> (1,729,1024)  [fc 1440->8192->1024]
  - `text_encoder` (1,seq)i64 -> (1,seq,1024)  [embedding table 51200x1024 f16]
  - `text_decoder` (1,seq,1024)+(24,2,1,16,T,64) -> hidden + new_kv_cache + logits (1,51200)
  - `coord_encoder/decoder`, `size_encoder/decoder` (region/detect heads, 1024-dim)
  - Weights: int8 DynamicQuantizeMatMul (u8 + f32 scales) + f16 activations (hence "int8" .mf)
  - Arch summary: dim 1024, 24 layers, 16 heads, head_dim 64, vocab 51200, RoPE; vision 720 dim,
    27x27=729 patches per 378x378 crop, MLP ff 2690; text ff 4096
- **Inference flow** (from onnx_vl.py): image -> BICUBIC-resize global crop 378x378 + optional
  (1,2)/(2,1)/(2,2) tiles; vision_encoder on all crops; concat global features with
  avg-pooled reconstructed tile features -> 1440; vision_projection -> 729x1024 embeds;
  text_decoder prefill with `initial_kv_cache` seed (1 token) + image embeds -> KV pos 730;
  prefill template (`\n\nCaption:` etc., from config.json) -> greedy argmax decode until eos.
- Captions/VQA validated on both official demo images; outputs coherent, ~6 s/image encode + ~10 s caption on CPU (8 threads).

### Test inputs (exact)

- Images:
  - demo-1: `refs/moondream/assets/demo-1.jpg` (773x767) - girl in a restaurant
  - demo-2: `refs/moondream/assets/demo-2.jpg` (822x966) - equipment console / rack
  - ocr_test: `refs/moondream-0.5b/ocr_test.png` (287x83, RGBA) - OCR test strip, a snippet of
    text captured from the Kate editor; source `mallomar-server/tests/sdk_conformance/data/ocr_test.png`
- Prompts (model templates from `.mf` `config.json`, applied by the official client):
  - caption: `\n\nCaption:` (tokens `[198, 198, 24334, 1159, 25]`)
  - caption_short: `\n\nShort caption:` (tokens `[198, 198, 16438, 8305, 25]`)
  - query: `\n\nQuestion:<question>\n\nAnswer:` (template prefix `[198,198,24361,25]` + encoded question + suffix `[198,198,33706,25]`)
  - questions: demo1/demo2 -> "What is the girl doing?", "What color is the girl's hair?", "What is this?";
    ocr_test -> "What text is written in the image?", "What does the text in the image say?"
- Decoding: greedy argmax (no sampling), stop at eos `50256`, max 512 tokens.
- Full prompt = `initial_kv_cache` seed (1 tok) + 729 image tokens + template tokens.

### Outputs (verbatim, from `refs/moondream-0.5b/results.json`)

demo-1:
- caption: " The image depicts a young girl with light blue hair and elf-like features, wearing a white tunic with a green stripe. She is holding a large, brown pastry, possibly a donut, in her hands. The girl is seated at a table, which has a dark brown surface, and is wearing a ring on her left ring finger. The background features a dimly lit restaurant or cafe, with a partially visible person holding a plate of food. The colors in the image are predominantly blue, green, and brown, creating a harmonious and visually appealing scene."
- caption_short: " A young elf, dressed in a white robe and green elf hat, holds a golden-brown pastry in a cozy restaurant, surrounded by other patrons."
- Q "What is the girl doing?" -> " The girl is holding a hot dog and appears to be eating it."
- Q "What color is the girl's hair?" -> " The girl's hair is blue."
- Q "What is this?" -> " The image features a young, pretty girl with grey hair, wearing a white and black dress. She is holding a hot dog in her hand and appears to be enjoying her meal."

demo-2:
- caption: " The image shows a DJ setup, featuring a black DJ console with a built-in DJ controller. The console is equipped with a variety of audio components, including a DJ controller, a control panel, and a power amplifier. The DJ console is also equipped with a DJ mixer and a control panel. The DJ setup is placed on a black floor mat, which contrasts with the black and blue tones of the equipment. In the background, a brick wall provides a backdrop to the DJ setup, and a black speaker is visible to the left of the DJ setup."
- caption_short: " A DJ setup features a black DJ console with a blue mixer, black cables, and a black stand, accompanied by a black amplifier and a black speaker."
- Q "What is the girl doing?" -> " The girl is standing in front of a DJ booth, operating the DJ equipment, and possibly waiting for her turn to play music." (hallucinated - no girl in image)
- Q "What color is the girl's hair?" -> " The girl's hair is blue." (hallucinated)
- Q "What is this?" -> " This image features a large, black, and shiny DJ setup with a variety of equipment and controls. The equipment includes a DJ controller, a mixer, and a mixer. There are also cables and wires connecting the various components, indicating a complex and well-equipped setup. The DJ setup is placed on a carpeted floor, providing a comfortable and ergonomic environment for the user."

Note: the repo README labels demo-2 a "computer server rack"; the model calls it a DJ setup. That is model content drift (and the girl-related queries hallucinate on demo-2), not a runtime failure.

ocr_test (OCR; 287x83 text strip, ~2.3 s encode):
- caption: " The image presents a digital text that reads "The bug sun was under the moon shining. Halo to hare hare hare [repeated ~200x]" - the long caption degrades into repetition after the first line
- caption_short: " A digital display reads "The big sun was under the moon shining. Hold to 895244015" in white text against a black background."
- Q "What text is written in the image?" -> " The text at the top of the image says, "The bug sun was under the moon shining." At the bottom of the image, there is a text that reads "Halo 89527244015.""
- Q "What does the text in the image say?" -> " The text at the bottom of the image says: "Halo to 89527244015"."
- Assessment: the model does read the text (2-3 short lines + a long number) but OCR is lossy: "big/bug" confusion, "Halo/Hold" confusion, digit errors (89527244015 vs 895244015). The 0.5B's OCR is approximate - expect errors on small/faint text. This matters for the port: OCR accuracy is a model-quality property, not a runtime property.
- NOTE: image preprocessing differs from the newer overlap-crop code in `vikhyat/moondream`
  torch repo (used by Moondream 2/3); the 0.5B ONNX runtime uses simple templates + BICUBIC.
