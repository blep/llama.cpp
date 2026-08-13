# Voxtral model inventory (local collection)

Status of the local Voxtral collection under `/home/blep/ai-models/`:

- **`llm/`** - GGUF files, mirroring the LM Studio layout convention:
  `<base>/<org>/<model-name>-GGUF/*.gguf` (consumed by llama.cpp / voxtral.cpp).
- **`hf/`** - raw Hugging Face safetensors repos for vLLM / transformers:
  `<base>/<org>/<model-name>` (mirrors the HF repo ID).

Downloaded 2026-08-13 from Hugging Face. All weights are the official Mistral
Voxtral models; the GGUF conversions come from the quantizer orgs listed per
model (Mistral publishes safetensors only - see "Sources" below).

## Two GGUF ecosystems in this collection

Voxtral = text decoder + audio encoder (the "audio tower"). Converters ship
them two ways, and the layout determines which runtime can consume the
files. The "Ecosystem" column in each table below uses these labels:

1. **llama.cpp (mtmd)** - split layout: text GGUF (`llama` arch) + separate
   `mmproj-*.gguf` audio encoder (`clip` arch, `projector_type=voxtral`).
   This is what this fork's `mtmd` tool consumes (`(tool_name) -hf <repo>`
   pulls model + mmproj together). Orgs: `ggml-org`, `bartowski`.
2. **voxtral.cpp / CrispASR** - merged layout: text + audio encoder in ONE
   file with a custom arch (`voxtral_realtime`, `voxtral_tts`), not loadable
   by llama.cpp today. These target the voxtral.cpp / CrispASR runtimes
   (or a future llama.cpp arch addition). Orgs: `cstr`, `andrijdavid`.

## Voxtral Mini (3B, 2025-07 release) - `mistralai/Voxtral-Mini-3B-2507`

General-purpose speech understanding (transcription, translation, Q&A).

| Quant | Ecosystem | Full path |
| :--- | :--- | :--- |
| bf16 | llama.cpp (mtmd) | `/home/blep/ai-models/llm/bartowski/mistralai_Voxtral-Mini-3B-2507-GGUF/mistralai_Voxtral-Mini-3B-2507-bf16.gguf` |
| Q8_0 | llama.cpp (mtmd) | `/home/blep/ai-models/llm/bartowski/mistralai_Voxtral-Mini-3B-2507-GGUF/mistralai_Voxtral-Mini-3B-2507-Q8_0.gguf` |
| Q6_K_L (best Q6) | llama.cpp (mtmd) | `/home/blep/ai-models/llm/bartowski/mistralai_Voxtral-Mini-3B-2507-GGUF/mistralai_Voxtral-Mini-3B-2507-Q6_K_L.gguf` |
| Q4_K_L (best Q4) | llama.cpp (mtmd) | `/home/blep/ai-models/llm/bartowski/mistralai_Voxtral-Mini-3B-2507-GGUF/mistralai_Voxtral-Mini-3B-2507-Q4_K_L.gguf` |
| mmproj (audio encoder, bf16) | llama.cpp (mtmd) | `/home/blep/ai-models/llm/bartowski/mistralai_Voxtral-Mini-3B-2507-GGUF/mmproj-mistralai_Voxtral-Mini-3B-2507-bf16.gguf` |

Official llama.cpp reference pair (the exact repo `docs/multimodal.md` cites
for `(tool_name) -hf ggml-org/Voxtral-Mini-3B-2507-GGUF`):

| Quant | Ecosystem | Full path |
| :--- | :--- | :--- |
| Q4_K_M | llama.cpp (mtmd) | `/home/blep/ai-models/llm/ggml-org/Voxtral-Mini-3B-2507-GGUF/Voxtral-Mini-3B-2507-Q4_K_M.gguf` |
| mmproj (audio encoder, Q8_0) | llama.cpp (mtmd) | `/home/blep/ai-models/llm/ggml-org/Voxtral-Mini-3B-2507-GGUF/mmproj-Voxtral-Mini-3B-2507-Q8_0.gguf` |

## Voxtral Small (24B, 2025-07 release) - `mistralai/Voxtral-Small-24B-2507`

Production-scale speech understanding; same capabilities as Mini, larger LM.

Download set trimmed 2026-08-13 (user decision): bf16 and Q8_0 are NOT
kept - this box has 61 GiB RAM (often ~18-30 GiB free with other work) and
the iGPU shares system RAM, so bf16 (~50 GB to run) never fits and Q8_0
(~28 GB to run) only fits when the box is idle. For STT, Q6_K_L vs bf16
WER deltas are negligible while saving ~28 GB. See "Sizes" below.

| Quant | Ecosystem | Full path |
| :--- | :--- | :--- |
| Q6_K_L (best Q6) | llama.cpp (mtmd) | `/home/blep/ai-models/llm/bartowski/mistralai_Voxtral-Small-24B-2507-GGUF/mistralai_Voxtral-Small-24B-2507-Q6_K_L.gguf` |
| Q4_K_L (best Q4) | llama.cpp (mtmd) | `/home/blep/ai-models/llm/bartowski/mistralai_Voxtral-Small-24B-2507-GGUF/mistralai_Voxtral-Small-24B-2507-Q4_K_L.gguf` |
| mmproj (audio encoder, bf16) | llama.cpp (mtmd) | `/home/blep/ai-models/llm/bartowski/mistralai_Voxtral-Small-24B-2507-GGUF/mmproj-mistralai_Voxtral-Small-24B-2507-bf16.gguf` |

## Voxtral Mini 4B Realtime (2026-02 release) - `mistralai/Voxtral-Mini-4B-Realtime-2602`

Streaming/low-latency STT (240 ms - 2.4 s configurable delay, 13 languages).
NOTE: these are **merged** single-file GGUFs (text + audio encoder together).

| Quant | Ecosystem | Full path |
| :--- | :--- | :--- |
| Q8_0 | voxtral.cpp / CrispASR | `/home/blep/ai-models/llm/andrijdavid/Voxtral-Mini-4B-Realtime-2602-GGUF/Q8_0.gguf` |
| Q6_K (best Q6 available) | voxtral.cpp / CrispASR | `/home/blep/ai-models/llm/andrijdavid/Voxtral-Mini-4B-Realtime-2602-GGUF/Q6_K.gguf` |
| Q4_K_M (best Q4 available) | voxtral.cpp / CrispASR | `/home/blep/ai-models/llm/andrijdavid/Voxtral-Mini-4B-Realtime-2602-GGUF/Q4_K_M.gguf` |

Missing: **bf16** - no f16/bf16 GGUF of this model exists on the Hub as of
2026-08-13 (checked cstr, freddm, andrijdavid, chris0173, acceldium). Largest
available quant is Q8_0. Converting from the 8.9 GB safetensors requires a
voxtral-aware converter (`convert_hf_to_gguf.py` in this fork does not
support the voxtral arch yet - `grep -c voxtral convert_hf_to_gguf.py` = 0).

## Voxtral 4B TTS (2026-03 release) - `mistralai/Voxtral-4B-TTS-2603`

Text-to-speech, 20 preset voices, 9 languages, streaming + batch.
NOTE: merged single-file GGUFs; the model carries the audio decoder, so these
files are NOT a text-GGUF + mmproj split.

| Quant | Ecosystem | Full path |
| :--- | :--- | :--- |
| f16 (closest available to bf16) | voxtral.cpp / CrispASR | `/home/blep/ai-models/llm/cstr/voxtral-4b-tts-GGUF/voxtral-4b-tts-f16.gguf` |
| Q8_0 | voxtral.cpp / CrispASR | `/home/blep/ai-models/llm/cstr/voxtral-4b-tts-GGUF/voxtral-4b-tts-q8_0.gguf` |
| Q4_K (best Q4 available) | voxtral.cpp / CrispASR | `/home/blep/ai-models/llm/cstr/voxtral-4b-tts-GGUF/voxtral-4b-tts-q4_k.gguf` |

Missing: **true bf16** (cstr ships f16; no bf16 GGUF on the Hub) and **Q6**
(no Q6 GGUF exists; cstr's repo is f16/Q8_0/Q4_K only, checked 2026-08-13).

## Voxtral Mini 4B Realtime - raw HF safetensors (vLLM / transformers)

The same model as the andrijdavid GGUF above, but in the original Hugging
Face format for vLLM / transformers. Ecosystem: **vLLM / transformers
(HF safetensors)** - not GGUF, not consumable by llama.cpp.

Location: `/home/blep/ai-models/hf/mistralai/Voxtral-Mini-4B-Realtime-2602/`

| File | Size | Purpose |
| :--- | :--- | :--- |
| `model.safetensors` | 8.9 GB | bf16 weights (text + causal audio encoder) |
| `config.json` | 1.6K | model config |
| `params.json` | 1.3K | tokenizer/model params |
| `generation_config.json` | 191 B | generation defaults |
| `processor_config.json` | 384 B | audio processor config |
| `tekken.json` | 14.9 MB | tokenizer |
| `README.md` | 14.9K | model card (provenance) |

Note: the HF repo also contains `consolidated.safetensors` (8.9 GB) - a
legacy duplicate of the same weights under Mistral's old naming, left over
from the original upload. NOT downloaded; vLLM/transformers load
`model.safetensors`. Fetch later with
`hf download mistralai/Voxtral-Mini-4B-Realtime-2602 --include 'consolidated.safetensors' --local-dir <dir>`
if a tool ever asks for it.

## Sources and provenance

| Repo | Model | Ecosystem | Layout | Files pulled | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| [bartowski/mistralai_Voxtral-Mini-3B-2507-GGUF](https://huggingface.co/bartowski/mistralai_Voxtral-Mini-3B-2507-GGUF) | Mini 3B | llama.cpp (mtmd) | split + mmproj | bf16, Q8_0, Q6_K_L, Q4_K_L, mmproj-bf16 | Q6_K_L/Q4_K_L = the _L variants (Q8_0 embed/output), strictly better than plain Q6_K/Q4_K_M |
| [bartowski/mistralai_Voxtral-Small-24B-2507-GGUF](https://huggingface.co/bartowski/mistralai_Voxtral-Small-24B-2507-GGUF) | Small 24B | llama.cpp (mtmd) | split + mmproj | Q6_K_L, Q4_K_L, mmproj-bf16 | bf16/Q8_0 not kept (RAM-limited box, see above) |
| [ggml-org/Voxtral-Mini-3B-2507-GGUF](https://huggingface.co/ggml-org/Voxtral-Mini-3B-2507-GGUF) | Mini 3B | llama.cpp (mtmd) | split + mmproj | Q4_K_M, mmproj-Q8_0 | official llama.cpp org; the pair `docs/multimodal.md` references |
| [andrijdavid/Voxtral-Mini-4B-Realtime-2602-GGUF](https://huggingface.co/andrijdavid/Voxtral-Mini-4B-Realtime-2602-GGUF) | Realtime | voxtral.cpp / CrispASR | merged | Q8_0, Q6_K, Q4_K_M | top 3 quants of the repo's Q2_K..Q8_0 ladder |
| [cstr/voxtral-4b-tts-GGUF](https://huggingface.co/cstr/voxtral-4b-tts-GGUF) | 4B TTS | voxtral.cpp / CrispASR | merged | f16, Q8_0, Q4_K | cstr = the open-source Voxtral team (original converter) |
| [mistralai/Voxtral-Mini-4B-Realtime-2602](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602) | Realtime | vLLM / transformers | safetensors (hf/) | model.safetensors + configs + tekken.json | official weights, raw HF format for vLLM testing; `consolidated.safetensors` legacy duplicate skipped |

Licenses: Voxtral Mini 3B / Small 24B / Mini 4B Realtime / 4B TTS are all
Apache 2.0 (open weights).

## Sizes and disk footprint

| Model | Pulled files | Total |
| :--- | :--- | :--- |
| Mini 3B (bartowski) | 19.9 GB (Hub) / **18.5 GiB (disk)** | 18.5 GiB |
| Small 24B (bartowski) | Q6_K_L 19.7 + Q4_K_L 14.8 + mmproj 1.4 = 35.9 GB (Hub) / **33.4 GiB (disk)** | 33.4 GiB |
| Mini 3B (ggml-org reference pair) | 3.2 GB (Hub) / **3.0 GiB (disk)** | 3.0 GiB |
| Realtime (andrijdavid) | 11.3 GB (Hub) / **10.5 GiB (disk)** | 10.5 GiB |
| 4B TTS (cstr) | 14.9 GB (Hub) / **13.8 GiB (disk)** | 13.8 GiB |
| **Total** | | **~79.2 GiB on disk** |

All downloads completed and verified 2026-08-13 (file sizes match the Hub
listings; headers checked with gguf recon - see below).

Both number sets above are provided: the Hub's decimal GB listings and the
actual on-disk GiB sizes (filesystem counts in binary).

## Verified metadata (gguf recon, 2026-08-13)

What the GGUF headers actually say, per file. The split files (`llama` text
arch + `clip` mmproj with `projector_type=voxtral`) are the llama.cpp (mtmd)
ecosystem; the merged files carry custom arches (`voxtral_realtime`,
`voxtral_tts`) and belong to the voxtral.cpp / CrispASR ecosystem.

| File | arch | Key metadata |
| :--- | :--- | :--- |
| `ggml-org/.../Voxtral-Mini-3B-2507-Q4_K_M.gguf` | `llama` | 30 blocks, emb 3072, FFN 8192, 32 heads / 8 KV, ctx 131072, vocab 131072; tensors: 183 Q4_K + 29 Q6_K + 61 F32 |
| `ggml-org/.../mmproj-Voxtral-Mini-3B-2507-Q8_0.gguf` | `clip` | `clip.projector_type=voxtral`, `clip.has_audio_encoder=True`; audio tower: 32 blocks, emb 1280, FFN 5120, 20 heads, 128 mel bins, proj 3072, stack_factor 4; 489 tensors (194 Q8_0 + 293 F32 + 2 F16) |
| `bartowski/...` Mini-3B text quants (bf16, Q8_0, Q6_K_L, Q4_K_L) | `llama` | same hyperparams as ggml-org Q4_K_M (30 blocks, emb 3072); bf16: 212 BF16 + 61 F32; Q8_0: 212 Q8_0 + 61 F32; Q6_K_L: 210 Q6_K + 2 Q8_0 (embed/output) + 61 F32; Q4_K_L: 182 Q4_K + 28 Q6_K + 2 Q8_0 + 61 F32 |
| `bartowski/.../mmproj-mistralai_Voxtral-Mini-3B-2507-bf16.gguf` | `clip` | same voxtral audio tower as the ggml-org mmproj; 489 tensors (194 BF16 + 293 F32 + 2 F16) |
| `andrijdavid/.../Q8_0.gguf`, `Q6_K.gguf`, `Q4_K_M.gguf` | `voxtral_realtime` | self-contained encoder+decoder GGUF, 712 tensors each. Encoder: 32 layers, dim 1280, hidden 5120, 32 heads, sliding win 750. Decoder: 26 layers, dim 3072, hidden 9216, 32/8 heads, sliding win 8192, rope 1e6. Audio: 16 kHz, 128 mel bins, hop 160, win 400, frame rate 12.5, delay tokens 6, pads 32/17. `voxtral_realtime` is a custom arch (not in llama.cpp) - these merged files target the voxtral.cpp/CrispASR ecosystem, NOT this fork's mtmd split flow |
| `cstr/.../voxtral-4b-tts-f16.gguf`, `-q8_0.gguf`, `-q4_k.gguf` | `voxtral_tts` | 401 tensors each, merged encoder+decoder+codec in one file. f16: 231 F16 + 170 F32; q8_0: 265 Q8_0 + 136 F32; q4_k: 265 Q4_K + 136 F32. `voxtral_tts` is a custom arch (not in llama.cpp) - cstr-ecosystem format, NOT the mtmd split flow |
| `bartowski/...` Small-24B text quants (Q6_K_L, Q4_K_L) | `llama` | 40 blocks, emb 5120 (Mistral Small 3 base), 363 tensors; Q6_K_L: 280 Q6_K + 2 Q8_0 + 81 F32; Q4_K_L: 240 Q4_K + 40 Q6_K + 2 Q8_0 + 81 F32 |
| `bartowski/.../mmproj-mistralai_Voxtral-Small-24B-2507-bf16.gguf` | `clip` | **identical audio tower to the Mini-3B mmproj** (32 blocks, emb 1280, 489 tensors, 194 BF16 + 293 F32 + 2 F16) - one shared voxtral encoder across the family |
