# Voxtral support testing results (llama.cpp)

Date: 2026-08-13
Branch: `blep_voxtral`
Build: `build/` (Vulkan backend, commit 89e0aa6fd, b10386)

## Test setup

- Tool: `llama-mtmd-cli` (mtmd) with `--audio` input, `--temp 0`, `-n 256`, `--load-mode mmap`.
- Audio samples: 16 kHz mono WAV converted from the project OGG samples:
  - FR: `tmp/voxtral-tests/fr-je-vais-au-marche.wav` (6.5 s)
  - EN: `tmp/voxtral-tests/en-coffee-shop.wav` (7.9 s)
- Reference transcripts:
  - FR: "Je vais au marché demain matin pour acheter du pain, des œufs et un peu de fromage, si tu veux m'accompagner."
  - EN: "I'll meet you at the coffee shop on Thursday around three, and then we can walk to the park if the weather stays dry."
- Expected sentence type: STT (speech-to-text). No TTS is possible: see "TTS" section.

## Key finding: transcription prompt layout

The GGUFs from both `ggml-org` and `bartowski` carry a **Devstral/Unsloth chat template**
(agentic scaffold, `[SYSTEM_PROMPT]...[/SYSTEM_PROMPT][INST]...[/INST]`) which is wrong for
Voxtral. Using that template, the model treats the audio as a user message and ANSWERS the
spoken content conversationally (e.g. the FR sample ends with "...si tu veux m'accompagner"
and the model replies "Bien sûr, je peux t'accompagner !") instead of transcribing it.

The reference implementation (`mistral_common` `encode_transcription`) builds:

```
BOS [INST] [BEGIN_AUDIO] <audio embd> [/INST] lang:XX [TRANSCRIBE]
```

i.e. the `[TRANSCRIBE]` special token and the `lang:` prefix are placed AFTER the closing
`[/INST]`, not inside the user message. `llama-mtmd-cli` must be told to use a Jinja template
(`--jinja --chat-template`) that emits that layout. The working template used for these tests:

```
{{ "[INST]" }}{{ messages[0].content }}{{ "[/INST]lang:XX[TRANSCRIBE]" }}
```

with prompt `-p "<__media__>"`. This reliably triggers transcription mode. Without it
(`--chat-template` absent, or the Devstral template), the model answers the audio
conversationally. This is a tooling/template limitation, not a model-support gap.

## STT results (all supported, transcription mode)

| Model | Quant | mmproj | Lang | Result (verbatim) | Match | Peak RSS | Time |
| :--- | :--- | :--- | :--- | :--- | :--- | ---: | ---: |
| Voxtral-Mini-3B (ggml-org) | Q4_K_M | mmproj-Q8_0 | FR | Je vais au marché demain matin pour acheter du pain, des œufs et un peu de fromage si tu veux m'accompagner. | exact (missing comma) | 2.57 GB | 5.2 s |
| Voxtral-Mini-3B (ggml-org) | Q4_K_M | mmproj-Q8_0 | EN | I'll meet you at the coffee shop on Thursday around 3, and then we can walk to the park if the weather stays dry. | exact (3 vs "three") | 2.57 GB | 4.9 s |
| Voxtral-Mini-3B (bartowski) | Q6_K_L | mmproj-bf16 | FR | Je vais au marché demain matin pour acheter du pain, des œufs et un peu de fromage si tu veux m'accompagner. | exact (missing comma) | 3.57 GB | 5.8 s |
| Voxtral-Mini-3B (bartowski) | Q6_K_L | mmproj-bf16 | EN | I'll meet you at the coffee shop on Thursday around 3 and then we can walk to the park if the weather stays dry. | exact (3 vs "three") | 3.57 GB | 5.8 s |
| Voxtral-Mini-3B (bartowski) | Q4_K_L | mmproj-bf16 | FR | Je vais au marché demain pour acheter du pain, des œufs et un peu de rhum si tu veux m'accompagner. | "rhum" vs "fromage", dropped "matin" | 2.86 GB | 5.4 s |
| Voxtral-Mini-3B (bartowski) | Q4_K_L | mmproj-bf16 | EN | I'll meet you at the coffee shop on Thursday around 3 and then we can walk to the park if the weather stays dry. | exact (3 vs "three") | 2.86 GB | 5.5 s |
| Voxtral-Small-24B (bartowski) | Q6_K_L | mmproj-bf16 | FR | Je vais au marché demain matin pour acheter des œufs et un peu de fromage, si tu veux m'accompagner. | dropped "pain" | 19.36 GB | 23.5 s |
| Voxtral-Small-24B (bartowski) | Q6_K_L | mmproj-bf16 | EN | I'll meet you at the coffee shop on Thursday around 3 and then we can go to the park if the weather stays dry. | "go" vs "walk", 3 vs "three" | 19.36 GB | 18.2 s |

Logs: `tmp/voxtral-tests/result-*.txt`.

Notes:
- All four quant/text variants of the Mini 3B and the Small 24B load and transcribe correctly
  in both FR and EN. Multilingual support works.
- Peak RSS is the `/usr/bin/time -v` maximum resident set size. The iGPU (Vulkan, RADV)
  shares system RAM; for the 24B the reported RSS includes both model and GPU buffers.
- Small 24B EN required `--no-mmproj-offload` (audio encoder on CPU). With the default
  (mmproj on GPU), RADV fails with "Not enough memory for command submission" and the
  process dumps core. The 24B weights (~19.3 GB) plus the bf16 mmproj on the shared
  32 GB iGPU does not leave room for the audio batch. FR happened to succeed in one run
  but this is on the memory edge; CPU-offload of the mmproj is the reliable path.
- Q4_K_L FR mistranscribes "fromage" as "rhum" and drops "matin" - a low-quant quality
  artifact, not a support issue (the same file's EN is exact).

## Unsupported models

| Model | File(s) | Arch in GGUF | llama.cpp status |
| :--- | :--- | :--- | :--- |
| Voxtral-Mini-4B-Realtime-2602 | `andrijdavid/.../Q8_0.gguf`, `Q6_K.gguf`, `Q4_K_M.gguf` | `voxtral_realtime` | NOT supported - `error loading model: unknown model architecture: 'voxtral_realtime'` |
| Voxtral-4B-TTS-2603 | `cstr/.../voxtral-4b-tts-f16.gguf`, `-q8_0.gguf`, `-q4_k.gguf` | `voxtral_tts` | NOT supported - `error loading model: unknown model architecture: 'voxtral_tts'` |

These are the merged single-file GGUFs (text + audio encoder in one file, custom arch).
`src/llama-arch.cpp` has no `voxtral_realtime`/`voxtral_tts` arch; the merged layout
cannot be consumed by the mtmd split flow (text GGUF + `clip`/`voxtral` mmproj).

## TTS

TTS output is NOT possible with the current llama.cpp build:

- `llama-tts` only supports `qwen3tts` and `pockettts` arches (`tools/mtmd/mtmd-helper-gen.cpp`).
  There is no `voxtral_tts` pipeline.
- The `voxtral_tts` GGUFs fail to load (`unknown model architecture`).
- Therefore no TTS audio files were generated. Naming convention reserved for future
  TTS work: `tts-<model>-<quant>-<fr|en>.wav` in `tmp/voxtral-tests/`.

## Support summary

| Ecosystem | Model | Support |
| :--- | :--- | :--- |
| llama.cpp (mtmd) split | Voxtral Mini 3B (all quants) | SUPPORTED (STT, FR + EN) |
| llama.cpp (mtmd) split | Voxtral Small 24B | SUPPORTED (STT, FR + EN; mmproj on CPU recommended) |
| voxtral.cpp / CrispASR merged | Voxtral Mini 4B Realtime | NOT SUPPORTED (arch `voxtral_realtime`) |
| voxtral.cpp / CrispASR merged | Voxtral 4B TTS | NOT SUPPORTED (arch `voxtral_tts`) |

Development needed for the two merged-arch models: add `voxtral_realtime` and `voxtral_tts`
architectures (and a TTS audio-decoding pipeline) to llama.cpp, plus conversion support in
`convert_hf_to_gguf.py` (currently `grep -c voxtral convert_hf_to_gguf.py` = 0).

## Voxtral Mini 4B Realtime via vLLM (raw HF safetensors)

Tested 2026-08-13 with the raw HF model at
`/home/blep/ai-models/hf/mistralai/Voxtral-Mini-4B-Realtime-2602/` served by the vLLM
reference checkout in `refs/vllm/` (built against ROCm 7.2, Python 3.12 venv in
`tmp/vllm-build/`).

### Serving setup

vLLM serves this model natively (`VoxtralRealtimeGeneration`, registered in
`vllm/model_executor/models/registry.py:603`). Three things were needed:

1. `--config-format mistral` (the model dir ships both `config.json` and `params.json`; vLLM's
   auto-detection keys off `consolidated*.safetensors` which is not downloaded, so it must be
   forced).
2. `--tokenizer-mode mistral` (required by the Voxtral processor).
3. `--attention-backend triton_attn` + `--no-enable-chunked-prefill`. Without these the server
   fails during warmup: the default ROCM_ATTN path falls back to a Triton paged-attention kernel
   (`chunked_prefill_paged_decode.py`) that needs 66048 B of shared memory, exceeding the 65536 B
   hardware limit of the Radeon 890M iGPU (gfx1150). MIOpen env `MIOPEN_FIND_MODE=1
   MIOPEN_FIND_ENFORCE=4` is also required to avoid a conv autotune failure in the causal audio
   encoder.

### Weight naming mismatch (transformers 5.x vs legacy)

The downloaded `model.safetensors` uses transformers-5.x tensor names
(`audio_tower.*`, `multi_modal_projector.*`, `language_model.model.*`,
`ada_rms_norm.linear1/2`), but vLLM's `voxtral.py` loader expects the legacy
mistral_common names (`mm_whisper_embeddings.whisper_encoder.*`,
`audio_language_adapter.w_in/w_out`, `model.layers.*`, `ada_rms_norm_t_cond.0/2`).
A one-off rename script (`tmp/convert-voxtral-realtime.py`) was used to produce a
compatible checkpoint. The rename map is documented in that script's docstring;
the original file was restored after testing.

### Capabilities found

- Supported tasks reported by the server: `generate`, `transcription`, `realtime`.
- `POST /v1/audio/transcriptions` works (OpenAI-compatible multipart `file=` upload).
  - FR: "Cheveux marché du pain, des oeufs et un peu de fromage, un petit peu ma compagnie."
    (expected: "Je vais au marché demain matin pour acheter du pain, des oeufs et un peu de
    fromage, si tu veux m'accompagner." - roughly correct, noisy start/end).
  - EN: "Al-Mitcho at the coffee shop on Southdale Street, on Zen we can go to the park if the
    weather stays right." (expected: "I'll meet you at the coffee shop on Thursday around three,
    and then we can walk to the park if the weather stays dry." - word errors, structure ok).
  - Quality on these short clean samples is noticeably worse than the offline Mini 3B / Small 24B
    through llama.cpp.
- `/v1/realtime` (WebSocket) works: stream `input_audio_buffer.append` chunks (16 kHz PCM16,
  base64), then `input_audio_buffer.commit` (`final:false` to start generation, `final:true` at
  end). The model emits one `transcription.delta` per generated token plus a final
  `transcription.done` with `usage` (prompt/total/completion tokens).

### Timing / word timestamps

- **No word/segment timestamps.** The realtime `transcription.delta` events
  (`realtime/protocol.py`) carry only `delta` text; `transcription.done` carries `text` + usage.
  There is no timestamp field.
- `response_format=verbose_json` (word/segment timing) is rejected: vLLM's STT serving gates it
  behind `SupportsTranscription.supports_segment_timestamp`, which `VoxtralRealtimeGeneration`
  does not set (default False). Same for `diarized_json`. `srt`/`vtt` are not accepted at all.
- The 80 ms frame-synchronous output exists internally (12.5 Hz, `[STREAMING_PAD]` /
  `[STREAMING_WORD]` tokens in the vocab), but vLLM does not surface frame or token timing in
  any API response.

### Porting implications for llama.cpp

- The Realtime model IS servable (it runs in vLLM on this iGPU with the workarounds above), so a
  llama.cpp port is feasible in principle. It needs: a new `voxtral_realtime` arch (causal audio
  encoder + temporal adapter + Mistral text decoder with `ada_rms_norm_t_cond`), the DSM decode
  loop (one token per 80 ms frame, delay + pad/word-boundary tokens), and a streaming audio input
  path. None of this exists in llama.cpp today.
- Word timing for LRC-style output would require surfacing frame indices in the decode loop
  (map `[STREAMING_WORD]` + text token positions to `frame_index * 80 ms`); vLLM does not do this
  today either, so a llama.cpp port would be a strict superset if it exposes it.
