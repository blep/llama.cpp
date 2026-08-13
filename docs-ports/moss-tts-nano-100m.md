# MOSS-TTS-Nano-100M port to llama.cpp (analysis / reference)

Source: `/home/blep/prj/mallomar-server/docs/model-llama-cpp-ports.md` (candidate: MOSS-TTS-Nano-100M)
Status: analysis phase - validate official reference output before porting.
Date: 2026-08-13

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
- Params: 100M, fp32 weights
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
