from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, MmprojModel, gguf


# Voxtral-Mini-4B-Realtime-2602 (realtime streaming ASR).
# Text decoder (arch `voxtral_rt_asr`) + causal audio tower (mmproj,
# arch `clip`), mirroring the offline Voxtral split in conversion/ultravox.py.
# The decoder is Ministral-like with a per-layer ada RMS-norm time-conditioning
# MLP; the tower is a causal whisper-style encoder (RoPE, sliding window,
# SILU FFN) + temporal adapter (downsample 4x + GELU MLP).
#
# Uses the transformers-5.x layout (config.json + model.safetensors): the
# decoder tensors are under `language_model.model.*` (stripped by the base
# filter_tensors), the audio tower under `audio_tower.*`, the adapter under
# `multi_modal_projector.*`. The tokenizer is tekken (mistral-common).


def _read_params(dir_model: Path) -> dict[str, Any]:
    params_path = dir_model / "params.json"
    if params_path.is_file():
        with open(params_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _read_special_tokens(dir_model: Path) -> dict[str, int]:
    tekken_path = dir_model / "tekken.json"
    if not tekken_path.is_file():
        return {}
    with open(tekken_path, "r", encoding="utf-8") as f:
        special = json.load(f)["special_tokens"]
    return {s["token_str"]: int(s["rank"]) for s in special}


def _mel_filter_bank(params: dict[str, Any]) -> Any:
    """Librosa-style mel filterbank (matching the reference mel preprocessing)."""
    import numpy as np

    enc = params["multimodal"]["whisper_model_args"]["encoder_args"]
    aud = enc["audio_encoding_args"]
    sample_rate = int(aud["sampling_rate"])
    window_size = int(aud["window_size"])
    n_mel = int(aud["num_mel_bins"])
    n_freq = 1 + window_size // 2

    def hertz_to_mel(freq_hz: np.ndarray) -> np.ndarray:
        min_log_hz = 1000.0
        min_log_mel = 15.0
        logstep = 27.0 / np.log(6.4)
        mel = min_log_mel + np.log(freq_hz / min_log_hz + 1e-9) * logstep
        mel[freq_hz < min_log_hz] = freq_hz[freq_hz < min_log_hz] * (min_log_mel / min_log_hz)
        return mel

    def mel_to_hertz(mels: np.ndarray) -> np.ndarray:
        min_log_hz = 1000.0
        min_log_mel = 15.0
        logstep = 27.0 / np.log(6.4)
        freq = min_log_hz * np.exp((mels - min_log_mel) / logstep)
        freq[mels < min_log_mel] = mels[mels < min_log_mel] * (min_log_hz / min_log_mel)
        return freq

    fft_freqs = np.linspace(0.0, float(sample_rate // 2), n_freq)
    mel_min = hertz_to_mel(np.asarray([0.0]))[0]
    mel_max = hertz_to_mel(np.asarray([8000.0]))[0]
    mel_freqs = np.linspace(mel_min, mel_max, n_mel + 2)
    filter_freqs = mel_to_hertz(mel_freqs)
    filter_diff = np.diff(filter_freqs)
    slopes = np.expand_dims(filter_freqs, 0) - np.expand_dims(fft_freqs, 1)
    down_slopes = -slopes[:, :-2] / filter_diff[:-1]
    up_slopes = slopes[:, 2:] / filter_diff[1:]
    fb = np.maximum(np.zeros(1), np.minimum(down_slopes, up_slopes))
    enorm = 2.0 / (filter_freqs[2: n_mel + 2] - filter_freqs[:n_mel])
    fb *= np.expand_dims(enorm, 0)
    return np.asarray(fb, dtype=np.float32)  # [n_freq, n_mel]


@ModelBase.register("VoxtralRealtimeForConditionalGeneration")
class VoxtralRealtimeAsrModel(TextModel):
    model_arch = gguf.MODEL_ARCH.VOXTRAL_RT_ASR
    model_name = "Voxtral Realtime ASR"

    def set_vocab(self):
        # tekken tokenizer via mistral-common (reads tekken.json)
        self._set_vocab_mistral()

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        arch = gguf.MODEL_ARCH_NAMES[self.model_arch]

        self.gguf_writer.add_rope_dimension_count(self.hparams["head_dim"])
        if (n_swa := self.hparams.get("sliding_window")) is not None:
            self.gguf_writer.add_sliding_window(n_swa)

        # decoder-side streaming decode config
        params = _read_params(self.dir_model)
        self.gguf_writer.add_uint32(f"{arch}.ada_norm_dim", params.get("ada_rms_norm_t_cond_dim", 32))
        self.gguf_writer.add_uint32(f"{arch}.delay_tokens", self.hparams.get("default_num_delay_tokens", 6))

        # streaming pad/word token ids ([STREAMING_PAD]=32, [STREAMING_WORD]=33)
        special = _read_special_tokens(self.dir_model)
        for token_str, key in (("[STREAMING_PAD]", "streaming_pad_token_id"), ("[STREAMING_WORD]", "streaming_word_token_id")):
            if token_str in special:
                self.gguf_writer.add_uint32(f"{arch}.streaming.{key}", special[token_str])


@ModelBase.register("VoxtralRealtimeForConditionalGeneration")
class VoxtralRealtimeAsrEncoderModel(MmprojModel):
    has_vision_encoder = False
    has_audio_encoder = True

    @classmethod
    def filter_tensors(cls, item):
        name, gen = item
        # normalize the transformers-5.x names to the mtmd `a.*` conventions:
        # audio_tower.embedder.conv{bid} -> audio_tower.conv{bid} (ultravox),
        # self_attn.o_proj -> self_attn.out_proj, and prefix the adapter like
        # WhisperEncoderModel (multi_modal_projector -> audio.multi_modal_projector)
        if "audio_tower.embedder." in name:
            name = name.replace("audio_tower.embedder.", "audio_tower.")
        if ".self_attn.o_proj." in name:
            name = name.replace(".self_attn.o_proj.", ".self_attn.out_proj.")
        if name.startswith("multi_modal_projector."):
            name = "audio." + name
        return super().filter_tensors((name, gen))

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.VOXTRAL_RT_ASR)
        self.gguf_writer.add_uint32("clip.audio.attention.head_dim", self.find_aparam(["head_dim"]))
        if (n_kv := self.find_aparam(["num_key_value_heads"], optional=True)) is not None:
            self.gguf_writer.add_uint32("clip.audio.attention.head_count_kv", n_kv)
        self.gguf_writer.add_audio_attention_layernorm_eps(self.find_aparam(["rms_norm_eps"]))
        self.gguf_writer.add_audio_num_mel_bins(self.find_aparam(["num_mel_bins"]))
        self.gguf_writer.add_uint32("clip.audio.projector.downsample_rate", self.global_config.get("downsample_factor", 4))
        if (n_swa := self.find_aparam(["sliding_window"], optional=True)) is not None:
            self.gguf_writer.add_uint32("clip.audio.attention.sliding_window", n_swa)
        rope = self.find_aparam(["rope_parameters"], optional=True) or {}
        self.gguf_writer.add_float32("clip.audio.rope.freq_base", float(rope.get("rope_theta", 1.0e6)))
        self.gguf_writer.add_float32("clip.audio.global_log_mel_max", 1.5)

    def tensor_force_quant(self, name: str, new_name: str, bid: int | None, n_dims: int) -> gguf.GGMLQuantizationType | bool:
        # the CPU conv path rejects bf16 weights (mul_mat(im2col_f32, bf16))
        if ".conv" in name and ".weight" in name:
            return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        # mel filterbank [n_freq, n_mel] (the ASR tool reads it from the mmproj)
        import torch

        yield ("a.mel_filters", torch.from_numpy(_mel_filter_bank(_read_params(self.dir_model))))
