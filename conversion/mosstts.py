from __future__ import annotations

from typing import Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, MmprojModel, gguf

# MOSS-TTS-Nano-100M ("global_local_transformer" TTS decoder) converts to TWO GGUFs,
# following the in-fork Qwen3-TTS pattern:
#   - main GGUF  (arch mosstts): the global transformer (GPT-2 core, RoPE, no pos_embd)
#   - mmproj GGUF (arch mmproj):  the gen-code component = 1-layer local transformer +
#                                 16 audio codebook tables/heads (a.gen.code.*, mtmd names)
# The model consumes a (seq, n_vq+1) grid: column 0 = text token id, columns 1..16 = code
# ids (1024 = pad). Input embeds = wte(text) + sum of the 16 per-codebook embeddings.

_N_CODEBOOKS = 16


def _hoist_gpt2_config(hparams: dict) -> dict:
    # the GPT-2 core hparams live in a nested gpt2_config; hoist them (setdefault so
    # keys like architectures=null do not clobber the root)
    if "gpt2_config" in hparams:
        for k, v in hparams["gpt2_config"].items():
            hparams.setdefault(k, v)
    return hparams


@ModelBase.register("MossTTSNanoForCausalLM")
class MossTTSNanoModel(TextModel):
    model_arch = gguf.MODEL_ARCH.MOSSTTS

    def __init__(self, dir_model, *args, **kwargs):
        hparams = kwargs.pop("hparams", None)
        if hparams is None:
            hparams = ModelBase.load_hparams(dir_model, is_mistral_format=False)
        super().__init__(dir_model, *args, hparams=_hoist_gpt2_config(hparams), **kwargs)

    def set_vocab(self):
        tokens, scores, toktypes = self._create_vocab_sentencepiece()
        self.gguf_writer.add_tokenizer_model("llama")
        self.gguf_writer.add_tokenizer_pre("default")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_scores(scores)
        self.gguf_writer.add_token_types(toktypes)
        special_vocab = gguf.SpecialVocab(self.dir_model, n_vocab=len(tokens))
        special_vocab.add_to_gguf(self.gguf_writer)

    def set_gguf_parameters(self):
        gc = self.hparams.get("gpt2_config", {})
        n_embd = gc.get("n_embd")
        n_head = gc.get("n_head")
        self.gguf_writer.add_block_count(gc.get("n_layer"))
        self.gguf_writer.add_context_length(gc.get("n_ctx"))
        self.gguf_writer.add_embedding_length(n_embd)
        self.gguf_writer.add_feed_forward_length(gc.get("n_inner"))
        self.gguf_writer.add_head_count(n_head)
        self.gguf_writer.add_layer_norm_eps(gc.get("layer_norm_epsilon"))
        # RoPE replaces the GPT-2 learned position embeddings; head_dim == n_embd / n_head
        self.gguf_writer.add_rope_dimension_count(n_embd // n_head)
        self.gguf_writer.add_rope_freq_base(gc.get("rope_base"))
        self.gguf_writer.add_file_type(self.ftype)

    @classmethod
    def filter_tensors(cls, item):
        name, gen = item
        # the audio generation component is converted separately (--mmproj)
        if name.startswith(("audio_embeddings.", "audio_lm_heads.", "local_transformer.")):
            return None
        return super().filter_tensors(item)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # global GPT-2 core. torch Linear weights are [out, in]; gguf-py stores the
        # reversed dims, which is exactly the ggml ne order llama.cpp expects, so no
        # transposition is applied.
        if name.startswith("transformer."):
            new_name = self.map_tensor_name(name)
            yield from super().modify_tensors(data_torch, new_name, bid)
            return

        # text output head is tied to wte (same values, emitted for the loader)
        if name == "text_lm_head.weight":
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.OUTPUT), data_torch)
            return

        yield from super().modify_tensors(data_torch, name, bid)


@ModelBase.register("MossTTSNanoForCausalLM")
class MossTTSNanoGenModel(MmprojModel):
    has_vision_encoder = False

    _code_embed_buffer: dict[int, Tensor] = {}
    _code_head_buffer: dict[int, Tensor] = {}

    def __init__(self, dir_model, *args, **kwargs):
        hparams = kwargs.pop("hparams", None)
        if hparams is None:
            hparams = ModelBase.load_hparams(dir_model, is_mistral_format=False)
        super().__init__(dir_model, *args, hparams=_hoist_gpt2_config(hparams), **kwargs)

    def set_vocab(self):
        # the gen-audio GGUF carries no tokenizer
        pass

    def set_gguf_parameters(self):
        gc = self.hparams.get("gpt2_config", {})
        n_embd = gc.get("n_embd")
        self.gguf_writer.add_file_type(self.ftype)
        self.gguf_writer.add_clip_has_gen_audio_encoder(True)
        self.gguf_writer.add_clip_gen_audio_projector_type(gguf.VisionProjectorType.MOSSTTS_GEN)
        # gen-code component: 1-layer local transformer + 16 codebooks
        self.gguf_writer.add_gen_audio_projection_dim(self.n_embd_text)
        self.gguf_writer.add_gen_audio_embedding_length(n_embd)
        self.gguf_writer.add_gen_audio_feed_forward_length(gc.get("n_inner"))
        self.gguf_writer.add_gen_audio_block_count(self.hparams.get("local_transformer_layers"))
        self.gguf_writer.add_gen_audio_head_count(gc.get("n_head"))
        self.gguf_writer.add_gen_audio_head_count_kv(gc.get("n_head"))
        self.gguf_writer.add_gen_audio_attention_layernorm_eps(gc.get("layer_norm_epsilon"))

    @classmethod
    def filter_tensors(cls, item):
        name, gen = item
        if name.startswith(("audio_embeddings.", "audio_lm_heads.", "local_transformer.")):
            return super().filter_tensors((name, gen))
        return None

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        T = gguf.MODEL_TENSOR

        # 16 per-codebook embeddings -> 3D [16, 1024, 768]
        if name.startswith("audio_embeddings."):
            idx = int(name.split(".")[1])
            self._code_embed_buffer[idx] = data_torch
            if len(self._code_embed_buffer) < _N_CODEBOOKS:
                return
            stacked = torch.stack([self._code_embed_buffer.pop(i) for i in range(_N_CODEBOOKS)], dim=0)
            yield (self.format_tensor_name(T.A_GEN_CODE_EMBD), stacked)
            return

        # 16 per-codebook output heads, tied to the embeddings -> 3D [16, 1024, 768]
        if name.startswith("audio_lm_heads."):
            idx = int(name.split(".")[1])
            self._code_head_buffer[idx] = data_torch
            if len(self._code_head_buffer) < _N_CODEBOOKS:
                return
            stacked = torch.stack([self._code_head_buffer.pop(i) for i in range(_N_CODEBOOKS)], dim=0)
            yield (self.format_tensor_name(T.A_GEN_CODE_HEAD), stacked)
            return

        # 1-layer local transformer + final norm
        if name.startswith("local_transformer."):
            yield from self._modify_local_transformer(data_torch, name)
            return

        yield from super().modify_tensors(data_torch, name, bid)

    def _modify_local_transformer(self, data_torch: Tensor, name: str) -> Iterable[tuple[str, Tensor]]:
        T = gguf.MODEL_TENSOR
        rest = name[len("local_transformer."):]

        if rest == "ln_f.weight":
            yield (self.format_tensor_name(T.A_GEN_CODE_OUTPUT_NORM), data_torch)
            return
        if rest == "ln_f.bias":
            yield (self.format_tensor_name(T.A_GEN_CODE_OUTPUT_NORM, suffix=".bias"), data_torch)
            return

        if not rest.startswith("h.0."):
            raise ValueError(f"Unexpected local transformer tensor: {name}")
        key = rest[len("h.0."):]

        if key == "ln_1.weight":
            yield (self.format_tensor_name(T.A_GEN_CODE_ATTN_NORM, 0), data_torch)
        elif key == "ln_1.bias":
            yield (self.format_tensor_name(T.A_GEN_CODE_ATTN_NORM, 0, ".bias"), data_torch)
        elif key == "attn.c_attn.weight":
            # merged QKV [2304, 768] split along dim 0 into q/k/v [768, 768]
            q, k, v = data_torch.chunk(3, dim=0)
            yield (self.format_tensor_name(T.A_GEN_CODE_ATTN_Q, 0), q)
            yield (self.format_tensor_name(T.A_GEN_CODE_ATTN_K, 0), k)
            yield (self.format_tensor_name(T.A_GEN_CODE_ATTN_V, 0), v)
        elif key == "attn.c_attn.bias":
            qb, kb, vb = data_torch.chunk(3, dim=0)
            yield (self.format_tensor_name(T.A_GEN_CODE_ATTN_Q, 0, ".bias"), qb)
            yield (self.format_tensor_name(T.A_GEN_CODE_ATTN_K, 0, ".bias"), kb)
            yield (self.format_tensor_name(T.A_GEN_CODE_ATTN_V, 0, ".bias"), vb)
        elif key == "attn.c_proj.weight":
            yield (self.format_tensor_name(T.A_GEN_CODE_ATTN_OUT, 0), data_torch)
        elif key == "attn.c_proj.bias":
            yield (self.format_tensor_name(T.A_GEN_CODE_ATTN_OUT, 0, ".bias"), data_torch)
        elif key == "ln_2.weight":
            yield (self.format_tensor_name(T.A_GEN_CODE_FFN_NORM, 0), data_torch)
        elif key == "ln_2.bias":
            yield (self.format_tensor_name(T.A_GEN_CODE_FFN_NORM, 0, ".bias"), data_torch)
        elif key == "mlp.fc_in.weight":
            yield (self.format_tensor_name(T.A_GEN_CODE_FFN_UP, 0), data_torch)
        elif key == "mlp.fc_in.bias":
            yield (self.format_tensor_name(T.A_GEN_CODE_FFN_UP, 0, ".bias"), data_torch)
        elif key == "mlp.fc_out.weight":
            yield (self.format_tensor_name(T.A_GEN_CODE_FFN_DOWN, 0), data_torch)
        elif key == "mlp.fc_out.bias":
            yield (self.format_tensor_name(T.A_GEN_CODE_FFN_DOWN, 0, ".bias"), data_torch)
        else:
            raise ValueError(f"Unexpected local transformer tensor: {name}")
