#!/root/miniconda3/bin/python
"""Plan the SM100 FlashMLA Mega-Attention KV cache without a GPU."""

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path("/root/autodl-tmp")
MODEL = ROOT / "DeepSeek-V4.1-Flash"
DEFAULT_JSON = ROOT / "dsv41_full_kv_cache_config.json"
DEFAULT_TEXT = ROOT / "dsv41_full_kv_cache_config.txt"
TARGET_BACKEND = "FLASHMLA_MEGA_ATTN_DSV41"

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
os.environ["XDG_CACHE_HOME"] = str(ROOT / ".cache")
os.environ["VLLM_CACHE_ROOT"] = str(ROOT / "vllm-cache")
sys.path.insert(0, str(ROOT / "vllm"))

import torch

from vllm.config import VllmConfig
from vllm.config.device import DeviceConfig
from vllm.v1.core.kv_cache_utils import get_kv_cache_configs
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
    get_kv_quant_mode,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--available-memory-gib", type=float, default=64.0)
    parser.add_argument(
        "--include-dspark-kv-cache",
        action="store_true",
        help="Include the three DeepSeek V4.1 DSpark draft-layer SWA caches.",
    )
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--text-output", type=Path, default=DEFAULT_TEXT)
    return parser.parse_args()


class DSparkPlanningConfig:
    method = "dspark"
    disable_eagle_block_drop = False

    def __init__(self, num_speculative_tokens: int):
        self.num_speculative_tokens = num_speculative_tokens

    def use_eagle(self) -> bool:
        return True

    def use_eagle_block_drop(self) -> bool:
        return True

    def use_multi_module_mtp(self) -> bool:
        return False


def spec_summary(name: str, spec, cache_format: str) -> dict:
    return {
        "name": name,
        "cache_format": cache_format,
        "spec_type": type(spec).__name__,
        "block_size": spec.block_size,
        "tokens_per_state": spec.tokens_per_state,
        "num_states": spec.num_states,
        "num_kv_heads": spec.num_kv_heads,
        "num_heads": spec.num_heads,
        "head_size": spec.head_size,
        "head_size_v": spec.head_size_v,
        "dtype": str(spec.dtype),
        "state_content_bytes": spec.state_content_size_bytes,
        "unpadded_page_size_bytes": spec.unpadded_page_size_bytes,
        "page_size_bytes": spec.page_size_bytes,
        "page_size_padded": spec.page_size_padded,
        "alignment": getattr(spec, "alignment", None),
        "block_stride_alignment": getattr(
            spec, "block_stride_alignment", None
        ),
        "cache_dtype_str": getattr(spec, "cache_dtype_str", None),
        "kv_quant_mode": spec.kv_quant_mode.name,
        "cache_role": str(getattr(spec, "cache_role", None)),
        "storage_block_size": getattr(spec, "storage_block_size", None),
        "prefix_cacheable": spec.prefix_cacheable,
    }


def main() -> None:
    args = parse_args()
    available_memory = int(args.available_memory_gib * 1024**3)

    with (MODEL / "config.json").open() as config_file:
        hf_config = json.load(config_file)
    text_config = hf_config["text_config"]
    num_layers = text_config["num_hidden_layers"]

    vllm_config = VllmConfig(
        device_config=DeviceConfig(device="cpu"),
    )
    vllm_config.model_config = SimpleNamespace(
        max_model_len=text_config["max_position_embeddings"],
        original_max_model_len=text_config["max_position_embeddings"],
        hf_config=SimpleNamespace(model_type="deepseek_v41"),
        get_total_num_hidden_layers=lambda: num_layers,
    )
    vllm_config.cache_config.block_size = 128
    vllm_config.cache_config.cache_dtype = "nvfp4_ds_mla"
    vllm_config.cache_config.kv_cache_layout = "BLHNC"
    vllm_config.cache_config.prefix_cache_retention_interval = None
    # CPU platforms disable HMA at runtime, but this script only executes the
    # pure-Python cache planner for the target SM100 layout.
    vllm_config.scheduler_config.disable_hybrid_kv_cache_manager = False
    vllm_config.attention_config.indexer_kv_dtype = "mxfp4"
    vllm_config.attention_config.indexer_sparse_logits = False

    dspark_num_layers = (
        text_config.get("n_mtp_layers")
        or text_config.get("num_nextn_predict_layers")
        or 3
    )
    dspark_num_speculative_tokens = text_config.get("dspark_block_size") or 5
    if args.include_dspark_kv_cache:
        vllm_config.speculative_config = DSparkPlanningConfig(
            dspark_num_speculative_tokens
        )

    specs = {}
    formats = {}
    compress_ratios = text_config["compress_ratios"]
    kv_source_layers = text_config["kv_source_layer_ids"]

    def make_swa_spec() -> SlidingWindowMLASpec:
        return SlidingWindowMLASpec(
            block_size=32,
            num_kv_heads=1,
            head_size=text_config["head_dim"],
            dtype=torch.uint8,
            sliding_window=text_config["sliding_window"],
            cache_dtype_str="nvfp4_ds_mla",
            state_content_bytes=528,
            alignment=512,
            model_version="deepseek_v4",
            kv_quant_mode=get_kv_quant_mode("nvfp4_ds_mla"),
        )

    for layer in range(num_layers):
        name = f"language_model.model.layers.{layer}.attn.swa_cache"
        specs[name] = make_swa_spec()
        formats[name] = (
            "FP8 SWA (V4.1 MXFP8: 512 E4M3 data + 16 UE8M0 scale "
            "bytes/token)"
        )

    for layer in kv_source_layers:
        ratio = compress_ratios[layer]
        compressor_name = f"language_model.model.layers.{layer}.attn"
        specs[compressor_name] = MLAAttentionSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=text_config["head_dim"],
            dtype=torch.uint8,
            tokens_per_state=ratio,
            cache_dtype_str="nvfp4_ds_mla",
            state_content_bytes=288,
            alignment=512,
            model_version="deepseek_v4",
            kv_quant_mode=get_kv_quant_mode("nvfp4_ds_mla"),
        )
        formats[compressor_name] = (
            "FP4 compressor (V4.1 NVFP4: 256 packed E2M1 data + 32 E4M3 "
            "scale bytes/token)"
        )

        if ratio > 1:
            rows_per_step = vllm_config.num_speculative_tokens + 2
            ring_capacity = max(8, 1 << (rows_per_step - 1).bit_length())
            state_cache_name = f"{compressor_name}.compressor.state_cache"
            specs[state_cache_name] = CircularBufferSpec(
                block_size=ring_capacity,
                num_kv_heads=1,
                head_size=2 * text_config["head_dim"],
                head_size_v=0,
                dtype=torch.float32,
            )
            formats[state_cache_name] = (
                "Compressor ring state (FP32 KV + score rows)"
            )

        indexer_name = f"{compressor_name}.indexer.k_cache"
        specs[indexer_name] = MLAAttentionSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=68,
            dtype=torch.uint8,
            tokens_per_state=ratio,
            alignment=512,
        )
        formats[indexer_name] = "FP4 indexer (MXFP4: 68 bytes/token)"

    dspark_layer_names = []
    if args.include_dspark_kv_cache:
        for layer in range(num_layers, num_layers + dspark_num_layers):
            assert compress_ratios[layer] == 0
            name = f"model.layers.{layer}.attn.swa_cache"
            specs[name] = make_swa_spec()
            formats[name] = (
                "DSpark FP8 SWA (V4.1 MXFP8: 512 E4M3 data + 16 UE8M0 "
                "scale bytes/token)"
            )
            dspark_layer_names.append(name)

    config = get_kv_cache_configs(
        vllm_config,
        kv_cache_specs=[specs],
        available_memory=[available_memory],
    )[0]
    config.kv_cache_layout = vllm_config.cache_config.kv_cache_layout

    groups = []
    for index, group in enumerate(config.kv_cache_groups):
        group_specs = (
            group.kv_cache_spec.kv_cache_specs
            if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
            else {name: group.kv_cache_spec for name in group.layer_names}
        )
        groups.append(
            {
                "index": index,
                "spec_type": type(group.kv_cache_spec).__name__,
                "block_size": group.kv_cache_spec.block_size,
                "page_size_bytes": group.kv_cache_spec.page_size_bytes,
                "is_eagle_group": group.is_eagle_group,
                "layers": [
                    spec_summary(name, group_specs[name], formats[name])
                    for name in group.layer_names
                ],
            }
        )

    result = {
        "model": str(MODEL),
        "architecture": hf_config["architectures"][0],
        "target_hardware": "SM100 (B200/GB200 class)",
        "target_attention_backend": TARGET_BACKEND,
        "target_backend_note": (
            "The ordinary FLASHMLA_SPARSE_DSV41 backend uses fp8_ds_mla for "
            "the compressed cache; the 288-byte NVFP4 record requires the "
            "SM100 Mega-Attention backend."
        ),
        "indexer_kv_dtype_note": (
            "mxfp4 is an explicit SM100 opt-in; DeepSeek indexer auto defaults "
            "to fp8."
        ),
        "include_dspark_kv_cache": args.include_dspark_kv_cache,
        "dspark_num_layers": (
            dspark_num_layers if args.include_dspark_kv_cache else 0
        ),
        "dspark_num_speculative_tokens": (
            dspark_num_speculative_tokens if args.include_dspark_kv_cache else 0
        ),
        "dspark_layer_names": dspark_layer_names,
        "num_hidden_layers": num_layers,
        "fake_available_memory_bytes": available_memory,
        "fake_available_memory_gib": args.available_memory_gib,
        "kv_cache_layout": config.kv_cache_layout,
        "cache_dtype": vllm_config.cache_config.cache_dtype,
        "indexer_kv_dtype": vllm_config.attention_config.indexer_kv_dtype,
        "input_spec_count": len(specs),
        "group_count": len(config.kv_cache_groups),
        "tensor_count": len(config.kv_cache_tensors),
        "num_blocks": config.num_blocks,
        "groups": groups,
        "tensors": [
            {
                "index": index,
                "size": tensor.size,
                "layers": tensor.layers,
                "layer_stride": tensor.layer_stride,
                "block_stride": tensor.block_stride,
                "offset": tensor.offset,
            }
            for index, tensor in enumerate(config.kv_cache_tensors)
        ],
    }

    args.json_output.write_text(json.dumps(result, indent=2) + "\n")
    args.text_output.write_text(repr(config) + "\n")

    print("get_kv_cache_configs: SUCCESS")
    print(f"input_specs={len(specs)}")
    print(f"groups={len(config.kv_cache_groups)}")
    print(f"tensors={len(config.kv_cache_tensors)}")
    print(f"num_blocks={config.num_blocks}")
    print(f"json={args.json_output}")
    print(f"text={args.text_output}")


if __name__ == "__main__":
    main()
