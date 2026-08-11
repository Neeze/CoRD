"""Script to inspect CoRD model architecture, module parameter breakdown, VRAM usage, and FLOP count."""

import argparse
import gc
import sys
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cord import CordConfig, CordForCausalLM


def get_vram_info() -> dict:
    """Return current GPU VRAM allocation and reservation in MB, if available."""
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    
    device_id = torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(device_id) / (1024 ** 2)
    reserved = torch.cuda.memory_reserved(device_id) / (1024 ** 2)
    max_allocated = torch.cuda.max_memory_allocated(device_id) / (1024 ** 2)
    total_memory = torch.cuda.get_device_properties(device_id).total_memory / (1024 ** 2)
    
    return {
        "cuda_available": True,
        "device_name": torch.cuda.get_device_name(device_id),
        "allocated_mb": allocated,
        "reserved_mb": reserved,
        "max_allocated_mb": max_allocated,
        "total_mb": total_memory,
    }


def print_vram_summary(title: str, vram_info: dict) -> None:
    """Print formatted VRAM usage status."""
    print(f"\n--- {title} ---")
    if not vram_info.get("cuda_available", False):
        print("  CUDA không khả dụng (Chạy trên CPU).")
        return
    
    print(f"  Thiết bị GPU          : {vram_info['device_name']}")
    print(f"  VRAM Đã cấp phát (Allocated): {vram_info['allocated_mb']:.2f} MB")
    print(f"  VRAM Đã giữ (Reserved)     : {vram_info['reserved_mb']:.2f} MB")
    print(f"  VRAM Đỉnh điểm (Peak Allocated): {vram_info['max_allocated_mb']:.2f} MB")
    print(f"  Tổng dung lượng VRAM      : {vram_info['total_mb']:.2f} MB")


def summarize_module_parameters(model: CordForCausalLM) -> None:
    """Print structured breakdown of parameters for each sub-module."""
    print("\n" + "=" * 80)
    print(" SUMMARY KIẾN TRÚC & THAM SỐ CÁC MODULE (MODULE PARAMETER BREAKDOWN)")
    print("=" * 80)

    unique_params = {id(p): p for p in model.parameters()}
    total_params = sum(p.numel() for p in unique_params.values())
    trainable_params = sum(p.numel() for p in unique_params.values() if p.requires_grad)

    print(f"Tổng số tham số (Total Params)      : {total_params:,} ({total_params / 1e6:.2f}M)")
    print(f"Số tham số huấn luyện (Trainable)  : {trainable_params:,} ({trainable_params / 1e6:.2f}M)")
    print("-" * 80)

    submodules = [
        ("Word Embeddings (model.embed_tokens)", model.model.embed_tokens),
        ("Concept Encoder (model.concept_encoder)", model.model.concept_encoder),
        ("Recurrent Core MacroBlock (model.recurrent_core)", model.model.recurrent_core),
        ("Decoder Layers (model.decoder_layers)", model.model.decoder_layers),
        ("Decoder Norm (model.decoder_norm)", model.model.decoder_norm),
        ("Value Head (model.value_head)", model.model.value_head),
        ("Uncertainty Head (model.uncertainty_head)", model.model.uncertainty_head),
        ("Halting Head (model.halting_head)", model.model.halting_head),
        ("Rollback Gate (model.rollback_gate)", model.model.rollback_gate),
        ("Merge Gate (model.merge_gate)", model.model.merge_gate),
        ("LM Head (lm_head)", model.lm_head),
    ]

    print(f"{'Tên Module / Thành phần':<45} | {'Số tham số':<15} | {'% Tổng số':<10}")
    print("-" * 80)

    for name, submod in submodules:
        sub_params = sum(p.numel() for p in submod.parameters())
        pct = (sub_params / total_params) * 100 if total_params > 0 else 0
        print(f"{name:<45} | {sub_params:>15,} | {pct:>9.2f}%")

    print("=" * 80)


def calculate_flops_and_run(
    model: CordForCausalLM,
    device: str,
    batch_size: int = 1,
    seq_len: int = 128,
    prefix_len: int = 64,
) -> float:
    """Run model forward pass inside PyTorch FlopCounterMode and return total FLOPs."""
    input_ids = torch.randint(
        0, model.config.vocab_size, (batch_size, seq_len), device=device, dtype=torch.long
    )
    prefix_lengths = torch.full((batch_size,), prefix_len, device=device, dtype=torch.long)
    attention_mask = torch.ones((batch_size, seq_len), device=device, dtype=torch.long)

    # Warmup pass
    with torch.no_grad():
        _ = model(input_ids=input_ids, prefix_lengths=prefix_lengths, attention_mask=attention_mask)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    print("\n" + "=" * 80)
    print(f" TÍNH TOÁN FLOPs MÔ HÌNH (Batch Size={batch_size}, Seq Len={seq_len}, Prefix Len={prefix_len})")
    print("=" * 80)

    flop_counter = FlopCounterMode(display=False)
    with torch.no_grad(), flop_counter:
        _ = model(input_ids=input_ids, prefix_lengths=prefix_lengths, attention_mask=attention_mask)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    total_flops = flop_counter.get_total_flops()
    gflops = total_flops / 1e9
    mflops = total_flops / 1e6

    print(f"Tổng số phép tính điểm động (Total FLOPs) : {total_flops:,}")
    print(f"GFLOPs                                     : {gflops:.4f} GFLOPs")
    print(f"MFLOPs                                     : {mflops:.2f} MFLOPs")
    print("=" * 80)

    return total_flops


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect CoRD Model Architecture, VRAM & FLOPs")
    parser.add_argument("--config", type=Path, default=None, help="Path to config json (e.g. configs/cord-50m.json)")
    parser.add_argument("--device", type=str, default="auto", help="Device to use: auto, cuda, cpu")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for inspection")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length")
    parser.add_argument("--prefix-len", type=int, default=64, help="Prefix length for concept encoder")
    parser.add_argument("--show-torchinfo", action="store_true", help="Print torchinfo summary table")
    args = parser.parse_args()

    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"[*] Đang chạy kiểm tra trên thiết bị: {device.upper()}")

    # 1. VRAM Check - Baseline
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    vram_baseline = get_vram_info()
    print_vram_summary("VRAM BAN ĐẦU (BASELINE)", vram_baseline)

    # Load Configuration & Model
    if args.config and args.config.exists():
        print(f"[*] Đang nạp cấu hình từ: {args.config}")
        config = CordConfig.from_json_file(args.config)
    else:
        print("[*] Sử dụng cấu hình mặc định (cord-50m.json nếu có, hoặc mặc định CordConfig)")
        default_config_path = Path(__file__).resolve().parents[1] / "configs" / "cord-50m.json"
        if default_config_path.exists():
            config = CordConfig.from_json_file(default_config_path)
        else:
            config = CordConfig()

    # 2. VRAM Check - After Model Instantiation on CPU
    print("\n[*] Đang khởi tạo mô hình CordForCausalLM...")
    model = CordForCausalLM(config)
    model.eval()

    vram_init = get_vram_info()
    print_vram_summary("VRAM SAU KHI KHỞI TẠO MÔ HÌNH (TRÊN CPU)", vram_init)

    # Move model to target device
    if device == "cuda":
        print("\n[*] Đang chuyển mô hình sang GPU CUDA...")
        model.to(device)
        vram_cuda = get_vram_info()
        print_vram_summary("VRAM SAU KHI LOAD MÔ HÌNH VÀO GPU (CUDA)", vram_cuda)

    # 3. Print Model Structure
    print("\n" + "=" * 80)
    print(" CẤU TRÚC MÔ HÌNH (MODEL ARCHITECTURE TREE)")
    print("=" * 80)
    print(model)

    # 4. Detailed Module Parameter Breakdown
    summarize_module_parameters(model)

    # 5. Torchinfo Summary (if requested or torchinfo available)
    try:
        from torchinfo import summary as torchinfo_summary

        print("\n" + "=" * 80)
        print(" TORCHINFO MODULE SUMMARY")
        print("=" * 80)
        dummy_input_ids = torch.randint(0, config.vocab_size, (args.batch_size, args.seq_len), device=device)
        dummy_prefix = torch.full((args.batch_size,), args.prefix_len, device=device, dtype=torch.long)
        
        info = torchinfo_summary(
            model,
            input_data={"input_ids": dummy_input_ids, "prefix_lengths": dummy_prefix},
            col_names=["input_size", "output_size", "num_params", "trainable"],
            depth=3,
            verbose=0,
        )
        print(info)
    except Exception as e:
        print(f"[!] Torchinfo summary skipped: {e}")

    # 6. FLOPs & Forward Pass Execution
    calculate_flops_and_run(
        model,
        device=device,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        prefix_len=args.prefix_len,
    )

    # 7. VRAM Peak Check after forward pass
    if device == "cuda":
        vram_peak = get_vram_info()
        print_vram_summary("VRAM ĐỈNH ĐIỂM SAU FORWARD PASS (PEAK GPU VRAM)", vram_peak)


if __name__ == "__main__":
    main()
