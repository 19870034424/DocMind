"""MixedPrecision -> None: 模型原生 bf16,无需 cast;梯度保持 bf16(省 3.3G)。
FSDP1 设置 MixedPrecision 时会把梯度升回 fp32(为 fp32 优化器),是 24G 单卡的显存杀手。
"""
P = "/root/autodl-tmp/RM-R1/rm_r1/verl/workers/fsdp_workers.py"
src = open(P, encoding="utf-8").read()
old = "        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=torch.bfloat16, buffer_dtype=buffer_dtype)"
assert src.count(old) == 1, f"anchor count={src.count(old)}"
src = src.replace(old, "        mixed_precision = None  # native bf16 model: no cast, grads stay bf16 (saves 3.3G)")
open(P, "w", encoding="utf-8").write(src)
print("MP_NONE_OK")
