"""每个 micro 步 backward 后调用 empty_cache,释放保留块给下一个 micro 的重算腾位。"""
DA = "/root/autodl-tmp/verl/verl/workers/actor/dp_actor.py"
src = open(DA, encoding="utf-8").read()
anchor = "                    loss.backward()"
assert src.count(anchor) == 1, f"anchor count={src.count(anchor)}"
assert "empty_cache" not in src, "already patched"
src = src.replace(anchor, anchor + "\n                    torch.cuda.empty_cache()  # single-GPU OOM mitigation")
open(DA, "w", encoding="utf-8").write(src)
print("CACHE_PATCH_OK")
