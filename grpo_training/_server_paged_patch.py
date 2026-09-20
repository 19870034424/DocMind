"""AdamW8bit -> PagedAdamW8bit: 优化器状态分页驻留 CPU,反传期间省 ~3G GPU。"""
P = "/root/autodl-tmp/RM-R1/rm_r1/verl/workers/fsdp_workers.py"
src = open(P, encoding="utf-8").read()
old = "actor_optimizer = bnb.optim.AdamW8bit(actor_module_fsdp.parameters(),"
new = "actor_optimizer = bnb.optim.PagedAdamW8bit(actor_module_fsdp.parameters(),"
assert src.count(old) == 1, f"anchor count={src.count(old)}"
src = src.replace(old, new)
src = src.replace('.write(\'AdamW8bit\')', '.write(\'PagedAdamW8bit\')')
open(P, "w", encoding="utf-8").write(src)
print("PAGED_OK")
