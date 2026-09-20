"""第二版探针: 修掉幂等 bug(逐锚点判断), 在 dp_actor 的 loss.backward() 后
加显存 + 梯度统计(是否 fp32/总量), 确认 +6.6G 的来源。
"""
import re

TRACE = "/root/autodl-tmp/mem_trace.txt"
DA = "/root/autodl-tmp/verl/verl/workers/actor/dp_actor.py"
IND = " " * 20

STAT = (IND + 'if self._probe_i <= 3: open("%s", "a").write('
        'f"micro{self._probe_i} after_bwd alloc={torch.cuda.memory_allocated()/2**30:.2f}G '
        'grads={sum(p.grad.numel()*p.grad.element_size() for p in self.actor_module.parameters() if p.grad is not None)/2**30:.2f}G '
        'gdtypes={str(set(str(p.grad.dtype) for p in self.actor_module.parameters() if p.grad is not None))}\\n")' % TRACE)

src = open(DA, encoding="utf-8").read()
assert "after_bwd" not in src, "after_bwd probe already present"
anchor = "                    loss.backward()"
assert src.count(anchor) == 1, f"backward anchor count={src.count(anchor)}"
src = src.replace(anchor, anchor + "\n" + STAT)
open(DA, "w", encoding="utf-8").write(src)
print("PROBE2_OK")
