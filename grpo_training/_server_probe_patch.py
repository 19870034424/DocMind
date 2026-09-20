"""给服务器上的 verl 打显存探针补丁(诊断 OOM 用)。

在 fsdp_workers.update_actor 入口/加载后、dp_actor 每个 micro 步前向后/反向后,
把 torch.cuda.memory_allocated/reserved 追加写到 /root/autodl-tmp/mem_trace.txt。
幂等: 重复运行会先还原再打(以 PROBE 标记识别)。
"""
import sys

TRACE = "/root/autodl-tmp/mem_trace.txt"


def probe(tag: str, indent: str) -> str:
    return (f'{indent}open("{TRACE}", "a").write('
            f'f"{tag} alloc={{torch.cuda.memory_allocated()/2**30:.2f}}G '
            f'reserved={{torch.cuda.memory_reserved()/2**30:.2f}}G\\n")')


def patch(path: str, old: str, new: str) -> None:
    src = open(path, encoding="utf-8").read()
    if "mem_trace.txt" in src:  # 已打过
        print(f"already patched: {path}")
        return
    assert src.count(old) == 1, f"anchor not unique in {path}: {old[:60]!r}"
    open(path, "w", encoding="utf-8").write(src.replace(old, new))
    print(f"patched: {path}")


# --- 1) fsdp_workers.update_actor ---
fw = "/root/autodl-tmp/RM-R1/rm_r1/verl/workers/fsdp_workers.py"
patch(fw,
      "    def update_actor(self, data: DataProto):\n"
      "        # Support all hardwares\n"
      "        data = data.to(torch.cuda.current_device())",
      "    def update_actor(self, data: DataProto):\n"
      "        # Support all hardwares\n"
      "        data = data.to(torch.cuda.current_device())\n"
      + probe("update_actor_entry", "        "))
patch(fw,
      "            load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=torch.cuda.current_device())\n"
      "\n"
      "        log_gpu_memory_usage('Before update policy', logger=logger)",
      "            load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=torch.cuda.current_device())\n"
      + probe("after_load", "        ") + "\n"
      "        log_gpu_memory_usage('Before update policy', logger=logger)")

# --- 2) dp_actor.update_policy micro 循环 ---
da = "/root/autodl-tmp/verl/verl/workers/actor/dp_actor.py"
IND = " " * 20
cnt = IND + "self._probe_i = getattr(self, '_probe_i', 0) + 1"
w_fwd = IND + ('if self._probe_i <= 3: open("%s", "a").write('
               'f"micro{self._probe_i} after_fwd alloc={torch.cuda.memory_allocated()/2**30:.2f}G '
               'reserved={torch.cuda.memory_reserved()/2**30:.2f}G\\n")' % TRACE)
w_bwd = IND + ('if self._probe_i <= 3: open("%s", "a").write('
               'f"micro{self._probe_i} after_bwd alloc={torch.cuda.memory_allocated()/2**30:.2f}G '
               'reserved={torch.cuda.memory_reserved()/2**30:.2f}G\\n")' % TRACE)
patch(da,
      "                    responses = data['responses']",
      cnt + "\n" + w_fwd + "\n                    responses = data['responses']")
patch(da,
      "                    loss.backward()",
      "                    loss.backward()\n" + w_bwd)
print("ALL_PROBES_OK")
