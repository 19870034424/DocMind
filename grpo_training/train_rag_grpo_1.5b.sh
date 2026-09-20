#!/bin/bash
# =================================================================
# DocMind RAG 回答模型 GRPO 训练(AutoDL 单卡 RTX 4090 24G)
# 基座: DeepSeek-R1-Distill-Qwen-1.5B   框架: verl(沿用 RM-R1 锁定版)
# =================================================================
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_USE_V1=0
export VERL_PPO_LOGGING_LEVEL="INFO"
export WANDB_DISABLED=true

N_GPU=1

# 服务器上先下载模型:
#   pip install -U huggingface_hub
#   HF_ENDPOINT=https://hf-mirror.com huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
#       --local-dir /root/autodl-tmp/r1-distill-qwen-1.5b
MODEL_PATH=/root/autodl-tmp/r1-distill-qwen-1.5b

LR=1.0e-6
GPU_MEM_UTILIZATION=0.35          # 1.5B+bf16, 给训练留显存; OOM 再调低
TOTAL_EPISODES=1
SAVE_EVERY_STEP=50
TRAIN_BS=8                        # rollout 批: 每步 8 道题
GROUP_N=8                         # GRPO 组大小: 每道题采样 8 个回答
PPO_MINI_BS=8                     # 8题x8回答=64条 rollout, 分两个 mini batch
TRAIN_PER_GPU=8
FORWARD_PER_GPU=8
MAX_PROMPT_LENGTH=4096
MAX_RESPONSE_LENGTH=2048          # 答案+引用不需要 8K, 省显存省时间
WARMUP_STYLE=constant

PROJECT_NAME=DocMind-RAG
EXPERIMENT_NAME=DocMind-R1Distill-1.5B-LR${LR}
SAVE_META_DIR=/root/autodl-tmp/docmind_grpo

REWARD_PATH=/root/autodl-tmp/RM-R1/grpo_training/rag_reward.py
REWARD_FUNC_NAME=compute_score

TRAIN_TASK=/root/autodl-tmp/RM-R1/grpo_training/data/train.jsonl
EVAL_TASK=/root/autodl-tmp/RM-R1/grpo_training/data/test.jsonl

MAX_NUM_BATCHED_TOKENS=$(($MAX_PROMPT_LENGTH + $MAX_RESPONSE_LENGTH))

ray stop; sleep 5; ray stop
ray start --head --node-ip-address 0.0.0.0 --num-gpus ${N_GPU}

python3 -m rm_r1.verl.trainer.main_ppo \
    data.train_files=${TRAIN_TASK} \
    data.val_files=${EVAL_TASK} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.train_batch_size=${TRAIN_BS} \
    data.return_raw_chat=true \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BS} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${TRAIN_PER_GPU} \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.optim.warmup_style=${WARMUP_STYLE} \
    actor_rollout_ref.rollout.n=${GROUP_N} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTILIZATION} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${FORWARD_PER_GPU} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${FORWARD_PER_GPU} \
    custom_reward_function.path=${REWARD_PATH} \
    custom_reward_function.name=${REWARD_FUNC_NAME} \
    trainer.project_name=${PROJECT_NAME} \
    trainer.logger='["console"]' \
    trainer.total_epochs=${TOTAL_EPISODES} \
    trainer.save_freq=${SAVE_EVERY_STEP} \
    trainer.test_freq=50 \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${N_GPU} \
    trainer.nnodes=1 \
    trainer.default_local_dir=${SAVE_META_DIR}/${EXPERIMENT_NAME}

ray stop
