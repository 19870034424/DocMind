#!/bin/bash
# =================================================================
# DocMind 裁判模型(RM-R1 式两两比较)GRPO 训练
# AutoDL 单卡 RTX 4090 24G   基座: DeepSeek-R1-Distill-Qwen-1.5B
# 与 train_rag_grpo_1.5b.sh(回答模型)是两个独立任务,数据/奖励不同
# =================================================================
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_USE_V1=0
export VERL_PPO_LOGGING_LEVEL="INFO"
export WANDB_DISABLED=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

N_GPU=1

MODEL_PATH=/root/autodl-tmp/r1-distill-qwen-1.5b

LR=1.0e-6
GPU_MEM_UTILIZATION=0.22          # 实测能 init 的下限(0.20/0.18 失败)
TOTAL_EPISODES=1
SAVE_EVERY_STEP=50
TRAIN_BS=8                       # rollout 批: 每步 8 道比较题
GROUP_N=8                        # GRPO 组大小: 每题采样 8 次判定
PPO_MINI_BS=8                    # 8题x8判定=64条 rollout
TRAIN_PER_GPU=1                  # lm_head logits 峰值限制,单卡必须为 1
FORWARD_PER_GPU=2
MAX_PROMPT_LENGTH=2560
MAX_RESPONSE_LENGTH=1024
WARMUP_STYLE=constant

PROJECT_NAME=DocMind-Judge
EXPERIMENT_NAME=DocMind-Judge-1.5B-LR${LR}
SAVE_META_DIR=/root/autodl-tmp/docmind_judge

REWARD_PATH=/root/autodl-tmp/RM-R1/grpo_training/judge_reward.py
REWARD_FUNC_NAME=compute_score

TRAIN_TASK=/root/autodl-tmp/RM-R1/grpo_training/data_judge/train.jsonl
EVAL_TASK=/root/autodl-tmp/RM-R1/grpo_training/data_judge/test.jsonl

MAX_NUM_BATCHED_TOKENS=$(($MAX_PROMPT_LENGTH + $MAX_RESPONSE_LENGTH))

ray stop; sleep 5; ray stop
ray start --head --node-ip-address 0.0.0.0 --num-gpus ${N_GPU}

python3 -m rm_r1.verl.trainer.main_ppo \
    data.train_files=${TRAIN_TASK} \
    data.val_files=${EVAL_TASK} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.train_batch_size=${TRAIN_BS} \
    data.truncation=left \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    +actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BS} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${TRAIN_PER_GPU} \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.optim.warmup_style=${WARMUP_STYLE} \
    actor_rollout_ref.rollout.n=${GROUP_N} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
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
