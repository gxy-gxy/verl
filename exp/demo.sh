#!/bin/bash
set -x
export VLLM_ATTENTION_BACKEND=XFORMERS

# # 全局配置优先级高，默认配置文件存放在/root/.config/pip/pip.conf
# pip config set global.index-url https://mirrors.tencent.com/pypi/simple/
# pip config set global.extra-index-url https://mirrors.tencent.com/repository/pypi/tencent_pypi/simple
# pip config set global.trusted-host mirrors.tencent.com
# pip install -e /mnt/geminiszgmcephfs/geminicephfs/pr-others-prctrans/xinyanguan/ywj-verl
# pip install liger_kernel==0.5.5
# pip install math-verify==0.7.0
# pip install antlr4-python3-runtime==4.9.3
# pip install nvidia-cublas-cu12==12.4.5.8
# pip uninstall -y megatron_core
# pip install pyext
# pip install pebble

export RAY_DEBUG=legacy
export GLOO_SOCKET_IFNAME=bond1
export NCCL_SOCKET_IFNAME=bond1

export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_CHECK_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_LL_THRESHOLD=16384
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_SOCKET_IFNAME=bond1
export UCX_NET_DEVICES=bond1
export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6
export NCCL_COLLNET_ENABLE=0
export SHARP_COLL_ENABLE_SAT=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_PXN_DISABLE=0
export NCCL_IB_TIMEOUT=22

export PATH="/opt/conda/bin":$PATH && conda --version
export PATH="/opt/conda/envs/deepscaler/bin":$PATH
which python

check_port() {
    (echo > /dev/tcp/$MASTER_ADDR/$PORT) >/dev/null 2>&1
    return $?
}

PORT=6379
export WORLD_SIZE=1
export RANK=0

if [ $RANK -eq 0 ]; then
    ray start --head --port $PORT
else
    while ! check_port; do
        echo "Port $PORT on $MASTER_ADDR is not open yet. Retrying in 5 seconds..."
        sleep 30s # wait for head node to start
    done
    ray start --address=$MASTER_ADDR:$PORT
fi

echo "Ray started on rank $RANK"

#############################################################
# 上面是运行环境
#############################################################

if [ $RANK -eq 0 ]; then
    wandb offline
    export WANDB_MODE=offline 
    export WANDB_DIR=/mnt/geminiszgmcephfs/geminicephfs/pr-others-prctrans/xinyanguan/

    export MODEL_PATH="/mnt/geminiszgmcephfs/geminicephfs/pr-others-prctrans/xinyanguan/hf_models/DeepSeek-R1-Distill-Qwen-1.5B"
    # Train over a single node, 8 A100-80GB GPUs.
    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        data.train_files=/mnt/geminiszgmcephfs/geminicephfs/pr-others-prctrans/xinyanguan/ywj-verl/data/math/train.parquet \
        data.val_files=/mnt/geminiszgmcephfs/geminicephfs/pr-others-prctrans/xinyanguan/ywj-verl/data/math/test.parquet \
        data.train_batch_size=8 \
        data.val_batch_size=128 \
        data.max_prompt_length=1024 \
        data.max_response_length=8192 \
        +data.diverse_prompt=True \
        actor_rollout_ref.rollout.max_num_batched_tokens=10350 \
        actor_rollout_ref.model.path=$MODEL_PATH  \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.model.use_liger=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=8 \
        actor_rollout_ref.actor.ppo_micro_batch_size=8 \
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.001 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.temperature=0.6 \
        +actor_rollout_ref.rollout.val_temperature=0.6 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
        actor_rollout_ref.rollout.n=8 \
        actor_rollout_ref.rollout.val_kwargs.n=8 \
        actor_rollout_ref.rollout.val_kwargs.do_sample=True \
        actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
        actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        algorithm.kl_ctrl.kl_coef=0.001 \
        trainer.critic_warmup=0 \
        trainer.logger=['console','wandb'] \
        trainer.project_name='ywj-verl' \
        trainer.experiment_name='debug' \
        trainer.val_before_train=False \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=${WORLD_SIZE} \
        trainer.save_freq=-1 \
        trainer.test_freq=-1 \
        trainer.default_hdfs_dir=null \
        trainer.total_epochs=30 "${@:1}" \
        trainer.default_local_dir="/tmp_dir"

    #############################################################
    # 当主进程训练结束后，执行 ray stop
    #############################################################
    echo "Training is done on rank 0, stopping Ray..."
    ray stop --force

else
    #############################################################
    # rank != 0 的其他进程，等待主进程停止后退出
    #############################################################
    echo "Worker rank $RANK is waiting for Ray to stop..."

    # 方式二（可选）：如果你的 Ray 版本较新，可以用 ray status 检测 
    while true; do
        ray status 1>/dev/null 2>&1
        if [ $? -ne 0 ]; then
            echo "Ray cluster no longer available. Exiting worker..."
            break
        fi
        sleep 5m
    done

fi

echo "Rank $RANK script ended."