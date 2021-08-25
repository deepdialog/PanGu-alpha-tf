#! /bin/bash

export CUDA_VISIBLE_DEVICES="0,1"
# Runs the "345M" parameter model

GPUS_PER_NODE=2
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DATA_PATH=/userhome/dataset/megatron/zhanghan/sample_100G_policy_3/Sample100GPolicy3_text_document
#DATA_PATH=/userhome/dataset/megatron/test_vocab4w/text_document
#CHECKPOINT_PATH=/ghome/yands/model/checkPoints/megatron-1.1-pangu
CHECKPOINT_PATH=/userhome/model/checkPoints/megatron-1.1-pangu-2.6B/merged_split
#CHECKPOINT_PATH=/userhome/model/panguAlpha_2.6b_fp16_NumpyCkpt/merged/

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

/opt/conda/bin/python -u -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_gpt2.py \
       --model-parallel-size 2 \
       --num-layers 31 \
       --hidden-size 2560 \
       --num-attention-heads 32 \
       --batch-size 1 \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --train-iters 500000 \
       --lr-decay-iters 320000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file /userhome/pclproject/gpt/Megatron-LM-1.1-Pangu/megatron/tokenizer/bpe_4w_pcl/vocab \
       --merge-file gpt2-merges.txt \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --lr-decay-style cosine \
       --min-lr 1.0e-5 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --warmup .01 \
       --checkpoint-activations \
       --log-interval 100 \
       --save-interval 1000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --attention-dropout 0.1 \
       --hidden-dropout 0.1 \
       --fp16 \
       --reset-attention-mask \
       --finetune



set +x
