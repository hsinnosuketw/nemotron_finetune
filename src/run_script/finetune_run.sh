python train_nemotron.py \
  --checkpoint_dir /datasets/soc-20250703225140/nemo_checkpoints/ \
  --data_dir /datasets/soc-20250703225140/dataset_split_without_sysprompt/ \
  --cache_dir /datasets/soc-20250703225140/ \
  --num_nodes 1 \
  --gpus_per_node 2 \
  --peft_scheme lora \
  --global_batch_size 2 \
  --micro_batch_size 1
