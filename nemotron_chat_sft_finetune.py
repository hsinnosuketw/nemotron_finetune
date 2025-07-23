import os
import ast
import json

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer

from pytorch_lightning import LightningDataModule
import pytorch_lightning as pl

from nemo.collections import llm
from nemo.collections.llm.gpt.data import FineTuningDataModule
import nemo_run as run

# --- 1. NeMo 2.0 Finetuning settings ---

# directory setting
TOKENIZER_NAME = "hf://nvidia/Llama-3_3-Nemotron-Super-49B-v1"
# CHECKPOINT_DIR = "/datasets/soc-20250703225140/nemo_checkpoints/"
CHECKPOINT_DIR = "/datasets/cc-20250630151645/nemo_checkpoints/"
# CHECKPOINT_DIR = "/datasets/soc-20250703225140/nemo_checkpoints/nemotron_49b_super_custom_finetune/2025-07-14_17-42-12/checkpoints/"

# include training, validation, and test.jsonl files
# with sysprompt
# DATA_DIR = "/datasets/soc-20250703225140/dataset_split_with_sysprompt/"

# without sysprompt
# DATA_DIR = "/datasets/soc-20250703225140/dataset_split_without_sysprompt/"
DATA_DIR = "/datasets/cc-20250630151645/dataset_split_with_partial_sysprompt/"


# CACHE_DIR = "/datasets/soc-20250703225140/"
CACHE_DIR = "/datasets/cc-20250630151645/"

# environment parameters setting
os.environ['HF_HOME'] = CACHE_DIR
os.environ['NEMO_HOME'] = CACHE_DIR

# model configurations
MODEL_CONFIG=llm.LlamaNemotronModel(llm.Llama33NemotronSuper49BConfig())
MODEL_SOURCE='hf://nvidia/Llama-3_3-Nemotron-Super-49B-v1'              
HF_REPO="nvidia/Llama-3_3-Nemotron-Super-49B-v1"

# device settings
NUM_NODES = 1           
GPUS_PER_NODE = 2     

# PEFT settings
PEFT_SCHEME = 'lora'      # 'lora' or 'none' (Full-FT)
PACKED_SEQUENCE = False   

# batch size settings
# gbs = mbs * gradient_accumulation_steps * num_gpus
GLOBAL_BATCH_SIZE = 2
MICRO_BATCH_SIZE = 1

LEN_DATASET = 3073

# training parameters
MIN_EPOCHS=3
MAX_EPOCHS=MIN_EPOCHS + 1

MAX_STEPS = (MAX_EPOCHS) * LEN_DATASET // 2
MIN_STEPS = (MIN_EPOCHS) * LEN_DATASET // 2

# parallelism settings
tensor_model_parallel_size = 2
pipeline_model_parallel_size = 1


# --- 2. NeMo 2.0 Model Loading Procedures ---

# Load model
llm.import_ckpt(model=MODEL_CONFIG, source=MODEL_SOURCE)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(HF_REPO)


# --- 2. NeMo 2.0 Recipe Finetuning Procedures ---
# Customize finetuning recipe

print("Building and initiating finetuning recipe...")

recipe = llm.llama33_nemotron_super_49b.finetune_recipe(
    name="nemotron_49b_super_custom_finetune",
    dir=CHECKPOINT_DIR,
    num_nodes=NUM_NODES,
    num_gpus_per_node=GPUS_PER_NODE,
    peft_scheme=PEFT_SCHEME,
    packed_sequence=PACKED_SEQUENCE,
)

# Data Loader Initiating
print("Overwriting dataset in recipe with customized dataset...")

# Initiating DataModule Instance
data_module = run.Config(
    llm.ChatDataModule,
    dataset_root=DATA_DIR,
    seq_length=recipe.model.config.seq_length,
    micro_batch_size=MICRO_BATCH_SIZE,
    global_batch_size=GLOBAL_BATCH_SIZE,
)

# recipe data setting
recipe.data = data_module

# recipe trainer setting
recipe.trainer.devices = GPUS_PER_NODE
recipe.trainer.max_epochs = MAX_EPOCHS
recipe.trainer.min_epochs = MIN_EPOCHS
recipe.trainer.max_steps = MAX_STEPS
recipe.trainer.min_steps = MIN_STEPS

recipe.trainer.enable_progress_bar=True
recipe.trainer.enable_model_summary=True

recipe.trainer.strategy.tensor_model_parallel_size = tensor_model_parallel_size
recipe.trainer.strategy.pipeline_model_parallel_size = pipeline_model_parallel_size
accumulate_steps = MICRO_BATCH_SIZE // MICRO_BATCH_SIZE
recipe.trainer.accumulate_grad_batches = accumulate_steps


# --- 3. NeMo 2.0 Recipe Finetuning Start ---
print("Recipe 設定完成，準備開始訓練...")
run.run(recipe, direct=True)

print("訓練完成！")