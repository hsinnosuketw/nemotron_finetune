import os
import argparse
import torch
from transformers import AutoTokenizer
from nemo.collections import llm
from nemo.collections.llm.gpt.data import FineTuningDataModule
import nemo_run as run


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune NeMo LLaMA 49B model")

    # 基本設定
    parser.add_argument("--tokenizer_name", type=str, default="hf://nvidia/Llama-3_3-Nemotron-Super-49B-v1")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Checkpoint directory to save/load NeMo checkpoints")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to your training/validation/test .jsonl files")
    parser.add_argument("--cache_dir", type=str, required=True, help="Path to Huggingface and NeMo cache directory")

    # 訓練設定
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--gpus_per_node", type=int, default=2)
    parser.add_argument("--peft_scheme", type=str, choices=["lora", "none"], default="lora")
    parser.add_argument("--packed_sequence", action="store_true")
    parser.add_argument("--global_batch_size", type=int, default=2)
    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument("--dataset_length", type=int, default=3073)
    parser.add_argument("--min_epochs", type=int, default=3)

    # 模型平行化設定
    parser.add_argument("--tensor_parallel", type=int, default=2)
    parser.add_argument("--pipeline_parallel", type=int, default=1)

    return parser.parse_args()


def setup_environment(cache_dir):
    os.environ['HF_HOME'] = cache_dir
    os.environ['NEMO_HOME'] = cache_dir


def load_model_and_tokenizer(model_source, hf_repo):
    model_config = llm.LlamaNemotronModel(llm.Llama33NemotronSuper49BConfig())
    llm.import_ckpt(model=model_config, source=model_source)
    tokenizer = AutoTokenizer.from_pretrained(hf_repo)
    return model_config, tokenizer


def create_recipe(args, model_config):
    max_epochs = args.min_epochs + 1
    max_steps = max_epochs * args.dataset_length // 2
    min_steps = args.min_epochs * args.dataset_length // 2

    recipe = llm.llama33_nemotron_super_49b.finetune_recipe(
        name="nemotron_49b_super_custom_finetune",
        dir=args.checkpoint_dir,
        num_nodes=args.num_nodes,
        num_gpus_per_node=args.gpus_per_node,
        peft_scheme=args.peft_scheme,
        packed_sequence=args.packed_sequence,
    )

    data_module = run.Config(
        llm.ChatDataModule,
        dataset_root=args.data_dir,
        seq_length=recipe.model.config.seq_length,
        micro_batch_size=args.micro_batch_size,
        global_batch_size=args.global_batch_size,
    )

    recipe.data = data_module
    recipe.trainer.devices = args.gpus_per_node
    recipe.trainer.max_epochs = max_epochs
    recipe.trainer.min_epochs = args.min_epochs
    recipe.trainer.max_steps = max_steps
    recipe.trainer.min_steps = min_steps
    recipe.trainer.enable_progress_bar = True
    recipe.trainer.enable_model_summary = True

    recipe.trainer.strategy.tensor_model_parallel_size = args.tensor_parallel
    recipe.trainer.strategy.pipeline_model_parallel_size = args.pipeline_parallel
    recipe.trainer.accumulate_grad_batches = args.micro_batch_size // args.micro_batch_size  # usually 1

    return recipe


def main():
    args = parse_args()
    setup_environment(args.cache_dir)

    print("Loading model and tokenizer...")
    model_config, tokenizer = load_model_and_tokenizer(args.tokenizer_name, args.tokenizer_name.split("hf://")[-1])

    print("Building and initiating finetuning recipe...")
    recipe = create_recipe(args, model_config)

    print("Starting training...")
    run.run(recipe, direct=True)
    print("Training completed!")


if __name__ == "__main__":
    main()
