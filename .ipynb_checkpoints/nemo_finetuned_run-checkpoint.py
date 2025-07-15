import torch
from pathlib import Path
from nemo.collections import llm
import nemo.lightning as nl
import nemo_run as run
from megatron.core.inference.common_inference_params import CommonInferenceParams

# 1. Find the path to the last saved checkpoint
try:
    sft_ckpt_path = str(next(
        d for d in Path("/datasets/soc-20250703225140/nemo_checkpoints/nemotron_49b_super_custom_finetune/2025-07-14_00-47-12/checkpoints/").iterdir() 
        if d.is_dir() and d.name.endswith("-last")
    ))
    print(f"We will load SFT checkpoint from: {sft_ckpt_path}")
except StopIteration:
    print("Error: Could not find a checkpoint directory ending with '-last'. Please check the path.")
    sft_ckpt_path = None

# 2. Define the prompts for inference
prompts = [
    "How many r's are in the word 'strawberry'?",
    "Which number is bigger? 10.119 or 10.19?",
    "What is the CPIC-recommended dose for metformin in a patient with the CYP2D6 *4/*4 genotype?", # A question from your dataset
]

def trainer() -> run.Config[nl.Trainer]:
    """Configures the Trainer for distributed inference."""
    # The strategy must match the model's training configuration.
    strategy = run.Config(
        nl.MegatronStrategy,
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=1,
    )
    # The trainer configures the environment where the model will run.
    trainer = run.Config(
        nl.Trainer,
        accelerator="gpu",
        devices=2,
        num_nodes=1,
        strategy=strategy,
        logger=False, # Logging is not needed for inference.
        plugins=[
            nl.MegatronMixedPrecision(
                precision="bf16-mixed",
                params_dtype=torch.bfloat16,
            )
        ]
    )
    return trainer

def configure_inference():
    """Configures the inference job using nemo_run.Partial."""
    if not sft_ckpt_path:
        raise FileNotFoundError("SFT Checkpoint path was not found. Cannot configure inference.")
        
    return run.Partial(
        llm.generate,
        path=str(sft_ckpt_path),
        trainer=trainer(),
        prompts=prompts,
        inference_params=CommonInferenceParams(
            num_tokens_to_generate=8192,
            top_p=0.95,
        ),
        output_path="sft_prediction.jsonl",
    )

def local_executor_torchrun(nodes: int = 1, devices: int = 1) -> run.LocalExecutor:
    """Configures the local executor to launch the job."""
    # Environment variables for optimizing distributed performance.
    env_vars = {
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        "NCCL_NVLS_ENABLE": "0",
        "NVTE_DP_AMAX_REDUCE_INTERVAL": "0",
        "NVTE_ASYNC_AMAX_REDUCTION": "1",
    }

    executor = run.LocalExecutor(
        ntasks_per_node=devices, 
        launcher="torchrun", 
        env_vars=env_vars
    )

    return executor

if __name__ == '__main__':
    # Ensure a checkpoint was found before trying to run
    if sft_ckpt_path:
        # The number of devices passed to the executor MUST match the number
        # of devices specified in the trainer configuration.
        run.run(configure_inference(), executor=local_executor_torchrun(devices=2))
