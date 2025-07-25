# ------------------------------------------------------------------
# Query-rewrite helper for the finetuned Nemotron 49B checkpoint
# ------------------------------------------------------------------
import re
from pathlib import Path
import torch
import nemo.lightning as nl
from megatron.core.inference.common_inference_params import CommonInferenceParams
from nemo.collections.llm import api

# template tokens used during SFT
_T = {
    "SYS_START": "<extra_id_0>",
    "TURN_START": "<extra_id_1>",
    "LABEL_START": "<extra_id_2>",
    "EOT": "\n",          # end-of-turn
    "EON": "\n",          # end-of-name
}

# regex to carve out the model answer that follows <extra_id_2>
_ANS_RE = re.compile(r"<extra_id_2>(.*?)<extra_id_\d+>", re.S)


def query_rewrite(
    *,
    question_prompt: str,
    system_prompt: str | None = None,
    num_tokens_to_generate: int,
    temperature: float = 1.0,
    top_p: float = 0.0,
    top_k: int = 1,
    ckpt_path: str | Path = (
        "/datasets/cc-20250630151645/nemo_checkpoints/"
        "nemotron_49b_super_custom_finetune/2025-07-24_04-00-28/checkpoints/"
        "model_name=0--val_loss=0.02-step=3199-consumed_samples=6400.0/"
    ),
) -> str:
    """
    Rewrite a user question into CPIC-style retrieval sub-queries.

    Returns the model’s answer text (already stripped of training tokens).
    """
    # ------------------------------------------------------------------ #
    # 1) Construct prompt exactly as in SFT                              #
    # ------------------------------------------------------------------ #
    sys_block = (
        _T["SYS_START"] + (system_prompt or "") + _T["EOT"]
    )
    user_block = (
        _T["TURN_START"] + "User" + _T["EON"]
        + question_prompt + _T["EOT"]
        + _T["LABEL_START"]        # model should write below this token
    )
    prompt = sys_block + user_block

    # ------------------------------------------------------------------ #
    # 2) Build / cache a Megatron Trainer once                           #
    # ------------------------------------------------------------------ #
    if not hasattr(query_rewrite, "_trainer"):
        strategy = nl.MegatronStrategy(
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            sequence_parallel=False,
            setup_optimizers=False,
            store_optimizer_states=False,
        )
        query_rewrite._trainer = nl.Trainer(
            accelerator="gpu",
            devices=2,
            strategy=strategy,
            plugins=nl.MegatronMixedPrecision(
                precision="bf16-mixed",
                params_dtype=torch.bfloat16,
                pipeline_dtype=torch.bfloat16,
                autocast_enabled=False,
                grad_reduce_in_fp32=False,
            ),
        )
    trainer = query_rewrite._trainer

    # ------------------------------------------------------------------ #
    # 3) Generate                                                        #
    # ------------------------------------------------------------------ #
    raw = api.generate(
        path=str(ckpt_path),
        prompts=[prompt],
        trainer=trainer,
        inference_params=CommonInferenceParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_tokens_to_generate=num_tokens_to_generate,
        ),
        text_only=True,
    )[0]  # single prompt → single result

    # ------------------------------------------------------------------ #
    # 4) Strip labels / extra tokens                                     #
    # ------------------------------------------------------------------ #
    m = _ANS_RE.search(raw)
    return (m.group(1) if m else raw).strip()
