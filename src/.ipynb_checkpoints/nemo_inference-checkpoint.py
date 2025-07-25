import torch
import torch.distributed
from megatron.core.inference.common_inference_params import CommonInferenceParams
import nemo.lightning as nl
import re
from prompt import system_prompt

# Question 0 from the training dataset
# user_question = "What is the dosing recommendation for pitavastatin in pediatric patients with a CYP3A4 poor metabolizer phenotype?"

# Question 9 from the training dataset
# user_question = "What is the suggested therapeutic response metric for IL-2 in renal cell carcinoma based on a patient's IL2RA genotype?"


# Question 3 from the testing dataset
# user_question = "How should fluoxetine dosing be adjusted for an individual with the HTR2A rs7997012 A/A genotype?"
# Question 13 from the testing dataset
# user_question = "What is the risk of treatment failure with standard rabeprazole dosing for a CYP2C19 ultrarapid metabolizer?"
user_question = "A 62-year-old female patient of African American descent with atrial fibrillation and a history of ischemic stroke requires anticoagulation therapy. She is initiated on warfarin, but genotyping reveals she is a CYP2C9 poor metabolizer (*3/*3 genotype) and carries the VKORC1 -1639G>A variant (homozygous A/A, associated with reduced warfarin dose requirements). Analyze the heightened risk of bleeding due to over-anticoagulation from these combined variants, discuss dose initiation and adjustment strategies per CPIC guidelines, and outline a monitoring plan including INR targets and potential switch to direct oral anticoagulants if challenges persist."


strategy = nl.MegatronStrategy(
    tensor_model_parallel_size=2,
    pipeline_model_parallel_size=1,
    context_parallel_size=1,
    sequence_parallel=False,
    setup_optimizers=False,
    store_optimizer_states=False,
)

trainer = nl.Trainer(
    accelerator="gpu",
    devices=2,
    num_nodes=1,
    strategy=strategy,
    plugins=nl.MegatronMixedPrecision(
        precision="bf16-mixed",
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        autocast_enabled=False,
        grad_reduce_in_fp32=False,
    ),
)

source = {
    "mask": "User",
    # "system": "",
    "system": system_prompt,
    "conversations": [
        {
            "from": "User",
            "value": user_question
        },
    ]
}

special_tokens = {
                "system_turn_start": "<extra_id_0>",
                "turn_start": "<extra_id_1>",
                "label_start": "<extra_id_2>",
                "end_of_turn": "\n",
                "end_of_name": "\n",
            }

from nemo.collections.nlp.data.language_modeling.megatron.gpt_sft_chat_dataset import _get_header_conversation_type_mask_role
# Apply prompt template to be the same format as training
header, conversation, data_type, mask_role = _get_header_conversation_type_mask_role(source, special_tokens)
prompts = [conversation]

from nemo.collections.llm import api
results = api.generate(
    # path="/datasets/soc-20250703225140/models/nvidia/Llama-3_3-Nemotron-Super-49B-v1/",
    path="/datasets/cc-20250630151645/nemo_checkpoints/nemotron_49b_super_custom_finetune/2025-07-24_04-00-28/checkpoints/model_name=0--val_loss=0.02-step=3199-consumed_samples=6400.0/",
    prompts=prompts,
    trainer=trainer,
    inference_params=CommonInferenceParams(
        temperature=1.0,
        top_p=0,  # greedy decoding
        top_k=1,  # greedy decoding
        num_tokens_to_generate=512,
    ),
    text_only=True,
)

# if torch.distributed.get_rank() == 0:

#     for i, r in enumerate(results):
#         print("=" * 50)
#         print(prompts[i])
#         print("*" * 50)
#         print(len(r))
#         print(r)
#         # if match:
#         #     print(match.group(0))
#         # else:
#         #     print(r)
#         # print("=" * 50)
#         # print("\n\n")
# 如果要額外過濾答案，可以在這裡寫 regex
# 例：擷取 <extra_id_2> 與下一個 <extra_id_x> 之間的內容
answer_pattern = re.compile(r"<extra_id_(.*?)</extra_id_2>", re.S)

for prompt_text, raw_text in zip(prompts, results):
    print("=" * 80)
    print(prompt_text)
    print("-" * 80)

    # 嘗試擷取「標示為回答」的區段；若找不到就整段輸出
    match = answer_pattern.search(raw_text)
    answer = match.group(1).strip() if match else raw_text.strip()

    print(f"[字元數] {len(answer)}")
    print(answer)
    print("=" * 80 + "\n")
