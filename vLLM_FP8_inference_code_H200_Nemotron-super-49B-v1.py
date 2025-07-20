from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# 1. Define Model and Engine Parameters
# ---------------------------------------
# Model identifier on Hugging Face Hub
model_name = "nvidia/Llama-3_3-Nemotron-Super-49B-v1-FP8"

# vLLM engine parameters based on Nvidia's recommendations
llm = LLM(
    model=model_name,
    trust_remote_code=True,
    tensor_parallel_size=8,
    quantization='modelopt',
    max_model_len=32768,
    gpu_memory_utilization=0.95,
    enforce_eager=True
)

# 2. Prepare the Input Prompt
# ---------------------------------------
# Load the tokenizer to apply the chat template
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Your input in chat format
messages = [
    {"role": "user", "content": "Who are you?"}
]

# Apply the chat template to format the input correctly for the model
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# 3. Define Sampling Parameters
# ---------------------------------------
# These parameters control the generation process, now with the correct temperature
sampling_params = SamplingParams(
    temperature=0.6,  # CORRECTED: Set temperature to Nvidia's recommended value
    top_p=0.95,
    max_tokens=1024
)

# 4. Run Inference
# ---------------------------------------
print("Running inference with vLLM...")
outputs = llm.generate(prompt, sampling_params)
print("Inference complete.")

# 5. Print the Output
# ---------------------------------------
for output in outputs:
    prompt_out = output.prompt
    generated_text = output.outputs[0].text
    print("\n--- Model Response ---")
    print(f"Prompt: {prompt_out!r}")
    print(f"Generated: {generated_text!r}")
    print("----------------------")