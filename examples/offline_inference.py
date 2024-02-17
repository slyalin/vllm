from vllm import LLM, SamplingParams
import torch
import time

# Sample prompts.
prompts = [
    "What is OpenVINO?",
    # "The president of the United States is",
    # "The capital of France is",
    # "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(max_tokens=256, ignore_eos=True)

# Create an LLM.
llm = LLM(model="facebook/opt-6.7b", dtype=torch.float32, device='cpu', trust_remote_code=True, seed=42)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.

generate_start = time.perf_counter()
outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
generate_end = time.perf_counter()

total_generation_time = (generate_end - generate_start) * 1000
print(f"Total generation time {total_generation_time} ms")

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
