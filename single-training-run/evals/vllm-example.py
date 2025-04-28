from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(
    temperature=0.0,          # deterministic demo
    max_tokens=1,             # generate a single dummy token
    logprobs=0,               # generated tokens
    prompt_logprobs=0         # prefix tokens
)


# load hellaswag queries
import pickle
#with open("hellaswag_queries.pkl", "rb") as f:
#    prompts = pickle.load(f)

import os
download_dir = None
if os.path.exists("/mnt/lustre/work"):
    download_dir = "/mnt/lustre/work/luxburg/sbordt10/.cache/vllm"

llm = LLM(model="allenai/OLMo-1B-hf", download_dir = download_dir, enable_prefix_caching=False)


outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

# if the "../results" directory does not exist, create it
if not os.path.exists("../results"):
    os.makedirs("../results")

# save the outputs
with open("../results/hellaswag_outputs.pkl", "wb") as f:
    pickle.dump(outputs, f)