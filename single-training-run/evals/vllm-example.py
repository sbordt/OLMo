from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(
    temperature=0.0,          # deterministic demo
    max_tokens=20,
    logprobs=1,               # generated tokens
    prompt_logprobs=1         # prefix tokens
)




llm = LLM(model="allenai/OLMo-1B-hf", kwargs={"download-dir": "/mnt/lustre/work/luxburg/sbordt10/.cache/vllm"})





outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")