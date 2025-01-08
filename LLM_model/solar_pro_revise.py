from llama_cpp import Llama

llm = Llama.from_pretrained(
	repo_id="MaziyarPanahi/solar-pro-preview-instruct-GGUF",
	filename="solar-pro-preview-instruct.IQ1_M.gguf",
)
