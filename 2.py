import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map="cpu",  # If you have a GPU, replace with "auto"
    torch_dtype=torch.float32,  # Use float32 for CPU
    trust_remote_code=True
)

# Creating a pipeline
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
    max_new_tokens=50,  # Instead of max_length
    do_sample=False,
)

prompt = "Write an email for a holiday request for a week for sister's marriage"

output = generator(prompt, max_new_tokens=100, truncation=True)

print(output[0]['generated_text'])
