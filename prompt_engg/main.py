##prompt enggg

#loading the model
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

#load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
 
 )

tokenizer=AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

#create a pipline

pipe=pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
    max_new_tokens=500,
    do_sample=False, #this tells no sampling will be done and the model will always return the most likely token

)


messages = [
 {"role": "user", "content": "Create a funny joke about chickens."}
 ]
 # Generate the output
output = pipe(messages)
print(output[0]["generated_text"])


##temoerature controls the randmness of the output. A lower temperature (e.g., 0.2) makes the model more deterministic, while a higher temperature (e.g., 0.8) makes it more random.

output = pipe(messages, do_sample=True, temperature=1)
print(output[0]["generated_text"])