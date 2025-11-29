from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "gpt2"  # replace with small local model or HF hosted LLM
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def query_llm(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=100)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Test a seed prompt
prompt = "You are Furina from Genshin Impact. How should I handle failure?"
print(query_llm(prompt))