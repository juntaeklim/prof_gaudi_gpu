from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token="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")
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token="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")

