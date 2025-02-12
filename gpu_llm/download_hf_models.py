from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Llama-3.1-8B-Instruct"
# model_name = "facebook/opt-6.7b"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="/", use_auth_token="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")
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir="/", use_auth_token="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")

assert True