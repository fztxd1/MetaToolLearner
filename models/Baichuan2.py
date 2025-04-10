import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from transformers.generation.utils import GenerationConfig
import os
from peft import PeftModel
import time

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class Baichuan2(object):
    def __init__(self, model_path, lora_model_path):
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
        tokenizer_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False, trust_remote_code=True)
        self.model.generation_config = GenerationConfig.from_pretrained("/root/autodl-tmp/model/Baichuan2-7B-Chat")

        if lora_model_path is not None:
            self.load_lora_model(lora_model_path)
        self.model.eval()

    def get_response(self, prompt):
        messages = []
        messages.append({"role": "user", "content": prompt})

        # print("Response: ",response)
        # print("\n")
        while True:
            try:
                response = self.model.chat(
                    self.tokenizer, 
                    messages)

                break 
            except Exception as e:
                time.sleep(5)
        return response

    def load_lora_model(self, lora_model_path):
        print(f"loading lora model from {lora_model_path}")
        self.model = PeftModel.from_pretrained(self.model, lora_model_path)