from transformers import AutoModelForCausalLM, AutoTokenizer

import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import GenerationConfig
from peft import PeftModel
import time

class Qwen(object):
    def __init__(self, model_path, lora_model_path):
        if torch.cuda.is_available():
            # self.device = torch.device(0)
            self.device = torch.device('cuda')
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if lora_model_path is not None:
            self.load_lora_model(lora_model_path)
        self.model.eval()

    def get_response(self, prompt):
        while True:
            try:
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

                generated_ids = self.model.generate(
                    model_inputs.input_ids,
                    max_new_tokens=512,
                    temperature=0.8,
                    top_p=0.8,
                    do_sample=True,
                )
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]

                response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                break
            except Exception as e:
                time.sleep(5)
        return response
    
    def load_lora_model(self, lora_model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(lora_model_path, legacy=True)
        print(f"loading lora model form{lora_model_path}")
        # self.model = PeftModel.from_pretrained(self.model, lora_model_path,torch_dtype=self.load_type,device_map='auto',).half()
        self.model = PeftModel.from_pretrained(self.model, lora_model_path)
