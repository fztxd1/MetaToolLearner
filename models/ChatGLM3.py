import os
from transformers import AutoTokenizer, AutoModel
import torch
from peft import PeftModel
import time

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class ChatGLM3(object):
    def __init__(self, model_path, lora_model_path):
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True, device_map="auto")
        tokenizer_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        if lora_model_path is not None:
            self.load_lora_model(lora_model_path)
        self.model.eval()

    def get_response(self, prompt):

        # print("Response: ",response)
        # print("\n")
        while True:
            try:
                response, history = self.model.chat(
                    self.tokenizer, 
                    prompt, 
                    history=[],        
                    max_length=4096,
                    top_p=0.8,
                    temperature=0.8)
                break  # 如果成功执行到这里，跳出循环
            except Exception as e:
                print(f"出现异常：{e}")
                print("等待 5 秒后再次运行...")
                time.sleep(5)  # 等待 5 秒
        return response

    def load_lora_model(self, lora_model_path):
        print(f"loading lora model from {lora_model_path}")
        self.model = PeftModel.from_pretrained(self.model, lora_model_path)