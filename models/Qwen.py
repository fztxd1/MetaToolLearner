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
                    # top_k=40,
                    top_p=0.8,
                    do_sample=True,
                    # num_beams=1,
                    # repetition_penalty=1.1,
                )
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]

                response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

                # input_text = generate_prompt(instruction=prompt, system_prompt=DEFAULT_SYSTEM_PROMPT)
                # negative_text = None 
                # # if args.negative_prompt is None \
                # #     else generate_prompt(instruction=raw_input_text, system_prompt=DEFAULT_SYSTEM_PROMPT)

                # inputs = self.tokenizer(input_text,return_tensors="pt")  #add_special_tokens=False ?
                # generation_output = self.model.generate(
                #                 input_ids = inputs["input_ids"].to(self.device),
                #                 attention_mask = inputs['attention_mask'].to(self.device),
                #                 eos_token_id=self.tokenizer.eos_token_id,
                #                 pad_token_id=self.tokenizer.pad_token_id,
                #                 generation_config = self.generation_config
                #             )
                # s = generation_output[0]
                # output = self.tokenizer.decode(s,skip_special_tokens=True)
                # response = output.split("[/INST]")[-1].strip()
                # print("Response: ",response)
                # print("\n")
                break
            except Exception as e:
                print(f"出现异常：{e}")
                print("等待 5 秒后再次运行...")
                time.sleep(5)  # 等待 5 秒
        return response
    
    def load_lora_model(self, lora_model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(lora_model_path, legacy=True)
        print(f"loading lora model form{lora_model_path}")
        # self.model = PeftModel.from_pretrained(self.model, lora_model_path,torch_dtype=self.load_type,device_map='auto',).half()
        self.model = PeftModel.from_pretrained(self.model, lora_model_path)
