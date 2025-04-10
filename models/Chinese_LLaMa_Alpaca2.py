DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant. 你是一个乐于助人的助手。"""

TEMPLATE = (
    "[INST] <<SYS>>\n"
    "{system_prompt}\n"
    "<</SYS>>\n\n"
    "{instruction} [/INST]"
)

import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import GenerationConfig
from peft import PeftModel
import time

def generate_prompt(instruction, system_prompt=DEFAULT_SYSTEM_PROMPT):
    return TEMPLATE.format_map({'instruction': instruction,'system_prompt': system_prompt})

class Chinese_LLaMa_Alpaca_2(object):
    def __init__(self, model_path, lora_model_path):
        tokenizer_path = model_path
        self.load_type = torch.float16
        if torch.cuda.is_available():
            # self.device = torch.device(0)
            self.device = torch.device('cuda')
        self.tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path, legacy=True)
        self.model = LlamaForCausalLM.from_pretrained(
                model_path,
                torch_dtype=self.load_type,
                low_cpu_mem_usage=True,
                device_map='auto',
                )
        self.model_vocab_size = self.model.get_input_embeddings().weight.size(0)
        self.tokenizer_vocab_size = len(self.tokenizer)
        # print(f"Vocab of the base model: {model_vocab_size}")
        # print(f"Vocab of the tokenizer: {tokenizer_vocab_size}")
        if self.model_vocab_size!=self.tokenizer_vocab_size:
            print("Resize model embeddings to fit tokenizer")
            self.model.resize_token_embeddings(self.tokenizer_vocab_size)
        self.generation_config = GenerationConfig(
            temperature=0.8,
            # top_k=40,
            top_p=0.8,
            do_sample=True,
            # num_beams=1,
            # repetition_penalty=1.1,
            max_new_tokens=4096
        )
        if lora_model_path is not None:
            self.load_lora_model(lora_model_path)
        self.model.eval()

    def get_response(self, prompt):
        while True:
            try:
                input_text = generate_prompt(instruction=prompt, system_prompt=DEFAULT_SYSTEM_PROMPT)
                negative_text = None 
                # if args.negative_prompt is None \
                #     else generate_prompt(instruction=raw_input_text, system_prompt=DEFAULT_SYSTEM_PROMPT)

                inputs = self.tokenizer(input_text,return_tensors="pt")  #add_special_tokens=False ?
                generation_output = self.model.generate(
                                input_ids = inputs["input_ids"].to(self.device),
                                attention_mask = inputs['attention_mask'].to(self.device),
                                eos_token_id=self.tokenizer.eos_token_id,
                                pad_token_id=self.tokenizer.pad_token_id,
                                generation_config = self.generation_config
                            )
                s = generation_output[0]
                output = self.tokenizer.decode(s,skip_special_tokens=True)
                response = output.split("[/INST]")[-1].strip()
                # print("Response: ",response)
                # print("\n")
                break
            except Exception as e:
                print(f"出现异常：{e}")
                print("等待 5 秒后再次运行...")
                time.sleep(5)  # 等待 5 秒
        return response
    
    def load_lora_model(self, lora_model_path):
        self.tokenizer = LlamaTokenizer.from_pretrained(lora_model_path, legacy=True)
        print(f"loading lora model form{lora_model_path}")
        # self.model = PeftModel.from_pretrained(self.model, lora_model_path,torch_dtype=self.load_type,device_map='auto',).half()
        self.model = PeftModel.from_pretrained(self.model, lora_model_path)
