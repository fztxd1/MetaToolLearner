import argparse
from models.ChatGLM3 import ChatGLM3
from models.Chinese_LLaMa_Alpaca2 import Chinese_LLaMa_Alpaca_2
from models.Qwen import Qwen
from models.Baichuan2 import Baichuan2
import json
import random
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default=None, type=str, required=True, choices=['ChatGLM3', 'Chinese_LLaMa_Alpaca2_7B', 'Chinese_LLaMa_Alpaca2_13B', 'Qwen_0.5B', 'Qwen_1.8B', 'Qwen_4B', 'Qwen_7B', 'Qwen_14B', 'Qwen_32B', 'Qwen_72B', 'Qwen_110B', 'Baichuan2_7B'])
parser.add_argument('--lora_model_path', default=None, type=str, help="If None, perform inference on the base model")
parser.add_argument('--model_path', default=None, type=str)
parser.add_argument('--evaluation_strategy', default=None, type=str, choices=['Base', 'CoT', 'ReAct', 'Lora', 'MFT'])
parser.add_argument('--Sampling_strategy', default=None, type=str, choices=['AT', 'WT'])
parser.add_argument('--test_data_file', default=None, type=str)
parser.add_argument('--save_path', default=None, type=str)
parser.add_argument('--test_num', default=None, type=str, required=True, choices=['test01', 'test02', 'test03', 'test04', 'test05'])

args = parser.parse_args()

if args.model_name == "ChatGLM3":
    LLM = ChatGLM3(args.model_path, args.lora_model_path)
if args.model_name == "Chinese_LLaMa_Alpaca2_7B" or args.model_name == "Chinese_LLaMa_Alpaca2_13B":
    LLM = Chinese_LLaMa_Alpaca_2(args.model_path, args.lora_model_path)
if args.model_name == "Qwen_0.5B" or args.model_name == "Qwen_1.8B" or args.model_name == "Qwen_4B" or args.model_name == "Qwen_7B" or args.model_name == "Qwen_14B" or args.model_name == "Qwen_32B" or args.model_name == "Qwen_72B" or args.model_name == "Qwen_110B":
    LLM = Qwen(args.model_path, args.lora_model_path)
if args.model_name == "Baichuan2_7B":  
    LLM = Baichuan2(args.model_path, args.lora_model_path)



seed_num = 0
if args.test_num == "test01":
    seed_num = 135246
if args.test_num == "test02":
    seed_num = 13524602
if args.test_num == "test03":
    seed_num = 13524603
if args.test_num == "test04":
    seed_num = 13524604
if args.test_num == "test05":
    seed_num = 13524605
random.seed(seed_num)

with open(args.test_data_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

with open("/root/autodl-fs/work/Tool-augmented/dataset/train.json", 'r', encoding='utf-8') as f:
    train_data = json.load(f)

tools_WT = {}
with open("/root/autodl-fs/work/Tool-augmented/dataset/all_tools.json", 'r', encoding='utf-8') as f:
    tools_WT = json.load(f)

tools_AT = []
for key, value in tools_WT.items():
    for item in value:
        if item not in tools_AT:
            tools_AT.append(item)

def get_random_tools_AT(tools, tool, num):
    filtered_list = [element for element in tools if element != tool]
    sampled_tools = random.sample(filtered_list, min(num, len(filtered_list)))
    random_index = random.randint(0, len(sampled_tools))
    sampled_tools.insert(random_index, tool)
    return sampled_tools

def get_random_tools_WT(tools, tool, tool_cls, num):
    tools_cls = tools[tool_cls] 
    filtered_list = [element for element in tools_cls if element != tool]
    sampled_tools = random.sample(filtered_list, min(num, len(filtered_list)))
    random_index = random.randint(0, len(sampled_tools))
    sampled_tools.insert(random_index, tool)
    return sampled_tools



save_result = []

# lora&base
prompt_template_Base = """你是一个能与工具交互的智能AI，回答下面的问题：
当你需要使用工具时，在回答的开头加上工具的引用，例如如果你想要使用查看新闻工具，应该以如下格式回答：
使用[查看新闻]工具，通过该功能或工具，用户可以浏览并获取最新的新闻报道、时事评论、娱乐资讯以及其他各个领域的资讯信息。
"""

prompt_template_CoT = """扮演一个能与工具交互的智能AI来解决问题，你需要一步一步给出思考过程，并最终用Answer给出答案，例如如果你想要使用查看新闻工具，应该以如下格式回答：Answer：使用[查看新闻]工具。
这是一个例子：
你是一个能与工具交互的智能AI，你能使用以下工具：设置手机勿扰模式，矩阵计算工具，查看课表，打开终端，制作电子表格，你需要一步一步的思考并给出思考过程
回答下面的问题：
我正在做一个项目，并需要整理大量的数据，我可以使用电子表格工具来创建一个结构化的表格，方便我进行数据分析。
Think：设置手机勿扰模式是一项功能或工具，它允许用户屏蔽来电、消息和通知，以确保在不受打扰的情况下专注于重要任务、工作或休息。用户可以自定义勿扰模式的时间段和设置，例如设定白天工作时间、睡眠时间或会议期间，手机将自动静音、屏蔽来电信息，让用户享受更好的工作和休息体验。
矩阵计算工具是一种功能强大的工具，用于进行矩阵运算和计算行列式、矩阵乘法、逆矩阵等数学操作。它能够快速且准确地完成复杂的矩阵计算任务，帮助用户进行数据分析、统计学研究、工程计算等领域的相关工作。该工具提供直观的用户界面，让用户可以轻松输入和编辑矩阵，并可自定义运算符和函数，以满足不同的计算需求。无论是学生、工程师还是研究人员，都可借助矩阵计算工具高效地完成数学运算并获得准确的结果。
查看课表使用电子课表应用来查看个人的课程安排和时间表
打开终端是一种用于访问计算机操作系统的命令行界面，可以通过输入命令来执行各种系统操作和管理任务。它提供了一个简洁而高效的方式来控制和调整计算机的设置、访问文件系统、安装/卸载软件、运行脚本和程序等。终端可以被用作开发者和系统管理员进行高级操作和故障排除的重要工具。
制作电子表格使用电子表格软件来创建、编辑和管理数据表格、图表和公式，以便更好地组织和分析数据，并与他人共享和协作。
Answer：使用[制作电子表格]工具

下面是你需要回答的问题：
"""

prompt_template_ReAct = """扮演一个能与工具交互的智能AI来解决问题，你需要用Think，Act，Observation来给出思考过程，并最终用Answer给出答案，例如如果你想要使用查看新闻工具，应该以如下格式回答：Answer：使用[查看新闻]工具。
这是一个例子：
你是一个能与工具交互的智能AI，你能使用以下工具：设置手机勿扰模式，矩阵计算工具，查看课表，打开终端，制作电子表格，你需要先思考并了解各个工具的具体作用，然后再用Answer给出答案
回答下面的问题：
我正在做一个项目，并需要整理大量的数据，我可以使用电子表格工具来创建一个结构化的表格，方便我进行数据分析。
Think：首先我需要知道各个工具的使用场景才能知道为了解决这个问题我最好使用哪些工具
Act：搜索设置手机勿扰模式，矩阵计算工具，查看课表，打开终端，制作电子表格
Observation：设置手机勿扰模式是一项功能或工具，它允许用户屏蔽来电、消息和通知，以确保在不受打扰的情况下专注于重要任务、工作或休息。用户可以自定义勿扰模式的时间段和设置，例如设定白天工作时间、睡眠时间或会议期间，手机将自动静音、屏蔽来电信息，让用户享受更好的工作和休息体验。
矩阵计算工具是一种功能强大的工具，用于进行矩阵运算和计算行列式、矩阵乘法、逆矩阵等数学操作。它能够快速且准确地完成复杂的矩阵计算任务，帮助用户进行数据分析、统计学研究、工程计算等领域的相关工作。该工具提供直观的用户界面，让用户可以轻松输入和编辑矩阵，并可自定义运算符和函数，以满足不同的计算需求。无论是学生、工程师还是研究人员，都可借助矩阵计算工具高效地完成数学运算并获得准确的结果。
查看课表使用电子课表应用来查看个人的课程安排和时间表
打开终端是一种用于访问计算机操作系统的命令行界面，可以通过输入命令来执行各种系统操作和管理任务。它提供了一个简洁而高效的方式来控制和调整计算机的设置、访问文件系统、安装/卸载软件、运行脚本和程序等。终端可以被用作开发者和系统管理员进行高级操作和故障排除的重要工具。
制作电子表格使用电子表格软件来创建、编辑和管理数据表格、图表和公式，以便更好地组织和分析数据，并与他人共享和协作。
Answer：使用[制作电子表格]工具

下面是你需要回答的问题：
"""

prompt_template_Lora = """你是一个能与工具交互的智能AI，回答下面的问题：
当你需要使用工具时，在回答的开头加上工具的引用，例如如果你想要使用查看新闻工具，应该以如下格式回答：
使用[查看新闻]工具，通过该功能或工具，用户可以浏览并获取最新的新闻报道、时事评论、娱乐资讯以及其他各个领域的资讯信息。
"""

if args.evaluation_strategy == "Base":
    prompt_template = prompt_template_Base
    for item in tqdm(data, desc="Processing"):
        tool, info, question, tool_cls = item["tool"], item["info"], item["question"], item["tool_cls"]
        if args.Sampling_strategy == "AT":
            sampled_tools = get_random_tools_AT(tools_AT, tool, 4)
        elif args.Sampling_strategy == "WT":
            sampled_tools = get_random_tools_WT(tools_WT, tool, tool_cls, 4)
        sampled_tools_text = "，".join(sampled_tools)
        question = f"你能使用以下工具：{sampled_tools_text}。请回答下面的问题：\n{question}"
        prompt = prompt_template + question
        response = LLM.get_response(prompt)
        # response = "test"
        # print("工具: ", sampled_tools)
        # print("response: ", response)
        # print("#####"*10)
        save_result.append(
            {
                "model_output": response,
                "label": item['tool'],
                "tools": sampled_tools_text
            }
        )

elif args.evaluation_strategy == "CoT":
    prompt_template = prompt_template_CoT
    for item in tqdm(data, desc="Processing"):
        tool, info, question, tool_cls = item["tool"], item["info"], item["question"], item["tool_cls"]
        if args.Sampling_strategy == "AT":
            sampled_tools = get_random_tools_AT(tools_AT, tool, 4)
        elif args.Sampling_strategy == "WT":
            sampled_tools = get_random_tools_WT(tools_WT, tool, tool_cls, 4)
        sampled_tools_text = "，".join(sampled_tools)
        question = f"你是一个能与工具交互的智能AI，你能使用以下工具：{sampled_tools_text}，你需要一步一步的思考并给出思考过程\n回答下面的问题：\n{question}"
        prompt = prompt_template + question
        response = LLM.get_response(prompt)
        # response = "test"
        # print("工具: ", sampled_tools)
        # print("response: ", response)
        # print("#####"*10)
        save_result.append(
            {
                "model_output": response,
                "label": item['tool'],
                "tools": sampled_tools_text
            }
        )

elif args.evaluation_strategy == "ReAct":
    prompt_template = prompt_template_ReAct
    for item in tqdm(data, desc="Processing"):
        tool, info, question, tool_cls = item["tool"], item["info"], item["question"], item["tool_cls"]
        if args.Sampling_strategy == "AT":
            sampled_tools = get_random_tools_AT(tools_AT, tool, 4)
        elif args.Sampling_strategy == "WT":
            sampled_tools = get_random_tools_WT(tools_WT, tool, tool_cls, 4)
        sampled_tools_text = "，".join(sampled_tools)
        ReAct_text = ""
        for sample_tool in sampled_tools:
            if sample_tool != tool:
                for train_data_item in train_data:
                    if sample_tool == train_data_item["tool"]:
                        ReAct_text += f"{train_data_item['info']}\n"
                        break
            else:
                ReAct_text += f"{info}\n"
        question = f"你是一个能与工具交互的智能AI，你能使用以下工具：{sampled_tools_text}，你需要先思考并了解各个工具的具体作用，然后再用Answer给出答案\n回答下面的问题：\n{question}\nThink 1：首先我需要知道各个工具的使用场景才能知道为了解决这个问题我最好使用哪些工具\nAct 1：搜索{sampled_tools_text}\nObservation：{ReAct_text}"
        prompt = prompt_template + question
        response = LLM.get_response(prompt)
        # response = "test"
        # print("工具: ", sampled_tools)
        # print("response: ", response)
        # print("#####"*10)
        save_result.append(
            {
                "model_output": response,
                "label": item['tool'],
                "tools": sampled_tools_text
            }
        )

elif args.evaluation_strategy == "Lora":
    prompt_template = prompt_template_Lora
    for item in tqdm(data, desc="Processing"):
        tool, info, question, tool_cls = item["tool"], item["info"], item["question"], item["tool_cls"]
        if args.Sampling_strategy == "AT":
            sampled_tools = get_random_tools_AT(tools_AT, tool, 4)
        elif args.Sampling_strategy == "WT":
            sampled_tools = get_random_tools_WT(tools_WT, tool, tool_cls, 4)
        sampled_tools_text = "，".join(sampled_tools)
        question = f"你能使用以下工具：{sampled_tools_text}。请回答下面的问题：\n{question}"
        prompt = prompt_template + question
        response = LLM.get_response(prompt)
        # response = "test"
        # print("工具: ", sampled_tools)
        # print("response: ", response)
        # print("#####"*10)
        save_result.append(
            {
                "model_output": response,
                "label": item['tool'],
                "tools": sampled_tools_text
            }
        )

elif args.evaluation_strategy == "MFT":
    prompt_template = prompt_template_Lora
    for item in tqdm(data, desc="Processing"):
        tool, info, question, tool_cls = item["tool"], item["info"], item["question"], item["tool_cls"]
        if args.Sampling_strategy == "AT":
            sampled_tools = get_random_tools_AT(tools_AT, tool, 4)
        elif args.Sampling_strategy == "WT":
            sampled_tools = get_random_tools_WT(tools_WT, tool, tool_cls, 4)
        sampled_tools_text = "，".join(sampled_tools)
        question = f"你能使用以下工具：{sampled_tools_text}。请回答下面的问题：\n{question}"
        prompt = prompt_template + question
        response = LLM.get_response(prompt)
        # response = "test"
        # print("工具: ", sampled_tools)
        # print("response: ", response)
        # print("#####"*10)
        save_result.append(
            {
                "model_output": response,
                "label": item['tool'],
                "tools": sampled_tools_text
            }
        )


with open(f"{args.save_path}/{args.model_name}-{args.evaluation_strategy}-{args.Sampling_strategy}.json", 'w', encoding='utf-8') as f:
    json.dump(save_result, f, ensure_ascii=False, indent=4)