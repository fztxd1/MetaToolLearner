import json
import re
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--evaluation_data_path', default=None, type=str, required = True)
parser.add_argument('--save_path', default=None, type=str, required = True)
args = parser.parse_args()

# with open(args.save_path, 'r', encoding='utf-8') as f:
#     result = json.load(f)

result = {}

for root, dirs, files in os.walk(args.evaluation_data_path):
    for file_name in files:
        if file_name.endswith('.json'):
            file_path = os.path.join(root, file_name)
            filename = os.path.basename(file_path)
            filename = os.path.splitext(filename)[0]
            match = re.match(r"([^-]+)-([^-]+)-([^-]+)", filename)
            # model_name, evaluation_strategy, Sampling_strategy = "", "", ""
            if match:
                model_name = match.group(1)
                evaluation_strategy = match.group(2)
                Sampling_strategy = match.group(3)
                print(f"model_name: {model_name}, evaluation_strategy: {evaluation_strategy}, Sampling_strategy: {Sampling_strategy}")
            else:
                print("未找到匹配的元素")

            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            accuracy_count = 0
            total_count = len(data)

            for item in data:
                label = item['label']
                # match = re.search(r'\[(.*?)\]', item['model_output'])
                # match = re.search(r"使用\[(.*?)\]工具", item['model_output'])
                matches = re.findall(r"使用\[(.*?)\]工具", str(item['model_output']))
                if matches:
                    model_output = matches[-1]
                else:
                    model_output = None
                # print("model_output: ", model_output)
                # print("label: ", item['label'])
                # print("#####"*10)
                if model_output == item['label']:
                    accuracy_count += 1


            print("accuracy_count: ", accuracy_count)
            print("total_count: ", total_count)
            print("precision", accuracy_count / total_count)
            print("precision", round((accuracy_count / total_count) * 100, 2))

            print()

            key = f"{model_name}-{Sampling_strategy}"
            if key not in result:
                result[key] =  [{
                        'evaluation_strategy': evaluation_strategy,
                        'accuracy_count': accuracy_count,
                        'total_count': total_count, 
                        'precision': round((accuracy_count / total_count) * 100, 2)
                    }]
            else: 
                result[key].append({
                        'evaluation_strategy': evaluation_strategy,
                        'accuracy_count': accuracy_count,
                        'total_count': total_count, 
                        'precision': round((accuracy_count / total_count) * 100, 2)
                    }
                )
# print(result)
with open(args.save_path, 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=4)




            # elif args.evaluation_strategy == 'CoT':
            #     # 执行 CoT 评估策略的代码
            #     print("CoT evaluation strategy selected.")
            #     for item in data:
            #         label = item['label']
            #         answer = re.search(r'^Answer.*', str(item['model_output']), re.MULTILINE)
            #         if answer:
            #             answer = answer.group(0)
            #             match = re.search(r"Answer：使用\[(.*?)\]工具", answer)
            #         else:
            #             answer = None
            #             match = None
            #         if match:
            #             model_output = match.group(1)
            #         else:
            #             model_output = None        
            #         print("model_output: ", model_output)
            #         print("label: ", label)
            #         print("#####"*10)
            #         if model_output and model_output.strip() == label.strip():
            #             accuracy_count += 1

            # elif args.evaluation_strategy == 'ReAct':
            #     # 执行 ReAct 评估策略的代码
            #     print("ReAct evaluation strategy selected.")
            # #     for item in data:
            # #         label = item['label']

            # #         lines = str(item['model_output']).split('\n')
            # #         if lines:
            # #             end_line = lines[-1]
            # #         else:
            # #             end_line = ""

            # #         match = re.search(r"Answer：使用\[(.*?)\]工具", end_line)
            # #         if match:
            # #             model_output = match.group(1)
            # #         else:
            # #             model_output = None
            # #         print("model_output: ", model_output)
            # #         print("label: ", item['label'])
            # #         print("#####"*10)
            # #         if model_output == item['label']:
            # #             accuracy_count += 1
            #     for item in data:
            #         label = item['label']
            #         answer = re.search(r'^Answer.*', str(item['model_output']), re.MULTILINE)
            #         if answer:
            #             answer = answer.group(0)
            #             match = re.search(r"Answer：使用\[(.*?)\]工具", answer)
            #         else:
            #             answer = None
            #             match = None
            #         if match:
            #             model_output = match.group(1)
            #         else:
            #             model_output = None       
            #         print("model_output: ", model_output)
            #         print("label: ", label)
            #         print("#####"*10)
            #         if model_output and model_output.strip() == label.strip():
            #             accuracy_count += 1

            # elif args.evaluation_strategy == 'Lora':
            #     # 执行基本评估策略的代码
            #     print("Base evaluation strategy selected.")
            #     for item in data:
            #         label = item['label']
            #         # match = re.search(r'\[(.*?)\]', item['model_output'])
            #         match = re.search(r"使用\[(.*?)\]工具", item['model_output'])
            #         if match:
            #             model_output = match.group(1)
            #         else:
            #             model_output = None
            #         print("model_output: ", model_output)
            #         print("label: ", item['label'])
            #         print("#####"*10)
            #         if model_output == item['label']:
            #             accuracy_count += 1