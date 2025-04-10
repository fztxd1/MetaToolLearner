python run.py \
    --model_name ChatGLM3 \
    --model_path /root/autodl-tmp/model/chatglm3-6b \
    --evaluation_strategy Base \
    --Sampling_strategy AT \
    --test_data_file /root/autodl-fs/work/Tool-augmented/dataset/test.json \
    --save_path /root/autodl-fs/work/Tool-augmented/experiment/test01 \
    --test_num test01

python run.py \
    --model_name ChatGLM3 \
    --model_path /root/autodl-tmp/model/chatglm3-6b \
    --evaluation_strategy CoT \
    --Sampling_strategy AT \
    --test_data_file /root/autodl-fs/work/Tool-augmented/dataset/test.json \
    --save_path /root/autodl-fs/work/Tool-augmented/experiment/test01 \
    --test_num test01

python run.py \
    --model_name ChatGLM3 \
    --model_path /root/autodl-tmp/model/chatglm3-6b \
    --evaluation_strategy ReAct \
    --Sampling_strategy AT \
    --test_data_file /root/autodl-fs/work/Tool-augmented/dataset/test.json \
    --save_path /root/autodl-fs/work/Tool-augmented/experiment/test01 \
    --test_num test01

python run.py \
    --model_name ChatGLM3 \
    --model_path /root/autodl-tmp/model/chatglm3-6b \
    --evaluation_strategy Base \
    --Sampling_strategy WT \
    --test_data_file /root/autodl-fs/work/Tool-augmented/dataset/test.json \
    --save_path /root/autodl-fs/work/Tool-augmented/experiment/test01 \
    --test_num test01

python run.py \
    --model_name ChatGLM3 \
    --model_path /root/autodl-tmp/model/chatglm3-6b \
    --evaluation_strategy CoT \
    --Sampling_strategy WT \
    --test_data_file /root/autodl-fs/work/Tool-augmented/dataset/test.json \
    --save_path /root/autodl-fs/work/Tool-augmented/experiment/test01 \
    --test_num test01

python run.py \
    --model_name ChatGLM3 \
    --model_path /root/autodl-tmp/model/chatglm3-6b \
    --evaluation_strategy ReAct \
    --Sampling_strategy WT \
    --test_data_file /root/autodl-fs/work/Tool-augmented/dataset/test.json \
    --save_path /root/autodl-fs/work/Tool-augmented/experiment/test01 \
    --test_num test01