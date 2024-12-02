import json
from bot_pipeline import BoT
import argparse
import os
import datetime

parser = argparse.ArgumentParser()

parser.add_argument('--llm_model',type=str,default='gpt-4o-mini',help='Model id of LLMs')
parser.add_argument('--embedding_model',type=str,default='text-embedding-3-large',help='Model id of embedding model')
parser.add_argument('--api_key',type=str,help='The api key of user')
parser.add_argument('--base_url',type=str,default='https://api.openai.com/v1/',help='We also support Open AI-like chat/embeddings APIs')
parser.add_argument('--rag_dir',type=str,default='./math',help='The path to save the meta buffer')
parser.add_argument('--run_test',type=str,default=False,help='Whether this is a test run that doesn\'t update the meta buffer')

args = parser.parse_args()

llm_model = args.llm_model
embedding_model = args.embedding_model
api_key = args.api_key
base_url = args.base_url
rag_dir = args.rag_dir
run_test = args.run_test

prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
data_dir = 'gsm8k/train.jsonl' if not run_test else 'gsm8k/test.jsonl'
output_name = 'train' if not run_test else 'test'
now = datetime.datetime.now()
timestamp_str = now.strftime("%Y-%m-%d-%H:%M:%S")
output_dir = 'inference_results'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

bot = BoT(
    user_input= None, 
    api_key = api_key,
    model_id = llm_model,
    embedding_model = embedding_model,
    base_url = base_url,
    rag_dir = rag_dir
)

for line in (open(data_dir)):
    input = json.loads(line)['question']
    user_input = prompt + input
    bot.update_input(user_input)
    result = bot.bot_inference() if not run_test else bot.bot_test()
    tmp = {'input':input,'result':result}
    with open(f'{output_dir}/GSM8K_{output_name}_{timestamp_str}.jsonl', 'a+', encoding='utf-8') as file:
        json_str = json.dumps(tmp)
        file.write(json_str + '\n')


# Evaluation
result_path = f'{output_dir}/GSM8K_{output_name}_{timestamp_str}.jsonl'

correct = 0
total = 0

with open(f'{output_dir}/GSM8K_{output_name}_{timestamp_str}_eval.txt', 'w') as output_file:
    for line_y, line_pred in zip(open(data_dir), open(result_path)):
        result = json.loads(line_pred)['result']
        true_y = json.loads(line_y)['ans']
        pred = result.split('\n')[-1]
        eval_result = '0'
        if str(true_y) in pred or str(int(true_y)) in pred:
            eval_result = '1'
            correct += 1
        total += 1
        output_file.write(f"{eval_result} | {int(true_y)} | {pred}\n")


    print(f'Total number:{total},Correct number:{correct},Accuracy:{correct/total}')
    output_file.write(f'Total number:{total},Correct number:{correct},Accuracy:{correct/total}')

