import json
from bot_pipeline import BoT
import argparse
import os
import datetime
from validation import check_answer, validator
from loguru import logger
from tqdm import tqdm
import sys

# 配置日志
logger.remove()  # 移除默认的处理器
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)
logger.add(
    "logs/run_inferences_{time}.log",
    rotation="500 MB",
    level="DEBUG"
)

parser = argparse.ArgumentParser()

parser.add_argument('--llm_model',type=str,default='gpt-4o-mini',help='Model id of LLMs')
parser.add_argument('--embedding_model',type=str,default='text-embedding-3-large',help='Model id of embedding model')
parser.add_argument('--api_key',type=str,help='The api key of user')
parser.add_argument('--base_url',type=str,default='https://api.openai.com/v1/',help='We also support Open AI-like chat/embeddings APIs')
parser.add_argument('--rag_dir',type=str,default='./math',help='The path to save the meta buffer')
parser.add_argument('--run_test',type=str,default=False,help='Whether this is a test run that doesn\'t update the meta buffer')
parser.add_argument('--distill_correct',type=str,default=False,help='Whether we only distill the template only when the generated solution is correct.')
args = parser.parse_args()

llm_model = args.llm_model
embedding_model = args.embedding_model
api_key = args.api_key
base_url = args.base_url
rag_dir = args.rag_dir
run_test = args.run_test
distill_correct = args.distill_correct

logger.info("Starting script with parameters:")
logger.info(f"LLM Model: {llm_model}")
logger.info(f"Embedding Model: {embedding_model}")
logger.info(f"Base URL: {base_url}")
logger.info(f"RAG Directory: {rag_dir}")
logger.info(f"Run Test: {run_test}")
logger.info(f"Distill Correct: {distill_correct}")

prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
data_dir = 'gsm8k/train.jsonl' if not run_test else 'gsm8k/test.jsonl'
output_name = 'train' if not run_test else 'test'
now = datetime.datetime.now()
timestamp_str = now.strftime("%Y-%m-%d-%H:%M:%S")
output_dir = 'inference_results'

logger.info(f"Using data directory: {data_dir}")
logger.info(f"Output name: {output_name}")
logger.info(f"Timestamp: {timestamp_str}")

if not os.path.exists(output_dir):
    logger.info(f"Creating output directory: {output_dir}")
    os.makedirs(output_dir)

try:
    logger.info("Initializing BoT instance...")
    bot = BoT(
        user_input=None, 
        api_key=api_key,
        model_id=llm_model,
        embedding_model=embedding_model,
        base_url=base_url,
        rag_dir=rag_dir
    )
    logger.success("BoT initialization successful")
except Exception as e:
    logger.error(f"Failed to initialize BoT: {str(e)}")
    raise

# 计算总行数用于进度条
total_lines = sum(1 for _ in open(data_dir))
logger.info(f"Total number of examples to process: {total_lines}")

correct_count = 0
total_count = 0

# 使用tqdm创建进度条
with tqdm(total=total_lines, desc="Processing examples") as pbar:
    for line in open(data_dir):
        total_count += 1
        try:
            input_data = json.loads(line)
            input_question = input_data['question']
            user_input = prompt + input_question
            
            logger.debug(f"Processing example {total_count}/{total_lines}")
            logger.debug(f"Input question: {input_question}")
            
            bot.update_input(user_input)
            result = bot.bot_test()
            
            # 动态更新策略
            if not run_test:
                if distill_correct:
                    if check_answer(input_data['ans'], result):
                        logger.info("Answer correct - updating meta buffer")
                        correct_count += 1
                        bot.bot_update()
                else:
                    logger.debug("Using naive update strategy")
                    bot.bot_update()
            
            # 保存结果
            tmp = {'input': input_question, 'result': result}
            output_file = f'{output_dir}/GSM8K_{output_name}_{timestamp_str}.jsonl'
            with open(output_file, 'a+', encoding='utf-8') as file:
                json_str = json.dumps(tmp)
                file.write(json_str + '\n')
            
            pbar.update(1)
            
        except Exception as e:
            logger.error(f"Error processing example {total_count}: {str(e)}")
            continue

# 记录最终统计信息
logger.info(f"Processing completed. Total examples: {total_count}")
if distill_correct:
    accuracy = (correct_count / total_count) * 100
    logger.info(f"Correct answers: {correct_count}")
    logger.info(f"Accuracy: {accuracy:.2f}%")

# 评估
logger.info("Starting evaluation...")
result_path = f'{output_dir}/GSM8K_{output_name}_{timestamp_str}.jsonl'
eval_path = f'{output_dir}/GSM8K_{output_name}_{timestamp_str}_eval.txt'
try:
    validator(data_dir, result_path, eval_path)
    logger.success("Evaluation completed successfully")
except Exception as e:
    logger.error(f"Error during evaluation: {str(e)}")

logger.info("Script execution completed")
