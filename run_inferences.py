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
parser.add_argument('--openai_api_key',type=str,help='The api key of user')
parser.add_argument('--openai_base_url',type=str,default='https://api.openai.com/v1/',help='We also support Open AI-like chat/embeddings APIs')
parser.add_argument('--local_api_key',type=str,help='The api key of user')
parser.add_argument('--local_base_url',type=str,default='http://10.10.100.15:8100/v1',help='We also support Open AI-like chat/embeddings APIs')
parser.add_argument('--rag_dir',type=str,default='./math',help='The path to save the meta buffer')
parser.add_argument('--run_test',type=str,default=False,help='Whether this is a test run that doesn\'t update the meta buffer')
parser.add_argument('--distill_correct',type=str,default=False,help='Whether we only distill the template only when the generated solution is correct.')
parser.add_argument('--local_llm_model_id',type=str,default='llama-3.1-8b',help='The model id of local LLM')
parser.add_argument('--local_llm_model_path',type=str,default='../hf_models/Meta-Llama-3___1-8B-Instruct',help='The path to the local LLM model')

args = parser.parse_args()

llm_model = args.llm_model
local_llm_model_id = args.local_llm_model_id
embedding_model = args.embedding_model
openai_api_key = args.openai_api_key
openai_base_url = args.openai_base_url
local_api_key = args.local_api_key
local_base_url = args.local_base_url
rag_dir = args.rag_dir  
run_test = args.run_test
distill_correct = args.distill_correct

logger.info("Starting script with parameters:")
logger.info(f"LLM Model: {llm_model}")
logger.info(f"Local LLM Model ID: {local_llm_model_id}")
logger.info(f"Embedding Model: {embedding_model}")
logger.info(f"OpenAI Base URL: {openai_base_url}")
logger.info(f"Local Base URL: {local_base_url}")
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



def increase_file_limit():
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (8192, hard))
        logger.info(f"Increased file limit from {soft} to 4096")
    except Exception as e:
        logger.warning(f"Could not increase file limit: {e}")

increase_file_limit()

try:
    logger.info("Initializing BoT instance...")
    bot = BoT(
        user_input=None, 
        local_api_key=args.local_api_key,
        local_base_url=args.local_base_url,
        model_id=args.local_llm_model_id,
        embedding_model=args.embedding_model,
        openai_api_key=args.openai_api_key,
        openai_base_url=args.openai_base_url,
        rag_dir=args.rag_dir
    )
    logger.success("BoT initialization successful")
except Exception as e:
    logger.error(f"Failed to initialize BoT: {str(e)}")
    raise

# 修改文件处理部分
output_file = f'{output_dir}/GSM8K_{output_name}_{timestamp_str}.jsonl'
correct_count = 0
total_count = 0

# 计算总行数
logger.info("Counting total lines in input file...")
with open(data_dir, 'r') as f:
    total_lines = sum(1 for _ in f)
logger.info(f"Total lines to process: {total_lines}")


try:
    # 使用单个 with 语句同时管理所有文件操作
    with open(data_dir, 'r') as input_file, \
         open(output_file, 'a+', encoding='utf-8') as output_f, \
         tqdm(total=total_lines, desc="Processing examples") as pbar:
        
        for line in input_file:
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
                json_str = json.dumps(tmp)
                output_f.write(json_str + '\n')
                output_f.flush()  # 确保及时写入
                
                pbar.update(1)
                
            except Exception as e:
                logger.exception(f"Error processing example {total_count}: {str(e)}")
                continue

except Exception as e:
    logger.error(f"Fatal error in main process: {str(e)}")
    raise

finally:
    # 记录最终统计信息
    logger.info(f"Processing completed. Total examples: {total_count}")
    if distill_correct:
        accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
        logger.info(f"Correct answers: {correct_count}")
        logger.info(f"Accuracy: {accuracy:.2f}%")

# 评估
logger.info("Starting evaluation...")
result_path = output_file  # 使用之前定义的输出文件路径
eval_path = f'{output_dir}/GSM8K_{output_name}_{timestamp_str}_eval.txt'
try:
    validator(data_dir, result_path, eval_path)
    logger.success("Evaluation completed successfully")
except Exception as e:
    logger.error(f"Error during evaluation: {str(e)}")

logger.info("Script execution completed")