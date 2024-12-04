import json
import os
import datetime
import sys
from bot_pipeline import BoT
from validation import check_answer, validator
from loguru import logger
from tqdm import tqdm
import argparse
import resource
import gc
import asyncio
import signal


parser = argparse.ArgumentParser()
parser.add_argument('--embedding_model',type=str,default='text-embedding-3-large',help='Model id of embedding model')
parser.add_argument('--openai_api_key',type=str,help='The api key of user')
parser.add_argument('--openai_base_url',type=str,default='https://api.openai.com/v1/',help='We also support Open AI-like chat/embeddings APIs')
parser.add_argument('--local_api_key',type=str,help='The api key of user')
parser.add_argument('--local_base_url',type=str,default='http://10.10.100.15:8100/v1',help='We also support Open AI-like chat/embeddings APIs')
parser.add_argument('--rag_dir',type=str,default='./math',help='The path to save the meta buffer')
parser.add_argument('--run_test',action='store_true',default=False,help='Whether this is a test run that doesn\'t update the meta buffer')
parser.add_argument('--distill_correct',action='store_true',default=False,help='Whether we only distill the template only when the generated solution is correct.')
parser.add_argument('--local_llm_model_id',type=str,default='llama-3.1-8b',help='The model id of local LLM')

args = parser.parse_args()

local_llm_model_id = args.local_llm_model_id
embedding_model = args.embedding_model
openai_api_key = args.openai_api_key
openai_base_url = args.openai_base_url
local_api_key = args.local_api_key
local_base_url = args.local_base_url
rag_dir = args.rag_dir  
run_test = args.run_test
distill_correct = args.distill_correct

# 配置日志
logger.remove()  # 移除默认的处理器
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)
logger.add(
    f"logs/run_inferences_{rag_dir}.log",
    rotation="500 MB",
    level="DEBUG"
)
logger.info("Starting script with parameters:")
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
output_dir = 'results/inference_results'

# 根据 RAG 目录生成状态文件路径
state_file = f'{output_dir}/state_{os.path.basename(rag_dir)}_{output_name}.json'

logger.info(f"Using data directory: {data_dir}")
logger.info(f"Output name: {output_name}")
logger.info(f"Timestamp: {timestamp_str}")
logger.info(f"State file: {state_file}")

if not os.path.exists(output_dir):
    logger.info(f"Creating output directory: {output_dir}")
    os.makedirs(output_dir)

def increase_file_limit():
    try:
        # 设置更高的限制
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        new_soft = min(hard, 65535)  # 设置一个更高的值
        resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
        logger.info(f"Increased file limit from {soft} to {new_soft}")
    except Exception as e:
        logger.warning(f"Could not increase file limit: {e}")
        # 尝试使用 ulimit 命令（仅在类Unix系统上有效）
        try:
            import subprocess
            subprocess.run(['ulimit', '-n', '65535'], shell=True)
            logger.info("Attempted to increase file limit using ulimit command")
        except Exception as e:
            logger.warning(f"Could not use ulimit command: {e}")

def cleanup_resources():
    """清理资源和文件描述符"""
    gc.collect()  # 触发垃圾回收
    
    # 清理 asyncio 事件循环
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.stop()
        if not loop.is_closed():
            loop.close()
    except Exception as e:
        logger.exception(f"Error cleaning up event loop: {e}")
    
    # 清理 HTTP 客户端
    try:
        import httpx
        httpx.Client().close()
    except Exception as e:
        logger.exception(f"Error cleaning up HTTP client: {e}")

def signal_handler(signum, frame):
    """处理程序终止信号"""
    logger.info("Received termination signal. Cleaning up...")
    cleanup_resources()
    sys.exit(0)

# 注册信号处理器
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

try:
    # 确保在开始时有一个新的事件循环
    if asyncio.get_event_loop().is_closed():
        asyncio.set_event_loop(asyncio.new_event_loop())
    
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
output_file = f'{output_dir}/GSM8K_{local_llm_model_id}_{output_name}_distill_{distill_correct}.jsonl'
correct_count = 0
total_count = 0
template_count = 0

# 计算总行数
logger.info("Counting total lines in input file...")
with open(data_dir, 'r') as f:
    total_lines = sum(1 for _ in f)
logger.info(f"Total lines to process: {total_lines}")

# 检查状态文件以确定从哪里开始
start_line = 0
if os.path.exists(state_file):
    with open(state_file, 'r') as sf:
        state_data = json.load(sf)
        start_line = state_data.get('last_processed_line', 0)
        correct_count = state_data.get('correct_count', 0)
        logger.info(f"Resuming from line {start_line}")

try:
    # 使用单个 with 语句同时管理所有文件操作
    with open(data_dir, 'r') as input_file, \
         open(output_file, 'a+', encoding='utf-8') as output_f:
        
        # 创建进度条，使用字典而不是列表来存储postfix
        pbar = tqdm(
            total=total_lines,
            initial=start_line,
            desc="Processing examples",
            ncols=100,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
        )
        
        # 从状态文件加载之前的进度
        previous_correct = correct_count  # 保存之前的正确数量
        previous_total = start_line      # 保存之前处理的总数
        
        # 初始化postfix为字典，使用之前的数据计算初始准确率
        initial_accuracy = (previous_correct / previous_total * 100) if previous_total > 0 else 0.0
        pbar.set_postfix(
            accuracy=f"{initial_accuracy:.2f}%",
            templates=f"0"
        )
        
        # 重置计数器，但保持之前的累计
        total_count = previous_total
        correct_count = previous_correct
        
        for line_number, line in enumerate(input_file, start=1):  # 从1开始计数
            if line_number <= start_line:  # 使用 <= 而不是 <
                continue  # 跳过已处理的行
            
            # 每处理100个样本进行一次清理
            if line_number % 100 == 0:
                cleanup_resources()
                # 重新创建事件循环
                if asyncio.get_event_loop().is_closed():
                    asyncio.set_event_loop(asyncio.new_event_loop())
                logger.debug(f"Performed resource cleanup at line {line_number}")
            
            total_count += 1
            try:
                # 添加输入数据验证
                if not line.strip():
                    logger.warning(f"Empty line found at line {total_count}, skipping...")
                    continue
                
                try:
                    input_data = json.loads(line)
                except json.JSONDecodeError as je:
                    logger.error(f"JSON decode error at line {total_count}: {str(je)}")
                    logger.debug(f"Problematic line content: {line[:100]}...")
                    continue
                
                # 验证输入数据格式
                if 'question' not in input_data:
                    logger.error(f"Missing 'question' field at line {total_count}")
                    continue
                
                input_question = input_data['question']
                if not input_question:
                    logger.warning(f"Empty question found at line {total_count}, skipping...")
                    continue
                
                user_input = prompt + " " + input_question  # 添加空格分隔
                
                logger.debug(f"Processing example {total_count}/{total_lines}")
                logger.debug(f"Input question: {input_question[:100]}...")  # 限制日志长度
                
                bot.update_input(user_input)
                try:
                    result = bot.bot_test()
                except Exception as e:
                    logger.error(f"Error in bot_test at line {line_number}: {str(e)}")
                    cleanup_resources()  # 发生错误时立即清理
                    result = bot.bot_test()
                
                if result is None:
                    logger.warning(f"Received None result for example {total_count}")
                    continue
                
                # 动态更新策略
                if not run_test:
                    if distill_correct:
                        if 'ans' not in input_data:
                            logger.error(f"Missing 'ans' field at line {total_count}")
                            continue
                            
                        if check_answer(input_data['ans'], result):
                            logger.info(f"Answer correct for example {total_count}")
                            correct_count += 1
                            bot.bot_update()
                    else:
                        logger.info("Using naive update strategy")
                        bot.bot_update()
                
                # 更新模板数量（添加错误处理）
                try:
                    template_file = f"{rag_dir}/kv_store_full_docs.json"
                    template_count = 0
                    if os.path.exists(template_file):
                        with open(template_file, 'r') as f:
                            templates = json.load(f)
                            template_count = len(templates)
                except Exception as te:
                    logger.error(f"Error reading template file: {str(te)}")
                    template_count = 0
                
                # 计算当前正确率（包含之前的数据）
                current_accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0.0
                
                # 保存结果到输出文件
                tmp = {
                    'input': input_question,
                    'result': result,
                    'correct': correct_count,
                    'total': total_count,
                    'accuracy': current_accuracy,
                    'templates': template_count,
                    'line_number': line_number  # 添加行号信息
                }
                
                # 写入结果
                json_str = json.dumps(tmp)
                output_f.write(json_str + '\n')
                output_f.flush()  # 确保立即写入
                
                # 更新进度条显示
                pbar.set_postfix(
                    accuracy=f"{current_accuracy:.2f}%",
                    templates=f"{template_count}"
                )
                
                pbar.update(1)
                
                # 更新状态文件
                with open(state_file, 'w') as sf:
                    json.dump({
                        'last_processed_line': line_number,
                        'correct_count': correct_count,
                        'total_count': total_count  # 添加总数信息
                    }, sf)
                
            except Exception as e:
                logger.exception(f"Error processing example {line_number}")
                cleanup_resources()  # 发生错误时立即清理
                continue
            
            # 定期刷新文件缓冲区
            if line_number % 10 == 0:
                output_f.flush()
                
        pbar.close()  # 确保进度条正确关闭

except Exception as e:
    logger.error(f"Fatal error in main process: {str(e)}")
    logger.exception("Full traceback:")
    cleanup_resources()  # 发生致命错误时清理
    raise

finally:
    # 确保在程序结束时清理所有资源
    cleanup_resources()

# 记录最终统计信息
logger.info(f"Final Statistics:")
logger.info(f"- Total Examples Processed: {total_count}")
logger.info(f"- Total Correct Answers: {correct_count}")
if total_count > 0:
    final_accuracy = (correct_count / total_count) * 100
    logger.info(f"- Final Accuracy: {final_accuracy:.2f}%")
logger.info(f"- Final Template Count: {template_count}")

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