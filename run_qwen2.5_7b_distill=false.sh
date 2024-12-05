#!/bin/bash

# 定义运行函数
run_with_retry() {
    local command="$1"
    local max_retries="${2:-5}"    # 默认最大重试5次
    local wait_time="${3:-10}"     # 默认等待10秒
    local retry_count=0

    while [ $retry_count -lt $max_retries ]; do
        echo "Starting attempt $((retry_count + 1)) of $max_retries"
        
        # 执行命令
        eval "$command"

        # 检查命令执行状态
        if [ $? -eq 0 ]; then
            echo "Script completed successfully"
            return 0
        else
            retry_count=$((retry_count + 1))
            if [ $retry_count -lt $max_retries ]; then
                echo "Script failed. Waiting $wait_time seconds before retry..."
                sleep $wait_time
            else
                echo "Maximum retry attempts ($max_retries) reached. Exiting."
                return 1
            fi
        fi
    done
}

# 构建命令字符串
COMMAND='python run_inferences.py \
--local_llm_model_id "qwen2_5-7b" \
--local_api_key "sk-URbCrxdzFA2eGJSsWjTeUndvTCZhQIAUWiMswgxrWTXv2jVu" \
--local_base_url "http://10.10.100.15:9997/v1" \
--openai_api_key "sk-URbCrxdzFA2eGJSsWjTeUndvTCZhQIAUWiMswgxrWTXv2jVu" \
--embedding_model "text-embedding-3-large" \
--openai_base_url "https://www.gpt-plus.top/v1" \
--rag_dir "results/Qwen2.5-7B-Instruct_selective-distill=false" \
--distill_correct False'

# 调用函数运行命令
run_with_retry "$COMMAND" 5 10  