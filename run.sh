# run.sh
export CUDA_VISIBLE_DEVICES=1,2,3
python run_inferences.py \
--llm_model '/home/ubuntu/workspace/model/LLama/Meta-Llama-3___1-8B-Instruct' \
--local_llm_model_id 'llama3.1-8b' \
--openai_api_key 'sk-URbCrxdzFA2eGJSsWjTeUndvTCZhQIAUWiMswgxrWTXv2jVu' \
--embedding_model 'text-embedding-3-large' \
--openai_base_url 'https://www.gpt-plus.top/v1' \
--local_api_key 'sk-URbCrxdzFA2eGJSsWjTeUndvTCZhQIAUWiMswgxrWTXv2jVu' \
--local_base_url 'http://10.10.100.15:8100/v1' \
--local_llm_model_path '/home/ubuntu/workspace/model/LLama/Meta-Llama-3___1-8B-Instruct' \
--rag_dir 'Llama-3.1-8B-Instruct_selective' \
--distill_correct True  