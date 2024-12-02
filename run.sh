# run.sh
export CUDA_VISIBLE_DEVICES=1,2,3
python run_inferences.py \
--llm_model '/home/ubuntu/workspace/model/LLama/Meta-Llama-3___1-8B-Instruct' \
--api_key 'sk-URbCrxdzFA2eGJSsWjTeUndvTCZhQIAUWiMswgxrWTXv2jVu' \
--embedding_model 'text-embedding-3-large' \
--base_url 'https://www.gpt-plus.top/v1' \
--rag_dir 'Llama-3.1-8B-Instruct_selective' \
--distill_correct True  
