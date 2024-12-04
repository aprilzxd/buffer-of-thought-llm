python run_inferences.py \
--local_llm_model_id 'qwen2_5-7b' \
--local_api_key 'sk-URbCrxdzFA2eGJSsWjTeUndvTCZhQIAUWiMswgxrWTXv2jVu' \
--local_base_url 'http://10.10.100.15:9997/v1' \
--openai_api_key 'sk-URbCrxdzFA2eGJSsWjTeUndvTCZhQIAUWiMswgxrWTXv2jVu' \
--embedding_model 'text-embedding-3-large' \
--openai_base_url 'https://www.gpt-plus.top/v1' \
--rag_dir 'results/Qwen2.5-7B-Instruct_selective-distill=false' \
--distill_correct False  