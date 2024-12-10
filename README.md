# SI-BoT: Self-Improved Buffer of Thoughts for LLM reasoning

### 1. Set up the environment


```bash
git clone https://github.com/aprilzxd/buffer-of-thought-llm.git
cd buffer-of-thought-llm
conda create -n BoT python==3.9 
pip install -r requirements2.txt
```
Note: if there are still uninstalled packages during running, please read the first several lines of requirements.txt and follow the instruction.

### 2. Download an LLM into a directory
For example:
```bash
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir ../hf_models/Qwen2.5-7B-Instruct
```
### 3. Change the number of problems you want to run

In run_inferences.py, change the 30 in "if count > 30" to the number of problems you want to run. Or comment out "if count > 30" and the following "break".

### 4. Run training 

```bash
python run_inferences.py --llm_model '../hf_models/Qwen2.5-7B-Instruct' --api_key 'your OpenAI API key' --rag_dir 'Qwen2.5-7B-Instruct_smart_update' --distill_correct
```

Here, **--llm_model** should be the directory of local models.

The **--api_key** is your OpenAI API key. We use OpenAI API as the embedding model.

The **--rag_dir** should be the directory for LightRAG output files.

The **--distill_correct** is an argument to control whether we use smart update. Including this argument means using smart update, while removing it means using naive update.

### 5. Run test

```bash
python run_inferences.py --llm_model '../hf_models/Qwen2.5-7B-Instruct' --api_key 'your OpenAI API key' --rag_dir 'Qwen2.5-7B-Instruct_smart_update' --run_test --distill_correct
```
Including **--run_test** means we run BoT on the test set of GSM8K. Please pay attention that here we should use the same rag_dir as in the training, so that the templates accumulated in the training stage can be utilized in the test stage.
## Acknowledgements

The implementation of our Meta Buffer and Buffer Manager is based on Original BoT repository. (https://github.com/YangLing0818/buffer-of-thought-llm). 

