import transformers
import torch
from meta_buffer_utilis import meta_distiller_prompt,extract_and_execute_code
from test_templates import game24,checkmate,word_sorting
from meta_buffer import MetaBuffer
from openai import OpenAI

from loguru import logger
class Pipeline:
    def __init__(self, api=True, model_id=None, bot_api_key=None, bot_base_url=None):
        self.api = api
        self.model_id = model_id
        self.bot_api_key = bot_api_key
        self.bot_base_url = bot_base_url
        # 在初始化时创建 OpenAI 客户端
        if self.api:
            self.client = OpenAI(
                api_key=self.bot_api_key,
                base_url=self.bot_base_url
            )
        else:
            self.client = None

    def get_respond(self, meta_prompt, user_prompt):
        if self.api:
            logger.info(f"meta_prompt: {meta_prompt}")
            logger.info(f"user_prompt: {user_prompt}")
            completion = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": meta_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            response = completion.choices[0].message.content
            return response
        else:
            messages = [
                {"role": "system", "content": meta_prompt},
                {"role": "user", "content": user_prompt},
            ]

            prompt = self.pipeline.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )

            terminators = self.pipeline.tokenizer.eos_token_id

            outputs = self.pipeline(
                prompt,
                max_new_tokens=2048,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.4,
                top_p=0.9,
            )
            respond = outputs[0]["generated_text"][len(prompt):]
            return respond


class BoT:
    def __init__(self, user_input,problem_id=0,local_api_key=None,local_base_url="http:/10.10.100.15:8100/v1",model_id='gpt-4o-mini',embedding_model='text-embedding-3-large',need_check=False,openai_api_key=None,openai_base_url='https://api.openai.com/v1/',rag_dir='./test'):
        self.bot_api_key = local_api_key
        self.model_id = model_id
        self.embedding_model = embedding_model
        self.bot_base_url = local_base_url
        self.pipeline = Pipeline(self.model_id,api_key=self.bot_api_key,base_url=self.bot_base_url,local_api=True)

        self.meta_buffer = MetaBuffer(self.model_id,self.embedding_model,api_key=local_api_key,base_url=local_base_url,rag_dir=rag_dir)

        self.user_input = user_input
        # Only for test use, stay tuned for our update
        self.problem_id = problem_id 
        self.need_check = need_check
        with open("./default_templates.txt") as f:
            self.meta_buffer.rag.insert(f.read())

    def update_input(self,new_input):
        self.user_input = new_input

    def problem_distillation(self):
        logger.info(f'User prompt:{self.user_input}')
        self.distilled_information = self.pipeline.get_respond(meta_distiller_prompt, self.user_input)
        logger.info(f'Distilled information:{self.distilled_information}')

    def buffer_retrieve(self):
        # For initial test use, we will later update the embedding retrieval version to support more, trail version
        if self.problem_id == 0:
            self.thought_template = game24
        elif self.problem_id == 1:
            self.thought_template = checkmate
        elif self.problem_id == 2:
            self.thought_template = word_sorting

    def buffer_instantiation(self):
        self.buffer_prompt = "You are an expert in problem analysis and can apply previous problem-solving approaches to new issues. The user will provide a specific task description and a meta buffer that holds multiple thought templates that will help to solve the problem. Your goal is to first extract most relevant thought template from meta buffer, analyze the user's task and generate a specific solution based on the thought template. Give a final answer that is easy to extract from the text."
        input = self.buffer_prompt + self.distilled_information
        self.result, self.context = self.meta_buffer.retrieve_and_instantiate(input)
        logger.info(self.result)

    def buffer_manager(self):
        self.problem_solution_pair = self.user_input + self.result
        logger.info(f"start thought distillation")
        self.thought_distillation()
        logger.info(f"end thought distillation")
        logger.info(f"start dynamic update")
        new_template = self.meta_buffer.dynamic_update(self.distilled_thought)
        logger.info(f"end dynamic update")
        return new_template
        

    def thought_distillation(self):
        thought_distillation_prompt = """You are an expert in problem analysis and generalization. Your task is to distill high-level thought templates that could be used to solve the provided problem and solution pairs. An example thought template for a solution concentration problem is provided below. It should be noted that you should only return the thought template without any extra output.
Example thought template:
### Problem Type 20: Solution Concentration Problem

**Definition**: This type of problem involves the relationship between a solvent (water or another liquid), solute, solution, and concentration.

**Quantitative Relationships**:
- Solution = Solvent + Solute
- Concentration = Solute ÷ Solution × 100%

**Solution Strategy**: Use the formulas and their variations to analyze and calculate the problem.

**Example**: There is 50 grams of a 16% sugar solution. How much water needs to be added to dilute it to a 10% sugar solution?

**Solution**:
Using the formula:
50 × 16% ÷ 10% - 50 = 30 grams of water need to be added."""
        self.distilled_thought = self.pipeline.get_respond(thought_distillation_prompt, self.problem_solution_pair)
        logger.info(f'Distilled thought: {self.distilled_thought}')

    def reasoner_instantiation(self):
        # Temporay using selection method to select answer extract method
        problem_id_list = [0,1,2]
        self.instantiation_instruct = """You are an expert in problem analysis and can apply previous problem-solving approaches to new issues. The user will provide a specific task description and a thought template. Your goal is to analyze the user's task and generate a specific solution based on the thought template. If the instantiated solution involves Python code, only provide the code and let the compiler handle it. If the solution does not involve code, provide a final answer that is easy to extract from the text.
It should be noted that all the python code should be within one code block, the answer should not include more than one code block! And strictly follow the thought-template to instantiate the python code but you should also adjust the input parameter according to the user input!"""

        self.formated_input = f"""Distilled information:
{self.distilled_information}
User Input:
{self.user_input}
Thought template:
{self.thought_template}

Instantiated Solution:
Please analyze the above user task description and thought template, and generate a specific, detailed solution. If the solution involves Python code, only provide the code. If not, provide a clear and extractable final answer."""

        self.inspector_prompt = """You are an excellent python programming master who are proficient in analyzing and editing python code, and you are also good at understanding the real-world problem. Your task is:
1. Analyze the given python code
2. Edit the input code to make sure the edited code is correct and could run and solve the problem correctly.  
Your respond should follow the format below:
```python
## Edited code here
```"""

        self.result = self.pipeline.get_respond(self.instantiation_instruct,self.formated_input)
        logger.info(f'Instantiated reasoning result: {self.result}')
        if self.problem_id in problem_id_list:
            self.final_result, code_str = extract_and_execute_code(self.result)
            if self.need_check:
                self.count = 0
                self.inter_input = f"User_input: {self.user_input}\n{code_str}\n{self.final_result}"
                self.inter_result = self.final_result
                while(('An error occurred' in self.inter_result) or (self.inter_result == '') or (self.inter_result == 'None')):
                    logger.info('The code cannot be executed correctly, here we continue the edit phase:',self.inter_result)
                    logger.info('The problem code is:',code_str)
                    self.inter_input = self.pipeline.get_respond(self.inspector_prompt,self.inter_input)
                    logger.info(self.inter_input)
                    self.inter_result, inter_code_str = extract_and_execute_code(self.inter_input)
                    self.inter_input = f"User_input: {self.user_input}\n{inter_code_str}\nThe result of code execution: {self.inter_result}"
                    self.count = self.count + 1
                    if self.count > 3:
                        break
                self.final_result = self.inter_result 
            logger.info(f'The result of code execution: {self.final_result}')
        else:
            self.final_result = self.result 

    def bot_run(self):
        self.problem_distillation()
        self.buffer_retrieve()
        self.reasoner_instantiation()
        return self.final_result

    def bot_inference(self):
        self.problem_distillation()
        self.buffer_instantiation()
        self.buffer_manager()
        logger.info(f'bot inference Final results: {self.result}')
        return self.result
    
    def bot_test(self):
        self.problem_distillation()
        self.buffer_instantiation()
        return self.result, self.context

    def bot_update(self):
        return self.buffer_manager()
