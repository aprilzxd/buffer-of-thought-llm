import os
from lightrag import LightRAG, QueryParam
from lightrag.llm import openai_complete_if_cache,openai_embedding,hf_model_complete, hf_embedding
import numpy as np
from lightrag.utils import EmbeddingFunc
from transformers import AutoModel, AutoTokenizer


class MetaBuffer:
    def __init__(self,llm_model,embedding_model,api_key=None,base_url="https://api.openai.com/v1/",rag_dir='./test'):
        self.api_key = api_key
        self.llm = llm_model
        self.embedding_model = embedding_model
        self.base_url = base_url
        if not os.path.exists(rag_dir):
            os.mkdir(rag_dir)
        self.rag = LightRAG(
            working_dir= rag_dir,
            llm_model_func=hf_model_complete, #self.llm_model_func,  # Use Hugging Face model for text generation
            llm_model_name=self.llm, # '../hf_models/Llama-3.2-1B', #'../../models/Qwen2.5-Math-7B-Instruct', # Model name from Hugging Face
            embedding_func=EmbeddingFunc(
                embedding_dim=3072,
                max_token_size=8192,
                func=self.embedding_func
            )
            # embedding_func=EmbeddingFunc(
            #         embedding_dim=1024,
            #         max_token_size=8192,
            #         func=lambda texts: hf_embedding(
            #             texts,
            #             tokenizer=AutoTokenizer.from_pretrained("../hf_models/bge-m3"),
            #             embed_model=AutoModel.from_pretrained("../hf_models/bge-m3")
            #         )
            #     ),
        )

    async def llm_model_func(self, prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
        return await openai_complete_if_cache(
            self.llm,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=self.api_key,
            base_url=self.base_url,
            **kwargs
        )

    async def embedding_func(self, texts: list[str]) -> np.ndarray:
        return await openai_embedding(
            texts,
            model= self.embedding_model,
            api_key= self.api_key,
            base_url= self.base_url
        )

    def retrieve_and_instantiate(self,input):
        response, context = self.rag.query(input, param=QueryParam(mode="hybrid"))
        return response, context

    def dynamic_update(self,thought_template):
        prompt = "Find most relevant thought template in the MetaBuffer according to the given thought template, and Determine whether there is a fundamental difference in the problem-solving approach between this and the most similar thought template in MetaBuffer. If there is, output \"True.\" If there is no fundamental difference, or if the two thought templates are highly similar, output \"False.\""
        input = prompt + thought_template

        # Perform naive search
        response, _ = self.rag.query(input, param=QueryParam(mode="hybrid"))
        print(response)
        if self.extract_similarity_decision(response):
            print('MetaBuffer Updated!')
            self.rag.insert(thought_template)
            return thought_template
        else:
            print('No need to Update!')
            return 'None'

    def extract_similarity_decision(self,text):
        """
        This function takes the input text of an example and extracts the final decision
        on whether the templates are similar or not (True or False).
        """
        # Convert the text to lowercase for easier matching
        text = text.lower()

        # Look for the conclusion part where the decision is made
        if "true" in text:
            return True
        else:
            return False
