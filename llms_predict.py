import pandas as pd
import os
# from OpenAIAPI import Summarizer
from llmsapi import llmsAPI
import logging
import time
from icecream import ic
from config import *
from tqdm import tqdm

# Configure the logger
logging.basicConfig(filename='summarization.log', filemode='a', format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

class DATA:
    def __init__(self):
        self.backstories_file = 'uqv100-backstories.tsv'

    def load_data(self):
        columns = ['UQV100Id', 'Backstory']
        self.df_data = pd.read_csv(UQV100_DATA_PATH + '/' + self.backstories_file,
                                sep='\t', 
                                usecols=columns,
                                index_col='UQV100Id')
        
        # ##DEBUG:
        # self.df_data = self.df_data.head(10)


class PROMPT:
    def __init__(self):
        pass
        
    def ZeroShot(self, backstory):
        context = '''BACKSTORY: ''' + backstory
        task = '''TASK: Predict the number of necessary documents to attend the user needs for the BACKSTORY. '''
        constraint_1 = '''1) You/the system must output a single character.'''
        constraint_2 = '''2) The character must be an integer to solve the task.'''
        constraint_3 = '''3) You/the system must not output any text/characters apart from the number.'''
        output_constraint = '''CONSTRAINTS: ''' + '\n' + constraint_1 + '\n' + constraint_2 + '\n' + constraint_3
        return context + '\n' + task + '\n' + output_constraint
    
    def get_prompt(self, prompt_type, backstory):
        if prompt_type == "ZeroShot":
            return self.ZeroShot(backstory)
        else:
            raise ValueError(f"The prompt type {prompt_type} is not available.")


class ChatLLM(DATA, PROMPT):
    def __init__(self, model, prompt_type):
        super().__init__()
        self.prompt_type = prompt_type
        self.model = model

    def get_predictions(self):
        predictions = []
        
        for i in tqdm(range(len(self.df_data)), desc="Predicting Effort Estimation", leave=False, position=1, ncols=100):  # Add tqdm to the loop
            backstory = self.df_data.iloc[i]['Backstory']
            prompt = super().get_prompt(self.prompt_type, backstory)
            llms_api = llmsAPI(prompt)
            output = llms_api.get_completion(llm=self.model)
            predictions.append(output)
            # time.sleep(1)

        self.df_data[self.model] = predictions
        
        
    def save_predictions(self):
        self.df_data.loc[:,[self.model]].to_csv(LLMS_PREDICTIONS + '/chatgpt_' + self.model + '_' + self.prompt_type + '.tsv', sep='\t')
        
    
    def main(self):
        super().load_data()
        self.get_predictions()
        self.save_predictions()

if __name__ == '__main__':
    for llm in tqdm(["gpt-3.5-turbo-0125", "gpt-4-turbo", "gpt-4o-2024-05-13"], desc="LLMs", position=0,ncols=100):
        LLmChat = ChatLLM(model=llm, prompt_type="ZeroShot")
        LLmChat.main()