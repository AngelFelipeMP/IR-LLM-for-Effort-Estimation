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
        self.crowdsource_annotations_file = 'uqv100-query-variations-and-estimates.tsv'

    def load_data_backstoris(self):
        columns = ['UQV100Id', 'Backstory']
        self.df_data = pd.read_csv(UQV100_DATA_PATH + '/' + self.backstories_file,
                                sep='\t', 
                                usecols=columns,
                                index_col='UQV100Id')
        
    def load_data_crowdsource_annotation(self):
        columns = ['UQV100Id', 'DocCount']
        self.df_annotations = pd.read_csv(UQV100_DATA_PATH + '/' + self.crowdsource_annotations_file,
                        sep='\t', 
                        usecols=columns,
                        index_col='UQV100Id')
        
    def count_crowdsource_annotation(self):
        self.df_annotations['DocCountList'] = self.df_annotations.groupby('UQV100Id')['DocCount'].apply(list)
        self.df_annotations['NumAnotations'] = self.df_annotations.apply(lambda row: len(row['DocCountList']), axis=1)
        self.df_annotations = self.df_annotations.reset_index().drop_duplicates(subset='UQV100Id', keep='first').set_index('UQV100Id')
        
    def add_num_annotations_to_backstories(self):
        self.df_data = self.df_data.join(self.df_annotations['NumAnotations'])
        
        # ic(self.df_data)
        # exit()
        
        ##DEBUG:
        self.df_data = self.df_data.head(10)

class PROMPT:
    def __init__(self):
        pass
        
    def ZeroShot(self, backstory):
        context = '''BACKSTORY: ''' + backstory
        task = '''TASK: Predict the number of necessary documents to satisfy the user need described by BACKSTORY.'''
        constraint_1 = '''1) You/the system must output a single character.'''
        constraint_2 = '''2) The character must be an integer to solve the task.'''
        constraint_3 = '''3) You/the system must not output any text/characters apart from the number.'''
        output_constraint = '''CONSTRAINTS: ''' + '\n' + constraint_1 + '\n' + constraint_2 + '\n' + constraint_3
        return context + '\n' + task + '\n' + output_constraint
    
    def ZeroShotBucket(self, backstory):
        context = '''BACKSTORY: ''' + backstory
        task = '''TASK: Predict the number of necessary documents to satisfy the user need described by BACKSTORY. '''
        constraint_1 = '''1) You/the system must output a single category.'''
        constraint_2_0 = '''2) Possible categories:'''
        constraint_2_1 = constraint_2_0 + '\n' + '''Zero: Find the answer in the search results listing, without reading any of the documents.'''
        constraint_2_2 = constraint_2_1 + '\n' + '''One: Find the answer by reading 1 document.'''
        constraint_2_3 = constraint_2_2 + '\n' + '''Two: Find the answer by reading 2 documents.'''
        constraint_2_4 = constraint_2_3 + '\n' + '''Few: Find the answer by reading from 3 to 5 documents.'''
        constraint_2_5 = constraint_2_4 + '\n' + '''Several: Find the answer by reading from 6 to 10 documents.'''
        constraint_2_6 = constraint_2_5 + '\n' + '''Many: Find the answer by reading from 11 to 100 documents.'''
        constraint_2   = constraint_2_6 + '\n' + '''Countless: Find the answer by reading 100+ documents.'''
        constraint_3 = '''3) You/the system must not output any text/characters apart from the category.'''
        
        output_constraint = '''CONSTRAINTS: ''' + '\n' + constraint_1 + '\n' + constraint_2 + '\n' + constraint_3
        # print('\n' + context + '\n' + task + '\n' + output_constraint)  # DEBUG: print the output constraint for validation purposes
        return context + '\n' + task + '\n' + output_constraint
    
    def ZeroShotDistribution(self, backstory, number=''):
        context = '''BACKSTORY: ''' + backstory
        
        task_1 = '''Retrieve a Python list with ''' + str(number)
        task_2 = '''predictions for the number of necessary documents to satisfy the user need described by BACKSTORY. '''
        task = '''TASK: ''' + task_1 + ''' ''' + task_2
        
        constraint_1 = '''1) You/the system must output a single Python list.'''
        constraint_2 = '''2) The Python list values must be integers to solve the task.'''
        constraint_3 = '''3) You/the system must not output any text/characters apart from the Python list.'''
        # constraint_4 = '''4) You/the system must ensure that the list contains exactly ''' + str(number) + ''' items/predictions.'''
        constraint_4 = '''4) You/the system must ensure that the list contains exactly ''' + str(number) + ''' items.'''

        constraint = '''CONSTRAINTS: ''' + '\n' + constraint_1 + '\n' + constraint_2 + '\n' + constraint_3 + '\n' + constraint_4
        
        return context + '\n' + task + '\n' + constraint
    
    def get_prompt(self, prompt_type, backstory, number=None):
        if prompt_type == "ZeroShot":
            return self.ZeroShot(backstory)
        elif prompt_type == "ZeroShotBucket":
            return self.ZeroShotBucket(backstory)
        elif prompt_type == "ZeroShotDistribution":
            return self.ZeroShotDistribution(backstory, number)
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
            num_annotations = self.df_data.iloc[i]['NumAnotations']
            # prompt = super().get_prompt(self.prompt_type, backstory, num_annotations) 
            prompt = super().get_prompt(self.prompt_type, backstory, 100) 
            llms_api = llmsAPI(prompt)
            output = llms_api.get_completion(llm=self.model, max_tokens=400)
            predictions.append(output)
            # time.sleep(1)

        self.df_data[self.model] = predictions
        
        
    def save_predictions(self):
        self.df_data.loc[:,[self.model]].to_csv(LLMS_PREDICTIONS + '/chatgpt_' + self.model + '_' + self.prompt_type + '.tsv', sep='\t')
        
    
    def main(self):
        super().load_data_backstoris()
        super().load_data_crowdsource_annotation()
        super().count_crowdsource_annotation()
        super().add_num_annotations_to_backstories()
        
        self.get_predictions()
        self.save_predictions()

if __name__ == '__main__':
    for llm in tqdm(["gpt-3.5-turbo-0125", "gpt-4-turbo", "gpt-4o-2024-05-13"], desc="LLMs", position=0,ncols=100):
        ##COMMENT: Modify line below to pick the right prompt_type
        LLmChatNumeric = ChatLLM(model=llm, prompt_type="ZeroShot")
        LLmChatNumeric.main()
        
        LLmChatCategory = ChatLLM(model=llm, prompt_type="ZeroShotBucket")
        LLmChatCategory.main()
    