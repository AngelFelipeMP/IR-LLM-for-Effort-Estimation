import pandas as pd
import os
# from OpenAIAPI import Summarizer
from openaiapi import OpenAIAPI
import logging
import time
from icecream import ic
from config import *
from tqdm import tqdm

# Configure the logger
logging.basicConfig(filename='summarization.log', filemode='a', format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

class ChatGPT_ZERO_SHOT:
    def __init__(self, model):
        self.backstories_file = 'uqv100-backstories.tsv'
        self.model = model
        self.prompt_type = 'zero-shot'

    def load_data(self):
        columns = ['UQV100Id', 'Backstory']
        self.df_data = pd.read_csv(UQV100_DATA_PATH + '/' + self.backstories_file,
                                sep='\t', 
                                usecols=columns,
                                index_col='UQV100Id')
    
    def prompt(self, backstory):
        context = '''BACKSTORY: '''
        task = '''TASK: Predict the number of necessary documents to attend the user needs for the BACKSTORY '''
        output_constraint = '''CONSTRAINT: You/the system must retrieve not text at all; it must only be a number, and it must be an integer.'''
        return context + backstory + '\n' + task + '\n' + output_constraint
        
    def get_predictions(self):
        predictions = []
        
        for i in tqdm(range(len(self.df_data)), desc="Predicting Effort Estimation"):  # Add tqdm to the loop
            backstory = self.df_data.iloc[i]['Backstory']
            backstory = self.df_data.iloc[i]['Backstory']
            prompt = self.prompt(backstory)
            OpenAIapi = OpenAIAPI(prompt)
            predictions.append(OpenAIapi.get_completion(model=self.model))
            # time.sleep(1)

        ic(len(predictions))
        ic(predictions)
        self.df_data[self.model] = predictions
        
        
    def save_predictions(self):
        self.df_data.to_csv(LLMS_PREDICTIONS + '/chatgpt_' + self.model + '_' + self.prompt_type + '.tsv')
    
    def main(self):
        self.load_data()
        self.get_predictions()
        self.save_predictions()
        
if __name__ == '__main__':
    ChatGPT = ChatGPT_ZERO_SHOT(model="gpt-3.5-turbo-0125")
    ChatGPT.main()
    
    ##TODO: DO not save the backstories with the predictions
    ##TODO: Get only the number from the text. Hence increase the max_tokens to 20
    ##TODO: Rethink about the code struct. I must have a code that will be robust for the following experiments
    

    # for i in range(len(text_list)):
    #     # https://arxiv.org/pdf/2301.13848.pdf
    #     context = '''BACKSTORY: '''
    #     task = '''TASK: Predict the number of necessary documents to attend the user needs for the BACKSTORY '''
    #     output_constraint = '''CONSTRAINT: You/the system must retrieve not text at all; it must only be a number, and it must be an integer.'''
    #     prompt = context + text_list[i] + '\n' + task + '\n' + output_constraint
        
    #     ic(prompt)

    #     OpenAIapi = OpenAIAPI(prompt)
    #     print(OpenAIapi.get_completion())