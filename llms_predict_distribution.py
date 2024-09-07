import pandas as pd
import os
# from OpenAIAPI import Summarizer
from llmsapi import llmsAPI
import logging
import time
from icecream import ic
from config import *
from tqdm import tqdm
from llms_predict import PROMPT
from llmsapi import llmsAPI

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
        

class ChatLLM(PROMPT):
    # def __init__(self, model, prompt_type):
    def __init__(self, llm, temperature, backstory, prompt_type):
        super().__init__()
        self.llm = llm
        self.temperature = temperature
        self.backstory = backstory
        self.prompt_type = prompt_type
        
        
    def predict(self):
        prompt = super().get_prompt(self.prompt_type, backstory) 
        llms_api = llmsAPI(prompt)
        return llms_api.get_completion(llm=self.llm, temperature=self.temperature)
        # time.sleep(1)

if __name__ == '__main__':
    
    uqv_data = DATA()
    uqv_data.load_data_backstoris()
    uqv_data.load_data_crowdsource_annotation()
    uqv_data.count_crowdsource_annotation()
    uqv_data.add_num_annotations_to_backstories()
    
    # #Hyperparameters
    # llms_list = ["gpt-3.5-turbo-0125", "gpt-4o-2024-05-13", "gpt-4-turbo"]
    # temperature_list = [0.0, 0.5, 1.0]
    # df = uqv_data.df_data #df.loc['UQV100.010':]
    # prompt_type = "ZeroShot"
    
    # ##DEBUG:
    llms_list = ["gpt-4-turbo"]
    temperature_list = [0.0]
    uqv_list = [
    "UQV100.044", "UQV100.034", "UQV100.047", "UQV100.060", "UQV100.031", 
    "UQV100.039", "UQV100.030", "UQV100.066", "UQV100.056", "UQV100.049", 
    "UQV100.020", "UQV100.041", "UQV100.045", "UQV100.058", "UQV100.050", 
    "UQV100.029", "UQV100.061", "UQV100.055", "UQV100.036", "UQV100.054", 
    "UQV100.038", "UQV100.069", "UQV100.043", "UQV100.062", "UQV100.063", 
    "UQV100.052", "UQV100.064", "UQV100.035", "UQV100.021", "UQV100.027", 
    "UQV100.028", "UQV100.033", "UQV100.022", "UQV100.024", "UQV100.068", 
    "UQV100.023", "UQV100.026", "UQV100.046", "UQV100.071", "UQV100.053", 
    "UQV100.048", "UQV100.032", "UQV100.037", "UQV100.040", "UQV100.065", 
    "UQV100.042", "UQV100.070", "UQV100.059", "UQV100.051", "UQV100.025", 
    "UQV100.057", "UQV100.067"]
    
    df = uqv_data.df_data.loc[uqv_list]
    prompt_type = "ZeroShot"
    
    for llm in tqdm(llms_list, desc="LLMs", position=0,ncols=100):
        for temperature in tqdm(temperature_list, desc="Temperatures", position=1, ncols=100):
            for index, backstory, num_anottaions in tqdm(df[['Backstory', 'NumAnotations']].itertuples(), desc="Backstories", position=2, ncols=100):
                
                data = []
                
                for num_pred in range(1, num_anottaions+1):
                    
                    LLmPred = ChatLLM(llm=llm,
                                        temperature=temperature,
                                        backstory=backstory,
                                        prompt_type=prompt_type)
                    
                    
                    data.append({'llm': llm, 'temperature': temperature, 'UQV100Id':index , 'num_annotations': num_anottaions, 'num_predictions': num_pred, 'prediction': LLmPred.predict()})
                    
    
                df_data = pd.DataFrame(data)
                df_data.to_csv(LLMS_PREDICTIONS_DIST + '/' + llm + '_' + str(temperature) + '_' + str(index) + '_' + str(num_anottaions) + '_' '.tsv', sep='\t')
                tqdm.write(f'Saved predictions for LLM: {llm}, Temperature: {temperature}, UQV100Id: {index}')