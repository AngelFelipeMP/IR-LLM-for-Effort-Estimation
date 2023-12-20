import pandas as pd
import os
# from OpenAIAPI import Summarizer
from openaiapi import OpenAIAPI
import logging
import time
from icecream import ic

# Configure the logger
logging.basicConfig(filename='summarization.log', filemode='a', format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)


if __name__ == '__main__':

    # df = pd.read_csv('[File Name]')

    #summary_list = []
    # text_list = list(df['text'])
    
    text_list = ['what is the meaning of life?']

    for i in range(len(text_list)):
        # https://arxiv.org/pdf/2301.13848.pdf
        context = '''TEXT: '''
        task = '''Summarize the TEXT in '''
        output_constraint = '''three sentences.'''
        output = '''Summary: '''
        prompt = context + text_list[i] + '\n' + task + output_constraint + '\n' + output

        OpenAIapi = OpenAIAPI(prompt)
        print(OpenAIapi.get_completion())