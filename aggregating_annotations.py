import pandas as pd
from config import *
from icecream import ic
from collections import Counter
import random
import statistics
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import seaborn as sns
from PIL import Image
import os

class UQV100_GOLD_LABELS_AGGREGATION:
    def __init__(self):
        self.query_and_estimates_file = 'uqv100-query-variations-and-estimates.tsv'
        self.path_uqv_data_DS = FAST_DAWID_SKENE_DATA + '/uqv100_dataset'
        

    def load_data(self):
        columns_average_estimates = ['UQV100Id', 'WorkerIdHash', 'DocCount','DocCountAverage']
        self.df = pd.read_csv(UQV100_DATA_PATH + '/' + self.query_and_estimates_file, 
                                sep='\t', 
                                usecols=columns_average_estimates)
    
    def gold_labels_dataframe(self):
        self.gold_df = pd.DataFrame(self.df['UQV100Id'].unique(), columns=['UQV100Id'])
        self.gold_df.set_index('UQV100Id', inplace=True)
    
    def majority_vote(self):
        self.gold_df['MajorityVote'] = self.df.groupby('UQV100Id')['DocCount'].apply(lambda x: x.mode().iloc[0])
        
    def median(self):
        self.gold_df['Median'] = self.df.groupby('UQV100Id')['DocCount'].apply(lambda x: x.median())
        self.gold_df['Median'] = self.gold_df['Median'].astype(int)
        
    def closer_integer_to_the_average(self):
        self.gold_df['CIA'] = self.df.groupby('UQV100Id')['DocCountAverage'].first().round()
        self.gold_df['CIA'] = self.gold_df['CIA'].astype(int)
        
    def braylan_lease(self):
        ##TODO: inclue braylan_lease into the class
        # braylan_lease_df = pd.read_csv(AGGREGATED_ANNOTATIONS + '/' + 'braylan_lease_aggregated_annotations' + '.tsv', 
        #                                 sep='\t',
        #                                 index_col='UQV100Id')
        
        # self.gold_df = pd.concat([self.gold_df, braylan_lease_df], axis=1)
        
        # include fds into the aggregation dataframe
        self.include_aggregation('braylan_lease_aggregated_annotations.tsv')
        
    def adapt_data_fast_dawid_skene(self):
        os.makedirs(self.path_uqv_data_DS, exist_ok=True)
        self.df_ds = self.df.loc[:, ['WorkerIdHash','UQV100Id', 'DocCount']]
        self.df_ds.to_csv(self.path_uqv_data_DS + '/' + 'crowd.csv', header=False, index=False)

    def fast_dawid_skene(self):
        code_line = 'python scripts/fast_dawid_skene.py'
        code_line = code_line + ' --dataset uqv100'
        code_line = code_line + ' --mode aggregate'
        code_line = code_line + ' --algorithm FDS'
        code_line = code_line + ' --output ' + AGGREGATED_ANNOTATIONS + '/' + 'uqv100_fast_dawid_skene_aggregate.csv'
        code_line = code_line + ' --verbose'
        # code_line = code_line + ' --print_result'
        
        os.chdir(FAST_DAWID_SKENE_REPO)
        os.system(code_line)
        os.chdir(REPO_PATH)
        
        # include fds into the aggregation dataframe
        self.include_aggregation('uqv100_fast_dawid_skene_aggregate.csv', 'FDS')
        
    def include_aggregation(self, filename, model=None):
        sep = '\t' if filename.endswith('.tsv') else ','
        
        df = pd.read_csv(AGGREGATED_ANNOTATIONS + '/' + filename, sep=sep)
        
        if 'UQV100Id' not in df.columns: 
            first_row = pd.DataFrame([df.columns], columns=df.columns)
            df = pd.concat([first_row, df], ignore_index=True)
            df.columns = ['UQV100Id', model]
        
        df.set_index('UQV100Id', inplace=True)
        
        self.gold_df = pd.concat([self.gold_df, df], axis=1)
    
    def save(self):
        self.gold_df.to_csv(AGGREGATED_ANNOTATIONS + '/' + 'aggregated_annotations.tsv', sep='\t')
        
        
    def main(self):
        self.load_data()
        self.gold_labels_dataframe()
        self.majority_vote()
        self.median()
        self.closer_integer_to_the_average()
        self.braylan_lease()
        self.adapt_data_fast_dawid_skene()
        self.fast_dawid_skene()
        self.save()

if __name__ == '__main__':
    UQV100 = UQV100_GOLD_LABELS_AGGREGATION()
    UQV100.main()