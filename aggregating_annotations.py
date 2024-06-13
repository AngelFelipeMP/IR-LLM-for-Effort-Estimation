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

class UQV100_GOLD_LABELS_AGGREGATION:
    def __init__(self):
        self.query_and_estimates_file = 'uqv100-query-variations-and-estimates.tsv'
        
    def load_data(self):
        columns_average_estimates = ['UQV100Id', 'DocCount', 'DocCountAverage']
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
        self.gold_df['Median'] = self.gold_df['Media'].astype(int)
        
    def closer_integer_to_the_average(self):
        self.gold_df['CIA'] = self.df.groupby('UQV100Id')['DocCountAverage'].first().round()
        self.gold_df['CIA'] = self.gold_df['CIA'].astype(int)
        
    def braylan_lease(self):
        ##TODO Load aggregated annotations
        pass
        
        
    def main(self):
        self.load_data()
        self.gold_labels_dataframe()
        self.first_value()
        self.majority_vote()
        self.median()
        self.closer_integer_to_the_average()

if __name__ == '__main__':
    UQV100 = UQV100_GOLD_LABELS_AGGREGATION()
    UQV100.main()