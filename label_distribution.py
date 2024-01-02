##TODO: Load average label per backstory
## -  get first one row per backstory/UQV100Id
## -  add a new column with the rounded average label

##TODO: Load all labels
## -  caculate the magority vote label
## -  if there is a tie, sabe a list of labels

##TODO: Create a graphic with the distribution of labels

import pandas as pd
from config import *
from icecream import ic
from collections import Counter

class UQV100_QUERIES_AND_ESTIMATES:
    def __init__(self):
        self.query_and_estimates_file = 'uqv100-query-variations-and-estimates.tsv'
        
        
    def load_data(self):
        columns_average_estimates = ['UQV100Id', 'DocCount', 'DocCountAverage']
        self.df = pd.read_csv(UQV100_DATA_PATH + '/' + self.query_and_estimates_file, 
                                sep='\t', 
                                usecols=columns_average_estimates,
                                index_col='UQV100Id')
        
    def data_transformation(self):
        self.df['DocCountAverageRound'] = self.df['DocCountAverage'].round()
        self.df['DocCountAverageRound'] = self.df['DocCountAverageRound'].astype(int)
        
        self.df['DocCountList'] = self.df.groupby('UQV100Id')['DocCount'].apply(list)
        self.df['NumAnotettors'] = self.df.apply(lambda row: len(row['DocCountList']), axis=1)
        self.df['DocMostFrequent'] = self.df['DocCountList'].apply(lambda x: [k for k, v in Counter(x).items() if v == max(Counter(x).values())])
        
        self.df['DocCountTie'] = self.df['DocMostFrequent'].apply(lambda x: 'TIE' if len(x)>1 else 'NO_TIE')
        
        self.df = self.df.loc[self.df['DocCountTie'] == 'TIE']
        self.df = self.df.reset_index().drop_duplicates(subset='UQV100Id', keep='first').set_index('UQV100Id')
        # self.df = self.df.drop_duplicates(subset='UQV100Id')
        
        ##NOTE: I STOPED HERE
        #       - the code is working
        #       - I need to check if I need to Analyse the labels futher
        #       - I may need to discrabe the label distribution for each sample, caculate the variance, standard deviation, etc.
        #       - I should check the values which are not tied


        ic(self.df)
        ic(len(self.df))
        
    def main(self):
        self.load_data()
        self.data_transformation()
    
if __name__ == '__main__':
    UQV100 = UQV100_QUERIES_AND_ESTIMATES()
    UQV100.main()
    
    
    
        # self.df_ = self.df.drop_duplicates(keep='first')
        # self.df = self.df.drop('DocCount', axis=1)
        
        # ic(self.df.head())
        
                # self.df['MajorityVote'] = self.df['DocCountList'].apply(lambda x: max(set(x), key = x.count))