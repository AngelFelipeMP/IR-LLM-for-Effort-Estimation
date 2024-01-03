##TODO: Load average label per backstory
## [X]  get first one row per backstory/UQV100Id
## [X]  add a new column with the rounded average label

##TODO: Load all labels
## [X]  caculate the magority vote label
## [X]  if there is a tie, sabe a list of labels

##TODO: I STOPED HERE
#  [X] I need to check if I need to Analyse the labels futher
#  [X] add a column with the frequence of the most frequent label
#  [X] add a coumn with the percentage of the most frequent label
#  [X] I will discrabe the label distribution for each sample, 
#       (X) standard deviation
#       (X) Coefficient of Variation
#       (X) Compare with Expected Values
#  [X] save file with all the transformations (one row per UQV100Id)
#  [X] save file with only Tied rows

##TODO: Write a report
#  [ ] write a report about the two possible gold labels
#  [ ] Discus about the caculated parameters (SD, CV, etc)
#  [ ] There is no sigle sample with the same label
#  [ ] What means high Spread and low Spread - mention the intervals
#  [ ] Should we Remove Tied Majority Vote samples?
#  [ ] Write down numbers of samples with Tied Majority Vote
#  [ ] Write down numbers of samples with 'NO_MATCH' -> Majority Vote and Average Labels
#  [ ] Num high/Low/Medium spread samples -> based on CV and ASD
#  [ ] print a histogram with the anottetion/distribution of labels for sample
#  [ ] print a histogram with the distribution of labels for Majority Vote and Average Labels - together (I want to see the difference)

##TODO: Create a graphic with the distribution of labels
#  [ ] I may change the name of the script to labels_analysis.py

import pandas as pd
from config import *
from icecream import ic
from collections import Counter
import random

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
        
        self.df['DocCountSD'] = self.df['DocCountList'].apply(lambda x: round(pd.Series(x).std(), 2))
        
        self.df['CoefficienOfVariation'] = self.df['DocCountList'].apply(lambda x: round(pd.Series(x).std() / pd.Series(x).mean(), 2))
        ##NOTE: Discus this interval
        self.df['DistSpreadBasedOnCV'] = self.df['CoefficienOfVariation'].apply(lambda x: 'Low' if x < 0.2 else 'Medium' if x < 0.4 else 'High')
        
        self.df['AverageSD'] = self.df.groupby('UQV100Id')['DocCountSD'].first().mean()
        ##NOTE: Discus this interval
        self.df['DistSpreadBasedOnASD'] = self.df['DocCountSD'].apply(lambda x: 'Low' if x < self.df['AverageSD'].iloc[0]*0.4 else 'Medium' if x < self.df['AverageSD'].iloc[0]*0.6 else 'High')
        
        self.df['NumAnotettors'] = self.df.apply(lambda row: len(row['DocCountList']), axis=1)
        
        self.df['DocCountMostFrequent'] = self.df['DocCountList'].apply(lambda x: [k for k, v in Counter(x).items() if v == max(Counter(x).values())])
        
        self.df['NumAnottetorsMostFrequent'] = self.df['DocCountList'].apply(lambda x: max(Counter(x).values()))
        
        self.df['PorcentageAnottetorsMostFrequent'] = round(self.df['NumAnottetorsMostFrequent'] / self.df['NumAnotettors'] * 100).astype(int)
        
        self.df['DocCountTie'] = self.df['DocCountMostFrequent'].apply(lambda x: 'TIE' if len(x)>1 else 'NO_TIE')
        
        self.df['MajorityVote'] = self.df['DocCountMostFrequent'].apply(lambda x: x[0] if len(x)==1  else random.choice(x))
        
        self.df['DistanceMVandAverage'] = abs(self.df['MajorityVote'] - self.df['DocCountAverageRound'])
        
        self.df['MatchMVandAverage'] = self.df['DistanceMVandAverage'].apply(lambda x: 'MATCH' if x == 0 else 'NO_MATCH')
        
        self.df['DocCountList'] =  self.df['DocCountList'].astype(str)
        
        self.df = self.df.reset_index().drop_duplicates(subset='UQV100Id', keep='first').set_index('UQV100Id')

        # ic(self.df)
        # ic(len(self.df))
        
    def save_data(self):
        # All backstories
        self.df.to_csv(LOGS_PATH + '/' + 'uqv100-labels-analysis.tsv', sep='\t')
        
    def graphics(self):
        pass
    
    def report(self):
        pass
        
    def main(self):
        self.load_data()
        self.data_transformation()
        self.save_data()
    
if __name__ == '__main__':
    UQV100 = UQV100_QUERIES_AND_ESTIMATES()
    UQV100.main()