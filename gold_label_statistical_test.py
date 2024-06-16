import pandas as pd
from config import *
from icecream import ic
from collections import Counter
import numpy as np
from scipy.stats import f_oneway, tukey_hsd

class UQV100_GOLD_LABELS_STATISTICAL_TEST:
    def __init__(self):
        self.gold_labels_path = AGGREGATED_ANNOTATIONS + '/' + 'aggregated_annotations' + '.tsv'
        # print(scipy.__version__)
        
    def load_data(self):
        self.df = pd.read_csv(self.gold_labels_path, 
                                sep='\t', 
                                index_col='UQV100Id')
        
        self.df.rename(columns={'MajorityVote': 'MV'}, inplace=True)
        
    
    def process_data(self):
        self.dist = []
        self.methods = []
        for column in self.df.columns:
            self.dist.append(self.df[column].to_list())
            self.methods.append(column)
    
    def anova_test(self):
        f , p_value = f_oneway(*self.dist)
        print(f'P value for ANOVA: {p_value}')

    def tukey_hsd_test(self):
        res = tukey_hsd(*self.dist)
        print(res)

        df = pd.DataFrame(res.pvalue, columns=self.methods)
        df['index'] = self.methods
        df.set_index('index', inplace=True)
        
        df.to_csv(AGGREGATED_ANNOTATIONS + '/tukey_hsd_pvalue.tsv')
    
    def main(self):
        self.load_data()
        self.process_data()
        self.anova_test()
        self.tukey_hsd_test()
        
if __name__ == '__main__':
    UQV100 = UQV100_GOLD_LABELS_STATISTICAL_TEST()
    UQV100.main()
    
    
    
    
    
    
    
                # self.gold_labels_dict = [(column,self.df[column].to_list()) for column in self.df.columns]
                # self.gold_labels_dist = [dist for _, dist in self.gold_labels_dict]
                # self.gold_labels_order = [(method,i) for i, method in enumerate(self.gold_labels_dict)]