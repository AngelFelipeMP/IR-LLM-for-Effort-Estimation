import sys
import os
import json
external_script_path = '/Users/angel_de_paula/repos/IR-LLM-for-Effort-Estimation'
sys.path.append(external_script_path)
import pandas as pd
import numpy as np
from icecream import ic
from config import UQV100_DATA_PATH, AGGREGATED_ANNOTATIONS
import experiments

class UQV100_GOLD_LABELS_BRAYLAN_AND_LEASE:
    def __init__(self):
        self.query_and_estimates_file = 'uqv100-query-variations-and-estimates.tsv'
        
    def load_data(self):
        columns_average_estimates = ['UQV100Id', 'WorkerIdHash', 'DocCount']
        self.df = pd.read_csv(UQV100_DATA_PATH + '/' + self.query_and_estimates_file, 
                                sep='\t', 
                                usecols=columns_average_estimates)
    
    def gold_labels_dataframe(self):
        self.gold_df = pd.DataFrame(self.df['UQV100Id'].unique(), columns=['UQV100Id'])
        self.gold_df.set_index('UQV100Id', inplace=True)
        
    def distance_fn(self, annotation1, annotation2):
        return abs(annotation1 - annotation2)
    
    def braylan_lease(self):
        self.aggregation_braylan_lease = experiments.RealExperiment(eval_fn=None,
                                                            label_colname='DocCount',
                                                            item_colname='UQV100Id', 
                                                            uid_colname='WorkerIdHash',
                                                            distance_fn=self.distance_fn)
        
        print('###### SETUP #######')
        self.aggregation_braylan_lease.setup(self.df)
        
        print('###### ANNODF #######')
        print(self.aggregation_braylan_lease.annodf)
        
        print('###### TRAIN #######')
        self.aggregation_braylan_lease.train()

        print('###### GL to PYTHON DICT #######')
        bau = self.aggregation_to_dict(self.aggregation_braylan_lease.bau_preds)
        sad = self.aggregation_to_dict(self.aggregation_braylan_lease.sad_preds)
        mas = self.aggregation_to_dict(self.aggregation_braylan_lease.mas_preds)
        
        print('###### SAVE GL - JSON #######')
        self.save_dict(bau, 'braylan_lease_BAU')
        self.save_dict(sad, 'braylan_lease_SAD')
        self.save_dict(mas, 'braylan_lease_MAS')
        
        print('###### SAVE GL - DateFrame #######')
        self.gold_df['BAU'] = list(bau.values())
        self.gold_df['SAD'] = list(sad.values())
        self.gold_df['MAS'] = list(mas.values())
        
        self.gold_df.to_csv(AGGREGATED_ANNOTATIONS + '/' + 'braylan_lease_aggregated_annotations.tsv', sep='\t')

    def aggregation_to_dict(self, aggregation):
        return {int(k): int(v) for k, v in aggregation.items()}
        
    
    def save_dict(self, file, name):
        with open(AGGREGATED_ANNOTATIONS + '/' + name + '.json', 'w') as f:
            json.dump(file, f, indent=4)
        
    def main(self):
        self.load_data()
        self.gold_labels_dataframe()
        self.braylan_lease()

if __name__ == '__main__':
    UQV100 = UQV100_GOLD_LABELS_BRAYLAN_AND_LEASE()
    UQV100.main()
