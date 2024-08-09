import pandas as pd
from config import *
from gold_label_statistical_test import UQV100_GOLD_LABELS_STATISTICAL_TEST
from gold_labels_plus_llms_plots import UQV100_GOLD_LABELS_PLUST_LLMS_HISTOGRAM
from icecream import ic

class UQV100_BUCKETING_PREDS(UQV100_GOLD_LABELS_STATISTICAL_TEST, UQV100_GOLD_LABELS_PLUST_LLMS_HISTOGRAM):
    def __init__(self):
        super().__init__()
        UQV100_GOLD_LABELS_PLUST_LLMS_HISTOGRAM.__init__(self)
        UQV100_GOLD_LABELS_STATISTICAL_TEST.__init__(self)
        
    
    def journal_bucket(self):
        # Define the bins and corresponding labels
        bins = [-float('inf'), 0, 1, 2, 5, 10, 100, float('inf')]
        labels = ['Zero', 'One', 'Two', 'Few', 'Several', 'Many', 'Countless']
        ending = 'JM'
        
        df = self.bucketing(bins, labels)
        self.save_bucketed_data(df, ending)
    
    
    def bucketing(self, bins, labels):
        df = self.df.copy()
        for column in self.df.columns:
            df[column] = pd.cut(df[column], bins=bins, labels=labels, right=False)
        
        return df
    
    def save_bucketed_data(self, df, ending): 
        df.to_csv(BUCKET_PREDS + '/' + 'bucketed_preds_' + ending + '.tsv', sep='\t')
    
    def main(self):
        UQV100_GOLD_LABELS_PLUST_LLMS_HISTOGRAM.load_llms_preds(self)
        UQV100_GOLD_LABELS_PLUST_LLMS_HISTOGRAM.rename_columns(self)
        UQV100_GOLD_LABELS_STATISTICAL_TEST.load_data(self)
        UQV100_GOLD_LABELS_PLUST_LLMS_HISTOGRAM.concat_aggregation_methods_and_llms(self)
        self.journal_bucket()
        
if __name__ == '__main__':
    UQV100 = UQV100_BUCKETING_PREDS()
    UQV100.main()