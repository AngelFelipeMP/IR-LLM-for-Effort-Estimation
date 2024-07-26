import pandas as pd
from config import *
# from icecream import ic
# from collections import Counter
# import numpy as np
from scipy.stats import f_oneway, tukey_hsd
# import scipy
import numpy as np
import krippendorff
from itertools import combinations
from icecream import ic
from nltk.metrics.agreement import AnnotationTask
from nltk.metrics import interval_distance

class UQV100_GOLD_LABELS_STATISTICAL_TEST:
    def __init__(self):
        self.gold_labels_path = AGGREGATED_ANNOTATIONS + '/' + 'aggregated_annotations' + '.tsv'
        
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
        # print(res)

        df = pd.DataFrame(res.pvalue, columns=self.methods)
        df['index'] = self.methods
        df.set_index('index', inplace=True)
        
        df.to_csv(AGGREGATED_ANNOTATIONS + '/tukey_hsd_pvalue.tsv')

    def krippendorff_alpha(self):
        # alpha_pairs = []
        alpha_matrix = pd.DataFrame(index=self.df.columns, columns=self.df.columns)

        columns = self.df.columns
        for col1, col2 in combinations(columns, 2):
            data_pair = np.array([self.df[col1], self.df[col2]])

            alpha_value = krippendorff.alpha(data_pair, level_of_measurement='interval')
            # ic(alpha_value)
            
            alpha_matrix.at[col1, col2] = alpha_value
            alpha_matrix.at[col2, col1] = alpha_value  # Symmetric matrix
        
        np.fill_diagonal(alpha_matrix.values, np.nan)
        alpha_matrix.to_csv(AGGREGATED_ANNOTATIONS + '/Krippendorffs_Alpha.tsv')

    def krippendorff_alpha_nltk(self):
        alpha_matrix = pd.DataFrame(index=self.df.columns, columns=self.df.columns)
        
        for col1, col2 in combinations(self.df.columns, 2):
            tuples_list = self.df.loc[:,[col1, col2]].stack().reset_index().apply(tuple, axis=1).tolist()
            tuples_list = [(row[1], row[0], row[2]) for row in tuples_list]
            t = AnnotationTask(tuples_list, distance=interval_distance)
            alpha_value = t.alpha()
            # print(f'({col1}-{col2}) {t.alpha()}')
            
            alpha_matrix.at[col1, col2] = alpha_value
            alpha_matrix.at[col2, col1] = alpha_value  # Symmetric matrix
        
        np.fill_diagonal(alpha_matrix.values, np.nan)
        alpha_matrix.to_csv(AGGREGATED_ANNOTATIONS + '/Krippendorffs_Alpha_NLTK.tsv')
            
    def person_correlation_coefficient(self):
        # Calculate the Pearson Correlation Coefficient for all possible column pairs (except the first)
        correlation_matrix = self.df.corr(method='pearson')
        correlation_matrix.to_csv(AGGREGATED_ANNOTATIONS + '/person_correlation_coefficient.tsv')

    
    def main(self):
        self.load_data()
        self.person_correlation_coefficient()
        self.krippendorff_alpha()
        self.krippendorff_alpha_nltk()
        self.process_data()
        self.anova_test()
        self.tukey_hsd_test()
        
if __name__ == '__main__':
    UQV100 = UQV100_GOLD_LABELS_STATISTICAL_TEST()
    UQV100.main()
    
    
    
    
    
    
    
    
