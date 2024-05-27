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


class UQV100_AGGREGATED_ANNOTATIONS_HISTOGRAM:
    def __init__(self):
        self.query_and_estimates_file = 'uqv100-query-variations-and-estimates.tsv'
        
        
    def load_data(self):
        columns_average_estimates = ['UQV100Id', 'DocCount']
        self.df = pd.read_csv(UQV100_DATA_PATH + '/' + self.query_and_estimates_file, 
                                sep='\t', 
                                usecols=columns_average_estimates)
        
    def histo_all_samples(self):
        annotations = self.df['DocCount'].to_list()
        sns.histplot(annotations,  discrete=True,  zorder=5)
        plt.title('Aggregated annotations')
        plt.xlabel('Annotations')
        
        # Display light grey horizontal grid lines across the plot
        plt.grid(axis='y', color='lightgrey', linestyle='-', linewidth=0.7, zorder=0)
        # save plot
        plt.savefig(GRAPHICS_PATH + '/histogram_all_annotations.png', bbox_inches='tight', dpi=300)
        # show plot
        plt.show()
        
    def histo_until_50(self):
        annotations = self.df['DocCount'].to_list()
        annotations_until_50 = [n for n in annotations if n<=50]

        # Create bins for each integer value up to 50
        bins = range(0, 51)
        
        # Set the figure size (width, height)
        plt.figure(figsize=(20, 8))
        
        ax = sns.histplot(annotations_until_50,  bins=bins, discrete=True,  zorder=5)
        
        plt.title('Aggregated annotations')
        plt.xlabel('Annotations')
        plt.xticks(range(0, 51))
        plt.yticks(range(0, 2800, 200))
        
        
        # Display light grey horizontal grid lines across the plot
        plt.grid(axis='y', color='lightgrey', linestyle='-', linewidth=0.7, zorder=0)
        
        # # Annotate each bar with its value
        # for patch in ax.patches:
        #     height = patch.get_height()
        #     ax.text(patch.get_x() + patch.get_width() / 2, height, int(height), 
        #             ha='center', va='bottom', rotation=45)

        
        # save plot
        plt.savefig(GRAPHICS_PATH + '/histogram_until_50.png', bbox_inches='tight', dpi=300)
        
        # show plot
        plt.show()
        
        
    def main(self):
        self.load_data()
        # self.histo_all_samples()
        self.histo_until_50()
    
if __name__ == '__main__':
    UQV100 = UQV100_AGGREGATED_ANNOTATIONS_HISTOGRAM()
    UQV100.main()