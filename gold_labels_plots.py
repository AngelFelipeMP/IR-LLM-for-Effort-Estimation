import pandas as pd
from config import *
from icecream import ic
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import seaborn as sns
import json


class UQV100_GOLD_LABELS_HISTOGRAM:
    def __init__(self):
        self.gold_labels_path = AGGREGATED_ANNOTATIONS + '/' + 'aggregated_annotations' + '.tsv'
        self.graphics_path = GRAPHICS_PATH
        
        
    def load_data(self):
        self.df = pd.read_csv(self.gold_labels_path, 
                                sep='\t', 
                                index_col='UQV100Id')
        
        # self.df.rename(columns={'MajorityVote': 'MV'}, inplace=True)
        
    def add_MAS_modified_column(self):
        with open(AGGREGATED_ANNOTATIONS + '/' + 'braylan_lease_MAS_first' + '.json', 'r') as f:
            mas_dict = json.load(f)
            
        mas_df = pd.DataFrame.from_dict(mas_dict, orient='index', columns=['MAS_mod'])
        mas_df.index = mas_df.index.astype(int)
        self.df.reset_index(inplace=True)
        self.df = pd.concat([self.df, mas_df], axis=1)
        self.df.set_index('UQV100Id', inplace=True)
        
        ic(self.df.head())
        
    def histogram_s(self):
        # Set a seed for reproducibility
        np.random.seed(8)
        
        # Get the Set3 color palette
        palette = sns.color_palette("Set3", len(self.df.columns))
        
        for i, column in enumerate(self.df.columns):

            annotations = self.df[column].to_list()
            
            # Calculate mean and median
            mean_val = np.mean(annotations)
            median_val = np.median(annotations)
            
            ax = sns.histplot(annotations,  discrete=True,  zorder=5, color=palette[i])
            
            # Add mean and median lines to the histogram
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=0.5, label=f'Mean: {mean_val:.2f}')
            ax.axvline(median_val, color='green', linestyle='-', linewidth=0.5, label=f'Median: {median_val:.2f}')
            
            plt.title('(' + column + ')' + ' Aggregated annotations')
            plt.xlabel('Gold labels')
            plt.xticks(range(sorted(set(annotations))[0], sorted(set(annotations))[-1] +1))
            plt.yticks(range(0, 70, 10))
            
            # Display light grey horizontal grid lines across the plot
            plt.grid(axis='y', color='lightgrey', linestyle='-', linewidth=0.7, zorder=0)
            ax.legend()
            
            # save plot
            plt.savefig(self.graphics_path + '/histogram_'+ column +'.png', bbox_inches='tight', dpi=300)
            
            # show plot
            plt.show()


    def process_data(self):
        self.df_seaborn = self.df.copy()
        columns = self.df_seaborn.columns
        first_cols = ['MV', 'CIA', 'Median', 'BAU', 'SAD', 'MAS', 'FDS']
        self.df_seaborn = self.df_seaborn[first_cols + [col for col in columns if col not in first_cols]]
        # self.df_seaborn = self.df_seaborn[['MV', 'CIA', 'Median','BAU', 'SAD', 'MAS']]
        
        # Use melt to transform the DataFrame
        self.df_seaborn =  self.df_seaborn.reset_index().melt(id_vars=['UQV100Id'], var_name='Aggregation Method', value_name='Category')

        # Drop the 'UQV100Id' column if it's not needed
        self.df_seaborn.drop(columns=['UQV100Id'], inplace=True)


    def multi_data_barplot(self):
        ax = sns.countplot(data=self.df_seaborn, x="Category", hue="Aggregation Method", dodge=True, zorder=5)
        
        # Add different shades of grey to the background to differentiate between groups of bars
        for i, category in enumerate(ax.get_xticks()):
            if i % 2 == 0:
                ax.axvspan(i - 0.5, i + 0.5, color='darkgrey', alpha=0.3, zorder=3)
            else:
                ax.axvspan(i - 0.5, i + 0.5, color='white', alpha=0.3, zorder=3)
        
        # Display light grey horizontal grid lines across the plot
        plt.grid(axis='y', color='lightgrey', linestyle='-', linewidth=0.7, zorder=0)
        ax.legend(loc='upper right')
        
        plt.title('Aggregated Methods')
        
        plt.savefig(self.graphics_path + '/histogram_all_agregation_methods'+'.png', bbox_inches='tight', dpi=350)
        plt.show()
        
        
    def multi_data_boxplot(self):
        sns.catplot(data=self.df_seaborn, x="Aggregation Method", y="Category", kind="box", palette="Set3")
        plt.title('Boxplot')
        
        # Rotate x-axis labels
        plt.xticks(rotation=30)
        
        plt.savefig(self.graphics_path + '/boxplot_all_agregation_methods'+'.png', bbox_inches='tight', dpi=350)
        plt.show()
        
    def multi_data_violin(self):
        sns.catplot(data=self.df_seaborn, x="Aggregation Method", y="Category", kind="violin", palette="Set3")
        plt.title('Violin')
        
        # Rotate x-axis labels
        plt.xticks(rotation=30)
        
        plt.savefig(self.graphics_path + '/violin_all_agregation_methods'+'.png', bbox_inches='tight', dpi=300)
        plt.show()
    
    
    def main(self):
        self.load_data()
        # self.add_MAS_modified_column()
        self.histogram_s()
        self.process_data()
        self.multi_data_barplot()
        self.multi_data_boxplot()
        self.multi_data_violin()
    
if __name__ == '__main__':
    UQV100 = UQV100_GOLD_LABELS_HISTOGRAM()
    UQV100.main()