import pandas  as pd
import os
import glob
from icecream import ic
from config import *
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class UQV100_DATA:
    def __init__(self):
        self.crowdsource_annotations_file = 'uqv100-query-variations-and-estimates.tsv'
        
    def load_data_crowdsource_annotation(self):
        columns = ['UQV100Id', 'DocCount', 'DocCountAverage']
        self.df_annotations = pd.read_csv(UQV100_DATA_PATH + '/' + self.crowdsource_annotations_file,
                        sep='\t', 
                        usecols=columns,
                        index_col='UQV100Id')
        
    def caculate_median(self):
        self.df_annotations['DocCountMedia'] = self.df_annotations.groupby('UQV100Id')['DocCount'].median()
        self.df_annotations = self.df_annotations.reset_index().drop_duplicates(subset='UQV100Id', keep='first').set_index('UQV100Id')
        
    def remove_DocCount_column(self):
        self.df_annotations = self.df_annotations.drop('DocCount', axis=1)
        
    def load_llms_predictions(self):
        tsv_files = glob.glob(os.path.join(LLMS_PREDICTIONS_DIST, "*.tsv"))
        df_list = [pd.read_csv(tsv_file, sep='\t') for tsv_file in tsv_files]
        self.df_llms_predictions = pd.concat(df_list, axis=0)
        self.df_llms_predictions.drop('Unnamed: 0', axis=1, inplace=True)

    def caculate_llms_predictions_average(self):
        self.df_llms_predictions['Average'] = self.df_llms_predictions.groupby(['llm', 'temperature', 'UQV100Id'])['prediction'].transform('mean')
        
    def caculate_llms_predictions_median(self):
        self.df_llms_predictions['Median'] = self.df_llms_predictions.groupby(['llm', 'temperature', 'UQV100Id'])['prediction'].transform('median')
        
    def keep_only_first_row(self):
        self.df_llms_predictions = self.df_llms_predictions.reset_index().drop_duplicates(subset=['llm', 'temperature', 'UQV100Id'], keep='first').reset_index(drop=True).drop('index', axis=1)
        
        
        
        
class UQV100_PLOTS:
    def __init__(self):
        self.graphics_path = GRAPHICS_PATH
        
    def load_and_process_data(self):
        data = UQV100_DATA()
        data.load_data_crowdsource_annotation()
        data.caculate_median()
        data.remove_DocCount_column()
        data.load_llms_predictions()
        
        #COMMENT: line below is temporaly added ZERO to avoid NaN values in 'prediction' column
        # data.df_llms_predictions['prediction'] = data.df_llms_predictions['prediction'].fillna(1)
        
        data.caculate_llms_predictions_average()
        data.caculate_llms_predictions_median()
        data.keep_only_first_row()
        
        self.df_crowdsource = data.df_annotations
        self.df_llms_predictions = data.df_llms_predictions
        
        # print(self.df_crowdsource)
        # print(self.df_llms_predictions)
    
        # #COMMENT: check for NaN values in 'prediction' column
        # nan = data.df_llms_predictions.loc[data.df_llms_predictions['prediction'].isna(), ['llm', 'temperature', 'UQV100Id']]
        # print(nan)
        
        
    #TODO make it proper to generate histograms for llms predictions
    def generate_histograms(self, df):
        for column in df.columns:
            plt.figure(figsize=(10, 6))
            
            min_val = int(np.floor(df[column].min()))
            max_val = int(np.ceil(df[column].max()))
            num = max_val - min_val
            print(f'Min: {min_val}, Max: {max_val}, Num: {num}')
            bins = [n for n in range(min_val, max_val+1)]
            print(f'bins {bins}')
            
            plt.hist(df[column], bins=bins, edgecolor='black', alpha=0.7)
            plt.title(f'Histogram of {column}')
            plt.xlabel(column)
            plt.ylabel('Frequency')
            plt.xticks(bins) 
            # plt.savefig(f'histogram_{column}.png', bbox_inches='tight', dpi=300)
            plt.show()
            
    def generate_histograms(self, df):
        colors = sns.color_palette("Set3", len(df.columns))  # Get a color palette with a fixed number of colors
        for i, column in enumerate(df.columns):
            plt.figure(figsize=(10, 6))
            
            min_val = int(np.floor(df[column].min()))
            max_val = int(np.ceil(df[column].max()))
            num = max_val - min_val
            print(f'Min: {min_val}, Max: {max_val}, Num: {num}')
            bins = [n for n in range(min_val, max_val+1)]
            print(f'bins {bins}')
            
            plt.hist(df[column], bins=bins, edgecolor='black', alpha=0.7, color=colors[i % len(colors)])
            plt.title(f'Histogram of {column}')
            plt.xlabel(column)
            plt.ylabel('Frequency')
            plt.xticks(bins) 
            # plt.savefig(f'histogram_{column}.png', bbox_inches='tight', dpi=300)
            plt.show()
            
            

    # def histogram(self, df):
    #     # Set a seed for reproducibility
    #     np.random.seed(8)
        
    #     # Get the Set3 color palette
    #     palette = sns.color_palette("Set3", len(df.columns))
        
    #     unique_values_flat = sorted(pd.unique(df.values.ravel()))
        
    #     for i, column in enumerate(df.columns):

    #         annotations = df[column].to_list()
            
    #         ax = sns.histplot(annotations,  discrete=True,  zorder=5, color=palette[i])
            
    #         # Add mean and median lines to the histogram
    #         ax.axvline(mean_val, color='red', linestyle='--', linewidth=0.5, label=f'Mean: {mean_val:.2f}')
    #         ax.axvline(median_val, color='green', linestyle='-', linewidth=0.5, label=f'Median: {median_val:.2f}')
            

    #         plt.title('(' + column + ')' + ' Predictions')

    #         plt.xlabel('Values')
    #         # plt.xticks(range(sorted(set(annotations))[0], sorted(set(annotations))[-1] +1))
    #         plt.xticks(annotations)
    #         plt.yticks(range(0, 110, 10))
            
    #         # Display light grey horizontal grid lines across the plot
    #         plt.grid(axis='y', color='lightgrey', linestyle='-', linewidth=0.7, zorder=0)
    #         ax.legend()
            
    #         # Ensure all categories are displayed on the x-axis
    #         ax.set_xlim(-0.5, len(unique_values_flat) - 0.5)
            
    #         # save plot
    #         plt.savefig(self.graphics_path + '/histogram_'+ column +'.png', bbox_inches='tight', dpi=400)
            
    #         # show plot
    #         plt.show()

if __name__ == '__main__':
    
    plots = UQV100_PLOTS()
    plots.load_and_process_data()
    plots.generate_histograms(plots.df_crowdsource)