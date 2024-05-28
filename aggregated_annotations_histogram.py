import pandas as pd
from config import *
from icecream import ic
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import seaborn as sns


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
        
        # Calculate mean and median
        mean_val = np.mean(annotations)
        median_val = np.median(annotations)
        
        ax = sns.histplot(annotations,  discrete=True,  zorder=5)
        
        # Add mean and median lines to the histogram
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=0.5, label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='green', linestyle='-', linewidth=0.5, label=f'Median: {median_val:.2f}')
        
        plt.title('Aggregated annotations')
        plt.xlabel('Annotations')
        
        # Display light grey horizontal grid lines across the plot
        plt.grid(axis='y', color='lightgrey', linestyle='-', linewidth=0.7, zorder=0)
        ax.legend()
        
        # save plot
        plt.savefig(GRAPHICS_PATH + '/histogram_all_annotations.png', bbox_inches='tight', dpi=300)
        # show plot
        plt.show()
        
    def histo_until_50(self):
        annotations = self.df['DocCount'].to_list()
        annotations_until_50 = [n for n in annotations if n<=50]
        
        # Calculate mean and median
        mean_val = np.mean(annotations)
        median_val = np.median(annotations)

        # Create bins for each integer value up to 50
        bins = range(0, 51)
        
        # Set the figure size (width, height)
        plt.figure(figsize=(20, 8))
        
        ax = sns.histplot(annotations_until_50,  bins=bins, discrete=True,  zorder=5)
        
        # Add mean and median lines to the histogram
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=0.5, label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='green', linestyle='-', linewidth=0.5, label=f'Median: {median_val:.2f}')
        
        plt.title('Aggregated annotations')
        plt.xlabel('Annotations')
        plt.xticks(range(0, 51))
        plt.yticks(range(0, 2800, 200))
        
        
        # Display light grey horizontal grid lines across the plot
        plt.grid(axis='y', color='lightgrey', linestyle='-', linewidth=0.7, zorder=0)
        ax.legend()
        
        # save plot
        plt.savefig(GRAPHICS_PATH + '/histogram_until_50.png', bbox_inches='tight', dpi=300)
        
        # show plot
        plt.show()
        
    def histo_boxplot_all(self):
        annotations = self.df['DocCount'].to_list()
        
        # Calculate mean and median
        mean_val = np.mean(annotations)
        median_val = np.median(annotations)
        
        fig, ax = plt.subplots()
        
        # Create the histogram on the main axis
        sns.histplot(annotations, discrete=True, zorder=5, ax=ax)
        
        # Add mean and median lines to the histogram
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=0.5, label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='green', linestyle='-', linewidth=0.5, label=f'Median: {median_val:.2f}')
        
        # Create a secondary axis for the boxplot
        ax_box = ax.inset_axes([0, 1.02, 1, 0.1])
        sns.boxplot(x=annotations, ax=ax_box, color='skyblue', zorder=6)
        
        # Add mean and median lines to the boxplot
        ax_box.axvline(mean_val, color='red', linestyle='--', linewidth=0.5, zorder=7)
        ax_box.axvline(median_val, color='green', linestyle='-', linewidth=0.5 ,zorder=7)
        
        # Remove the x-axis labels for the boxplot
        ax_box.set_xlabel('')
        ax_box.set_xticks([])
        ax_box.set_yticks([])
        
        # Match the x-axis limits of the boxplot to those of the histogram
        ax_box.set_xlim(ax.get_xlim())
        
        # Main plot settings
        ax.set_title('Aggregated annotations')
        ax.set_xlabel('Annotations')
        ax.grid(axis='y', color='lightgrey', linestyle='-', linewidth=0.7, zorder=0)
        ax.legend()
        
        # Save plot
        plt.savefig(GRAPHICS_PATH + '/histogram_boxplot_all_annotations.png', bbox_inches='tight', dpi=300)
        
        # Show plot
        plt.show()
        
    def histo_boxplot_until_50(self):
        annotations = self.df['DocCount'].to_list()
        annotations_until_50 = [n for n in annotations if n<=50]
        
        # Calculate mean and median
        mean_val = np.mean(annotations)
        median_val = np.median(annotations)
        
        # Create bins for each integer value up to 50
        bins = range(0, 51)
        
        fig, ax = plt.subplots(figsize=(20, 8))
        
        # Create the histogram on the main axis
        sns.histplot(annotations_until_50, bins=bins, discrete=True, zorder=5, ax=ax)
        
        # Add mean and median lines to the histogram
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=0.5, label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='green', linestyle='-', linewidth=0.5, label=f'Median: {median_val:.2f}')
        
        # Create a secondary axis for the boxplot
        ax_box = ax.inset_axes([0, 1.02, 1, 0.1])
        sns.boxplot(x=annotations_until_50, ax=ax_box, color='skyblue', zorder=6)
        
        # Add mean and median lines to the boxplot
        ax_box.axvline(mean_val, color='red', linestyle='--', linewidth=0.5, zorder=7)
        ax_box.axvline(median_val, color='green', linestyle='-', linewidth=0.5 ,zorder=7)
        
        # Remove the x-axis labels for the boxplot
        ax_box.set_xlabel('')
        ax_box.set_xticks([])
        ax_box.set_yticks([])
        
        # Match the x-axis limits of the boxplot to those of the histogram
        ax_box.set_xlim(ax.get_xlim())
        
        # Main plot settings
        ax.set_title('Aggregated annotations')
        ax.set_xlabel('Annotations')
        ax.set_xticks(range(0, 51))
        ax.set_yticks(range(0, 2800, 200))
        ax.grid(axis='y', color='lightgrey', linestyle='-', linewidth=0.7, zorder=0)
        ax.legend()
        
        # Save plot
        plt.savefig(GRAPHICS_PATH + '/histogram_boxplot_util_50.png', bbox_inches='tight', dpi=300)
        
        # Show plot
        plt.show()
        
    def main(self):
        self.load_data()
        self.histo_all_samples()
        self.histo_until_50()
        self.histo_boxplot()
        self.histo_boxplot_until_50()
    
if __name__ == '__main__':
    UQV100 = UQV100_AGGREGATED_ANNOTATIONS_HISTOGRAM()
    UQV100.main()