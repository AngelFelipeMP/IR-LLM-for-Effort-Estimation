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

class UQV100_QUERIES_AND_ESTIMATES:
    def __init__(self):
        self.query_and_estimates_file = 'uqv100-query-variations-and-estimates.tsv'
        
    def load_data(self):
        columns_average_estimates = ['UQV100Id', 'WorkerIdHash']
        self.df = pd.read_csv(UQV100_DATA_PATH + '/' + self.query_and_estimates_file, 
                                sep='\t', 
                                usecols=columns_average_estimates)
        
    def annotation_matrix(self):
        self.df['value'] = 1
        self.pivot_df =  self.df.pivot(index='WorkerIdHash', columns='UQV100Id', values='value')
        self.pivot_df = self.pivot_df.fillna(0)
        self.pivot_df = self.pivot_df.astype(int)
        
    def heatmap(self):
        cmap = ListedColormap(['red', 'green'])

        # Increase the default figure size to ensure all cells are included
        plt.figure(figsize=(20, 50))

        # Create the heatmap without annotations due to space constraints
        sns.heatmap(self.pivot_df, cmap=cmap, cbar=False)
        
        # Save the figure to a temporary image file
        temp_filename = GRAPHICS_PATH + '/heatmap_temp.png'
        plt.savefig(temp_filename, dpi=300, bbox_inches='tight')

        # Close the plot to free up memory
        plt.close()

        # Open the saved image using PIL
        with Image.open(temp_filename) as img:
            # Rotate the image by 90 degrees counter-clockwise
            rotated_img = img.rotate(270, expand=True)
            
            # Save the rotated image to the final file
            rotated_filename = GRAPHICS_PATH + '/heatmap_rotated.png'
            rotated_img.save(rotated_filename)

        
    # def histogram(self):
    #     # Calculate the sum of each column (assuming numerical data)
    #     column_sums = self.pivot_df.sum(axis=1)

    #     # Create a histogram based on the column sums
    #     plt.hist(column_sums) # Adjust the number of bins as needed

    #     # Add titles and labels as appropriate
    #     plt.title('Histogram of Column Sums')
    #     plt.xlabel('Sum of column values')
    #     plt.ylabel('Frequency')

    #     # Show the plot
    #     plt.show()
        
    # def histogram(self):
    #     column_sums = self.pivot_df.sum(axis=1)

    #     # Define bin edges with a step size of 10 units; adjust the range as needed
    #     # min_sum = column_sums.min()
    #     # max_sum = column_sums.max()
    #     # bins = range(int(min_sum), int(max_sum) + 10, 10)
    #     bins = range(int(0), int(100) + 10, 10)

    #     # Create a histogram with the specified bins
    #     plt.hist(column_sums, bins=bins, edgecolor='black')

    #     # Add titles and labels
    #     plt.title('Histogram of worker annotaton')
    #     plt.xlabel('Samples annotated')
    #     plt.ylabel('Number of workers')

    #     # Show the plot
    #     plt.show()
    
    def histogram(self):
        column_sums = self.pivot_df.sum(axis=1)

        # Define bin edges with a step size of 10 units; adjust the range as needed
        bins = range(0, 101, 10)

        # Calculate the weight for each data point such that the sum of the weights is 1.
        # Each weight is the percentage contribution of the single data point.
        total_sum = len(column_sums)
        weights = (np.ones_like(column_sums) / total_sum) * 100

        # Create a histogram with the specified bins and weights
        counts, _ = np.histogram(column_sums, bins=bins)
        n, bins, patches = plt.hist(column_sums, bins=bins, weights=weights, edgecolor='black', zorder=3)

        # Annotate each bar with its frequency count
        for count, patch in zip(counts, patches):
            x = patch.get_x() + patch.get_width() / 2
            y = patch.get_height()
            plt.annotate(str(count), (x, y), xytext=(0, 5), textcoords="offset points", ha='center', va='bottom')

        # Add titles and labels
        plt.title('Histogram of worker annotation')
        plt.xlabel('Samples annotated')
        plt.ylabel('Percentage of workers')
        plt.xticks(bins)
        
        # Display light grey horizontal grid lines across the plot
        plt.grid(axis='y', color='lightgrey', linestyle='-', linewidth=0.7, zorder=0)
        
        # Increase the top margin to add extra space above the tallest bar
        plt.margins(y=0.1) # Adds 10% padding to the top of the y-axis
        
        # Save the figure before calling plt.show()
        plt.savefig(GRAPHICS_PATH + '/histogram.png', bbox_inches='tight', dpi=300)
    
    def data_transformation(self):
        pass
    
    def save_data(self):
        # All backstories
        self.df.to_csv(LOGS_PATH + '/' + 'uqv100-sparsity-analysis_09-05-2024.tsv', sep='\t')

    def report(self):
        pass
        
    def main(self):
        self.load_data()
        self.annotation_matrix()
        self.heatmap()
        self.histogram()
        # self.data_transformation()
        # self.save_data()
    
if __name__ == '__main__':
    UQV100 = UQV100_QUERIES_AND_ESTIMATES()
    UQV100.main()