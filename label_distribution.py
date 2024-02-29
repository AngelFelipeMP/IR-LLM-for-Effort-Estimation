import pandas as pd
from config import *
from icecream import ic
from collections import Counter
import random
import statistics
import matplotlib.pyplot as plt
import numpy as np

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
        self.df['CloserIntegerToTheAverage'] = self.df['DocCountAverage'].round()
        self.df['CloserIntegerToTheAverage'] = self.df['CloserIntegerToTheAverage'].astype(int)
        
        self.df['DocCountList'] = self.df.groupby('UQV100Id')['DocCount'].apply(list)
        
        self.df['DocCountSD'] = self.df['DocCountList'].apply(lambda x: round(pd.Series(x).std(), 2))
        
        self.df['CoefficientOfVariation'] = self.df['DocCountList'].apply(lambda x: round(pd.Series(x).std() / pd.Series(x).mean(), 2))
        ##NOTE: Discus this interval
        self.df['DistSpreadBasedOnCV'] = self.df['CoefficientOfVariation'].apply(lambda x: 'Low' if x < 0.2 else 'Medium' if x < 0.4 else 'High')
        
        self.df['AverageSD'] = self.df.groupby('UQV100Id')['DocCountSD'].first().mean()
        
        self.df['SAD-SDn/ASD'] = self.df['DocCountSD'] / self.df['AverageSD'] 
        
        ##NOTE: Discus this interval
        self.df['DistSpreadBasedOnASD'] = self.df['DocCountSD'].apply(lambda x: 'Low' if x < self.df['AverageSD'].iloc[0]*0.8 else 'Medium' if x < self.df['AverageSD'].iloc[0]*1.2 else 'High')
        
        self.df['NumAnotettors'] = self.df.apply(lambda row: len(row['DocCountList']), axis=1)
        
        self.df['DocCountMostFrequent'] = self.df['DocCountList'].apply(lambda x: [k for k, v in Counter(x).items() if v == max(Counter(x).values())])
        
        self.df['NumAnottetorsMostFrequent'] = self.df['DocCountList'].apply(lambda x: max(Counter(x).values()))
        
        self.df['PorcentageAnottetorsMostFrequent'] = round(self.df['NumAnottetorsMostFrequent'] / self.df['NumAnotettors'] * 100).astype(int)
        
        self.df['DocCountTie'] = self.df['DocCountMostFrequent'].apply(lambda x: 'TIE' if len(x)>1 else 'NO_TIE')
        
        self.df['MajorityVote'] = self.df['DocCountMostFrequent'].apply(lambda x: x[0] if len(x)==1  else random.choice(x))
        
        self.df['DistanceMVandCIA'] = abs(self.df['MajorityVote'] - self.df['CloserIntegerToTheAverage'])
        
        self.df['MatchMVandCIA'] = self.df['DistanceMVandCIA'].apply(lambda x: 'MATCH' if x == 0 else 'NO_MATCH')
        
        self.df['AverageLinkageDistanceCIA'] = self.df.apply(lambda x: statistics.mean([abs(value - x['CloserIntegerToTheAverage']) for value in x['DocCountList']]), axis=1)
        
        self.df['AverageLinkageDistanceMV'] = self.df.apply(lambda x: statistics.mean([abs(value - x['MajorityVote']) for value in x['DocCountList']]), axis=1)
        
        self.df['CentroidLinkagDistanceCIA'] = (self.df['DocCountAverage'] - self.df['CloserIntegerToTheAverage']).abs()
        
        self.df['CentroidLinkagDistanceMV'] = (self.df['DocCountAverage'] - self.df['MajorityVote']).abs()
        
        # self.df['DocCountList'] =  self.df['DocCountList'].astype(str) ##TODO: check if I need it!
        
        self.df = self.df.reset_index().drop_duplicates(subset='UQV100Id', keep='first').set_index('UQV100Id')

    def clasterDistance(self):
        self.df_clustering = pd.DataFrame({
        'GoldLabel': ['CIA', 'MV'],
        'AverageLinkageClustering': [self.df['AverageLinkageDistanceCIA'].mean(), self.df['AverageLinkageDistanceMV'].mean()],
        'CentroidLinkageClustering': [self.df['CentroidLinkagDistanceCIA'].mean(), self.df['CentroidLinkagDistanceMV'].mean()],
        'Average': [sum([self.df['AverageLinkageDistanceCIA'].mean(), self.df['CentroidLinkagDistanceCIA'].mean()])/2, 
                    sum([self.df['AverageLinkageDistanceMV'].mean(), self.df['CentroidLinkagDistanceMV'].mean()])/2]
        })
        
    def save_data(self):
        # All backstories
        self.df.to_csv(LOGS_PATH + '/' + 'uqv100-labels-analysis_29-02-2023.tsv', sep='\t')
        self.df_clustering.to_csv(LOGS_PATH + '/' + 'claster_distance.tsv', sep='\t')
        
    def boxplot_columns(self):
        # Create the figure and axes objects
        fig, axes = plt.subplots(nrows=10, ncols=10, figsize=(20, 20))  # Adjust the size as needed
        axes = axes.flatten()  # Flatten the axes array for easy iteration

        for i, ((data_list, cia, mv), ax) in enumerate(zip(self.df.loc[:,['DocCountList','CloserIntegerToTheAverage', 'MajorityVote']].values.tolist(), axes)):
                if i < len(self.df):
                    # Generate boxplot for each list
                    ax.boxplot(data_list, positions=[1], widths=0.5)

                    # Add additional lines at specified values
                    ax.axhline(y=cia, color='r', linestyle='--')
                    ax.axhline(y=mv, color='g', linestyle='-.')

                    # Set title or any other properties for each subplot
                    ax.set_title(f'Boxplot {i + 1}')
                    ax.set_xticks([])  # Hide x ticks

        # Adjust layout to prevent overlap
        plt.tight_layout()
        
        # Save figure
        plt.savefig(GRAPHICS_PATH + '/boxplots_columns.png')

        # Display the plot
        plt.show()
        
    def boxplot_line(self):
        doc_count_list = self.df['DocCountList'].tolist()
        closer_integer_to_the_average = self.df['CloserIntegerToTheAverage'].tolist()
        majority_vote = self.df['MajorityVote'].tolist()

        # Create the figure and axes object
        fig, ax = plt.subplots(figsize=(20, 10))  # Adjust the size as needed

        # Plot all boxplots on the same axes
        for i, data_list in enumerate(doc_count_list):
            position = i + 1  # Position for the current boxplot
            ax.boxplot(data_list, positions=[position], widths=0.5)

            # Add additional lines at specified values from the other two lists
            cia_value = closer_integer_to_the_average[i]
            mv_value = majority_vote[i]

            # Draw lines across the y-axis at the position of the current boxplot
            ax.plot([position-0.25, position+0.25], [cia_value, cia_value], color='red', linestyle='--')
            ax.plot([position-0.25, position+0.25], [mv_value, mv_value], color='green', linestyle='-.')

        # Customize the x-axis to properly space the boxplots
        ax.set_xlim(0, len(doc_count_list) + 1)
        ax.set_xticks(range(1, len(doc_count_list) + 1))
        ax.set_xticklabels([f'{i + 1}' for i in range(len(doc_count_list))])

        # Optional: rotate the x-axis tick labels if they overlap
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Save figure
        plt.savefig(GRAPHICS_PATH + '/boxplots_line.png')
        
        # Display the plot
        plt.show()
    
    def boxplot_line_test(self):
        # Preprocess data by filtering out outliers over 100
        filtered_doc_count_list = [[value for value in sublist if value <= 40]
                                for sublist in self.df['DocCountList'].tolist()]

        closer_integer_to_the_average = self.df['CloserIntegerToTheAverage'].tolist()
        majority_vote = self.df['MajorityVote'].tolist()

        # Create the figure and axes object
        fig, ax = plt.subplots(figsize=(20, 10))  # Adjust the size as needed

        # Plot all boxplots on the same axes without outliers over 100
        for i, data_list in enumerate(filtered_doc_count_list):
            position = i + 1  # Position for the current boxplot
            ax.boxplot(data_list, positions=[position], widths=0.5)

            # Add additional lines at specified values from the other two lists
            cia_value = closer_integer_to_the_average[i]
            mv_value = majority_vote[i]

            # Draw lines across the y-axis at the position of the current boxplot
            ax.plot([position-0.25, position+0.25], [cia_value, cia_value], color='red', linestyle='--')
            ax.plot([position-0.25, position+0.25], [mv_value, mv_value], color='green', linestyle='-.')

        # Customize the x-axis to properly space the boxplots
        ax.set_xlim(0, len(filtered_doc_count_list) + 1)
        ax.set_xticks(range(1, len(filtered_doc_count_list) + 1))
        ax.set_xticklabels([f'{i + 1}' for i in range(len(filtered_doc_count_list))])

        # Optional: rotate the x-axis tick labels if they overlap
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Adjust layout to prevent overlap
        plt.tight_layout()
        
        # Save figure
        plt.savefig(GRAPHICS_PATH + '/boxplots_line_test.png')

        # Display the plot
        plt.show()
    
    
    
    
    def report(self):
        pass
        
    def main(self):
        self.load_data()
        self.data_transformation()
        self.clasterDistance()
        self.save_data()
        # self.boxplot_columns()
        # self.boxplot_line()
        
        self.boxplot_line_test()
    
if __name__ == '__main__':
    UQV100 = UQV100_QUERIES_AND_ESTIMATES()
    UQV100.main()
    
    
    
    
#     import matplotlib.pyplot as plt

# # Assuming self.df is your dataframe and has 'DocCountList', 'CloserIntegerToTheAverage', 'MajorityVote' as columns

# # Convert your DataFrame columns to lists
# doc_count_list = self.df['DocCountList'].tolist()
# closer_integer_to_the_average = self.df['CloserIntegerToTheAverage'].tolist()
# majority_vote = self.df['MajorityVote'].tolist()

# # Create the figure and axes object
# fig, ax = plt.subplots(figsize=(20, 10))  # Adjust the size as needed

# # Plot all boxplots on the same axes
# for i, data_list in enumerate(doc_count_list):
#     position = i + 1  # Position for the current boxplot
#     ax.boxplot(data_list, positions=[position], widths=0.5)

#     # Add additional lines at specified values from the other two lists
#     cia_value = closer_integer_to_the_average[i]
#     mv_value = majority_vote[i]

#     # Draw lines across the y-axis at the position of the current boxplot
#     ax.plot([position-0.25, position+0.25], [cia_value, cia_value], color='red', linestyle='--')
#     ax.plot([position-0.25, position+0.25], [mv_value, mv_value], color='green', linestyle='-.')

# # Customize the x-axis to properly space the boxplots
# ax.set_xlim(0, len(doc_count_list) + 1)
# ax.set_xticks(range(1, len(doc_count_list) + 1))
# ax.set_xticklabels([f'{i + 1}' for i in range(len(doc_count_list))])

# # Optional: rotate the x-axis tick labels if they overlap
# plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# # Adjust layout to prevent overlap
# plt.tight_layout()

# # Display the plot
# plt.show()