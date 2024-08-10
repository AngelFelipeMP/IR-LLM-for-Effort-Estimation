import pandas as pd
from config import *
from icecream import ic
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import seaborn as sns
import json
import glob

class UQV100_BUCKET_HISTOGRAM:
    def __init__(self):
        self.graphics_path = BUCKET_GRAPHICS_PATH
        self.llms_pred_path = LLMS_PREDICTIONS
        
    def pred_settings(self, prompt_type, bucket_method, category_order):
        self.category_order = category_order
        self.prompt_type = prompt_type
        self.pos_bucketed_aggregations_and_preds = BUCKET_PREDS + '/' + 'bucketed_preds_' + bucket_method + '.tsv'
        
        
    def load_pos_bucketed_data(self):
        self.df_pos = pd.read_csv(self.pos_bucketed_aggregations_and_preds, 
                                sep='\t', 
                                index_col='UQV100Id')
        
    def load_llms_preds(self):
        tsv_files = glob.glob(os.path.join(self.llms_pred_path, "*"+ self.prompt_type + ".tsv"))
        df_llms = [pd.read_csv(tsv_file, sep='\t', index_col=0) for tsv_file in tsv_files]
        self.llms_preds_df = pd.concat(df_llms, axis=1)
        
    def rename_columns(self):
        self.llms_preds_df.columns = ['*' + col.split('-', 2)[0].upper() + '-' + col.split('-', 2)[1] for col in self.llms_preds_df.columns]
        
    def concat_pos_bucket_and_CategoryLlms(self):
        self.df = pd.concat([self.df_pos, self.llms_preds_df], axis=1)
        
        
    def histogram(self):
        # Set a seed for reproducibility
        np.random.seed(8)
        
        # Get the Set3 color palette
        palette = sns.color_palette("Set3", len(self.df.columns))
        
        for i, column in enumerate(self.df.columns):

            annotations = self.df[column].to_list()
            
            # Convert annotations to a categorical type with the specified order
            annotations = pd.Categorical(annotations, categories=self.category_order, ordered=True)
            
            ax = sns.histplot(annotations,  discrete=True,  zorder=5, color=palette[i])
        
            if 'GPT' in column:
                plt.title('(' + column + ')' + ' Predictions')
            else:
                plt.title('(' + column + ')' + ' Aggregated annotations')
            plt.xlabel('Categories')
            
            plt.xticks(self.category_order)
            plt.yticks(range(0, 110, 10))
            
            # Display light grey horizontal grid lines across the plot
            plt.grid(axis='y', color='lightgrey', linestyle='-', linewidth=0.7, zorder=0)
            # ax.legend()
            
            # Ensure all categories are displayed on the x-axis
            ax.set_xlim(-0.5, len(self.category_order) - 0.5)
            
            # save plot
            plt.savefig(self.graphics_path + '/histogram_bucketed_'+ column +'.png', bbox_inches='tight', dpi=400)
            
            # show plot
            plt.show()

        
    def main(self):
        self.load_pos_bucketed_data()
        self.load_llms_preds()
        self.rename_columns()
        self.concat_pos_bucket_and_CategoryLlms()
        self.histogram()
        
if __name__ == '__main__':
    uqv100_bucket_histogram = UQV100_BUCKET_HISTOGRAM()
    uqv100_bucket_histogram.pred_settings(
        prompt_type='ZeroShotBucket', 
        bucket_method='JM',
        category_order = ['Zero', 'One', 'Two', 'Few', 'Several', 'Many', 'Countless']
    )
    uqv100_bucket_histogram.main()