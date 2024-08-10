import pandas as pd
from config import *
from icecream import ic
from gold_labels_plots import UQV100_GOLD_LABELS_HISTOGRAM
import glob

# #DRAFIT for BUCKETED PREDS
# class UQV100_GOLD_LABELS_PLUST_LLMS_HISTOGRAM(UQV100_GOLD_LABELS_HISTOGRAM):
#     def __init__(self, prompt_type):
#         super().__init__()
#         self.prompt_type = prompt_type
#         self.llms_pred_path = LLMS_PREDICTIONS
#         self.graphics_path = BUCKET_GRAPHICS_PATH
#         self.gold_labels_path = BUCKET_PREDS + '/' + 'bucketed_preds_JM' + '.tsv'
    
#     def load_llms_preds(self):
#         tsv_files = glob.glob(os.path.join(self.llms_pred_path, "*"+ self.prompt_type + ".tsv"))
#         df_llms = [pd.read_csv(tsv_file, sep='\t', index_col=0) for tsv_file in tsv_files]
#         self.llms_preds_df = pd.concat(df_llms, axis=1)
        
#     def rename_columns(self):
#         self.llms_preds_df.columns = [col.split('-', 2)[0].upper() + '-' + col.split('-', 2)[1] for col in self.llms_preds_df.columns]
        
#     def concat_aggregation_methods_and_llms(self):
#         self.df = pd.concat([self.df, self.llms_preds_df], axis=1)

#     def main(self):
#         self.load_llms_preds()
#         self.rename_columns()
#         super().load_data()
#         self.concat_aggregation_methods_and_llms()
#         super().histogram_s()
#         super().process_data()
#         super().multi_data_barplot()
#         super().multi_data_boxplot()
#         super().multi_data_violin()
        
        
class UQV100_GOLD_LABELS_PLUST_LLMS_HISTOGRAM(UQV100_GOLD_LABELS_HISTOGRAM):
    def __init__(self):
        super().__init__()
        self.llms_pred_path = LLMS_PREDICTIONS
        ##TODO: add a setting parameters fuction to choose prompt type
        self.prompt_type = 'ZeroShot'
    
    def load_llms_preds(self):
        tsv_files = glob.glob(os.path.join(self.llms_pred_path, "*"+ self.prompt_type + ".tsv"))
        # tsv_files = glob.glob(os.path.join(self.llms_pred_path, "*.tsv"))
        df_llms = [pd.read_csv(tsv_file, sep='\t', index_col=0) for tsv_file in tsv_files]
        self.llms_preds_df = pd.concat(df_llms, axis=1)
        
    def rename_columns(self):
        self.llms_preds_df.columns = [col.split('-', 2)[0].upper() + '-' + col.split('-', 2)[1] for col in self.llms_preds_df.columns]
        
    def concat_aggregation_methods_and_llms(self):
        self.df = pd.concat([self.df, self.llms_preds_df], axis=1)

    def main(self):
        self.load_llms_preds()
        self.rename_columns()
        super().load_data()
        self.concat_aggregation_methods_and_llms()
        super().histogram_s()
        super().process_data()
        super().multi_data_barplot()
        super().multi_data_boxplot()
        super().multi_data_violin()
        
        
if __name__ == '__main__':
    UQV100 = UQV100_GOLD_LABELS_PLUST_LLMS_HISTOGRAM()
    UQV100.main()