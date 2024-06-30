import pandas as pd
from config import *
from gold_label_statistical_test import UQV100_GOLD_LABELS_STATISTICAL_TEST
from gold_labels_plus_llms_plots import UQV100_GOLD_LABELS_PLUST_LLMS_HISTOGRAM
import glob
from icecream import ic

class UQV100_GOLD_LABELS_PLUST_LLMS_STATISTICAL_TEST(UQV100_GOLD_LABELS_STATISTICAL_TEST, UQV100_GOLD_LABELS_PLUST_LLMS_HISTOGRAM):
    def __init__(self):
        super().__init__()
        UQV100_GOLD_LABELS_PLUST_LLMS_HISTOGRAM.__init__(self)
        UQV100_GOLD_LABELS_STATISTICAL_TEST.__init__(self)
        
    def main(self):
        UQV100_GOLD_LABELS_PLUST_LLMS_HISTOGRAM.load_llms_preds(self)
        UQV100_GOLD_LABELS_PLUST_LLMS_HISTOGRAM.rename_columns(self)
        UQV100_GOLD_LABELS_STATISTICAL_TEST.load_data(self)
        UQV100_GOLD_LABELS_PLUST_LLMS_HISTOGRAM.concat_aggregation_methods_and_llms(self)
        UQV100_GOLD_LABELS_STATISTICAL_TEST.process_data(self)
        UQV100_GOLD_LABELS_STATISTICAL_TEST.anova_test(self)
        UQV100_GOLD_LABELS_STATISTICAL_TEST.tukey_hsd_test(self)
        
if __name__ == '__main__':
    UQV100 = UQV100_GOLD_LABELS_PLUST_LLMS_STATISTICAL_TEST()
    UQV100.main()