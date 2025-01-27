import torch.nn as nn
import pandas as pd


def check_parameters(net):
    '''
        Returns module parameters. Mb
    '''
    parameters = sum(param.numel() for param in net.parameters())
    return parameters / 10**6


def concate_two_csv (f_csv, s_csv, pth_to_save):
    df1 = pd.read_csv(f_csv)
    df2 = pd.read_csv(s_csv)
    combined_df = pd.concat([df1, df2], ignore_index=True)
    combined_df.to_csv(pth_to_save, index=False)
    print(f"Concated in {pth_to_save}")