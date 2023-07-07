import numpy as np
import torch
import torch.nn as nn
from info import *

def preprocess(df):
    timeseries = df.iloc[:, :8].values
    timeseries = np.reshape(timeseries, (shots, sensors, signal_length))
    
    tabular = df.iloc[:,9:36]
    tabular.drop(columns=drop_columns, inplace=True)
    tabular = tabular.drop_duplicates().values
    
    
    temp = df.iloc[:, 10].values
    y = np.array([temp[i] for i in range(0, len(temp), signal_length)])
    y = y/100
    
    return timeseries, tabular, y

class FlatGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_first):
        super(FlatGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first)
    
    def forward(self, x):
        out, _ = self.gru(x)
        return out.view(out.shape[0], -1)

class FILAB_multimodal(nn.Module):
    def __init__(self):
        super(FILAB_multimodal, self).__init__()

        self.tabular_layer = nn.Sequential(
            nn.Linear(tabular[0], tabular[1]),
            nn.BatchNorm1d(tabular[1]),
            nn.ReLU(),
            nn.Linear(tabular[1], tabular[0]),
            nn.BatchNorm1d(tabular[0]),
            nn.ReLU(),
            nn.Linear(tabular[0], tabular[2]),
            nn.BatchNorm1d(tabular[2]),
            nn.ReLU() # (batch size, 16,)
        )

        self.timeseries_cnn = nn.Sequential(
            nn.Conv1d(in_channels=cnn[0], out_channels=cnn[1], kernel_size=cnn[2]),
            nn.BatchNorm1d(cnn[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=cnn[1], out_channels=cnn[3], kernel_size=cnn[2]),
            nn.BatchNorm1d(cnn[3]),
            nn.ReLU(),
            nn.Conv1d(in_channels=cnn[3], out_channels=cnn[4], kernel_size=cnn[2]),
            nn.BatchNorm1d(cnn[4]),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1) #  -> squeeze(-1) 해줘야 됨.
        )

        self.timeseries_rnn = nn.Sequential(
            FlatGRU(input_size=signal_length, hidden_size=rnn[0], num_layers=rnn[1], batch_first=rnn[2]),
            nn.LayerNorm(rnn[3]),
            FlatGRU(input_size=rnn[3], hidden_size=rnn[4], num_layers=rnn[1], batch_first=rnn[2]),
            nn.LayerNorm(rnn[4]),
            nn.Linear(rnn[4], rnn[5]),
            nn.ReLU(),
            nn.Linear(rnn[5], rnn[4])
        )

        self.calculator = nn.Sequential(
            nn.Linear(cal[0], cal[1]),
            nn.ReLU(),
            nn.Linear(cal[1], cal[2]),
            nn.ReLU(),
            nn.Linear(cal[2], cal[3])
        )
        
    def forward(self, x_timeseries, x_tabular):
        x1 = self.tabular_layer(x_tabular)
        x2 = self.timeseries_cnn(x_timeseries)
        x2 = x2.squeeze(-1)
        x3 = self.timeseries_rnn(x_timeseries)
        
        x = torch.cat((x1, x2, x3), dim=-1)
        
        x = self.calculator(x)
        
        return x.squeeze(-1)