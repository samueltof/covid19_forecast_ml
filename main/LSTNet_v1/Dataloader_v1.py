import torch
from torch.utils.data import Dataset

class MarketDataset(Dataset):
    
    def __init__(self, data, history_len):
        self.data = data
        self.history_len = history_len
        
    def __len__(self):
        self.len = len(self.data) - self.history_len  
        return self.len
    
    def __getitem__(self, index):
        x_cols = ['Open-N', 'High-N', 'Low-N', 'Close-N', 'Volume-N']
        y_cols = ['Close-N']
        x = self.data.iloc[index: index+self.history_len, :][x_cols].values
        y = self.data.iloc[index+self.history_len, :][y_cols].values.astype('float')
        x = torch.tensor(x).float()
        y = torch.tensor(y).float()
        return x, y