from torch.utils.data import Dataset

class ModalDataset(Dataset):
    def __init__(self, timeseries_data, tabular_data, targets):
        assert len(timeseries_data) == len(tabular_data)
        self.timeseries_data = timeseries_data
        self.tabular_data = tabular_data
        self.targets = targets
        
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, index):
        return self.timeseries_data[index], self.tabular_data[index], self.targets[index]