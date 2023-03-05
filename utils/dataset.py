import torch as th
from torch.utils.data import Dataset, DataLoader

class LinearClassificationDataset(Dataset):
  def __init__(self, X,y):
    self.X = th.from_numpy(X)
    self.y = th.from_numpy(y)
    # self.X = th.from_numpy(X)
    # self.y = th.from_numpy(y.to_numpy())

  def __len__(self):
      return len(self.X)    

  def __getitem__(self, idx):
      # label = self.data[idx,-1]          
      # data = self.data[idx,:-1]
      y = self.y[idx]          
      X = self.X[idx]
      return X,y.type(th.LongTensor)
  

class ConvClassificationDataset(Dataset):
  def __init__(self, X,y):
    self.X = X
    self.y = y
    # self.X = th.from_numpy(X)
    # self.y = th.from_numpy(y.to_numpy())

    
  def __len__(self):
      return len(self.X)    

  def __getitem__(self, idx):
      # label = self.data[idx,-1]          
      # data = self.data[idx,:-1]
      y = self.y[idx]          
      X = self.X[idx]
      return X,y.type(th.LongTensor)
    
class TimeseriesDataset(Dataset):   
    # def __init__(self, X, y, seq_len=1):
    def __init__(self, X, seq_len=1):
        self.X = X
        # self.y = y
        self.seq_len = seq_len

    def __len__(self):
        # return self.X.__len__() - (self.seq_len-1)
        return self.X.__len__() - (self.seq_len)

    def __getitem__(self, index):
        # return self.X[index:index+self.seq_len], self.y[index+self.seq_len-1]
        # return self.X[index:index+self.seq_len].type(th.float), self.X[index+self.seq_len-1].type(th.float)
        return self.X[index:index+self.seq_len].type(th.float), self.X[index+self.seq_len].type(th.float)
        # return self.X[index:index+self.seq_len], self.y[index:index+self.seq_len]  

def dataloader(data_train,data_val,data_test,batch_size,shuffle = True, drop_last = False):
    train_dataloader = DataLoader(data_train, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    test_dataloader = DataLoader(data_test, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    val_dataloader = DataLoader(data_val, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)   

    train_features, train_y = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Regression batch shape: {train_y.size()}")
    return train_dataloader,val_dataloader,test_dataloader,train_features.size()

