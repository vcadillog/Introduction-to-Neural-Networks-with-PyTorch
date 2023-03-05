import torch as th
import torch.nn as nn


class LinearClassification(nn.Module):
  def __init__(self, input_dim: int, 
               output_dim: int) -> None:
    super(LinearClassification, self).__init__()
    
    self.input_to_hidden = nn.Linear(input_dim, 128)
    self.hidden_layer_1 = nn.Linear(128, 64)
    self.hidden_layer_2 = nn.Linear(64, 32)    
    self.hidden_to_output = nn.Linear(32, output_dim)    
    self.relu = nn.ReLU()    
    self.dropout = nn.Dropout(p=0.2)

    for m in self.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data)
                
  def forward(self, x: th.Tensor) -> th.Tensor:   
    x = self.input_to_hidden(x)
    x = self.relu(x)
    x = self.dropout(x)
    x = self.hidden_layer_1(x)
    x = self.relu(x)
    x = self.dropout(x)
    x = self.hidden_layer_2(x)
    x = self.relu(x)
    x = self.dropout(x)        
    x = self.hidden_to_output(x)        
    return x  

class ConvClassification(nn.Module):
  def __init__(self, input_dim: int, 
               output_dim: int) -> None:
    super(ConvClassification, self).__init__()    
    
    self.input_to_hidden = nn.Conv2d(input_dim , 32 , kernel_size = (3,3) , stride = (1,1))
    self.pool = nn.MaxPool2d(3, stride = 2)
    self.cnn1 = nn.Conv2d(32 , 64 , kernel_size = (3,3) , stride = (1,1))
    self.fc1 = nn.Linear(64*4*4, 32)
    self.fc2 = nn.Linear(32, 16)
    self.hidden_to_output = nn.Linear(16, output_dim)         
    self.relu = nn.ReLU()    
    self.dropout = nn.Dropout(p=0.2)

    for m in self.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data)


  def forward(self, x: th.Tensor) -> th.Tensor:        
    x = self.input_to_hidden(x)
    x = self.relu(x)
    x = self.pool(x) 
    x = self.cnn1(x)    
    x = self.relu(x)
    x = self.pool(x)    
    x = th.flatten(x, 1)
    x = self.fc1(x)    
    x = self.relu(x)
    x = self.dropout(x)
    x = self.fc2(x)    
    x = self.relu(x)
    x = self.dropout(x)    
    x = self.hidden_to_output(x)      
    return x    

class RecurrentRegression(nn.Module):
  def __init__(self, input_dim: int, seq_len: int,device) -> None:
    super(RecurrentRegression, self).__init__()
    self.hidden_size = 128
    self.num_layers = 2
    self.input_to_hidden = nn.RNN(input_dim,num_layers = self.num_layers,hidden_size = self.hidden_size, batch_first = True)    
    self.fc1 = nn.Linear(self.hidden_size*seq_len, 64)
    self.fc2 = nn.Linear(64, 32)
    self.hidden_to_output = nn.Linear(32, input_dim)    
    self.tanh = nn.Tanh()    
    self.dropout = nn.Dropout(p=0.2)

    for m in self.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data)

    self.h_0 = th.zeros([self.num_layers,64,self.hidden_size]).to(device)
                
  def forward(self, x: th.Tensor) -> th.Tensor:    
    
    x , self.h_0 = self.input_to_hidden(x,self.h_0)       
    x  = x.reshape(x.size(0),-1)
    x = self.fc1(x)
    x = self.tanh(x)
    x = self.dropout(x)
    x = self.fc2(x)
    x = self.tanh(x)
    x = self.dropout(x)
    x = self.hidden_to_output(x)        
    return x  

