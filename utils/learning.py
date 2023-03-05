import torch as th

class learning_loop():
    def __init__(self, N_EPOCH,model,device,train_data,val_data,test_data,optimizer,criterion, directory,mode ) -> None:
        self.N_EPOCH = N_EPOCH
        self.model = model
        self.device = device
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.optimizer = optimizer
        self.criterion = criterion        
        self.mode = mode     
        self.writer1=SummaryWriter('logs/'+directory,flush_secs=60)
        self.writer2=SummaryWriter('logs/'+directory,flush_secs=60)   

    def train(self):
        for epoch in range(1, self.N_EPOCH+1):
            for (_,train) , (_,test) in zip(enumerate(self.train_data, 0),enumerate(self.val_data, 0)):
                self.model.train()
                X_train_ts, y_train_ts = train
                X_val_ts, y_val_ts = test

                X_train_ts, y_train_ts = X_train_ts.to(self.device), y_train_ts.to(self.device)
                X_val_ts, y_val_ts = X_val_ts.to(self.device), y_val_ts.to(self.device)

                out = self.model(X_train_ts.to(th.float32))      
                loss = self.criterion(out, y_train_ts)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                if self.mode == 'classification':
                    acc = (th.argmax(out, dim=1) == y_train_ts).float().mean().item()
                
                self.model.eval()
                with th.no_grad():
                    out_val = self.model(X_val_ts.to(th.float32))
                    loss_val = self.criterion(out_val, y_val_ts)
                    if self.mode == 'classification':
                        acc_val = (th.argmax(out_val, dim=1) == y_val_ts).float().mean().item()
               
            self.writer1.add_scalar('Loss/train', loss.item(), epoch)
            self.writer2.add_scalar('Loss/test', loss_val.item(), epoch)  
                       
            if self.mode == 'classification':
                self.writer1.add_scalar('Accuracy/train', acc*100 , epoch)
                self.writer2.add_scalar('Accuracy/test', acc_val*100, epoch)                           

            if epoch % 20 == 0:
                    if self.mode == 'classification':
                        print('Epoch : {:3d} / {}, Loss : {:.4f}, Accuracy : {:.2f} %, Val Loss : {:.4f}, Val Accuracy : {:.2f} %'.format(
                            epoch, self.N_EPOCH, loss.item(), acc*100, loss_val.item(), acc_val*100))
                    else:
                        print('Epoch : {:3d} / {}, Loss : {:.4f},  Val Loss : {:.4f}'.format(
                            epoch, self.N_EPOCH, loss.item(), loss_val.item()))
                
        self.writer.close()        

    def test(self):
        acc_avg = 0
        loss_avg = 0
        y_list = [] # i,y_test,y_pred
        
        for (i,test) in enumerate(self.test_data, 0):
            
            X_test_ts, y_test_ts = test           
            X_test_ts, y_test_ts = X_test_ts.to(self.device), y_test_ts.to(self.device)

            
            self.model.eval()
            with th.no_grad():
                out_test = self.model(X_test_ts.to(th.float32))
                loss_test = self.criterion(out_test, y_test_ts)
                if self.mode == 'classification':
                    acc_test = (th.argmax(out_test, dim=1) == y_test_ts).float().mean().item()
            
            if self.mode == 'classification':
                acc_avg += acc_test            
            else:
                y_list.append([i,y_test_ts.numpy(),out_test.numpy()])
                
            loss_avg += loss_test
        if self.mode == 'classification':
            print('Val avg Loss : {:.4f}, Val avg Accuracy : {:.2f} %'.format(
                loss_avg/i,acc_avg/i*100))  
        else:
            print('Val avg Loss : {:.4f}'.format(
                loss_avg/i))  
            return y_list
