# Introduction-to-Neural-Networks-with-PyTorch
A brief introduction to basic neural networks, Fully Connected, Convolutional and Recurrent using PyTorch library.

To use the code download the jupyter notebook in your local machine or in Google Colab.

The code contains 3 neural networks in the utils/networks module:
1. Network with Fully connected layers to classify numerical data.
2. Network with Convolutional layers to classify MNIST rotated image data.
3. Network with Recurrent layers to do multivariate forecasting of weather data.

This work also contains saved models of the 3 neural networks, to load the model run all the code specific to a network e.g. for a Recurrent Network (RNN) run all the code and skip the training loop and saving model cells.
- RNN_Regression.train()
- th.save(RNN_Regression.model.state_dict(), 'models/rnn-regression.pth')

The log data can be visualized in a Tensorboard.

Fully connected network:

![Alt text](https://github.com/vcadillog/Introduction-to-Neural-Networks-with-PyTorch/blob/main/images/DNN_loss.png)
![Alt text](https://github.com/vcadillog/Introduction-to-Neural-Networks-with-PyTorch/blob/main/images/DNN_acc.png)

- Cian: Validation 
- Gray: Training 

Convolutional network:

![Alt text](https://github.com/vcadillog/Introduction-to-Neural-Networks-with-PyTorch/blob/main/images/CNN_loss.png)
![Alt text](https://github.com/vcadillog/Introduction-to-Neural-Networks-with-PyTorch/blob/main/images/CNN_acc.png)

- Orange: Validation
- Magenta: Training

Recurrent network:

Successful forecast feature:
![Alt text](https://github.com/vcadillog/Introduction-to-Neural-Networks-with-PyTorch/blob/main/images/RNN_f10.png)

Unsuccessful forecast feature:
![Alt text](https://github.com/vcadillog/Introduction-to-Neural-Networks-with-PyTorch/blob/main/images/RNN_f11.png)

- Orange: Test
- Gray: Predicted


[1] Weather data extracted from: https://github.com/cure-lab/LTSF-Linear
 
