from model import NeuralNetwork
import numpy as np
import torch

model = NeuralNetwork()
# model.load_state_dict(torch.load("model.pt"))

input_data = sepal_length,sepal_width,petal_length,petal_width = np.array([2,1,3,4])
input_tensor = torch.FloatTensor(input_data).unsqueeze(0)

with torch.no_grad():
    y_pred = model.forward(input_tensor)
    
_,predicted = torch.max(y_pred,1)

if predicted == 0:
    print("setosa")

elif predicted == 1:
    print("versicolor")

else:
    print("virginica")