import torch
from torch import nn
import numpy as np
import pandas as pd
from model import NeuralNetwork
from sklearn.model_selection import train_test_split

model = NeuralNetwork()

my_df = pd.read_csv("iris.csv")
my_df["variety"] = my_df["variety"].replace("Setosa",0.0)
my_df["variety"] = my_df["variety"].replace("Versicolor",1.0)
my_df["variety"] = my_df["variety"].replace("Virginica",2.0)

X = my_df.drop("variety",axis=1) #features
Y = my_df["variety"] #labels (target)

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,shuffle=True,random_state=41)

X_train = torch.FloatTensor(X_train.values)
X_test = torch.FloatTensor(X_test.values)

Y_train = torch.LongTensor(Y_train.values)
Y_test = torch.LongTensor(Y_test.values)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)

#train
losses = []
best_loss = float("inf")
for epoch in range(1000):
    y_pred = model.forward(X_train)
    loss = criterion(y_pred,Y_train)
    losses.append(loss.item())
    
    if epoch % 10 == 0:
        print(f"Epoch : {epoch} and loss : {loss}")
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if loss.item() < best_loss :
        best_loss = loss.item()
        torch.save(model.state_dict(),"model.pt")
        
#test 
with torch.no_grad():
    y_test_pred = model.forward(X_test)
    predic = torch.argmax(y_test_pred,dim=1)
    accuracy = (predic == Y_test).sum().item()/len(Y_test)
    print(f"test accuracy {accuracy*100}")