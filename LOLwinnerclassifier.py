#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import export_graphviz
from six import StringIO 
from IPython.display import Image 
from sklearn import neighbors
from sklearn.svm import SVC
import datetime
import pydotplus
import os 
iris = pd.read_csv('C://Users//Thinkpad//Desktop//new_data.csv')



# In[7]:


X = iris.drop(['gameId','creationTime','seasonId','winner'], axis=1).values
y = iris['winner'].values
min_max_scaler = preprocessing.MaxAbsScaler()
X = min_max_scaler.fit_transform(X)
X= torch.FloatTensor(X)
y = torch.LongTensor(y)

iris = pd.read_csv('C://Users//Thinkpad//Desktop//test_set.csv')
new_feature = iris.drop(['gameId','creationTime','seasonId','winner'], axis=1).values
new_label = iris['winner'].values
new_feature = min_max_scaler.fit_transform(new_feature)
new_feature = torch.FloatTensor(new_feature)
new_label = torch.LongTensor(new_label)

starttime_ANN = datetime.datetime.now()
#long running
#do something other
class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=17, out_features=20)
        self.fc2 = nn.Linear(in_features=20, out_features=25)
        self.fc3 = nn.Linear(in_features=25, out_features=25)
        self.output = nn.Linear(in_features=25, out_features=4)
    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = self.output(x)
        x = F.softmax(x)
        return x
model = ANN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
epochs = 500
loss_arr = []
for i in range(epochs):
    y_hat = model.forward(X)
    loss = criterion(y_hat, y)
    loss_arr.append(loss)
    if i % 5== 0:
        print(f'Epoch: {i} Loss: {loss}')
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
new_out = model(new_feature)
new_y = torch.max(new_out, 1)
print("The accuracy of ANN model is: ", accuracy_score(new_label, new_y[1]))
endtime_ANN = datetime.datetime.now()
print("The time used by training the ANN model is ",endtime_ANN - starttime_ANN)


# In[25]:


starttime_DT = datetime.datetime.now()
DT = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort=False)
# Train Decision Tree Classifer
DT = DT.fit(X,y)
#Predict the response for test dataset
y_new_pred = DT.predict(new_feature)
print("The accuracy of DT model is: ",accuracy_score(new_label, y_new_pred))
endtime_DT = datetime.datetime.now()
print("The time used by training the DT model is ",endtime_DT - starttime_DT)


# In[6]:


feature_cols = ['gameDuration', 'firstBlood', 'firstTower','firstInhibitor','firstBaron','firstDragon','firstRiftHerald','t1_towerKills','t1_inhibitorKills','t1_baronKills','t1_dragonKills','t1_riftHeraldKills','t2_towerKills','t2_inhibitorKills','t2_baronKills','t2_dragonKills','t2_riftHeraldKills']
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
# Configure environment variables
dot_data = StringIO()
export_graphviz(DT, out_file=dot_data, filled=True, rounded=True,special_characters=True,feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue()) 
graph.write_png('C://Users//Thinkpad//Desktop//diabetes.png')
Image(graph.create_png())


# In[31]:


starttime_SVM = datetime.datetime.now()
SVM = SVC()
SVM.set_params(kernel='linear', gamma='scale',probability=True).fit(X, y) 
A_svm=SVM.predict(new_feature)
print("The accuracy of SVM model is: ",accuracy_score(new_label, A_svm))
endtime_SVM = datetime.datetime.now()
print("The time used by training the SVM model is ",endtime_SVM - starttime_SVM)


# In[41]:


starttime_knn= datetime.datetime.now()
knn = neighbors.KNeighborsClassifier(n_neighbors=4,weights='uniform',algorithm='brute',p=1)
knn.fit(X, y)   #Training the data set
A_knn=knn.predict(new_feature)     #Predict
print("The accuracy of knn model is: ",accuracy_score(new_label, A_knn))
endtime_knn = datetime.datetime.now()
print("The time used by training the knn model is ",endtime_knn - starttime_knn)


# In[36]:


starttime_MLP = datetime.datetime.now()
MLP = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 5), random_state=1,activation='relu',max_iter=100)
MLP.fit(X, y)
A_mlp=MLP.predict(new_feature)
print("The accuracy of MLP model is: ",accuracy_score(new_label, A_mlp))
endtime_MLP = datetime.datetime.now()
print("The time used by training the MLP model is ",endtime_MLP - starttime_MLP)


# In[ ]:




