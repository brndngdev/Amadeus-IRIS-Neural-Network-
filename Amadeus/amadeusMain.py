import torch 
import torch.nn as nn # importing torch neural network lirary
import torch.nn.functional as F # lets me move data forward for my later forward function)
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


# model class that inherits nn module (comes with torch)
# we're gonna create a class for model and instantiate that class pretty much

class Model(nn.Module):
    # Input layer (4 features of the flower in the IRIS database)
    # -> Hidden layer 1 (number of neurons)
    # -> Hidden layer 2 (n)
    # -> Output layer (3 classes of IRIS flowers)

    def __init__(self, in_features = 4 , h1 = 6, h2 = 8, out_features = 3):
        super().__init__() # instantiates nn.module
        self.fc1 = nn.Linear(in_features, h1) # fc1 means fully connected, we're moving from input layer to layer 1 (in_features, h1)
        self.fc2 = nn.Linear(h1, h2) # moving from fully connected layer 1 of the neural network to 2 (h1, h2)
        self.out = nn.Linear(h2, out_features) # moving from layer 2 to the output layer (h2, out_features)

    # we need a forward function to move forward between layers

    def forward(self, x): # pushing between layers
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.out(x)

        return x

# picking a seed for randomisation purposes
torch.manual_seed(69)
# instantiates model
model = Model()

url = "https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv"
dataframe = pd.read_csv(url) # reads the comma seperated value dataframe
print(dataframe) # .head() gives 5 five things and .tail() gives last 5 , print(dataframe.to_string()) gives the entire database as a string object
# changes last column from string to int
dataframe["variety"] = dataframe["variety"].map(
    {"Setosa": 0, "Versicolor": 1, "Virginica": 2}
)

X = dataframe.drop("variety", axis=1).to_numpy()
y = dataframe["variety"].to_numpy()

# train test splitting (the reason we changed variety from a string to an integer)

X = dataframe.drop("variety", axis = 1).to_numpy() # dropping the variety column and converting to numpy array
y = dataframe["variety"].to_numpy() # making the variety column a numpy array

print(X)
print(y)

# makes training data 80% / 0.8 and makes the testing data 0.3 / 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=69) # random split makes sure its the same every time i run it

# converting to tensor data type (faster for deep learning)
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

# setting criteria of accuracy
criteria = nn.CrossEntropyLoss()
# choosing learning rate
optimiser = torch.optim.Adam(model.parameters(), lr = 0.01) # the lower the learning rate the longer the model takes.

# 1 epoch is 1 run through all the training data in the dataset
epochs = 100
losses = [] # append losses to list
for i in range(epochs):
    y_prediction = model.forward(X_train) # get predicted result

    # measure loss/error 
    loss = criteria(y_prediction, y_train) # pytorch calculates how far my predictions are from the correct answers AND how it was calculated
    # append a loss onto the loss list, will graph with mpl laterr
    losses.append(loss.detach().numpy()) # use the detatch() function because pytorch also tracks how the loss was calculated, which isnt needed in the lossess array, so using detatch() makes a disconnected tensor

    # print every 10 epochs
    if i % 10 == 0:
        print("Epoch: {} and loss: {}".format(i, loss))

    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

# =================================================================================================================================
# QUICK SIDE TANGENT TO UNDERSTAND KEY CONCEPTS
# =================================================================================================================================
# Weight: How much influence an input has on a neuron's output
# >> Our neural network takes sepal length and widthm and petal length and width.
# >> A neuron takes each input and assigns a weight, then multiplies itby a weight, adds them together and adds a bias.
# >> Weights initially are random by PyTorch but later the optimiser (Adam optimiser) adjusts them and the ml algo gets better
#
# > Gradient:
# >> A gradient tells our neural network which direction a weight should change to maximally reduce loss
# >> It also tells us how sensitive the loss is to change
# >>>> So sometimes if you change the weight, you change the loss too.
# >>>> If weight is 0.8 and loss is 10, and you change weight to 0.9 and loss is 9, youve decreased loss which is useful. 
# >>>> Gradient measures this
#
# Propagation:
# >> There are Two types of propagation, forward and backward propagation.
# >> Propagation is just how data travels throughout the neural network
#
# > Forward Propagation:
# >>> Input -> Hidden layers -> Output
# >>> So in line 76 when we call the forward function we perform forward propagation
#
# > Backward Propagation:
# >>> After Examining how wrong the predictions are, the network goes through its calculations to determine HOW weight affects the loss.
# >>> This process is called BACKPROPAGATION.
# 
# >> Backpropagation
# >>>> Backpropagation is the process that calculates gradients
# >>>> Calculates gradients from the loss and weights
#
# ====================================================================================================================================

plt.plot(range(epochs), losses)
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.title("Training Loss")
plt.show()