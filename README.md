# Amadeus-IRIS-Neural-Network-
A small neural network I produced that can visualise its data using mpl and uses pandas and pytorch for data analysis.

The project files also contain a little overview on key neural network/ machine learning/ deep learning concepts like weight, propagation and backpropagation etc.
It pulls its dataframe form an online database on IRIS flowers and their attributes!

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
