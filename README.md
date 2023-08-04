# Programming-Test-Learning-Activations-in-Neural-Networks

Abstract—
The choice of Activation Functions (AF) has proven
to be an important factor that affects the performance of an
Artificial Neural Network (ANN). Use a 1-hidden layer neural
network model that adapts to the most suitable activation
function according to the data-set. The ANN model can learn for
itself the best AF to use by exploiting a flexible functional form,
k0 + k1 ∗ x with parameters k0, k1 being learned from multiple
runs. You can use this code-base for implementation guidelines
and help.

I. BACKGROUND

Selection of the best performing AF for classification task
is essentially a naive (or brute-force) procedure wherein, a
popularly used AF is picked and used in the network for
approximating the optimal function. If this function fails, the
process is repeated with a different AF, till the network learns
to approximate the ideal function. It is interesting to inquire
and inspect whether there exists a possibility of building a
framework which uses the inherent clues and insights from
data and bring about the most suitable AF. The possibilities
of such an approach could not only save significant time and
effort for tuning the model, but will also open up new ways
for discovering essential features of not-so-popular AFs.

II. MATHEMATICAL FRAMEWORK

A. Compact Representation

Let the proposed Ada-Act activation function be mathematically defined as:
g(x) = k0 + k1x (1)
where the coefficients k0, k1 and k2 are learned during training
via back-propagation of error gradients.
For the purpose of demonstration, consider a feed-forward
neural network consisting of an input layer L0 consisting
of m nodes for m features, two hidden layers L1 and L2
consisting of n and p nodes respectively, and an output
layer L3 consisting of k nodes for k classes. Let zi and
ai denote the inputs to and the activations of the nodes in
layer Li respectively. Let wi and bi denote the weights and
the biases applied to the nodes of layer Li−1, and let the
activations of layer L0 be the input features of the training
examples. Finally, let K denote the column matrix containing
the equation coefficients: 
k0
k1
k2

and let t denote the number of
training examples being taken in one batch. Then the forwardpropagation equations will be:
z1 = a0 × w1 + b1
a1 = g(z1)
z2 = a1 × w2 + b2
a2 = g(z2)
z3 = a2 × w3 + b3
a3 = Sof tmax(z3)
where × denotes the matrix multiplication operation and
Sof tmax() denotes the Softmax activation function.
For back-propagation, let the loss function used in this
model be the Categorical Cross-Entropy Loss, and let dfi
denote the gradient matrix of the loss with respect to the matrix
fi
, where f can be substituted with z, a, b, or w. and let there
be matrices dK2 and dK1 of dimension 3×1. Then the backpropagation equations will be:
dz3 = a3 − y
dw3 =
1
t
a
T
2 × dz3
db3 = avgcol(dz3)
da2 = dz3 × w
T
3
dz2 = g
0
(z2) ∗ da2
dw2 =
1
t
a
T
1 × dz2
db2 = avgcol(dz2)
dK2 =



avge(da2)
avge(da2 ∗ z2)
avge(da2 ∗ z
2
2
)


 (2)
da1 = dz2 × w
T
2
dz1 = g
0
(z1) ∗ da1
dw1 =
1
t
a
T
0 × dz1
db1 = avgcol(dz1)
hr@happymonk.co
dK1 =



avge(da1)
avge(da1 ∗ z1)
avge(da1 ∗ z
2
1
)


 (3)
dK = dK2 + dK1 (4)

where ∗ is the element-wise multiplication operation, T is the
matrix transpose operation, avgcol(x) is the function which
returns the column-wise average of the elements present in
the matrix x, and avge(x) is the function which returns the
average of all the elements present in the matrix x.
Consider the learning rate of the model to be α. The update
equations for the model parameters will be:

w1 = w1 − α.dw1
b1 = b1 − α.db1
w2 = w2 − α.dw2
b2 = b2 − α.db2
w3 = w3 − α.dw3
b3 = b3 − α.db3
K = K − α.dK
