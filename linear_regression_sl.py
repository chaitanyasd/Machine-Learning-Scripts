# Linear regression with one variable
from numpy import loadtxt, zeros, ones, array
from pylab import scatter, show, title, xlabel, ylabel, plot

def cost(X, Y, theta):
	'''Compute cost'''
	m = Y.size
	# find h(x)-y
	predictions = X.dot(theta).flatten() #convert to row major representation
	sqrErrors = (predictions - Y) ** 2
	# print(predictions,"\n",Y,"\n",sqrErrors)
	J = (1/(2.0*m)) * sqrErrors.sum()
	return J

def gradient_descent(X, Y, theta, alpha, iterations):
	'''gradient descent'''
	m = Y.size
	J_history = zeros((iterations, 1))
	for i in range(iterations):
		predictions = X.dot(theta).flatten()
		t1 = (predictions - Y) * X[:,0]
		t2 = (predictions - Y) * X[:,1]

		theta[0][0] = theta[0][0] - alpha*(1.0/m)*t1.sum()
		theta[1][0] = theta[1][0] - alpha*(1.0/m)*t2.sum()

		J_history[i, 0] = cost(X, Y, theta) 
	return theta, J_history


data = loadtxt("data.txt", delimiter=",")
#print (data)

scatter(data[:,0], data[:,1], marker='x')
title('Price Distribution')
xlabel('Size of house')
ylabel('Price')
#show()

X = data[:, 0]
Y = data[:, 1]
m = Y.size

it = ones((m,2))
it[:, 1] = X
theta = zeros((2,1))

iterations = 1500
alpha = 0.01

#print (X,Y,theta)
print (cost(it, Y, theta))

theta, J_history = gradient_descent(it, Y, theta, alpha, iterations)
print (theta,"....\n",J_history)
print (cost(it, Y, theta))

result = it.dot(theta)
plot(data[:, 0], result, c='r')
show()
