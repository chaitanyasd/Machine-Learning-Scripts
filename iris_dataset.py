from numpy import zeros, ones, array, mean, std, arange
import matplotlib.pyplot as plt
from pylab import plot, show, xlabel, ylabel
from sklearn import datasets

def normalize(X):
	'''
	MEAN NORMALIZATION
	Xi = (Xi - MEAN) / STANDARD_DEVIATION
	'''
	mean_r = [] 							# holds all means for each Xi
	std_r = []								# holds all standard deviation for Xi
	X_norm = X
	no_columns = X.shape[1]					# no of columns
	for i in range(no_columns):
		m = mean(X[:, i]) 					# Ui
		s = std(X[:, i])  					# Si
		mean_r.append(m)
		std_r.append(s)
		X_norm[:, i] = (X_norm[:, i] - m) / s 		
	return X_norm, mean_r, std_r

def cost(X, Y, theta):
	m = Y.size
	predictions = X.dot(theta)				# find h(Xi)
	sqErrors = (predictions - Y)**2			# find (h(Xi)-y)^2
	J = (1.0 / (2 * m)) * sqErrors.sum()	# find cost = (alpha/m)*[sum( (h(Xi)-y)^2 )]
	#print(J)
	return J

def gradient_descent(X, Y, theta, alpha, iterations):
	# X.shape = 3 as X[:,0] = 1
	m = Y.size
	JHist = zeros((iterations,1))
	for i in range(iterations):
		predictions = X.dot(theta)    		# h(x)
		theta_size = theta.size
		for j in range(theta_size):
			temp = X[:, j]
			temp.shape = (m, 1)
			error = (predictions - Y) * temp
			theta[j][0] = theta[j][0] - alpha*(1.0/m)*error.sum()
		JHist[i,0] = cost(X, Y, theta)
	return theta, JHist

data = datasets.load_iris().data

X = data[:,:3]
Y = data[:,3]
m = Y.size
Y.shape = (m,1)

x, mean_r, std_r = normalize(X)

it = ones((m,4))
it[:,1:4] = x

iterations = 10000
alpha = 0.01
theta = zeros((4,1))
theta, J_history = gradient_descent(it, Y, theta, alpha, iterations)
print ("THETA - \n",theta)
print ("\nCOST - \n",cost(it,Y,theta))

plot(arange(iterations), J_history)
xlabel('Iterations')
ylabel('Cost Function')
show()