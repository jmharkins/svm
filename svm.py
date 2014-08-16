import numpy as np
# define h, which calculates the transpose of theta times x for a given
# theta vector and x vector
def h(theta, x):
	# if np.transpose(theta) * x >= 0:
	# 	return 1
	# else:
	# 	return 0
	return theta.T * x

# helper function which calculates the "cost 0" and "cost 1" functions
# as specified in the video, and returns them as a tuple or row vector
def costhelp(h):
	cost1 = -log(h)
	cost0 = -log(1-h)
	return np.mat([[cost0, cost1]])

# define cost function which takes c, theta: a 3x1 column vector of candidate
# points for a decision boundary, and the data, which is given as a 3xn matrix
# containing 1, x_1 and x_2 for n observations, and y, a nx1 column vector of 
# classes corresponding to the x observations.
def costfn(c,theta,x,y):
	#calculate first term (c times summation)
	cost = 0;
	for i in range(0, x.shape[1]):
		# calculate cost0 and cost1 for x^(i) and the given theta
		cterms = costhelp(h(theta, x[:,i]))
		cost = cost + (y[i,:] * cterms[1]) + ((1 - y[i,:]) * cterms[0])
	cost = c * cost;
	# calculate second term (sum of theta_j squared for all j)
	finalterm = 0;
	for j in range(0, x.shape[0]):
		finalterm = finalterm + (theta[j,:])^2
	finalterm = finalterm * 0.5
	cost = cost + finalterm
	return cost