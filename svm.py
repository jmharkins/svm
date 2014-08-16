import numpy as np
theta = np.mat([[-3, 0, 1]])
data = 
def h(theta, x):
	# if np.transpose(theta) * x >= 0:
	# 	return 1
	# else:
	# 	return 0
	return theta.T * x
def costhelp(h):
	cost1 = -log(h)
	cost2 = -log(1-h)
	return np.mat([[cost1, cost2]])

