
import math
import numpy as np
import csv
from scipy.optimize import minimize

class Logistic:
	def __init__(self, dataFrame):
		self.X = dataFrame[:,0:-1]
		self.Y = dataFrame[:,-1]
		self.dataFrame = dataFrame
		self.nrows = float(dataFrame.shape[0])
		self.X = np.append(self.X, np.ones((self.nrows,1)), 1)
		self.ncols = self.X.shape[1]

	def sigmoid(self, x):
		return(1/(1+np.exp(-x)))

	def cost(self, theta):
		g  = self.sigmoid(np.dot(self.X, theta)).transpose()
		J = (1/self.nrows)*np.sum(np.dot(np.log(g), -self.Y) - np.dot(np.log(1-g), (1-self.Y)))
		return J
	#def main():
	#	print(sigmoid(5))
f = open("../data/data.csv","r")
data = np.loadtxt(f, skiprows=1, delimiter=",")
log = Logistic(data)
theta = np.zeros(log.ncols)
print(log.cost(theta))
res = minimize(log.cost, theta, method = "Nelder-Mead")
print(res)