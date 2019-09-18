#!/usr/bin/python
import openpyxl as opxl
import numpy as np
import sys
from sklearn.model_selection import KFold
#import matplotlib as plt
def computeBinary(difference,initial):	 
	m = len(difference)
	binary = np.zeros((m,1))
	for i in range(m):
		# if 33% increase or 6ml increase we classify as growth
		if (difference[i] > 6) or (difference[i] > initial[i]*0.33):
			binary[i] = 1.0
	return binary		
		
def accuracy(binary):
	binary = np.absolute(binary)
	incorrect = np.sum(binary)
	correct = len(binary) - incorrect
	accuracy = correct/len(binary)
	return accuracy
		
		
	

def computeCost (X,y,theta):
    m = len(y)
    p1 = np.matmul(X,theta)-y
    p2 = p1.transpose()
    J=np.matmul(p2,p1)/(2*m)
        # print("J: ",J)
    return J
#############################################
def gradient_descent(X,y,theta,alpha,num_iters):
    m = len(y)
    J_history=np.zeros((num_iters,1))
    for i in range(0,num_iters):
        temp=np.matmul(X,theta)
        #print(temp, temp.shape)
        error=temp-y
        new_X=np.matmul(X.transpose(),error )
        theta_temp= ( (alpha/m)*new_X )
        #print(theta_temp)
        #print(theta_temp, "test")
        theta= theta-theta_temp;
    
        J_history[i][0]=computeCost (X,y,theta)
    
    if i!=0 and i!=1:
        if J_history[i][0]-J_history[i-1][0]< alpha: 
            print("convergence criterion is satisfied")
    return(J_history,theta)
###################################################### 
splits = 5      
kfold = KFold(splits, True, 1)  
       
np.set_printoptions(threshold=sys.maxsize,precision = 2,linewidth=200)
MAXROW = 76
SELECTED = ['E','D','F','G','H','K','L','N','Q','R','S','T','U','V','W','X','Y','Z','AA','AB','AC','AD','AE','AZ']
std_col = ['G','J','M','AI','AL','AO']
wb = opxl.load_workbook("linear_table_edited.xlsx")
data_set = wb['Sheet3']
std_sheet = wb['Sheet4']
X_matrix = []
for y in range(2,MAXROW+1):
    row = []
    if y!=8 and y!=13 and y!=23 and y!=32 and y!=44 and y!=50 and y!=53 and y!=55 and y!= 58 and y!= 67:
        for x in SELECTED:
            #print(x+str(y))
            cell_idx = x+str(y)
            # print(cell_idx)
            row.append(float(data_set[cell_idx].value))
        # print(row)
        X_matrix.append(row)

X_matrix = np.matrix(X_matrix)
y_zero = []
initial_volumes = []
end_volumes = []
for y in range(2,MAXROW+1):
    if y!=8 and y!=13 and y!=23 and y!=32 and y!=44 and y!=50 and y!=53 and y!=55 and y!= 58 and y!= 67:
        """
        cell_idx = 'G'+str(y)
        IPH_0hr = float(std_sheet[cell_idx].value)
        cell_idx = 'J'+str(y)
        IVH_0hr = float(std_sheet[cell_idx].value)
        """
        cell_idx = 'M'+str(y)
        ICH_0hr = float(std_sheet[cell_idx].value)
        """
        cell_idx = 'AI'+str(y)
        IPH_24hr = float(std_sheet[cell_idx].value)
        cell_idx = 'AL'+str(y)
        IVH_24hr = float(std_sheet[cell_idx].value)
        """
        cell_idx = 'AO'+str(y)
        ICH_24hr = float(std_sheet[cell_idx].value)

        diff = ICH_24hr-(ICH_0hr)
        initial_volumes.append(ICH_0hr)
        end_volumes.append(ICH_24hr)
        y_zero.append(diff)
        # print("row: ",y)
        # print(diff)
        # cell_idx = 'G'+str(y)
        # print(float(std_sheet[cell_idx].value))
#print(matrix)
initial_volumes = np.matrix(initial_volumes)
initial_volumes = np.transpose(initial_volumes)
end_volumes = np.matrix(end_volumes)
end_volumes = np.transpose(end_volumes)
y_zero = np.matrix(y_zero)
y_zero = np.transpose(y_zero)
# print(matrix.shape)

# enumerate splits
split_data = kfold.split(X_matrix)
alpha=0.0000001
num_iters=10000
ones = np.full((65,1),1)
X_matrix = np.hstack((ones,X_matrix))
all_thetas = []
iter = 0
# train is a set of current training index
# test is a set of current testing indec
for train, test in split_data:
	iter += 1
	current_theta = np.full((25,1),1)
	current_X = np.matrix(X_matrix)
	current_X = current_X[train,:]	# data set of selected training data
	current_Y = y_zero[train,:]		# difference for selected training data
	test_X = np.matrix(X_matrix)
	test_X = test_X[test,:]		# data set of selected testing data
	test_Y = y_zero[test,:]		# difference for selected testing data
	(J_history,current_theta)= gradient_descent(current_X,current_Y,current_theta,alpha,num_iters)
	all_thetas.append(current_theta)
	test_difference = np.matmul(test_X,current_theta)
	original_binary = computeBinary(y_zero[test,:],initial_volumes[test,:])	# classification using given data
	learned_binary = computeBinary(test_difference,initial_volumes[test,:])
	percent = accuracy(original_binary-learned_binary)
	message = "%.2f%% accuracy on the training data" % percent	# accuracy for each iteration
	print(iter, "iteration")
	print(message)
	
#print("Y dimension:",y_zero.shape)
average_thetas = np.zeros((len(all_thetas[0]),1))
for i in range(splits):
	for j in range(len(all_thetas[0])):
		average_thetas[j] = average_thetas[j]+ all_thetas[i][j,0]
average_thetas /= splits		# cross validates among 5 splits

final_difference = np.matmul(X_matrix,average_thetas)
final_original_binary = computeBinary(y_zero,initial_volumes)	# classification using given data
final_learned_binary = computeBinary(final_difference,initial_volumes)
percent = accuracy(final_original_binary-final_learned_binary)
message = "%.2f%% accuracy on the all data" % percent	# accuracy for each iteration
print("\nfinal iteration")
print(message)

