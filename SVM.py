#!/usr/bin/python
import openpyxl as opxl
import numpy as np
import sys
import matplotlib.pyplot as plt
import optunity
#import optunity.metrics
import sklearn.svm
 
#import matplotlib as plt
def computeBinary(difference,initial):	 
	m = len(difference)
	binary = []
	for i in range(m):
		# if 33% increase or 6ml increase we classify as growth
		if (difference[i] > 6) or (difference[i] > initial[i]*0.33):
			binary.append(1)
		else:
			binary.append(0)
	return binary		
		
def accuracy(binary):
	samples = float(len(binary))
	binary = np.absolute(binary)
	incorrect = np.sum(binary)
	correct = len(binary) - incorrect	
	accuracy = correct/samples
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
            #print("convergence criterion is satisfied")
            pass
    return(J_history,theta)
###################################################### 

       
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
        cell_idx = 'M'+str(y)
        ICH_0hr = float(std_sheet[cell_idx].value)
        cell_idx = 'AO'+str(y)
        ICH_24hr = float(std_sheet[cell_idx].value)

        diff = ICH_24hr-(ICH_0hr)
        initial_volumes.append(ICH_0hr)
        end_volumes.append(ICH_24hr)
        y_zero.append(diff)
    
    
#print(matrix)
initial_volumes = np.matrix(initial_volumes)
initial_volumes = np.transpose(initial_volumes)
end_volumes = np.matrix(end_volumes)
end_volumes = np.transpose(end_volumes)
y_zero = np.matrix(y_zero)
y_zero = np.transpose(y_zero)
y_binary = computeBinary(y_zero,initial_volumes)
#print(y_binary)
# print(matrix.shape)

'''alpha=0.0000001
num_iters=10000'''
ones = np.full((65,1),1)
X_matrix = np.hstack((ones,X_matrix))
theta = np.full((25,1),1)
Cs = np.arange(100000.0)
Cs = Cs/100
min = 100
minC = 0




# score function: twice iterated 10-fold cross-validated accuracy
@optunity.cross_validated(x=X_matrix, y=y_binary, num_folds=4, num_iter=2)
def svm_auc(x_train, y_train, x_test, y_test, logC, logGamma):
    model = sklearn.svm.SVC(C=10 ** logC, gamma=10 ** logGamma,kernel='rbf').fit(x_train, y_train)
    decision_values = model.decision_function(x_test)
    return optunity.metrics.roc_auc(y_test, decision_values)

# perform tuning
hps, _, _ = optunity.maximize(svm_auc, num_evals=200, logC=[-5, 10], logGamma=[-100, 100])


# train model on the full training set with tuned hyperparameters
optimal_model = sklearn.svm.SVC(C=10 ** hps['logC'], gamma=10 ** hps['logGamma']).fit(X_matrix, y_binary)

output = []
ourguess = []
for j in range(65):
		
			test = optimal_model.predict(X_matrix[j,:])
			ourguess.append(test[0])
			
ourguess = np.array(ourguess)
binary = []
for i in range(len(ourguess)):
	binary.append(ourguess[i]-y_binary[i])
print(binary)
percent = accuracy(binary)
print(percent)

