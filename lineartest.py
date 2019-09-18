#!/usr/bin/python
import openpyxl as opxl
import numpy as np
import sys

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
        print(theta)
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
    
       
np.set_printoptions(threshold=sys.maxsize,precision = 2,linewidth=200)
MAXROW = 97
SELECTED = ['A']
std_col = ['A']
wb = opxl.load_workbook("textdataHematoma.xlsx")#"linear_table_edited.xlsx")
data_set = wb['Sheet1']
std_sheet = wb['Sheet2']
X_matrix = []
for y in range(1,MAXROW+1):
    row = []
    
    for x in SELECTED:
        #print(x+str(y))
        cell_idx = x+str(y)
        # print(cell_idx)
        row.append(float(data_set[cell_idx].value))
        # print(row)
    X_matrix.append(row)

X_matrix = np.matrix(X_matrix)
y_zero = []
for y in range(1,MAXROW+1):
    cell_idx = 'A'+str(y)
    ICH_0hr = float(std_sheet[cell_idx].value)
    y_zero.append(ICH_0hr)
        # print("row: ",y)
        # print(diff)
        # cell_idx = 'G'+str(y)
        # print(float(std_sheet[cell_idx].value))
#print(matrix)
y_zero = np.matrix(y_zero)
y_zero = np.transpose(y_zero)
# print(matrix.shape)
print("Y dimension:",y_zero.shape)
# print(y_zero.shape)
#print(X_matrix)
#print(y_zero)

theta = np.full((2,1),1)
print(theta.shape)

ones = np.full((97,1),1)
X_matrix = np.hstack((ones,X_matrix))
#print(X_matrix)
alpha=0.3
num_iters=1500

(J_history,theta)= gradient_descent(X_matrix,y_zero,theta,alpha,num_iters)
#print(X_matrix)
#print(theta)
#print(y_zero)
error= ( y_zero-np.matmul(X_matrix,theta) / y_zero )*100
print(np.matmul(X_matrix,theta))
print("you can find the percentage of error using method of linear regression for 65 patients in the vector that was printed above")
#print("convergence criterion for J is satisfied")
