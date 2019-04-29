import numpy as np
def median(lst): 
    sortedLst = sorted(lst) 
    lstLen = len(lst) 
    index = (lstLen - 1) // 2 

    if (lstLen % 2): 
     return sortedLst[index] 
    else: 
     return (sortedLst[index] + sortedLst[index + 1])/2.0 
    
def transarr(arr):
    arr = arr.reshape(-1)

def separate(arr):
    temp = np.array()
    newarr = []
    for i in range(0, 4):
        for j in range(0, 4):
            temp = arr[i*7 : (i+1)*7 -1 , j*7 : (j+1)*7 -1]
            transarr(temp)
            newarr[i][j] = median(temp)



