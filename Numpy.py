import numpy as np
arr = np.array( [[1, 2, 3],
                 [4, 5, 6]] )

print("Mang 2 chieu: ")
print(arr)

cparr = arr.reshape(-1)

print("Mang 1 chieu:")
print(cparr)