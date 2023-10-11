import numpy as np

def exercise1():
    # Định nghĩa x
    x = [1,2,3]
    
    # Định nghĩa y
    y = [-1,-2,3]
    
    x_norm2 = np.sqrt(1*1+2*2+3*3)

    # Tính chuẩn y và gán vào biến y_norm2
    y_norm2 = np.sqrt(1*1+2*2+3*3)

    return [round(x_norm2, 2), round(x_norm2, 2)]

x_norm2, y_norm2 = exercise1()
print(round(x_norm2, 2), round(y_norm2,2))