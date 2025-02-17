import numpy as np

def normalize_array(arr, axis=0):
    '''
    axis=0: 按列归一化
    axis=1: 按行归一化
    axis=None: 按整个数组归一化
    '''
    # 计算指定轴上的最小值和最大值  
    min_val = np.min(arr, axis=axis, keepdims=True)  
    max_val = np.max(arr, axis=axis, keepdims=True)  

    normalized_arr = (arr - min_val) / (max_val - min_val)

    return normalized_arr

def find_nearest_index(array, values):
    array = np.asarray(array)
    if not isinstance(values, list):
        index=(np.abs(array - values)).argmin()
        return index
    else:
        index = []
        for value in values:
            index.append((np.abs(array - value)).argmin())
        return index

def find_max_and_index(lst):  
    if len(lst) == 0:
        return None, None  
    max_value = lst[0]  
    max_index = 0  
    for index, value in enumerate(lst):  
        if value > max_value:  
            max_value = value  
            max_index = index  
    return max_value, max_index 