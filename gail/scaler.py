import numpy as np

def scale_obs(
        data, 
        is_scale,
        min_feature = np.array([-2.5, -3.5, -np.pi, 0, -0.5, -2.5, -3.5]), 
        max_feature = np.array([2.0, 3.5, np.pi, 0.16, 0.5, 2.0, 3.5]),
        isZero2One = True
    ):
    """
    data: shape=(?, 1, ?, 7)
    min, max: len 7 array
    return: array in range [0, 1] if isZero2One is True (default)
                          [-1, 1] if isZero2One is False
    """
    if not is_scale:
        return data
    feature_dim = data.shape[-1] # 5 or 7
    min_feature = min_feature[:feature_dim]
    #print(min_feature)
    max_feature = max_feature[:feature_dim]
    result = (data - min_feature)/(max_feature - min_feature)
    return result if isZero2One else result * 2 - 1

def inverse_obs(
        data,
        is_scale,
        _min = np.array([-2.5, -3.5, -np.pi, 0, -0.5, -2.5, -3.5]), 
        _max = np.array([2.0, 3.5, np.pi, 0.16, 0.5, 2.0, 3.5]),
        isZero2One = True
    ):
    """
    inverse of MinMaxScaler
    return: array in range [0, 1] if isZero2One is True (default)
                          [-1, 1] if isZero2One is False
    """
    if not is_scale:
        return data
    feature_dim = data.shape[-1] # 5 or 7
    _min = _min[:feature_dim]
    _max = _max[:feature_dim]
    data = data if isZero2One else (data + 1) / 2
    
    result = _min + data * (_max - _min)
    return result

def scale_u(data, is_scale, min, max):
    if not is_scale:
        return data
    #min = np.min(data)
    #max = np.max(data)
    return (data - min) * 2 / (max - min) - 1

def inverse_u(data, is_scale, min, max):
    if not is_scale:
        return data
    return (data + 1) * (max - min)/ 2 + min
