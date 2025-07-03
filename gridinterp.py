import pandas as pd
import numpy as np
from scipy.interpolate import griddata, interp2d
import matplotlib.pyplot as plt

def load_data(iv_file, strike_file):
    iv_df = pd.read_csv(iv_file, header=0)
    strike_df = pd.read_csv(strike_file, header=0)
    
    # 获取tenor列表(假设列名就是tenor)
    tenors = iv_df.columns.tolist()
    
    return iv_df, strike_df, tenors

def create_grid_interpolation(iv_df, strike_df, tenors, method='linear'):
    # 创建用于插值的点
    points = []
    values = []
    
    # 遍历每一行数据构建插值点
    for row_idx in range(len(iv_df)):
        for tenor_idx, tenor in enumerate(tenors):
            # 获取iv和对应的strike
            iv = iv_df.iloc[row_idx, tenor_idx]
            strike = strike_df.iloc[row_idx, tenor_idx]
            
            # 确保数据有效(不是NaN)
            if pd.notna(iv) and pd.notna(strike):
                # 使用tenor的索引作为x坐标(可以根据实际tenor值转换)
                tenor_value = float(tenor) if tenor.replace('.', '', 1).isdigit() else tenor_idx
                points.append([tenor_value, strike])
                values.append(iv)
    
    # 转换为numpy数组
    points = np.array(points)
    values = np.array(values)
    
    # 创建插值函数
    def interpolator(tenor_query, strike_query):
        tenor_value = float(tenor_query) if isinstance(tenor_query, (int, float, str)) and str(tenor_query).replace('.', '', 1).isdigit() else tenors.index(tenor_query)
        
        # 使用griddata进行插值
        return griddata(points, values, [[tenor_value, strike_query]], method=method)[0]
    
    return interpolator, points, values
