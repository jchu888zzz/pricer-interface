from bisect import bisect
import numpy as np
import scipy.optimize as optimize

def find_idx(grid:list,value:list) -> int:
    idx=bisect(grid,value)
    if value <=grid[0]:
        return 0
    elif grid[0] < value <grid[-1]:
        return idx-1
    else:
        return len(grid)-1
    
def first_occ(item:list,value:list) -> int:
    
    if value in item:
        idx=np.argwhere(item==value)[0][0]
        return int(idx)
    else:
        return len(item)-1
    
def positive_integral_constant_by_part(values:np.ndarray,values_grid:np.ndarray,t:float) -> float:
    """ works for integral defined on positive real numbers  with positive values"""
    delta=np.array( [values_grid[0]] +[x-y for x,y in zip(values_grid[1:],values_grid)] )
    integral_value=np.cumsum(values*delta)

    if any(values_grid<0) or t<0:
        raise ValueError(' Only takes positives time or grid_values ')

    if t<=values_grid[0]:
        return t*values[0]
        
    elif  values_grid[0]< t <= values_grid[-1]:
        idx=find_idx(values_grid,t)
        return integral_value[idx]+values[idx]*(t-values_grid[idx])
        
    else:
        return integral_value[-1]+values[-1]*(t-values_grid[-1])

def inverse_positive_integral_constant_by_part(values:np.ndarray,values_grid:np.ndarray,
                                               y:float,standard:float) -> float:
    """ works for integral defined on positive real numbers  with positive values"""

    if y<0:
        raise ValueError('y must be positive')
    
    low=0
    upper=10

    if y> positive_integral_constant_by_part(values,values_grid,upper):
        return standard
    else:
        func= lambda t : positive_integral_constant_by_part(values,values_grid,t) -y
        res=optimize.brentq(func,low,upper)

        return res
    