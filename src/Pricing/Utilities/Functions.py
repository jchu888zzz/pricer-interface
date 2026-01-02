import numpy as np
import scipy.optimize as optimize

def find_idx(grid: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Vectorized version that handles arrays of values"""
    idx = np.searchsorted(grid, values, side='right')
    result = np.where(values <= grid[0], 0,
                     np.where(values >= grid[-1], len(grid) - 1, idx - 1))
    return result.astype(int)
    
def first_occ(item:list,value:list) -> int:
    
    if value in item:
        idx=np.argwhere(item==value)[0][0]
        return int(idx)
    else:
        return len(item)-1
    

def first_occ_vec(matrix: np.ndarray, value) -> np.ndarray:
    """
    Vectorized version that finds the first occurrence of a value in each row of a matrix.

    """
    matrix = np.asarray(matrix)

    # Find all occurrences: shape (n_rows, n_cols) of booleans
    matches = matrix == value

    # For each row, find the first TRUE value (column index)
    # argmax returns index of first TRUE (and first element if all FALSE)
    first_indices = np.argmax(matches, axis=1)

    # Check which rows actually have a match
    # (if all FALSE, argmax returns 0, so we need to verify)
    has_match = np.any(matches, axis=1)

    # For rows without match, set to last column index
    default_idx = matrix.shape[1] - 1
    result = np.where(has_match, first_indices, default_idx)

    return result
    
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
    