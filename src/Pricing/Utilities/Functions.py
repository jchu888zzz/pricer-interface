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


def _evaluate_integral_cst_internal(t: float | np.ndarray, values: np.ndarray, 
                                     values_grid: np.ndarray, integral_value: np.ndarray,
                                     grid_start: float, grid_end: float, eps: float) -> float | np.ndarray:
    """
    Internal helper function for evaluating piecewise smooth integral.
    
    Shared implementation between integral_cst_by_part and IntegralCSTPrecalculated.evaluate
    to avoid code duplication.
    
    Parameters:
    -----------
    t : float or np.ndarray
        Time point(s) at which to evaluate
    values : np.ndarray
        Function values at grid points
    values_grid : np.ndarray
        Time grid points
    integral_value : np.ndarray
        Precomputed cumulative integral values
    grid_start : float
        First grid point
    grid_end : float
        Last grid point
    eps : float
        Smoothing parameter
    
    Returns:
    --------
    float or np.ndarray
        Integral value(s)
    """
    # Handle scalar input
    if np.isscalar(t):
        if t < 0:
            raise ValueError('t must be non-negative')
        
        if t <= grid_start:
            return float(t * values[0])
        elif t >= grid_end:
            return float(integral_value[-1] + values[-1] * (t - grid_end))
        else:
            idx = np.searchsorted(values_grid, t, side='right') - 1
            delta_i = values_grid[idx+1] - values_grid[idx]
            delta = t - values_grid[idx]
            value_diff = values[idx+1] - values[idx]
            
            smooth_time = eps * delta_i
            if smooth_time <= 0:
                increment = values[idx] * delta
            elif delta <= smooth_time:
                increment = values[idx] * delta + 0.5 * value_diff * delta**2 / smooth_time
            else:
                increment = (values[idx] + 0.5*value_diff) * smooth_time + values[idx+1] * (delta - smooth_time)
            
            return float(integral_value[idx] + increment)
    
    # Handle array input
    else:
        t = np.asarray(t, dtype=float)
        if np.any(t < 0):
            raise ValueError('All t values must be non-negative')
        
        result = np.zeros_like(t, dtype=float)
        
        # Case 1: t <= grid_start
        mask1 = t <= grid_start
        result[mask1] = t[mask1] * values[0]
        
        # Case 3: t >= grid_end
        mask3 = t >= grid_end
        if np.any(mask3):
            result[mask3] = integral_value[-1] + values[-1] * (t[mask3] - grid_end)
        
        # Case 2: within grid
        mask2 = ~(mask1 | mask3)
        if np.any(mask2):
            t_subset = t[mask2]
            idx = np.searchsorted(values_grid, t_subset, side='right') - 1
            
            delta_i = values_grid[idx+1] - values_grid[idx]
            delta = t_subset - values_grid[idx]
            value_diff = values[idx+1] - values[idx]
            
            smooth_time = eps * delta_i
            smooth_time = np.maximum(smooth_time, 1e-12)
            
            increment = np.zeros_like(delta)
            
            mask_smooth = delta <= smooth_time
            if np.any(mask_smooth):
                delta_smooth = delta[mask_smooth]
                smooth_t = smooth_time[mask_smooth]
                increment[mask_smooth] = (values[idx[mask_smooth]] * delta_smooth + 
                                         0.5 * value_diff[mask_smooth] * delta_smooth**2 / smooth_t)
            
            mask_const = ~mask_smooth
            if np.any(mask_const):
                delta_const = delta[mask_const]
                smooth_t = smooth_time[mask_const]
                increment[mask_const] = ((values[idx[mask_const]] + 0.5*value_diff[mask_const]) * smooth_t + 
                                        values[idx[mask_const]+1] * (delta_const - smooth_t))
            
            result[mask2] = integral_value[idx] + increment
        
        return result


class IntegralCSTPrecalculated:
    """
    Precomputed integral table for piecewise smooth functions.
    Caches cumulative integral values to speed up repeated evaluations.
    
    Use this when calling integral_cst_by_part many times with the same curve.
    """
    def __init__(self, values:np.ndarray, values_grid:np.ndarray, eps:float=0.0):
        """
        Precompute integral table.
        
        Parameters:
        -----------
        values : np.ndarray
            Function values at grid points
        values_grid : np.ndarray
            Time grid points (must be increasing)
        eps : float
            Smoothing parameter (should match eps used in evaluate calls)
        """
        self.values = np.asarray(values, dtype=float)
        self.values_grid = np.asarray(values_grid, dtype=float)
        self.eps = eps
        self.grid_start = values_grid[0]
        self.grid_end = values_grid[-1]
        
        if np.any(values_grid < 0):
            raise ValueError('Only takes positive grid values')
        
        # Precompute delta and modified values (one-time cost)
        self.delta = np.diff(values_grid, prepend=values_grid[0])
        modified_values = (1 - 0.5*eps)*values[1:] + 0.5*eps*values[:-1]
        modified_values = np.insert(modified_values, 0, values[0])
        
        # Cache cumulative integral
        self.integral_value = np.cumsum(modified_values * self.delta)
    
    def evaluate(self, t: float | np.ndarray) -> float | np.ndarray:
        """
        Evaluate integral at time point(s) using precomputed table.
        Much faster than calling integral_cst_by_part repeatedly.
        """
        return _evaluate_integral_cst_internal(t, self.values, self.values_grid, 
                                               self.integral_value, self.grid_start, 
                                               self.grid_end, self.eps)


def integral_cst_by_part(values:np.ndarray, values_grid:np.ndarray, 
                   t: float | np.ndarray, eps=0.0) -> float | np.ndarray:
    """
    Vectorized computation of piecewise smooth integral with linear smoothing phase.
    
    Computes integral with smoothing effect controlled by eps parameter.
    For each interval [t_i, t_{i+1}], the rate linearly transitions from
    values[i] to values[i+1] over the first eps*(t_{i+1} - t_i) time,
    then continues as values[i+1].
    
    Parameters:
    -----------
    values : np.ndarray
        Function values at each grid point
    values_grid : np.ndarray
        Time grid points (must be increasing)
    t : float or np.ndarray
        Time point(s) at which to evaluate the integral
    eps : float
        Smoothing parameter (0 = piecewise constant, 1 = full linear interpolation)
    
    Returns:
    --------
    float or np.ndarray
        Integral value(s) (scalar if t is scalar, array if t is array)
    """
    values = np.asarray(values, dtype=float)
    values_grid = np.asarray(values_grid, dtype=float)
    
    if np.any(values_grid < 0):
        raise ValueError('Only takes positive grid values')
    
    # Precompute integral table (same as IntegralCSTPrecalculated)
    delta = np.diff(values_grid, prepend=values_grid[0])
    modified_values = (1 - 0.5*eps)*values[1:] + 0.5*eps*values[:-1]
    modified_values = np.insert(modified_values, 0, values[0])
    integral_value = np.cumsum(modified_values * delta)
    
    grid_start = values_grid[0]
    grid_end = values_grid[-1]
    
    # Use shared internal function
    return _evaluate_integral_cst_internal(t, values, values_grid, integral_value, 
                                           grid_start, grid_end, eps)


def inverse_integral_cst_by_part(values:np.ndarray, values_grid:np.ndarray,
                                               y: float | np.ndarray, 
                                               standard: float | np.ndarray) -> float | np.ndarray:
    """
    Solve for t such that positive_integral_constant_by_part(values, values_grid, t) = y.
    
    Works for integral defined on positive real numbers with positive values.
    Vectorized to handle both scalar and array inputs for y and standard.
    
    Parameters:
    -----------
    values : np.ndarray
        Function values at grid points
    values_grid : np.ndarray
        Time grid points
    y : float or np.ndarray
        Target integral value(s)
    standard : float or np.ndarray
        Fallback value(s) if y exceeds maximum integral
    
    Returns:
    --------
    float or np.ndarray
        Solved t value(s) (scalar if y is scalar, array if y is array)
    """
    # Precompute maximum integral value (constant for all evaluations)
    upper_bound = 10.0
    max_integral = integral_cst_by_part(values, values_grid, upper_bound)
    
    # Handle scalar input
    if np.isscalar(y):
        if y < 0:
            raise ValueError('y must be positive')
        
        if y > max_integral:
            return float(standard) if np.isscalar(standard) else float(standard[0])
        
        # Use brentq for accurate scalar root-finding
        func = lambda t: integral_cst_by_part(values, values_grid, t) - y
        result = optimize.brentq(func, 0.0, upper_bound)
        return float(result)
    
    # Handle array input - vectorized root-finding using bisection
    else:
        y = np.asarray(y, dtype=float)
        standard = np.asarray(standard, dtype=float)
        
        if np.any(y < 0):
            raise ValueError('All y values must be non-negative')
        
        result = np.empty_like(y, dtype=float)
        
        # Identify which values exceed max integral
        mask_exceed = y > max_integral
        result[mask_exceed] = standard[mask_exceed] if standard.ndim > 0 else standard
        
        # For remaining values, use vectorized bisection
        mask_solve = ~mask_exceed
        if np.any(mask_solve):
            y_subset = y[mask_solve]
            
            # Vectorized bisection method
            low = np.zeros_like(y_subset)
            high = np.full_like(y_subset, upper_bound)
            
            # Bisection iterations (20 iterations gives ~1e-6 precision)
            for _ in range(10):
                mid = (low + high) / 2.0
                f_mid = integral_cst_by_part(values, values_grid, mid) - y_subset
                
                # Update bounds based on function values
                low = np.where(f_mid < 0, mid, low)
                high = np.where(f_mid >= 0, mid, high)
            
            result[mask_solve] = (low + high) / 2.0
        
        return result
    