# Compute Price Refactoring Summary

## Overview
Split `compute_price()` into separate `compute_price_bond()` and `compute_price_swap()` functions while reducing code duplication through common helper functions.

## Changes

### CallableFeature.py (LSM-based callable pricing)

**New Functions:**
- `_compute_price_common(contract, dic_arg, dic_arg_helper)` - Shared setup
- `compute_price_bond(dic_prep, basis_option, regressor_class)` - Callable bond pricing
- `compute_price_swap(dic_prep, basis_option, regressor_class)` - Callable swap pricing

**Helper Functions:**
- `_compute_price_bond(...)` - Internal implementation for bond
- `_compute_price_swap(...)` - Internal implementation for swap

**Router Function:**
- `compute_price(dic_prep, basis_option, regressor_class, swap)` - Routes to bond or swap

**Advantages:**
- Each public function (`compute_price_bond`, `compute_price_swap`) has a clear, specific purpose
- Common LSM setup is shared in `_compute_price_common()`
- Bond vs swap differences are isolated in `_compute_price_bond()` and `_compute_price_swap()`
- Original router function maintained for backward compatibility

**Code Duplication Removed:**
- Common steps: `contract.update_arg_pricing()`, `contract.compute_cashflows()`, cashflow adjustment, stop index computation, probability calculation
- These are handled once in helper functions instead of duplicated

### Bullet.py (Direct discounting for bullet pricing)

**New Functions:**
- `_compute_price_common(contract, dic_arg)` - Shared setup
- `compute_price_bond(dic_prep)` - Bullet bond pricing
- `compute_price_swap(dic_prep)` - Bullet swap pricing

**Router Function:**
- `compute_price(dic_prep, swap)` - Routes to bond or swap

**Advantages:**
- Simpler structure than callable (no LSM, no basis functions)
- Shared argument preparation and coupon calculation
- Explicit validation that funding_leg exists for swap mode
- Very clean separation of concerns

**Code Duplication Removed:**
- `contract.update_arg_pricing()`, `contract.compute_cashflows()`, coupon calculation
- Result dictionary construction

## Function Signatures

### CallableFeature
```python
# Public functions
def compute_price_bond(dic_prep: dict, basis_option: str, regressor_class: Ridge | KNeighborsRegressor) -> dict
def compute_price_swap(dic_prep: dict, basis_option: str, regressor_class: Ridge | KNeighborsRegressor) -> dict
def compute_price(dic_prep: dict, basis_option: str, regressor_class: Ridge | KNeighborsRegressor, swap: bool) -> dict

# Internal helpers
def _compute_price_common(contract, dic_arg, dic_arg_helper) -> (dic_arg, cashflows, dic_arg_helper)
def _compute_price_bond(contract, dic_prep, dic_arg, cashflows, dic_arg_helper, basis_option, regressor_class, risky_curve, deg=3) -> dict
def _compute_price_swap(contract, dic_prep, dic_arg, cashflows, dic_arg_helper, basis_option, regressor_class, risky_curve, deg=3) -> dict
```

### Bullet
```python
# Public functions
def compute_price_bond(dic_prep: dict) -> dict
def compute_price_swap(dic_prep: dict) -> dict
def compute_price(dic_prep: dict, swap: bool) -> dict

# Internal helper
def _compute_price_common(contract, dic_arg) -> None (sets contract.res_coupon)
```

## Benefits

1. **Clarity**: Each function does one thing and is obviously correct
2. **Maintainability**: Changes to bond vs swap logic are isolated
3. **Type Safety**: Function signatures are explicit about what's required
4. **DRY**: Common operations extracted to helper functions
5. **Backward Compatibility**: Original router functions maintained
6. **Testability**: Can test bond and swap paths independently
7. **IDE Support**: Better autocompletion and documentation per function

## Usage

### Before (still supported)
```python
result = CallableFeature.compute_price(dic_prep, 'polynomial', regressor, swap=True)
result = Bullet.compute_price(dic_prep, swap=True)
```

### After (recommended)
```python
result = CallableFeature.compute_price_swap(dic_prep, 'polynomial', regressor)
result = CallableFeature.compute_price_bond(dic_prep, 'polynomial', regressor)
result = Bullet.compute_price_swap(dic_prep)
result = Bullet.compute_price_bond(dic_prep)
```

Both approaches work - the router function maintains backward compatibility.
