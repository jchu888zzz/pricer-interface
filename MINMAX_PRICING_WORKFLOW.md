# MinMax Pricing Workflow Design

## Overview
MinMax is a structured product that applies min/max constraints to an underlying value. It can be priced as either a **bullet** structure (fixed maturity) or a **callable** structure (early redemption allowed). The pricing workflow routes to the appropriate algorithm based on contract characteristics.

## Architecture

```
MinMax Contract Input
    ↓
[1] DETERMINATION PHASE
    ├─ Is contract callable? (has call_dates)
    │  ├─ YES → Callable path (LSM algorithm)
    │  └─ NO  → Bullet path (direct discounting)
    ↓
[2] PRECOMPUTATION PHASE
    ├─ Callable: CallableFeature.precomputation()
    │  ├─ Generate rate simulations
    │  ├─ Generate helper simulations
    │  ├─ Compute underlying values
    │  ├─ Prep discount factors for call dates
    │  └─ Setup funding leg (if swap)
    │
    └─ Bullet: Bullet.precomputation()
       ├─ Generate rate simulations
       ├─ Compute underlying values
       └─ Setup funding leg (if swap)
    ↓
[3] PRICING DECISION PHASE
    ├─ Is structure a Swap?
    │  ├─ YES → Swap variant
    │  │   ├─ Call compute_price_swap()
    │  │   └─ Deduct funding leg price
    │  │
    │  └─ NO → Bond variant
    │      ├─ Call compute_price_bond()
    │      └─ Use risky discounting
    ↓
[4] EXECUTION PHASE
    ├─ Callable:
    │  ├─ Build LSM regressions
    │  ├─ Compute stop indices (optimal exercise)
    │  ├─ Adjust cashflows for early redemption
    │  └─ Calculate final price
    │
    └─ Bullet:
       ├─ Apply min/max constraints
       ├─ Compute mean cashflows
       └─ Discount to present value
    ↓
Result Dictionary
```

## Detailed Workflow Steps

### Phase 1: Determination (Product Module)

**File**: `src/Pricing/Rates/Payoffs/Types/MinMax.py` (or product module)

**Decision Logic**:
```python
def precomputation(calc_date, model, params, risky_curve, risky):
    contract = MinMax(params)

    # Check if callable
    if hasattr(contract, 'call_dates') and contract.call_dates:
        return CallableFeature.precomputation(
            calc_date, contract, model, risky_curve,
            structure_choice=params['structure_type']
        )
    else:
        return Bullet.precomputation(
            calc_date, contract, model, risky_curve,
            structure_choice=params['structure_type']
        )
```

**Key Decision Factor**: Presence of `call_dates` attribute
- Set during `__init__` via `self.get_callable_info(parameters)`
- If no call dates defined in parameters → bullet path
- If call dates defined → callable path

### Phase 2: Precomputation (Feature Module)

#### Callable Path: `CallableFeature.precomputation()`

**Generates**:
- `dic_arg`: Main simulation data (10,000 scenarios)
  - `rates`: Hull-White rate paths
  - `undl`: Underlying values (min/max applied)
  - `zc_exercise`: Discount factors at call dates
  - `zc_continuation`: Discount factors after call dates
  - `measure_change_factor`: For bond pricing

- `dic_arg_helper`: Helper simulation data (10,000 scenarios, seed=42)
  - Same structure as `dic_arg` but with different seed
  - Used exclusively for LSM regression fitting

- `contract`: Updated with
  - `paygrid`: Time fractions to payment dates
  - `fwds`: Expected underlying values
  - `zc`: Zero coupon discount factors

- `funding_leg` (if swap)
  - Funding spread funding structure

**Computational Cost**: High (2× 10K simulations + helper data prep)

#### Bullet Path: `Bullet.precomputation()`

**Generates**:
- `dic_arg`: Simulation data (10,000 scenarios)
  - `rates`: Hull-White rate paths
  - `undl`: Underlying values

- `contract`: Updated with
  - `paygrid`: Time fractions to payment dates
  - `fwds`: Expected underlying values
  - `zc`: Zero coupon discount factors
  - `proba_recall`: Probability of reaching maturity (always 1.0 for bullet)
  - `res_capital`: Capital return (always 1.0 for bullet)

- `funding_leg` (if swap)

**Computational Cost**: Low (1× 10K simulations, minimal prep)

### Phase 3: Pricing Decision (Product Module)

**File**: `src/Pricing/Rates/Payoffs/Types/MinMax.py` (or product module)

**Decision Logic**:
```python
def compute_price(dic_prep, risky_curve):
    is_swap = contract.structure_type == 'Swap'

    if 'dic_arg_helper' in dic_prep:  # Callable
        if is_swap:
            return CallableFeature.compute_price_swap(
                dic_prep, 'polynomial', REGRESSOR_CLASS
            )
        else:
            return CallableFeature.compute_price_bond(
                dic_prep, 'polynomial', REGRESSOR_CLASS
            )
    else:  # Bullet
        if is_swap:
            return Bullet.compute_price_swap(dic_prep)
        else:
            return Bullet.compute_price_bond(dic_prep)
```

**Indicators**:
- **Callable detection**: Presence of `dic_arg_helper` in `dic_prep`
- **Swap detection**: Contract property or dictionary key presence

### Phase 4: Execution (Feature Module)

#### Callable Path: `CallableFeature.compute_price_bond/swap()`

**Algorithm**: Least Squares Monte Carlo (LSM)

**Steps**:
1. **Setup Phase**
   ```
   - Update pricing arguments with contract parameters
   - Compute min/max-constrained cashflows
   - Prepare regression inputs (helper simulations)
   ```

2. **Regression Phase**
   ```
   For each call date (reverse chronological):
   - Fit polynomial/Laguerre regression on continuation value
   - Regression input: Underlying values at call date
   - Regression output: Discounted future cashflows
   ```

3. **Decision Phase**
   ```
   For each scenario:
   - At each call date, predict future value via regression
   - Compare predicted future value vs immediate exercise payoff
   - Exercise if immediate payoff > predicted future
   - Update early redemption probability
   ```

4. **Valuation Phase**
   ```
   - Adjust cashflows for early redemption decisions
   - Compute mean cashflows across scenarios
   - Calculate early redemption probabilities

   For Bond:
   - Compute bond measure change factor
   - Apply risky discounting
   - Calculate funding spread

   For Swap:
   - Compute structure price (funding-leg-adjusted)
   - Compute funding leg price
   - Return net price
   ```

**Complexity**: O(n_scenarios × n_call_dates × n_basis_polynomials)

#### Bullet Path: `Bullet.compute_price_bond/swap()`

**Algorithm**: Direct Discounting

**Steps**:
1. **Setup Phase**
   ```
   - Update pricing arguments with contract parameters
   - Compute min/max-constrained cashflows
   ```

2. **Valuation Phase**
   ```
   - Compute mean cashflows across scenarios
   - Set probability of reaching maturity = 1.0
   - No early redemption

   For Bond:
   - Add capital repayment to cashflows
   - Use risky discount factors
   - Fixed funding spread (no early redemption adjustment)

   For Swap:
   - Compute structure price
   - Deduct funding leg price (fixed spread)
   - Return net price
   ```

**Complexity**: O(n_scenarios × n_payment_dates)

## Data Flow

### Product Module Entry Point

```python
# User provides contract parameters
params = {
    'coupon': '2.5%',
    'floor': '0%',          # MinMax-specific
    'cap': '10%',           # MinMax-specific
    'first_call_date': '06.01.2027',  # Makes it callable
    'structure_type': 'Swap'
}

# Product module handles routing
dic_prep = MinMax_module.precomputation(calc_date, model, params, risky_curve, risky=True)
result = MinMax_module.compute_price(dic_prep, risky_curve)
result = MinMax_module.solve_coupon(dic_prep, risky_curve)
```

### Callable-Specific Data

```
dic_prep = {
    'contract': MinMax(...),           # Updated with paygrid, fwds, zc
    'dic_arg': {...},                  # Main simulations (10K paths)
    'dic_arg_helper': {...},           # Helper simulations (10K paths, seed=42)
    'risky_curve': Risky_Curve(...),
    'funding_leg': Leg(...)            # Only if swap
}
```

### Bullet-Specific Data

```
dic_prep = {
    'contract': MinMax(...),           # Updated with paygrid, fwds, zc, proba_recall, res_capital
    'dic_arg': {...},                  # Simulations (10K paths)
    'risky_curve': Risky_Curve(...),
    'funding_leg': Leg(...)            # Only if swap
}
```

## Implementation Pattern for MinMax

### Minimal Product Module (MinMax.py)

```python
import numpy as np
import QuantLib as ql
from sklearn.neighbors import KNeighborsRegressor

from Pricing.Rates.Payoffs.Types import Base
from Pricing.Rates.Payoffs import CallableFeature, Bullet
from Pricing.Utilities import InputConverter

REGRESSOR_CLASS = KNeighborsRegressor(n_neighbors=30)

class MinMax(Base.Payoff):
    def __init__(self, parameters: dict):
        self.typename = 'MinMax'
        self.get_common_parameters(parameters)
        self.get_callable_info(parameters)
        if 'floor' in parameters.keys():
            self.floor = InputConverter.set_param(parameters['floor'], 0)
        if 'cap' in parameters.keys():
            self.cap = InputConverter.set_param(parameters['cap'], 0)

    def compute_cashflows(self, dic_arg: dict) -> np.ndarray:
        """Apply min/max constraints to underlying."""
        undl = dic_arg['undl']
        return np.maximum(
            self.floor,
            np.minimum(undl.T, self.cap)
        ) * np.tile(self.delta, (undl.shape[1], 1))

    def update_arg_pricing(self, coupon: float, dic_arg: dict) -> dict:
        """MinMax doesn't price on coupon variation."""
        res = dic_arg.copy()
        res.update({'x': coupon})
        return res


def precomputation(calc_date: ql.Date, model, params: dict, risky_curve, risky: bool):
    """Route between bullet and callable precomputation."""
    contract = MinMax(params)

    if hasattr(contract, 'call_dates') and contract.call_dates:
        # Callable: use LSM-aware precomputation
        return CallableFeature.precomputation(
            calc_date, contract, model, risky_curve,
            structure_choice=params['structure_type']
        )
    else:
        # Bullet: simple precomputation
        return Bullet.precomputation(
            calc_date, contract, model, risky_curve,
            structure_choice=params['structure_type']
        )


def compute_price(dic_prep: dict, risky_curve):
    """Route between bond/swap and bullet/callable pricing."""
    contract = dic_prep['contract']
    is_swap = contract.structure_type == 'Swap'

    # Determine pricing path
    if 'dic_arg_helper' in dic_prep:  # Callable indicator
        if is_swap:
            return CallableFeature.compute_price_swap(
                dic_prep, 'polynomial', REGRESSOR_CLASS
            )
        else:
            return CallableFeature.compute_price_bond(
                dic_prep, 'polynomial', REGRESSOR_CLASS
            )
    else:  # Bullet
        if is_swap:
            return Bullet.compute_price_swap(dic_prep)
        else:
            return Bullet.compute_price_bond(dic_prep)


def solve_coupon(dic_prep: dict, risky_curve):
    """Route between bond/swap and bullet/callable coupon solving."""
    contract = dic_prep['contract']
    is_swap = contract.structure_type == 'Swap'

    if 'dic_arg_helper' in dic_prep:  # Callable
        return CallableFeature.solve_coupon(
            dic_prep, 'polynomial', REGRESSOR_CLASS,
            swap=is_swap, use_memory=False
        )
    else:  # Bullet
        return Bullet.solve_coupon(dic_prep, swap=is_swap)
```

## Decision Tree for Product Developers

```
MinMax Contract Received
  │
  ├─ Read contract.call_dates
  │  │
  │  ├─ Callable (has call_dates)?
  │  │  ├─ YES: Use CallableFeature.precomputation()
  │  │  └─ NO:  Use Bullet.precomputation()
  │  │
  │  └─ Store result in dic_prep
  │
  ├─ Read contract.structure_type
  │  │
  │  ├─ Is Swap?
  │  │  ├─ YES: Call *_swap() functions
  │  │  └─ NO:  Call *_bond() functions
  │  │
  │  └─ Return appropriate result
```

## Performance Characteristics

| Scenario | Precomputation | Pricing | Memory |
|----------|---|---|---|
| Bullet Bond | 1-2s | 100ms | Low (1 sim) |
| Bullet Swap | 1-2s | 150ms | Low (1 sim + funding) |
| Callable Bond | 3-5s | 500ms-1s | Medium (2 sims + regression) |
| Callable Swap | 3-5s | 1-2s | Medium (2 sims + funding + regression) |

## Error Handling Strategy

### Validation in Precomputation

```python
# Check for required fields
if 'undl' not in dic_arg:
    raise ValueError("Underlying values missing in dic_arg")

# Check for swap consistency
if structure_type == 'Swap' and 'funding_leg' not in dic_prep:
    raise ValueError("Swap pricing requires funding_leg in dic_prep")

# Check for callable consistency
if 'dic_arg_helper' in dic_prep and 'zc_exercise' not in dic_arg:
    raise ValueError("Callable pricing requires zc_exercise in dic_arg")
```

### Validation in Pricing

```python
# Verify data integrity before computation
if contract.cap < contract.floor:
    raise ValueError(f"Cap ({contract.cap}) must be >= Floor ({contract.floor})")

# Ensure probabilities are valid
if not np.allclose(contract.proba_recall.sum(), 1.0):
    warnings.warn("Probabilities don't sum to 1.0, normalizing...")
    contract.proba_recall /= contract.proba_recall.sum()
```

## Testing Strategy

### Unit Tests

```python
def test_minmax_callable_bond():
    # Setup: MinMax with call dates, Bond structure
    # Verify: Uses CallableFeature.compute_price_bond()
    # Assert: Result has proper structure

def test_minmax_bullet_swap():
    # Setup: MinMax without call dates, Swap structure
    # Verify: Uses Bullet.compute_price_swap()
    # Assert: Result includes funding_table

def test_minmax_floor_cap():
    # Setup: MinMax with floor=1%, cap=5%
    # Verify: Cashflows are clipped correctly

def test_minmax_callable_early_exercise():
    # Setup: MinMax with aggressive cap
    # Verify: Early exercise decisions are reasonable
    # Assert: Probability < 1.0 for at least one date
```

### Integration Tests

```python
def test_minmax_callable_vs_bullet():
    # Same parameters, but one callable, one bullet
    # Bullet should be cheaper (no early exercise option)

def test_minmax_bond_vs_swap():
    # Same parameters, but different structure_type
    # Swap price should differ by funding spread

def test_minmax_solve_coupon():
    # Verify solve_coupon returns valid results
    # Check coupon is between bounds (1% - 50%)
```

## Summary

**Key Design Principles**:
1. **Routing by Structure**: Determine callable vs bullet at precomputation time
2. **Feature Purity**: Each feature module handles its algorithm independently
3. **Consistent API**: Bond/swap decision deferred to pricing functions
4. **Data-Driven**: Presence of data keys determine execution path (robust to refactoring)
5. **Minimal Product Code**: Products route; features execute

**Recommended Usage**:
```python
# In product-specific module (MinMax.py, Digit.py, RangeAccrual.py, etc.)
dic_prep = precomputation(...)      # Routes to Callable or Bullet
price = compute_price(dic_prep)      # Routes to bond or swap
coupon = solve_coupon(dic_prep)      # Routes to bond or swap
```
