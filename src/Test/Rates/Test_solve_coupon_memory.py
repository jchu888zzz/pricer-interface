import sys
from pathlib import Path
pricing_root=Path(__file__).parent.parent.parent
sys.path.insert(0,str(pricing_root))

import time
import QuantLib as ql
from copy import deepcopy
from sklearn.linear_model import Ridge

from Pricing.Rates import GetResults
from Pricing.Rates.Model import HullWhite
from Pricing.Rates.Payoffs.Types import FixedRate, Digit, MinMax, RangeAccrual
from Pricing.Rates.Payoffs import CallableFeature

REGRESSOR_CLASS = Ridge(alpha=0.5, fit_intercept=True)

def test_solve_coupon_comparison(mkt_data: dict, params: dict, module):
    """
    Test both solve_coupon implementations (recompute vs memory) and compare results.
    Calls CallableFeature.solve_coupon directly with correct parameters.
    """
    prep_model = HullWhite.get_model(mkt_data['calc_date'], mkt_data,
                                      params['currency'], None)

    is_swap = params['structure_type'] == 'Swap'

    # Precomputation
    dic_prep = module.precomputation(prep_model['calc_date'], prep_model['model'],
                                      params, prep_model['risky_curve'], risky=True)

    # Add risky_curve to dic_prep for CallableFeature
    dic_prep['risky_curve'] = prep_model['risky_curve']

    # Test 1: Original implementation (recompute after optimization)
    start_time = time.time()
    coupon_recompute, funding_recompute = CallableFeature.solve_coupon(
        dic_prep,
        basis_option='polynomial',
        regressor_class=REGRESSOR_CLASS,
        swap=is_swap,
        use_memory=False
    )
    time_recompute = time.time() - start_time

    # Test 2: Memory implementation (use last iteration value)
    # Need to re-prep since solve_coupon may modify contract state
    dic_prep_memory = module.precomputation(prep_model['calc_date'], prep_model['model'],
                                             params, prep_model['risky_curve'], risky=True)
    dic_prep_memory['risky_curve'] = prep_model['risky_curve']

    start_time = time.time()
    coupon_memory, funding_memory = CallableFeature.solve_coupon(
        dic_prep_memory,
        basis_option='polynomial',
        regressor_class=REGRESSOR_CLASS,
        swap=is_swap,
        use_memory=True
    )
    time_memory = time.time() - start_time

    return {
        'coupon_recompute': coupon_recompute,
        'funding_recompute': funding_recompute,
        'time_recompute': time_recompute,
        'coupon_memory': coupon_memory,
        'funding_memory': funding_memory,
        'time_memory': time_memory,
        'coupon_diff': abs(coupon_recompute - coupon_memory),
        'funding_diff': abs(funding_recompute - funding_memory),
        'time_saved': time_recompute - time_memory
    }


def run_tests():
    # Setup market data
    DataPath = r"C:\Users\jorda\OneDrive\Documents\pricer_interface-main\snapshot"
    calc_date = ql.Date(11, 11, 2025)
    mkt_data = GetResults.retrieve_data(path_folder=DataPath, date=calc_date)

    # Test configurations
    test_configs = [
        {
            'name': 'FixedRate 10Y NC3 Swap',
            'module': FixedRate,
            'params': {
                "NC": "3",
                "UF": "2.0%",
                "currency": "EUR",
                "fixing_days_offset": "-10",
                "frequency": "Annually",
                "in-fine": "false",
                "issue_date": "06.01.2026",
                "maturity": "10",
                "multi-call": "true",
                "structure_type": "Swap",
                "yearly_buffer": "0.0%"
            }
        },
        {
            'name': 'FixedRate 10Y NC3 Bond',
            'module': FixedRate,
            'params': {
                "NC": "3",
                "UF": "2.0%",
                "currency": "EUR",
                "fixing_days_offset": "-10",
                "frequency": "Annually",
                "in-fine": "false",
                "issue_date": "06.01.2026",
                "maturity": "10",
                "multi-call": "true",
                "structure_type": "Bond",
                "yearly_buffer": "0.0%"
            }
        },
        {
            'name': 'FixedRate 5Y NC2 Swap',
            'module': FixedRate,
            'params': {
                "NC": "2",
                "UF": "1.5%",
                "currency": "EUR",
                "fixing_days_offset": "-10",
                "frequency": "Annually",
                "in-fine": "false",
                "issue_date": "06.01.2026",
                "maturity": "5",
                "multi-call": "true",
                "structure_type": "Swap",
                "yearly_buffer": "0.0%"
            }
        },
    ]

    print("=" * 80)
    print("COMPARISON: solve_coupon RECOMPUTE vs MEMORY implementations")
    print("=" * 80)

    for config in test_configs:
        print(f"\n{'-' * 60}")
        print(f"Test: {config['name']}")
        print(f"{'-' * 60}")

        try:
            result = test_solve_coupon_comparison(mkt_data, config['params'], config['module'])

            print(f"\n  RECOMPUTE (original):")
            print(f"    Coupon:  {result['coupon_recompute']:.6f} ({result['coupon_recompute']*100:.4f}%)")
            print(f"    Funding: {result['funding_recompute']:.6f} ({result['funding_recompute']*100:.4f}%)")
            print(f"    Time:    {result['time_recompute']:.3f}s")

            print(f"\n  MEMORY (last iteration):")
            print(f"    Coupon:  {result['coupon_memory']:.6f} ({result['coupon_memory']*100:.4f}%)")
            print(f"    Funding: {result['funding_memory']:.6f} ({result['funding_memory']*100:.4f}%)")
            print(f"    Time:    {result['time_memory']:.3f}s")

            print(f"\n  DIFFERENCES:")
            print(f"    Coupon diff:  {result['coupon_diff']:.8f} ({result['coupon_diff']*10000:.4f} bps)")
            print(f"    Funding diff: {result['funding_diff']:.8f} ({result['funding_diff']*10000:.4f} bps)")
            print(f"    Time saved:   {result['time_saved']:.3f}s ({result['time_saved']/result['time_recompute']*100:.1f}% faster)")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_tests()
