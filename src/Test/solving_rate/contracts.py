test_autocall_bond={'_source_tab':'Autocall',
                'param':{'issue_date':'30.11.2025',
                'maturity':'10',
                'fixing_days_offset':'-5',
                'frequency':'Annually',
                'coupon_level':'3.5%',
                'autocall_level':'2.25%',
                'underlying1':'EUR CMS 10Y',
                'memory_effect':'false',
                'fixing_type':'in arrears',
                'NC':'1',
                'currency':'EUR',
                'structure_type':'Bond',
                'yearly_buffer': '0.0%',
                'UF':'3%',
                'solving_choice':'Solve coupon'}}

test_autocall_swap={'_source_tab':'Autocall',
                'param':{'issue_date':'30.11.2025',
                'maturity':'10',
                'fixing_days_offset':'-5',
                'frequency':'Annually',
                'coupon_level':'3.5%',
                'autocall_level':'2.25%',
                'underlying1':'EUR CMS 10Y',
                'memory_effect':'false',
                'fixing_type':'in arrears',
                'NC':'1',
                'currency':'EUR',
                'structure_type':'Swap',
                'yearly_buffer': '0.0%',
                'UF':'3%',
                'solving_choice':'Solve coupon'}}


test_digit_bond={'_source_tab':'Digit',
                'param':{'issue_date':'30.11.2025',
                'maturity':'10',
                'fixing_days_offset':'-5',
                'frequency':'Annually',
                'coupon_level':'3.5%',
                'underlying1':'EUR CMS 10Y',
                'memory_effect':'false',
                'fixing_type':'in arrears',
                'currency':'EUR',
                'structure_type':'Bond',
                'yearly_buffer': '0.0%',
                'UF':'3%',
                'solving_choice':'Solve coupon'}}
data_callable={'multi-call': 'true',
                'NC':'5'}
test_digit_bond['param'].update(data_callable)

test_digit_swap={'_source_tab':'Digit',
                'param':{'issue_date':'30.11.2025',
                'maturity':'10',
                'fixing_days_offset':'-5',
                'frequency':'Annually',
                'coupon_level':'3.5%',
                'underlying1':'EUR CMS 10Y',
                'memory_effect':'false',
                'fixing_type':'in arrears',
                'currency':'EUR',
                'structure_type':'Swap',
                'yearly_buffer': '0.0%',
                'UF':'3%',
                'solving_choice':'Solve coupon'}}

data_callable={'multi-call': 'true',
                'NC':'5'}
test_digit_swap['param'].update(data_callable)

test_range_bond={'_source_tab':'RangeAccrual',
                'param':{'issue_date':'30.11.2025',
                        'maturity':'5',
                        'fixing_days_offset':'-5',
                        'frequency':'Annually',
                        'lower_bound':'2%',
                        'upper_bound':'4.5%',
                        'underlying1':'EUR CMS 10Y',
                        'fixing_type':'in arrears',
                        'currency':'EUR',
                        'structure_type':'Bond',
                        'yearly_buffer': '0.0%',
                        'UF':'3%',
                        'solving_choice':'Solve coupon'}}
data_callable={'multi-call': 'true',
                'NC':'3'}
test_range_bond['param'].update(data_callable)

test_range_swap={'_source_tab':'RangeAccrual',
                'param':{'issue_date':'30.11.2025',
                        'maturity':'5',
                        'fixing_days_offset':'-5',
                        'frequency':'Annually',
                        'lower_bound':'2%',
                        'upper_bound':'4.5%',
                        'underlying1':'EUR CMS 10Y',
                        'fixing_type':'in arrears',
                        'currency':'EUR',
                        'structure_type':'Swap',
                        'yearly_buffer': '0.0%',
                        'UF':'3%',
                        'solving_choice':'Solve coupon'}}
data_callable={'multi-call': 'true',
                'NC':'3'}
test_range_swap['param'].update(data_callable)



test_fixed_bond={'_source_tab':'FixedRate',
                'param':{'issue_date':'30.11.2025',
                        'maturity':'8',
                        'fixing_days_offset':'-5',
                        'frequency':'Annually',
                        'fixing_type':'in arrears',
                        'currency':'EUR',
                        'structure_type':'Bond',
                        'yearly_buffer': '0.0%',
                        'UF':'3%',
                        'solving_choice':'Solve coupon'}}
data_callable={'multi-call': 'true',
                'NC':'3'}
test_fixed_bond['param'].update(data_callable)

test_fixed_swap={'_source_tab':'FixedRate',
                'param':{'issue_date':'30.11.2025',
                        'maturity':'8',
                        'fixing_days_offset':'-5',
                        'frequency':'Annually',
                        'fixing_type':'in arrears',
                        'currency':'EUR',
                        'structure_type':'Swap',
                        'yearly_buffer': '0.0%',
                        'UF':'3%',
                        'solving_choice':'Solve coupon'}}
data_callable={'multi-call': 'true',
                'NC':'3'}
test_fixed_swap['param'].update(data_callable)