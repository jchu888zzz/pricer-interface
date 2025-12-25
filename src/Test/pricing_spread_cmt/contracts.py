test_tarn_bond={'_source_tab':'TARN',
                'param':{'issue_date':'30.11.2025',
                'maturity':'10',
                'fixing_days_offset':'-5',
                'frequency':'Annually',
                'coupon':'3%',
                'coupon_level':'1.5%',
                'nb_guaranteed_coupon':'2',
                'target':"10%",
                'guaranteed_coupon':"3%",
                'underlying1':'BFRTEC10',
                'underlying2':'EUR CMS 10Y',
                'fixing_type':'in arrears',
                'currency':'EUR',
                'structure_type':'Bond'}}

test_tarn_swap={'_source_tab':'TARN',
                'param':{'issue_date':'30.11.2025',
                'maturity':'10',
                'fixing_days_offset':'-5',
                'frequency':'Annually',
                'coupon':'3%',
                'coupon_level':'1.5%',
                'nb_guaranteed_coupon':'2',
                'target':"10%",
                'guaranteed_coupon':"3%",
                'underlying1':'BFRTEC10',
                'underlying2':'EUR CMS 10Y',
                'fixing_type':'in arrears',
                'currency':'EUR',
                'structure_type':'Swap',
                'funding_spread':'100bps'}}


test_autocall_bond={'_source_tab':'Autocall',
                'param':{'issue_date':'30.11.2025',
                'maturity':'10',
                'fixing_days_offset':'-5',
                'frequency':'Annually',
                'coupon_level':'1.5%',
                'coupon':'10%',
                'autocall_level':'1%',
                'underlying1':'BFRTEC10',
                'underlying2':'EUR CMS 10Y',
                'memory_effect':'false',
                'fixing_type':'in arrears',
                'NC':'1',
                'currency':'EUR',
                'structure_type':'Bond'}}

test_autocall_swap={'_source_tab':'Autocall',
                'param':{'issue_date':'30.11.2025',
                'maturity':'10',
                'fixing_days_offset':'-5',
                'frequency':'Annually',
                'coupon_level':'1.5%',
                'coupon':'3%',
                'autocall_level':'1%',
                'underlying1':'BFRTEC10',
                'underlying2':'EUR CMS 10Y',
                'memory_effect':'false',
                'fixing_type':'in arrears',
                'NC':'1',
                'currency':'EUR',
                'structure_type':'Swap',
                'funding_spread':'100bps'}}



test_digit_bond={'_source_tab':'Digit',
                'param':{'issue_date':'30.11.2025',
                'maturity':'10',
                'fixing_days_offset':'-5',
                'frequency':'Annually',
                'coupon_level':'1.5%',
                'coupon':'6%',
                'underlying1':'BFRTEC10',
                'underlying2':'EUR CMS 10Y',
                'memory_effect':'false',
                'fixing_type':'in arrears',
                'currency':'EUR',
                'structure_type':'Bond'}}
data_callable={'multi-call': 'true',
                'NC':'5'}
test_digit_bond['param'].update(data_callable)


test_digit_swap={'_source_tab':'Digit',
                'param':{'issue_date':'30.11.2025',
                        'maturity':'10',
                        'fixing_days_offset':'-5',
                        'frequency':'Annually',
                        'coupon_level':'1.5%',
                        'coupon':'6%',
                        'underlying1':'BFRTEC10',
                        'underlying2':'EUR CMS 10Y',
                        'memory_effect':'false',
                        'fixing_type':'in arrears',
                        'currency':'EUR',
                        'structure_type':'Swap',
                        'funding_spread':'90bps'}}
data_callable={'multi-call': 'true',
                'NC':'5'}

test_range_bond={'_source_tab':'RangeAccrual',
                'param':{'issue_date':'30.11.2025',
                        'maturity':'5',
                        'fixing_days_offset':'-5',
                        'frequency':'Annually',
                        'lower_bound':'0%',
                        'upper_bound':'2%',
                        'coupon':'4%',
                        'underlying1':'BFRTEC10',
                        'underlying2':'EUR CMS 10Y',
                        'fixing_type':'in arrears',
                        'currency':'EUR',
                        'structure_type':'Bond'}}
data_callable={'multi-call': 'true',
                'NC':'3'}
test_range_bond['param'].update(data_callable)


test_range_swap={'_source_tab':'RangeAccrual',
                'param':{'issue_date':'30.11.2025',
                        'maturity':'5',
                        'fixing_days_offset':'-5',
                        'frequency':'Annually',
                        'lower_bound':'0%',
                        'upper_bound':'2%',
                        'coupon':'4%',
                        'underlying1':'BFRTEC10',
                        'underlying2':'EUR CMS 10Y',
                        'fixing_type':'in arrears',
                        'currency':'EUR',
                        'structure_type':'Swap',
                        'funding_spread':'70bps'}}
data_callable={'multi-call': 'true',
                'NC':'3'}
test_range_swap['param'].update(data_callable)