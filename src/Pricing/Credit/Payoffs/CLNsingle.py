import numpy as np
import QuantLib as ql

import Pricing.Utilities.Dates as Dates
import Pricing.Funding_leg.Funding as Fund
from Pricing.Credit.Model.Intensity import Intensity
import Pricing.Utilities.InputConverter as InputConverter

class CLN_single:
    def __init__(self,parameters:dict):
        self.currency=parameters['currency']
        self.coupon=InputConverter.set_param(parameters['coupon'],0)
        self.issue_date=InputConverter.convert_date(parameters['issue_date'])
        self.maturity=parameters['maturity']+'Y'
        self.freq=InputConverter.freq_converter(parameters['frequency'])
        self.underlying=parameters['underlying']
        self.option_type=InputConverter.check_option_type(parameters['option_type'])

        #start=Dates.find_next_cds_standard(self.issue_date)
        self.pay_dates,self.fix_dates,self.paygrid=Dates.compute_schedule(self.issue_date,self.issue_date,
                                                                           self.issue_date+ql.Period(self.maturity),
                                                                           self.freq,0)
        
    def compute_bond_price(self,RiskyCurve,coupon:float,include_details=True):
        
        proba=np.array([self.model.compute_survival_proba(t)  for t in self.paygrid])
        res_coupon=coupon*proba
        ZC=RiskyCurve.Discount_Factor(self.paygrid)        
        price=sum(res_coupon*ZC) + proba[-1]*ZC[-1] +(1-proba[-1])*ZC[-1]*(self.model.R)
        
        if include_details:
            details={'Payment Dates':self.pay_dates,
                'Coupons':res_coupon,
                'Probability':proba,
                'Zero Coupon':ZC,
                'Spread':self.spread}
            return price,details
        return price
    
    def compute_prep_for_pricing(self,calc_date:ql.Date,RiskyCurve,model:Intensity,funding_rate:str):
        cal=ql.Actual364()
        self.fixgrid=np.array([cal.yearFraction(calc_date,x) for x in self.fix_dates if x> calc_date])
        self.paygrid=np.array([cal.yearFraction(calc_date,x) for x in self.pay_dates if x> calc_date])
        self.spread=Fund.compute_funding_spread(RiskyCurve,self.paygrid[-1])
        self.model=model

        if self.option_type=='Swap':
            self.fund_leg=Fund.Funding(self.issue_date,self.maturity,funding_rate)
            self.fund_leg.compute_funding_data(RiskyCurve.classic_curve,calc_date)
