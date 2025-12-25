import Pricing.Utilities.InputConverter as InputConverter
import Pricing.Utilities.Dates as Dates
import Pricing.Funding_leg.Funding as Fund
import numpy as np
import QuantLib as ql


class CLN_Tranche_Zero:
    def __init__(self,parameters:dict):
        self.currency=parameters['currency']
        self.coupon=InputConverter.set_param(parameters['coupon'],0)
        self.issue_date=InputConverter.convert_date(parameters['issue_date'])
        self.maturity_date=InputConverter.convert_date(parameters['maturity_date'])
        self.freq=InputConverter.freq_converter(parameters['frequency'])
        self.underlying=parameters['underlying']
        self.attach_point=int(parameters['attachment_point'])
        self.detach_point=int(parameters['detachment_point'])
        self.option_type=InputConverter.check_option_type(parameters['option_type'])

        #start=Dates.find_next_cds_standard(self.issue_date)
        self.pay_dates,self.fix_dates,self.paygrid=Dates.compute_schedule(self.issue_date,self.issue_date,
                                                                           self.maturity_date,
                                                                           self.freq,0)
        
    def compute_bond_price(self,RiskyCurve,coupon:float,include_details=True):
        
        temp=np.zeros_like(self.paygrid)
        for i,_ in enumerate(self.paygrid):
            temp[i]=np.mean([1- min(1,max(x-self.attach_point,0)/(self.detach_point-self.attach_point)) 
                             for x in self.default_matrix[:,i] ])
        delta=np.array([self.paygrid[0]] + [x-y for x,y in zip(self.paygrid[1:],self.paygrid)])
        res_coupon=coupon*temp*delta
        ZC=RiskyCurve.Discount_Factor(self.paygrid)        
        price=sum(res_coupon*ZC) + temp[-1]*ZC[-1]
        
        if include_details:
            details={'Payment Dates':self.pay_dates,
                'Coupons':res_coupon,
                'Probability':temp,
                'Zero Coupon':ZC,
                'Spread':self.fund_spread}
            return price,details
        return price
    
    def compute_prep_for_pricing(self,calc_date:ql.Date,RiskyCurve,model,funding_rate:str):
        cal=ql.Thirty360(ql.Thirty360.BondBasis)
        self.fixgrid=np.array([cal.yearFraction(calc_date,x) for x in self.fix_dates if x> calc_date])
        self.paygrid=np.array([cal.yearFraction(calc_date,x) for x in self.pay_dates if x> calc_date])
        self.fund_spread=Fund.compute_funding_spread(RiskyCurve,self.paygrid[-1])

        rho=model.solve_correlation(self.attach_point/model.nb_entities,self.detach_point/model.nb_entities)
        default_times=model.generate_default_time(rho,nb_simu=10000)

        self.default_matrix=np.zeros((default_times.shape[0],len(self.paygrid)))
        for i,t in enumerate(self.paygrid):
            self.default_matrix[:,i]=np.sum(default_times<=t,axis=1)


        if self.option_type=='Swap':
            self.fund_leg=Fund.Funding(self.issue_date,self.maturity,funding_rate)
            self.fund_leg.compute_funding_data(RiskyCurve.classic_curve,calc_date)

