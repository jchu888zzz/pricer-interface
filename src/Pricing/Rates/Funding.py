import numpy as np
from bisect import bisect
import QuantLib as ql

from Pricing.Utilities import Functions,Dates
from Pricing.Rates.Payoffs import Base
from collections import Counter

def get_compound_rate(rates:np.ndarray,d1:ql.Date,d2:ql.Date,N=360):

    bdays=ql.TARGET().businessDayList(d1,d2)
    delta=np.array([x-y for x,y in zip(bdays[1:],bdays)])
    dc=ql.Actual360().dayCount(d1,d2)
    res=np.array([(np.prod(1+r*delta/N)-1)*N/dc for r in rates])
    return res


_DIC_CURRENCY={'EUR':'3M','USD':'Overnight'}
class Leg:

    def __init__(self,contract,currency:str):
        self.type_rate=_DIC_CURRENCY[currency]
        self.contract=contract

        self.maturity_date=contract.issue_date+ql.Period(contract.maturity)
        if self.type_rate=='Overnight':
            self.pay_dates,self.fix_dates=Dates.compute_schedule(contract.issue_date,
                                                                 contract.issue_date+ql.Period(contract.maturity),
                                                                 freq=ql.Period('3M'),
                                                                 offset=-2,
                                                                 fixing_type='in arrears')
        else:
            self.pay_dates,self.fix_dates=Dates.compute_schedule(contract.issue_date,
                                                                 contract.issue_date+ql.Period(contract.maturity),
                                                                 freq=ql.Period('3M'),
                                                                 offset=-5,
                                                                 fixing_type='in advance')

    def precomputation(self,calc_date:ql.Date,model,data_rates:dict):
        rates,schedule=data_rates['rates'],data_rates['schedule'][1:]

        cal=ql.Actual360()
        idx=Functions.find_idx(self.fix_dates,calc_date)
        self.fix_dates=self.fix_dates[idx:]
        self.pay_dates=self.pay_dates[idx:]
        
        fixgrid=np.array([max(0,cal.yearFraction(calc_date,x)) for x in self.fix_dates ])
        paygrid=np.array([cal.yearFraction(calc_date,x) for x in self.pay_dates])
        self.delta=np.array([paygrid[0]]+[x-y for x,y in zip(paygrid[1:],paygrid)] )

        t_maturity=cal.yearFraction(calc_date,self.maturity_date)
        if self.type_rate=='Overnight':
            self.fwds=np.zeros((len(self.fix_dates),rates.shape[0]))
            start_date=self.fix_dates[0]
            start_idx=Functions.find_idx(schedule,start_date)
            for i,end_date in enumerate(self.fix_dates[1:]):
                end_idx=Functions.find_idx(schedule,end_date)
                self.fwds[i]=get_compound_rate(rates[:,start_idx:end_idx],start_date,end_date,N=360)
                start_date=end_date
                start_idx=end_idx

        elif self.type_rate=='3M':            
            idxs=Functions.find_idx(schedule,self.fix_dates)
            self.fwds=model.compute_deposit_from_rates(rates[:,idxs],fixgrid,'3M')
            # self.fwds=np.array([model.compute_deposit_from_rates(rates[:,i],t,'3M')
            #                             for i,t in zip(idxs,fixgrid) ])
        else:
            raise ValueError(f'{self.type_rate} not implemented')
        
        contract=self.contract
        if not hasattr(contract,'call_dates'):
            return
        else:
            idxs=[Functions.find_idx(schedule,d) for d in self.pay_dates]
            self.measure_change_factor=np.array([Base.compute_measure_change_factor(model,rates[:,i],t,t_maturity) 
                                        for i,t in zip(idxs,paygrid) ])
            
            self.zc_continuation=[None]*len(contract.call_dates)
            for i,d in enumerate(contract.call_dates):
                fix_idx=Functions.find_idx(schedule,d)
                t=cal.yearFraction(calc_date,schedule[fix_idx])
                pay_idx=bisect(self.pay_dates,d)
                self.zc_continuation[i]=model.compute_discount_factor_from_rates(rates[:,idx],t,paygrid[pay_idx:])

    def compute_cashflows(self,spread:float):
        
        return (self.fwds+spread)*np.tile(self.delta,(self.fwds.shape[0],1))
    

    def compute_values(self,spread:float):
        self.proba=np.ones(len(self.pay_dates))
        self.coupons=np.mean(self.compute_cashflows(spread),axis=0)

    def compute_values_for_early_redemption(self,stop_idxs:list[int],spread:float):
        contract=self.contract
        mask_alive=np.ones((len(stop_idxs),len(self.pay_dates)))
        for i,idx in enumerate(stop_idxs):
            j=bisect(self.pay_dates,contract.pay_dates[idx])
            mask_alive[i,j:]=0
        
        self.coupons=np.zeros(len(self.pay_dates))
        self.proba=np.ones(len(self.pay_dates))

        for i,d in enumerate(self.pay_dates):
            cashflows=self.fwds[i]+spread
            if d <=contract.call_dates[0]:
                self.coupons[i]= np.mean(cashflows)*self.delta[i]
            else:
                idx=Functions.find_idx(contract.pay_dates,d)
                self.proba[i]=min(1,np.mean([b/x for x,b in zip(self.measure_change_factor[i],mask_alive[:,i])]))
                amount=np.mean([ x*y for x,y in zip(cashflows/self.measure_change_factor[i],mask_alive[:,i])])
                self.coupons[i]=amount*self.delta[i]

    