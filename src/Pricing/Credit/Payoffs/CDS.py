
import numpy as np
import QuantLib as ql

import Pricing.Utilities.Dates as Dates
import Pricing.Utilities.Functions as Functions
from Pricing.Credit.Model.Intensity import Intensity

import Pricing.Utilities.InputConverter as InputConverter

class CDS:
    def __init__(self,parameters:dict):
        self.currency=parameters['currency']
        self.coupon=InputConverter.set_param(parameters['coupon'],0)
        self.issue_date=InputConverter.convert_date(parameters['issue_date'])
        self.maturity=parameters['maturity']+'Y'
        self.underlying=parameters['underlying']

        start=Dates.find_next_cds_standard(self.issue_date)
        end=start+ql.Period(self.maturity)
        self.schedule=Dates.compute_target_schedule(start,end,ql.Period('3M'))


    def compute_theorical_price(self,model:Intensity,
                                calc_date:ql.Date,cal=ql.Actual365Fixed()) -> float:

        tgrid=np.array([cal.yearFraction(calc_date,d) for d in self.schedule if d>calc_date])
        B=[1] + [model.DF(t)*model.compute_survival_proba(t) for t in tgrid]
        df_interp=model.Curve.interpolate
        t_eps=1e-3

        cl,fl,acc=0,0,0
        for i,(t1,t2) in enumerate(zip(tgrid,tgrid[1:])):
            delta=t2-t1
            idx=Functions.find_idx(model.standard_tgrid,t1)
            r=-(np.log(df_interp(t1))-np.log(df_interp(t1-t_eps)))/t_eps
            cl+=model.values[idx]/(model.values[idx]+r)*(B[i]-B[i+1])
            fl+=delta**np.exp(-r*(delta+10/360))*model.compute_survival_proba(t1)
            acc+=(1/365)*model.values[idx]/(model.values[idx]+r)*( (B[i]-B[i+1])/(model.values[idx]+r) -delta*B[i+1] + 0.5*(B[i]-B[i+1]))

        return (1-model.R)*cl - self.coupon*(fl+acc)