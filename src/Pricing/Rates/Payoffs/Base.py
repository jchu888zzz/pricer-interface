import numpy as np
import QuantLib as ql
from collections import Counter
from bisect import bisect
import typing
from scipy.optimize import minimize

from Pricing.Utilities import Dates,InputConverter,Functions

#Coupon Functions
def get_stop_matrix(stop_idxs:list[int],shape:tuple[int,int])-> np.ndarray:
    res=np.zeros(shape)
    for x,i in zip(res,stop_idxs):
        x[i]=1
    
    return res

def memory_feature(b:list)->np.ndarray:
    
    tracker=0
    temp=np.nonzero(b)[0]+1
    res=np.zeros(len(b))

    for item in temp:
        res[item-1]=item-tracker
        tracker=item
        
    return res

def to_zero(v:list,idx:int) -> list:
    v[idx:]=0
    return v

def adjust_to_stop_idxs(matrix:np.ndarray,stop_idxs:list,infine:bool):
    Stop_matrix=get_stop_matrix(stop_idxs,matrix.shape)
    if infine :
        res=matrix*Stop_matrix
    else:
        res=np.array([to_zero(x,idx+1) for x,idx  in zip(matrix,stop_idxs)])
    return res

def compute_cdt_digit(simu:np.ndarray,coupon_lvl:float,infine:bool,memory_effect:bool):

        coupon_cdt=(simu.T <= coupon_lvl)
        if infine and memory_effect:
            return np.array([ np.cumsum(memory_feature(x)) for x in coupon_cdt])
        if infine:
            return np.cumsum(coupon_cdt,axis=1)    
        if memory_effect:
            return np.array([memory_feature(x)for x in coupon_cdt])
        
        return coupon_cdt

#Contract Functions + Payoff Class
def compute_measure_change_factor(model,rates:np.ndarray,t:float,T:float) -> np.ndarray:
    Pt_T=model.compute_discount_factor_from_rates(rates,t,T)
    return (1/Pt_T)*model.DF(T)/model.DF(t)

def compute_bond_measure_change(measure_change_factor:np.ndarray,stop_idxs:np.ndarray)-> np.ndarray:
    Stop_matrix=get_stop_matrix(stop_idxs,measure_change_factor.T.shape)
    res=np.array([ np.mean([b/y for y,b in zip(measure_change_factor[i],Stop_matrix[:,i])]) 
                    for i in range(len(measure_change_factor))])
    return res

def prep_undl(contract,model,data_rates:dict,include_rates=True) -> dict:
    if contract.hasunderlying :
        if hasattr(contract,'fixing_depth'):

            fix_dates=np.insert(contract.fix_dates,0,contract.issue_date)
            # only range accrual 
            if contract.spreadunderlying:
                dic_arg=model.compute_spread_undl_from_rates_with_depth(data_rates,fix_dates,
                                                                        contract.underlying_name1,contract.underlying_name2,
                                                                        nb_sub_fix_points=contract.fixing_depth,
                                                                        include_rates=include_rates)
            else:
                dic_arg=model.compute_single_undl_from_rates_with_depth(data_rates,fix_dates,
                                                                    contract.underlying_name1,
                                                                    nb_sub_fix_points=contract.fixing_depth,
                                                                    include_rates=include_rates)
            densities=contract.compute_densities(dic_arg['undl'])
            dic_arg.update({'undl':dic_arg['undl'][:,-1,:],'densities':densities})
            return dic_arg

        else:
            if contract.spreadunderlying:
                return model.compute_spread_undl_from_rates(data_rates,contract.fix_dates,
                                                                contract.underlying_name1,contract.underlying_name2,
                                                                include_rates=include_rates)
            else:
                return model.compute_single_undl_from_rates(data_rates,contract.fix_dates,
                                                                contract.underlying_name1,
                                                                include_rates=include_rates)
            
class Payoff :

    def get_common_parameters(self,parameters:dict):
        self.parameters=parameters
        self.currency=parameters['currency']
        self.issue_date=InputConverter.convert_date(parameters['issue_date'])
        self.freq=InputConverter.freq_converter(parameters['frequency'])
        self.maturity=parameters['maturity']+'Y'
        if 'fixing_type' in parameters.keys():
            self.fixing_type=parameters['fixing_type']
    
        if 'coupon' in parameters.keys():
            self.coupon=InputConverter.set_param(parameters['coupon'],0)
        if 'UF' in parameters.keys():
            self.UF=InputConverter.set_param(parameters['UF'],0)

        if 'funding_spread' in parameters.keys():
            self.funding_spread=InputConverter.set_param(parameters['funding_spread'],0)

        if "yearly_buffer" in parameters.keys():
            self.yearly_buffer=InputConverter.set_param(parameters['yearly_buffer'],0)

        self.infine=False
        if 'in-fine' in parameters.keys():
            if parameters['in-fine']=='true':
                self.infine=True
        
        self.offset=int(InputConverter.set_param(parameters['fixing_days_offset'],0))
        self.structure_type=InputConverter.check_option_type(parameters['structure_type'])
        self.get_undl_info(parameters)
        self.pay_dates,self.fix_dates=Dates.compute_schedule(self.issue_date,self.issue_date+ql.Period(self.maturity),
                                                            self.freq,self.offset,self.fixing_type)
    
    def get_callable_info(self,parameters:dict):
        if 'first_call_date' in parameters.keys():
            first_call_date=InputConverter.convert_date(parameters['first_call_date'])
            frequency=InputConverter.freq_converter(parameters['call_frequency'])
            self.call_dates=Dates.compute_target_schedule(first_call_date,self.pay_dates[-1],frequency)[:-1]
            #self.call_idxs=[Functions.find_idx(self.fix_dates,x) for x in self.call_dates ]
            self.is_callable=True  
            return
        if 'NC' in parameters.keys():
            self.is_callable=True
            non_call=max(int(parameters['NC'])-1,0)
            self.call_dates=[self.fix_dates[non_call]]
            if 'multi-call' in parameters.keys():
                if parameters['multi-call']=='true':
                    self.call_dates+=list(self.fix_dates[non_call+1:-1])
            #self.call_idxs=[Functions.find_idx(self.fix_dates,x) for x in self.call_dates ]
            return
        
    def get_guaranteed_coupon_info(self,parameters:dict):
        self.sum_coupon_floor=0
        self.NC=0
        if "nb_guaranteed_coupon" in parameters.keys():
            nb_guaranteed_coupon=int(parameters['nb_guaranteed_coupon'])
            self.guar_coupon_dates=self.pay_dates[:nb_guaranteed_coupon]
            self.guar_coupon=InputConverter.set_param(parameters['guaranteed_coupon'],0)

            self.NC=nb_guaranteed_coupon
            self.sum_coupon_floor=self.guar_coupon*nb_guaranteed_coupon

    def get_undl_info(self,parameters:dict):
        self.hasunderlying=False
        self.spreadunderlying=False
        if "underlying1" in parameters.keys():
            self.underlying_name1=parameters['underlying1']
            self.hasunderlying=True

        if "underlying2" in parameters.keys():
            # self.cur2,self.rate_type2,self.tenor2=parameters['underlying2'].split()
            self.underlying_name2=parameters['underlying2']
            self.hasunderlying=True
            self.spreadunderlying=True


    def get_memory_effect(self,parameters):
        self.memory=False
        if 'memory_effect' in parameters.keys():
            if parameters['memory_effect']=='true':
                self.memory=True

    def _update(self,calc_date:ql.Date,cal=ql.Thirty360(ql.Thirty360.BondBasis)):

        idx=bisect(self.pay_dates,calc_date)
        self.fix_dates=self.fix_dates[idx:]
        self.pay_dates=self.pay_dates[idx:]
        if calc_date > self.issue_date:
            start=calc_date
        else:
            start=self.issue_date
        
        grid=np.array([cal.yearFraction(start,x) for x in self.pay_dates])
        self.delta=np.array([grid[0]] + [x-y for x,y in zip(grid[1:],grid)])

        if hasattr(self,'call_dates'):
            self.call_dates=[x for x in self.call_dates if x> start ]

        if hasattr(self,"guaranteed_coupon_dates"):
            self.guar_coupon_dates=[x for x in self.guar_coupon_dates if x> start]
            self.NC=len(self.guar_coupon_dates) if self.guar_coupon_dates else 0

    def compute_funding_adjustment(self,calc_date:ql.Date):
        if self.issue_date> calc_date+ql.Period('2M'):
            self.funding_adjustment= 0.95
            return
        self.funding_adjustment= 1

    def compute_recall_proba(self,stop_idxs):
        dic_res=Counter(stop_idxs)
        res=np.zeros(len(self.pay_dates))
        for key,value in dic_res.items():
            res[key]=value/len(stop_idxs)
        return res

#Result and spread functions
def organize_structure_table(contract,ZC) -> dict:

    res={'Payment Dates':contract.pay_dates}
    if hasattr(contract,'fwds'):
        res.update({'Model Forward':contract.fwds})
    res.update({
            'Early Redemption Proba':contract.proba_recall,
            'Cash Flows':contract.res_coupon,
            'Zero Coupon':ZC})
    
    return res

def organize_funding_table(funding_leg,ZC:np.ndarray)-> dict:
    res={'Payment Dates':funding_leg.pay_dates,
        'Model Forward':np.mean(funding_leg.fwds,axis=0),
        'Proba':funding_leg.proba,
        'Cash Flows': funding_leg.coupons,
        'Zero Coupon':ZC}

    return res

def get_funding_spread_early_redemption(risky_curve,pay_dates:list[ql.Date],proba:np.ndarray,adjustment:float) ->np.ndarray:

    pay_grid=[risky_curve.calendar.yearFraction(risky_curve.calc_date,d) for d in pay_dates]
    c=0.0004
    interp_spreads=risky_curve.funding_spread_interp(pay_grid)
    res = sum(proba*interp_spreads) - c
    return res*adjustment

def get_funding_spread(risky_curve,maturity_date:ql.Date,adjustment:float):

    t_maturity=risky_curve.calendar.yearFraction(risky_curve.calc_date,maturity_date)
    return risky_curve.funding_spread_interp(t_maturity)*adjustment 

def optimize_coupon(func_to_solve:typing.Callable):
    init=0.04
    bounds=[(0.01,.5)]
    opt=minimize(func_to_solve,x0=init,method='Nelder-Mead',bounds=bounds)
    return opt.x[0]

