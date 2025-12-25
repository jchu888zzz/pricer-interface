import pandas as pd
import QuantLib as ql
import numpy as np
from datetime import datetime

import Pricing.Rates.Model.HullWhite as HullWhite
from Pricing.Curves import Classic,CMT
import Pricing.Rates.Model.ConvexityAdjustment as CA
import Pricing.Rates.Instruments as Rate_Instruments
from Pricing.Utilities import Dates,Functions

DIC_UNDL={'BFRTEC10':{'tag':'BFR','tenor':'10Y','currency':'EUR','vol_shift':0.8},
        'SOLDE10E':{'tag':'GDBR_CMT','tenor':'10Y','currency':'EUR','vol_shift':0.8},
        'SOLBE10E':{'tag':'BGB_CMT','tenor':'10Y','currency':'EUR','vol_shift':0.8},
        'SOITA10Y':{'tag':'BTP_CMT','tenor':'10Y','currency':'EUR','vol_shift':0.8},
        'SOLIT1OE':{'tag':'BTP_CMT','tenor':'10Y','currency':'EUR','vol_shift':0.8},
        'H15T10Y':{'tag':'CMT','tenor':'10Y','currency':'USD','vol_shift':0.8}}

def select_rates(rates:np.ndarray,grid_simu:list,tgrid:np.ndarray,depth=1):
    if depth==1:
        idx=[Functions.find_idx(grid_simu,t) for t in tgrid]
        return rates[:,idx].T
    else:
        tgrid1=np.insert(tgrid,0,0)
        res=np.zeros((len(tgrid),depth,rates.shape[0]))
        for i,(t1,t2) in enumerate(zip(tgrid1,tgrid1[1:])):
            subgrid=np.linspace(t1,t2,depth)
            sub_idx=[Functions.find_idx(grid_simu,t) for t in subgrid]
            res[i]=(rates[:,sub_idx].T)
        return res
    
def compute_correlation(df:pd.DataFrame,ticker1:str,ticker2:str) -> float:
    
    shift_wednesday={1:3,2:2,3:1,4:0,5:-1,6:-2,7:-3}
    effectiveDate = ql.Date.todaysDate() - ql.Period('2Y')
    effectiveDate=effectiveDate+shift_wednesday[effectiveDate.weekday()]
    terminationDate = ql.Date.todaysDate()
    schedule = ql.MakeSchedule(effectiveDate,terminationDate,ql.Period('1W'), calendar=ql.TARGET(), 
                                convention=ql.ModifiedFollowing, terminalDateConvention =ql.ModifiedFollowing, 
                                rule=ql.DateGeneration.Forward,endOfMonth=False)
    def format(date:ql.Date)->datetime.date:
        return datetime(date.year(),date.month(),date.dayOfMonth()).date()
    
    schedule=[format(x) for x in schedule ]
    df=df.loc[[df.index[Functions.find_idx(df.index.map(lambda x: x.date()),x)] for x in schedule]]
    df=df.diff()
    df=df.dropna()
    res=np.corrcoef(df[ticker1].to_numpy(),df[ticker2].to_numpy())
    
    return res

def get_model(dic_df:dict,undl:str,calc_date) -> dict:

    currency=DIC_UNDL[undl]['currency']
    curve,risky_curve=Classic.get_curves(calc_date,dic_df,currency)

    swaptions_rate=Rate_Instruments.get_swaptions(dic_df['swaption'],curve,calc_date,currency)
    swaptions_rate=[x for x in swaptions_rate if x.striketype=='ATM']
    model_rate=HullWhite.Calibration(curve,swaptions_rate)

    cmt_instruments=CMT.get_instru(dic_df['cmt'],calc_date,DIC_UNDL[undl]['tag'])
    cmt_curve=CMT.Curve(cmt_instruments,'EUR')
    cmt_curve.retrieve_interp(dic_df['cmt'],calc_date,tag=DIC_UNDL[undl]['tag'])

    df_cmt=dic_df['swaption'].copy()
    df_cmt['Quote']*=DIC_UNDL[undl]['vol_shift']
    swaptions_cmt=Rate_Instruments.get_swaptions(df_cmt,cmt_curve,calc_date,currency)
    swaptions_rate=[x for x in swaptions_rate if x.striketype!='ATM']
    model_cmt=HullWhite.Calibration(cmt_curve,swaptions_cmt)

    path="//Umilp-p2.cdm.cm-cic.fr/cic-lai-lae-cigogne$/1_Structuration/6_Lexifi/Snapshot_data/Historical_prices.xlsx"
    ticker1=undl+' Index'
    if currency=="USD":
        ticker2="USISSO10 Index"
    elif currency=="EUR":
        ticker2='EUAMDB10 Index'
    else:
        raise ValueError(f'{currency} Not implemented')

    cov_matrix=compute_correlation(pd.read_excel(path,sheet_name='Rate Undl',index_col=0),
                                            ticker1,ticker2)

    model=HW_CMT(model_rate,model_cmt,cov_matrix)
    return {'risky_curve':risky_curve,
            'curve':curve,
            'model':model,
            'calc_date':calc_date}

class HW_CMT:

    def __init__(self,model_rate,model_cmt,cov_matrix):
        self.model_rate=model_rate
        self.model_cmt=model_cmt
        self.DF=model_rate.DF
        self.compute_discount_factor_from_rates=model_rate.compute_discount_factor_from_rates
        self.compute_deposit_from_rates=model_rate.compute_deposit_from_rates
        self.cov_matrix=cov_matrix
    
    def generate_rates(self,calc_date:ql.Date,maturity_date:ql.Date,
                       cal=ql.Thirty360(ql.Thirty360.BondBasis),Nbsimu=10000,seed=5) -> dict:

        rng=np.random.default_rng(int(seed))
        T=cal.yearFraction(calc_date,maturity_date)
        schedule=Dates.compute_target_schedule(calc_date,maturity_date,ql.Period('1D'))
        grid=np.array([cal.yearFraction(calc_date,x) for x in schedule[1:] ])
        Z=rng.multivariate_normal(mean=[0,0],cov=self.cov_matrix,size=(len(grid),Nbsimu))

        model_rate=self.model_rate
        model_cmt=self.model_cmt

        h=1e-3
        rates=np.zeros((Nbsimu,len(grid)))
        rates_cmt=np.zeros((Nbsimu,len(grid)))
        rates[:,0]=-(np.log(model_rate.DF(h))-np.log(model_rate.DF(0)))/h
        rates_cmt[:,0]=-(np.log(model_cmt.DF(h))-np.log(model_cmt.DF(0)))/h
        
        prev_alpha=model_rate.alpha_T(grid[0],T)
        prev_alpha_cmt=model_cmt.alpha_T(grid[0],T)

        prev_var=model_rate.var_(grid[0])
        prev_var_cmt=model_cmt.var_(grid[0])
        for i in range(1,len(grid)):
            delta=grid[i]-grid[i-1]
            alpha=model_rate.alpha_T(grid[i],T)
            var=model_rate.var_(grid[i])
            rates[:,i]=( rates[:,i-1]*np.exp(-model_rate.a*delta) +alpha - prev_alpha*np.exp(-model_rate.a*delta) +
                      np.sqrt(var-prev_var*np.exp(-2*model_rate.a*delta))*Z[:,:,0][i] )
            prev_alpha=alpha
            prev_var=var

            alpha_cmt=model_cmt.alpha_T(grid[i],T)
            var_cmt=model_cmt.var_(grid[i])
            rates_cmt[:,i]=( rates_cmt[:,i-1]*np.exp(-model_cmt.a*delta) +alpha_cmt - prev_alpha_cmt*np.exp(-model_cmt.a*delta) +
                      np.sqrt(var_cmt-prev_var_cmt*np.exp(-2*model_cmt.a*delta))*Z[:,:,1][i] )
            prev_alpha_cmt=alpha_cmt
            prev_var_cmt=var_cmt

        return {'rates':rates,'rates_cmt':rates_cmt,'schedule':schedule,'grid':grid}
    
    def compute_cmt_from_rates(self,rates_cmt:np.ndarray,t:float,tenor1:str,option='adjusted'):
        model_cmt=self.model_cmt
        curve_cmt=model_cmt.Curve
        if option=='unadjusted':
            return model_cmt.compute_cms_from_rates(rates_cmt,t,tenor1)
        elif option=='adjusted':
            fwd_adjusted=curve_cmt.ajusted_fwd_cms_interp(tenor1,t)
            undl=model_cmt.compute_cms_from_rates(rates_cmt,t,tenor1)
            adjustment= fwd_adjusted -np.mean(undl)
            return undl+adjustment
        else:
            raise ValueError('compute_cmt_from_rates: option not implemented')
    
    def compute_single_undl_from_rates(self,data_rates:dict,fixgrid:list,undl1:str,include_rates=True) ->tuple[np.ndarray]:
        """ result shape (len(fixgrid),nb simu)"""
        tenor1=DIC_UNDL[undl1]['tenor']
        rates_cmt=select_rates(data_rates['rates_cmt'],data_rates['grid'],fixgrid,1)
        undl=np.array([self.compute_cmt_from_rates(rates_cmt[i],fixgrid[i],tenor1,'adjusted') for i in range(len(fixgrid))])

        if not include_rates:
            return {'undl':undl}
        else:
            rates=select_rates(data_rates['rates'],data_rates['grid'],fixgrid,1)
            return {'undl':undl,'rates':rates}

    #Only CMT -CMS
    def compute_spread_undl_from_rates(self,data_rates:dict,fixgrid:list,
                                       undl1:str,undl2:str,include_rates=True) ->tuple[np.ndarray]:
        """ result shape (len(fixgrid),nb simu)"""
        tenor1=DIC_UNDL[undl1]['tenor']
        cur2,rate_type2,tenor2=undl2.split()

        rates_cmt=select_rates(data_rates['rates_cmt'],data_rates['grid'],fixgrid,1)

        model_rate=self.model_rate
        rates=select_rates(data_rates['rates'],data_rates['grid'],fixgrid,1)
        
        undl=np.array([self.compute_cmt_from_rates(rates_cmt[i],fixgrid[i],tenor1,option='adjusted')-
                       model_rate.compute_cms_from_rates(rates[i],fixgrid[i],tenor2) for i in range(len(fixgrid))])

        if not include_rates:
            return {'undl':undl}
        else:
            rates=select_rates(data_rates['rates'],data_rates['grid'],fixgrid,1)
            return {'undl':undl,'rates':rates}
        
    def compute_single_undl_from_rates_with_depth(self,data_rates:dict,fixgrid:list,undl1:str,fixing_depth:int,
                                                  include_rates=True)->np.ndarray:
        """ result shape (len(fixgrid),fixing_depth,nb simu)"""
        tenor1=DIC_UNDL[undl1]['tenor']
        rates_cmt=select_rates(data_rates['rates_cmt'],data_rates['grid'],fixgrid,fixing_depth)
        rates=select_rates(data_rates['rates'],data_rates['grid'],fixgrid,fixing_depth)

        res=np.zeros_like(rates_cmt)
        
        tgrid=np.insert(fixgrid,0,0)
        for i,t in enumerate(tgrid[:-1]):
            sub_fixgrid=np.linspace(t,tgrid[i+1],fixing_depth) # nbweeks 52
            res[i]=np.array([ self.compute_cmt_from_rates(rates_cmt[i][j],sub_fixgrid[j],tenor1,'adjusted') for j in range(len(sub_fixgrid))])
        
        if not include_rates:
            return {'undl':res}
        else:
            return {'undl':res,'rates':rates[:,-1,:]}
    
    #Only CMT -CMS
    def compute_spread_undl_from_rates_with_depth(self,data_rates:dict,fixgrid:list,undl1:str,
                                                  undl2:str,fixing_depth:int,include_rates=True)->np.ndarray:
        """ result shape (len(fixgrid),fixing_depth,nb simu)"""
        tenor1=DIC_UNDL[undl1]['tenor']
        cur2,rate_type2,tenor2=undl2.split()

        rates_cmt=select_rates(data_rates['rates_cmt'],data_rates['grid'],fixgrid,fixing_depth)
        model_rate=self.model_rate
        rates=select_rates(data_rates['rates'],data_rates['grid'],fixgrid,fixing_depth)
        res=np.zeros_like(rates_cmt)
        
        tgrid=np.insert(fixgrid,0,0)
        for i,t in enumerate(tgrid[:-1]):
            sub_fixgrid=np.linspace(t,tgrid[i+1],fixing_depth)
            res[i]=np.array([ self.compute_cmt_from_rates(rates_cmt[i][j],sub_fixgrid[j],tenor1,'adjusted')-
                             model_rate.compute_cms_from_rates(rates[i][j],sub_fixgrid[j],tenor2) for j in range(len(sub_fixgrid))])

        if not include_rates:
            return {'undl':res}
        else:
            return {'undl':res,'rates':rates[:,-1,:]}
