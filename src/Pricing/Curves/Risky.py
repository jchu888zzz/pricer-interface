import numpy as np
import QuantLib as ql
import pandas as pd

from Pricing.Curves import Classic
from Pricing.Utilities import Data_File
from Pricing.Credit.Model import Intensity 
import Pricing.Credit.Instruments as Credit_instruments


def GetCurve(File:pd.ExcelFile,cur_name:str,calc_date:ql.Date,option:str):

    curve=Classic.GetCurve(File,cur_name,calc_date,option)
    df=Data_File.concatenate_dataframe_from_mktdata_file(File,sheet_list=['Grille CIC_'+cur_name])
    entity=Credit_instruments.retrieve_credit_single_from_dataframe(df,'CIC_EUR',calc_date)
    
    model_credit=Intensity.Calibration(curve,entity)
    
    res=Risky_Curve(curve,model_credit)

    return res

class Risky_Curve:

    def __init__(self,Curve:Classic.Yield_Curve,model:Intensity.Intensity):
        
        self.curve=Curve
        self.model=model

    def Discount_Factor(self,tgrid:np.ndarray,risky:bool):
        R=self.model.entity.R
        DF=self.curve.Discount_Factor(tgrid)
        if not risky:
            return DF
            # return self.risky_DF(t)
        else:
             default_proba=np.array([1-self.model.compute_survival_proba(t) for t in tgrid ])
             return DF*(1-default_proba*(1-R))
        
    def adjustment(self,t:float,T: np.ndarray):
        R=self.model.entity.R
        default_proba=self.model.compute_survival_proba(t)-self.model.compute_survival_proba(T)

        return (1-default_proba*(1-R))
        # R=self.entity.R
        # return  1-(self.compute_default_proba(T)/self.compute_default_proba(t))*(1-R)

    
    # def get_default_proba1(self,t1:float,t2:float) -> float:

    #     temp1=Functions.positive_integral_constant_by_part(self.cr_spread,self.tenor_grid,t1)
    #     temp2=Functions.positive_integral_constant_by_part(self.cr_spread,self.tenor_grid,t2)
    #     return 1- np.exp(-(temp2-temp1))
    
