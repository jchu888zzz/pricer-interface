
from PySide6.QtCore import QObject,Signal
import os
import QuantLib as ql
import pandas as pd
import numpy as np

from Pricing.Rates.Payoffs.Autocallable import TARN, Autocall
from Pricing.Rates.Payoffs.Callable import Digit, FixedRate,RangeAccrual
from Pricing.Utilities import Dates,Data_File
from Pricing.Rates.Model import HullWhiteCMT,HullWhite

from Pricing.Equity import EQDataPrep
from Pricing.Equity.Model import HestonModel

class SnapshotWorker(QObject):
    finished=Signal(dict)
    finished_msg=Signal(str)

    def __init__(self,calc_date:ql.Date):
        super().__init__()
        self.path_folder=r"C:\Users\jorda\OneDrive\Documents\pricer_interface-main\snapshot"
        self.calc_date=calc_date

    def get_filename(self,calc_date:ql.Date):
        date_formatted=Dates.ql_to_string(calc_date,'%Y-%m-%d')
        return os.path.join(self.path_folder,"mkt_data_"+date_formatted+ ".xlsx")
    
    def run(self):

        calc_date=ql.Date(11,11,2025)
        try:
            path=self.get_filename(calc_date)
        except ValueError:
            try:
                calc_date=ql.TARGET().advance(calc_date,-ql.Period('1D'))
                path=self.get_filename(calc_date)
            except ValueError:
                raise ValueError('No File for this date or previous business date')

        with pd.ExcelFile(path) as File:
            df_curve=Data_File.concatenate_dataframe_from_mktdata_file(File,
                                                        sheet_list=('Deposits','Futures','Swaps'))

            df_swaption=Data_File.concatenate_dataframe_from_mktdata_file(File,
                                                                sheet_list=('Swaptions',))
            df_caps=Data_File.concatenate_dataframe_from_mktdata_file(File,
                                                            sheet_list=('Caps and Floors',))
            df_issuer=Data_File.concatenate_dataframe_from_mktdata_file(File,
                                                            sheet_list=['Grille CIC_EUR','Grille CIC_USD'])
            df_cmt=Data_File.concatenate_dataframe_from_mktdata_file(File,
                                                            sheet_list=('Deposits','CMT','Swaps'))
            
        res={"curve":df_curve,
            "swaption":df_swaption,
            "caps":df_caps,
            "cmt":df_cmt,
            "issuer":df_issuer,
            'calc_date':calc_date}

        self.finished.emit(res)
        self.finished_msg.emit("Market Data imported")

class RateWorker(QObject):
    finished=Signal(dict)
    
    def __init__(self,data:dict,dic_contract:dict):
        super().__init__()
        self.data=data
        self.dic_contract=dic_contract

    def _calibrate(self) -> dict:
        return HullWhite.get_model(self.data,self.dic_contract['param']['currency'],
                                    self.data['calc_date'])

    def compute_price(self):
        
        PAYOFF_MAPPING={'Autocall':Autocall.Process,
                        'TARN':TARN.Process,
                        'Digit':Digit.Process,
                        'RangeAccrual':RangeAccrual.Process,
                        'FixedRate':FixedRate.Process}
        
        payoff_type=self.dic_contract['_source_tab']
        payoff_class=PAYOFF_MAPPING.get(payoff_type)
        if not payoff_class:
            raise ValueError(f'{payoff_type} not implemented')

        prep_model=self._calibrate()
        res=payoff_class.compute_price(prep_model,self.dic_contract['param'])
        self.finished.emit(res)

    def solve_coupon(self):
        
        PAYOFF_MAPPING={'Autocall':Autocall.Process,
                        'Digit':Digit.Process,
                        'RangeAccrual':RangeAccrual.Process,
                        'FixedRate':FixedRate.Process}
        
        payoff_type=self.dic_contract['_source_tab']
        payoff_class=PAYOFF_MAPPING.get(payoff_type)
        if not payoff_class:
            raise ValueError(f'{payoff_type} not implemented')

        prep_model=self._calibrate()
        res=payoff_class.solve_coupon(prep_model,self.dic_contract['param'])
        self.finished.emit(res)

        
class CMTWorker(QObject):
    finished=Signal(dict)
    
    def __init__(self,data:dict,dic_contract:dict):
        super().__init__()
        self.data=data
        self.dic_contract=dic_contract

    def _calibrate(self):
        return HullWhiteCMT.get_model(self.data,self.dic_contract['param']['underlying1'],
                                    self.data['calc_date'])

    def compute_price(self):
        
        PAYOFF_MAPPING={'Autocall':Autocall.Process,
                        'TARN':TARN.Process,
                        'Digit':Digit.Process,
                        'RangeAccrual':RangeAccrual.Process}
        
        payoff_type=self.dic_contract['_source_tab']
        payoff_class=PAYOFF_MAPPING.get(payoff_type)
        if not payoff_class:
            raise ValueError(f'{payoff_type} not implemented')

        prep_model=self._calibrate()
        res=payoff_class.compute_price(prep_model,self.dic_contract['param'])
        self.finished.emit(res)   

    def solve_coupon(self):
        
        PAYOFF_MAPPING={'Autocall':Autocall.Process,
                        'Digit':Digit.Process,
                        'RangeAccrual':RangeAccrual.Process}
        
        payoff_type=self.dic_contract['_source_tab']
        payoff_class=PAYOFF_MAPPING.get(payoff_type)
        if not payoff_class:
            raise ValueError(f'{payoff_type} not implemented')

        prep_model=self._calibrate()
        res=payoff_class.solve_coupon(prep_model,self.dic_contract['param'])
        self.finished.emit(res)


##Equity part
def get_callprobas(S_fix, call_lvl, S0):
    Nsim, n = S_fix.shape
    probas = np.zeros(n)
    redeemed = []
    for i in range(n - 1):
        idx_call = np.where(S_fix[:,i] >= call_lvl*S0)[0]
        idx_call = np.setdiff1d(idx_call, redeemed)
        probas[i] = len(idx_call)/Nsim
        redeemed = np.union1d(redeemed, idx_call)
    probas[-1] = 1 - np.sum(probas, axis = 0)
    return probas

def geteq_fund2(T_pay, probas, fund_curve, discount):
    c=0.0004
    tenor_spread = fund_curve(T_pay)
    fund2 = (np.sum(tenor_spread*probas, axis = 0) - c)*discount
    return tenor_spread, fund2


class EquityWorker(QObject):
    finished=Signal(dict)

    def __init__(self,input_data:dict):   
        super().__init__()
        self.param=input_data.copy()
        
        self.param["MC"] = 1/365, 10000
        value_date = ql.TARGET().advance(ql.Date.todaysDate(), -1, ql.Days)
        self.param["value_date"] = value_date
        if input_data['currency']=="EUR":
            spread_data = pd.read_excel(os.path.join(EQDataPrep.spread_path, "Refi_CIC_EUR.xls"),sheet_name="CIC_EUR")
        elif input_data['currency']=="USD":
            spread_data=pd.read_excel(os.path.join(EQDataPrep.spread_path, "Refi_CIC_EUR.xls"),sheet_name="CIC_USD")
        else:
            raise ValueError(" Unavailable currency")
        self.param["fund_curve"] = EQDataPrep.spread_prep(spread_data)
        path_markit_names=r"\\Umilp-p2.cdm.cm-cic.fr\cic-lai-lae-cigogne$\1_Structuration\19_Quant\Methodo Funding\Markit\Names.xlsx"
        self.param["ref_table"] = pd.read_excel(path_markit_names, header = 0)


    def compute_result(self):
        dt, Nsim = self.param["MC"]
        value_date = self.param["value_date"]
        fund_curve = self.param["fund_curve"]
        ref_table = self.param["ref_table"]

        #convert param
        start_date = ql.Date(self.param["issue_date"], "%d.%m.%Y")
        trade_date = ql.Date(self.param["initial_strike_date"], "%d.%m.%Y")
        mat = int(self.param["maturity"])
        call_lvl = float(self.param["autocall_level"])*0.01

        freq_dic={'Annually':'1Y','Semi-annually':'6M',
            'Quarterly':'3M','Monthly':'1M'}
        freq = freq_dic[self.param["frequency"]]
        per_nocall = int(self.param["periods_no_autocall"])
        offset = int(self.param["fixing_offset"])

        pay_dates = list(ql.Schedule(start_date, start_date + ql.Period(mat, ql.Years), ql.Period(freq), ql.TARGET(), ql.Following, ql.Following, ql.DateGeneration.Forward, False))[1 + per_nocall:]
        fix_dates = list(map(lambda date: ql.TARGET().advance(date, offset, ql.Days), pay_dates))
        T_pay = np.array(list(map(lambda date: ql.Actual365Fixed().yearFraction(value_date, date), pay_dates)))
        T_fix = np.array(list(map(lambda date: ql.Actual365Fixed().yearFraction(value_date, date), fix_dates)))

        stock = self.param["underlying"]
        vol_surface, forward, df = EQDataPrep.prep_data_markit(value_date.ISO(), EQDataPrep.markit_path, stock, ref_table)

        params = HestonModel.calib_heston(value_date.ISO(), vol_surface)

        Xt = HestonModel.diffuse_heston_1D(params, dt, Nsim, T_fix[-1], 10)
        idx_fix = np.array(T_fix/dt, dtype = int)
        S_fix = forward(T_fix)*Xt[:, idx_fix]
        T_strike = ql.Actual365Fixed().yearFraction(value_date, trade_date)
        S0 = forward(T_strike)
        probas = get_callprobas(S_fix, call_lvl, S0)

        if start_date> value_date+ql.Period('2M'):
            discount=0.95
        else:
            discount=1
        duration = np.sum(T_pay*probas, axis = 0)
        new_spreads, new_fund = geteq_fund2(T_pay, probas, fund_curve, discount)

        res= {"duration": duration,
                "Payment Dates":pay_dates,
                "Early Redemption Proba": probas,
                "Forwards":forward(T_pay),
                'Zero Coupon': df(T_pay), 
                "funding_spread": new_fund}
        
        self.finished.emit(res)



