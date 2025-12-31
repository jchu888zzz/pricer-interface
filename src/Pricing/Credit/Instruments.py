import pandas as pd
import QuantLib as ql
import numpy as np
import Pricing.Utilities.Data_File as Data_File
import Pricing.Utilities.Dates as Dates

def retrieve_credit_index_from_datafile(File:pd.ExcelFile,name_index:str):
    " select strictly by  undl name and create index class"

    df=Data_File.concatenate_df(File,name_index,sheet_list=('Credit Spreads','CDO','Credit Index Defaults'))
    df['underlying']=df['Description'].map(lambda x : x.split(' ')[1])
    mask=df['underlying'].eq(name_index)
    df=df[mask]

    Attachments=[]
    Detachments=[]
    correls=[]
    for i,line in df.iterrows():
        char=line['Description'].split()
        quote=line['Quote']
        name=char[1]
        if name==name_index:
            if char[0]=='CDS_index_nb_defaults':
                nb_default=int(quote)
            elif char[0]=='Credit_spread':
                spread=quote
            elif char[0]=='CDO_tranche_base_correlation':
                Attachments.append(float(char[2]))
                Detachments.append(float(char[3]))
                correls.append(quote)
    
    Attachments=tuple(sorted(Attachments))
    Detachments=tuple(sorted(Detachments))
    correls=tuple(sorted(correls))
    return Credit_Index(name,Attachments,Detachments,correls,nb_default,spread)


class Credit_Index:
    
    def __init__(self,undl:str,attachments:tuple,detachments:tuple,
                 correls:tuple,nb_default:int,spread:float):
        self.undl=undl
        self.attach,self.detach=attachments,detachments
        self.base_correls=correls
        self.spread=spread
        self.nb_default=nb_default

        date=Dates.GetLastBusinessDate()
        start=Dates.find_next_cds_standard(date)
        end=start+ql.Period('5Y')
        self.schedule=Dates.compute_target_schedule(start,end,ql.Period('3M'))

        dic_word={'iTraxx_Main':125,'iTraxx_Xover':75,'CDX_IG':125,'CDX_HY':100}

        for key in dic_word.keys():
            if self.undl.startswith(key):
                self.nb_entities=dic_word[key]
    
    def __repr__(self):
        return (f'{self.undl}\n'
                f'Attachments:{self.attach}\n'
                f'Detachments:{self.detach}\n'
                f'correl:{self.rho}\n'
                f'Spread:{self.spread}\n'
                f'Nb defaults:{self.nb_default}\n')
        
class Credit_Single:

    def __init__(self,undl:str,dic_data:dict,calc_date:ql.Date):
        self.undl=undl
        self.calc_date=calc_date
        self.retrieve_info(dic_data,calc_date)
        if 'sub' in undl.lower():
            self.R=0.2
        else:
            self.R=0.4 
    
    def retrieve_info(self,dic_data:dict,calc_date:ql.Date,cal=ql.Actual365Fixed()):

        self.tenors=tuple(sorted(list(dic_data.keys()),key=lambda x: ql.Period(x)))
        self.quotes=tuple([dic_data[t] for t in self.tenors])

        start=Dates.find_next_cds_standard(calc_date)
        end=start+ql.Period(self.tenors[-1])
        self.schedule=Dates.compute_target_schedule(start,end,ql.Period('3M'))

        self.tgrid=np.array([cal.yearFraction(calc_date,d) for d in self.schedule])

    def __repr__(self):
        return (f'Credit {self.undl}\n'
                f'Tenors {self.tenors}\n'
                f'Quotes {self.quotes}\n')


def retrieve_credit_single_from_dataframe(df:pd.DataFrame,undl_name:str,calc_date:ql.Date)-> Credit_Single:
    tenors=[]
    quotes=[]
    for i,line in df.iterrows():
        char=line['Description'].split()
        name=char[1]
        quote=line['Quote']
        tenor=char[-1]
        if name==undl_name:
            if char[0]=='Credit_spread':
                tenors.append(tenor)
                quotes.append(quote)

    return Credit_Single(undl_name,{x:y for x,y in zip(tenors,quotes)},calc_date)  
