import numpy as np
import QuantLib as ql
from bisect import bisect,bisect_left
from datetime import datetime

def find_next_cds_standard(date:ql.Date) -> ql.Date:
    standard_months=[3,6,9,12]
    year=date.year()
    idx=bisect_left(standard_months,date.month())
    if date.dayOfMonth()>20 and date.month() in standard_months:
        idx+=1
        if idx>=len(standard_months):
            idx=0
            year+=1

    return ql.Date(20,standard_months[idx],year)

def compute_target_schedule(start:ql.Date,end:ql.Date,frequency:ql.Period)->list[ql.Date]:        
    calendar = ql.TARGET()
    convention = ql.ModifiedFollowing
    endConvention = ql.ModifiedFollowing
    rule = ql.DateGeneration.Backward
    endOfMonth = False
    schedule = ql.Schedule(start, end, frequency, calendar, convention, endConvention, rule, endOfMonth)

    return list(schedule)

def compute_schedule(issue_date:ql.Date,redemption_date:ql.Date,freq:ql.Period,
                     offset:float,fixing_type:str)->tuple[list[ql.Date],list[ql.Date]]:

    schedule=list(ql.MakeSchedule(issue_date,redemption_date,freq))[1:]
    pay_dates=[ql.TARGET().advance(d,ql.Period('0D')) for d in schedule]
    if fixing_type=='in arrears':
        fix_dates=[ql.TARGET().adjust(x+offset,ql.ModifiedFollowing) for x in schedule]
        return pay_dates,fix_dates
    elif fixing_type=='in advance':
        fix_dates=[ql.TARGET().adjust(x-freq+offset,ql.ModifiedFollowing) for x in schedule]
        return pay_dates,fix_dates
    else:
        raise ValueError(f'{fixing_type} not valid input as fixing type')


def date_converter(date:ql.Date,sep='/')-> str:
    return str(date.dayOfMonth())+sep +str(date.month()) + sep+str(date.year())

def date_to_ql(x:datetime)->ql.Date:
    return ql.Date(x.day,x.month,x.year)

def string_to_ql(char:str,format:str)->ql.Date:
    date=datetime.strptime(char,format)
    return ql.Date(date.day,date.month,date.year)

def ql_to_string(date:ql.Date,format:str) -> str:
    temp=datetime(date.year(),date.month(),date.dayOfMonth())
    return temp.strftime(format)

def GetLastBusinessDate()->ql.Date:
    today=datetime.today()
    cal=ql.TARGET()
    res=cal.adjust(date_to_ql(today)-ql.Period('1D'),ql.Preceding)
    return res

def shift_from_today(shift:str)->str:
    today=datetime.today()
    cal=ql.TARGET()
    res=cal.adjust(date_to_ql(today)+ql.Period(shift),ql.Following)
    res=date_converter(res,sep='.')

    return res