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

def compute_target_schedule(start:ql.Date,end:ql.Date,frequency:ql.Period)->np.ndarray[ql.Date]:        
    calendar = ql.TARGET()
    convention = ql.ModifiedFollowing
    endConvention = ql.ModifiedFollowing
    rule = ql.DateGeneration.Backward
    endOfMonth = False
    schedule = ql.Schedule(start, end, frequency, calendar, convention, endConvention, rule, endOfMonth)

    return np.array(schedule)

def compute_schedule(issue_date:ql.Date,redemption_date:ql.Date,freq:ql.Period,
                     offset:float,fixing_type:str)->tuple[list[ql.Date],list[ql.Date]]:

    pay_dates=compute_target_schedule(issue_date,redemption_date,freq)[1:]
    if fixing_type=='in arrears':
        fix_dates=np.array([ql.TARGET().adjust(x+offset,ql.ModifiedFollowing) for x in pay_dates])
        return pay_dates,fix_dates
    elif fixing_type=='in advance':
        fix_dates=np.array([ql.TARGET().adjust(x-freq+offset,ql.ModifiedFollowing) for x in pay_dates])
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

def ql_linspace(start_date: ql.Date, end_date: ql.Date, num: int,
                calendar: ql.Calendar = None, convention: ql.BusinessDayConvention = None,
                adjust_endpoints: bool = False) -> np.ndarray:
    """
    Create a QuantLib date schedule equivalent to np.linspace.

    Generates evenly-spaced dates between start_date and end_date, similar to np.linspace
    for date ranges. By default, preserves the exact start and end dates.

    Args:
        start_date: Starting QuantLib date
        end_date: Ending QuantLib date
        num: Number of dates to generate (including start and end)
        calendar: QuantLib calendar for business day adjustments (default: TARGET)
        convention: Business day convention (default: ModifiedFollowing)
        adjust_endpoints: If True, applies business day adjustment to start/end dates.
                         If False (default), preserves exact start/end dates.

    Returns:
        NumPy array of QuantLib dates, evenly spaced from start_date to end_date.
        First and last elements are guaranteed to be start_date and end_date
        (unless adjust_endpoints=True).

    Example:
        >>> start = ql.Date(1, 1, 2024)
        >>> end = ql.Date(1, 1, 2025)
        >>> dates = ql_linspace(start, end, num=5)
        >>> # Returns: [Date(1,1,2024), Date(4,2,2024), Date(7,2,2024),
        >>> #           Date(10,2,2024), Date(1,1,2025)]

    Performance Note:
        - Efficient NumPy-based calculation of date spacing
        - Business day adjustments applied to intermediate dates only
        - O(num) time complexity
    """
    if calendar is None:
        calendar = ql.TARGET()
    if convention is None:
        convention = ql.ModifiedFollowing

    if num < 2:
        raise ValueError("num must be at least 2 (start and end dates)")

    # Calculate total days between dates
    total_days = (end_date - start_date)

    # Generate evenly-spaced day offsets using linspace
    day_offsets = np.linspace(0, total_days, num, dtype=int)

    # Create dates
    dates = []
    for i, offset in enumerate(day_offsets):
        raw_date = start_date + int(offset)

        # Preserve exact start and end dates unless adjust_endpoints is True
        if i == 0:
            dates.append(start_date if not adjust_endpoints else calendar.adjust(raw_date, convention))
        elif i == len(day_offsets) - 1:
            dates.append(end_date if not adjust_endpoints else calendar.adjust(raw_date, convention))
        else:
            # For intermediate dates, apply business day adjustment
            dates.append(calendar.adjust(raw_date, convention))

    return np.array(dates)