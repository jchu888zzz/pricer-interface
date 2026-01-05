from Pricing.Utilities import Dates,Functions

DIC_UNDL={'BFRTEC10':{'tag':'BFR','tenor':'10Y','currency':'EUR','vol_shift':0.8},
        'SOLDE10E':{'tag':'GDBR_CMT','tenor':'10Y','currency':'EUR','vol_shift':0.8},
        'SOLBE10E':{'tag':'BGB_CMT','tenor':'10Y','currency':'EUR','vol_shift':0.8},
        'SOITA10Y':{'tag':'BTP_CMT','tenor':'10Y','currency':'EUR','vol_shift':0.8},
        'SOLIT1OE':{'tag':'BTP_CMT','tenor':'10Y','currency':'EUR','vol_shift':0.8},
        'H15T10Y':{'tag':'CMT','tenor':'10Y','currency':'USD','vol_shift':0.8}}

_DIC_FREQ_SWAPTION={"EUR":{"delta_fix":1,"delta_float":0.5},
                    "USD":{"delta_fix":0.5,"delta_float":0.25}}

def select_rates(rates:np.ndarray,simu_dates:np.ndarray[ql.Date],fix_dates:np.ndarray[ql.Date],
                nb_sub_fix_points:None| int) -> np.ndarray:
    if not nb_sub_fix_points:
        idx=Functions.find_idx(simu_dates,fix_dates)
        return rates[:,idx].T
    else:
        res=[]
        for i,(d1,d2) in enumerate(zip(fix_dates,fix_dates[1:])):
            sub_schedule=Dates.ql_linspace(d1,d2,nb_sub_fix_points)
            sub_idx=Functions.find_idx(simu_dates,sub_schedule)
            res.append(rates[:,sub_idx].T)
        return np.array(res)
    
def compute_correlation(df:pd.DataFrame,ticker1:str,ticker2:str) -> np.ndarray:
    
    shift_wednesday={1:3,2:2,3:1,4:0,5:-1,6:-2,7:-3}
    effectiveDate = ql.Date.todaysDate() - ql.Period('2Y')
    effectiveDate=effectiveDate+shift_wednesday[effectiveDate.weekday()]
    terminationDate = ql.Date.todaysDate()
    schedule = ql.MakeSchedule(effectiveDate,terminationDate,ql.Period('1W'), calendar=ql.TARGET(), 
                                convention=ql.ModifiedFollowing, terminalDateConvention =ql.ModifiedFollowing, 
                                rule=ql.DateGeneration.Forward,endOfMonth=False)
    def format(date:ql.Date)->datetime.date:
        return datetime(date.year(),date.month(),date.dayOfMonth()).date()
    
    schedule=np.array([format(x) for x in schedule ])
    dates=np.array([x.date() for x in df.index])
    idxs=Functions.find_idx(dates,schedule)
    df=df.loc[df.index[idxs]]
    df=df.diff()
    df=df.dropna()
    res=np.corrcoef(df[ticker1].to_numpy(),df[ticker2].to_numpy())
    
    return res

def get_model(calc_date:ql.Date,mkt_data:dict,currency:str,undl:str|None) -> dict:

    currency=DIC_UNDL[undl]['currency']
    tenor=DIC_UNDL[undl]['tenor']
    tag=DIC_UNDL[undl]['tag']
    curve,risky_curve=Classic.get_curves(calc_date,mkt_data,currency,'Classical')

    swaptions_rate=Rate_Instruments.select_and_prepare_swaptions(mkt_data['swaption'],curve,calc_date,currency)
    swaptions_rate=[x for x in swaptions_rate if x.strike_type=='ATM' and x.tenor==tenor]
    model_rate=HullWhite.Calibration(curve,swaptions_rate)

    cmt_curve=CMT.get_curve(calc_date,mkt_data['cmt'],currency,tag)

    df_cmt=mkt_data['swaption'].copy()
    df_cmt['Quote']*=DIC_UNDL[undl]['vol_shift']
    swaptions_cmt=Rate_Instruments.select_and_prepare_swaptions(df_cmt,cmt_curve,calc_date,currency)
    swaptions_cmt=[x for x in swaptions_cmt if x.strike_type=='ATM' and x.tenor==tenor]
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

    # cov_matrix=np.array([[1,0.8],[0.8,1]])

    model=HW_CMT(model_rate,model_cmt,cov_matrix)
    return {'risky_curve':risky_curve,
            'curve':curve,
            'model':model,
            'calc_date':calc_date}

class HW_CMT:

    def __init__(self,model_rate:HullWhite.HW,model_cmt:HullWhite.HW,cov_matrix):
        self.model_rate=model_rate
        self.model_cmt=model_cmt
        self.curve=model_rate.curve
        self.DF=model_rate.DF
        self.compute_discount_factor_from_rates=model_rate.compute_discount_factor_from_rates
        self.compute_deposit_from_rates=model_rate.compute_deposit_from_rates
        self.cov_matrix=cov_matrix
    
    def generate_rates(self,calc_date:ql.Date,maturity_date:ql.Date,
                        cal=ql.Thirty360(ql.Thirty360.BondBasis),Nbsimu=10000,seed=5) -> dict:

        rng=np.random.default_rng(int(seed))
        T_maturity=cal.yearFraction(calc_date,maturity_date)
        schedule=Dates.compute_target_schedule(calc_date,maturity_date,ql.Period('1D'))
        grid=np.array([cal.yearFraction(calc_date,x) for x in schedule[1:] ])
        Z=rng.multivariate_normal(mean=[0,0],cov=self.cov_matrix,size=(len(grid),Nbsimu))

        model_rate=self.model_rate
        model_cmt=self.model_cmt

        rates=np.zeros((Nbsimu,len(grid)))
        rates_cmt=np.zeros((Nbsimu,len(grid)))
        rates[:,0]=model_rate.instantaneous_f(0,h=0.05)
        rates_cmt[:,0]=model_cmt.instantaneous_f(0,h=0.05)
        
        prev_alpha=model_rate.alpha_T(grid[0],T_maturity)
        prev_alpha_cmt=model_cmt.alpha_T(grid[0],T_maturity)

        prev_var=model_rate.var_(grid[0])
        prev_var_cmt=model_cmt.var_(grid[0])
        for i in range(1,len(grid)):
            delta=grid[i]-grid[i-1]
            alpha=model_rate.alpha_T(grid[i],T_maturity)
            var=model_rate.var_(grid[i])
            rates[:,i]=( rates[:,i-1]*np.exp(-model_rate.a*delta) +alpha - prev_alpha*np.exp(-model_rate.a*delta) +
                        np.sqrt(var-prev_var*np.exp(-2*model_rate.a*delta))*Z[:,:,0][i] )
            prev_alpha=alpha
            prev_var=var

            alpha_cmt=model_cmt.alpha_T(grid[i],T_maturity)
            var_cmt=model_cmt.var_(grid[i])
            rates_cmt[:,i]=( rates_cmt[:,i-1]*np.exp(-model_cmt.a*delta) +alpha_cmt - prev_alpha_cmt*np.exp(-model_cmt.a*delta) +
                        np.sqrt(var_cmt-prev_var_cmt*np.exp(-2*model_cmt.a*delta))*Z[:,:,1][i] )
            prev_alpha_cmt=alpha_cmt
            prev_var_cmt=var_cmt

        return {'rates':rates,'rates_cmt':rates_cmt,'schedule':schedule}
    
    def compute_cmt_from_rates(self,rates_cmt:np.ndarray,t:float,tenor1:str,option:str):
        model_cmt=self.model_cmt
        curve_cmt=model_cmt.curve
        if option=='unadjusted':
            return model_cmt.compute_cms_from_rates(rates_cmt,t,tenor1,1,1)
        elif option=='adjusted':
            fwd_adjusted=curve_cmt.forward_cms(t,tenor1,option)
            undl=model_cmt.compute_cms_from_rates(rates_cmt,t,tenor1,1,1)
            adjustment= fwd_adjusted -np.mean(undl)
            return undl+adjustment
        else:
            raise ValueError(f'compute_cmt_from_rates: {option} not implemented')
    
    def compute_single_undl_from_rates(self, data_rates:dict, fix_dates:list[ql.Date], undl1:str,
                                        nb_sub_fix_points:int|None=None, include_rates=True,
                                        option='adjusted') -> dict:
        """
        Compute CMT underlying rates from simulated short rates.

        Args:
            nb_sub_fix_points: If None, returns shape (len(fix_dates), nb_simu)
                                If int, returns shape (len(fix_dates)-1, nb_sub_fix_points, nb_simu)
        """
        tenor1 = DIC_UNDL[undl1]['tenor']
        nb_simu=data_rates['rates'].shape[0]
        calendar = self.curve.calendar
        calc_date = self.curve.calc_date

        if nb_sub_fix_points is None:
            # Simple case: one rate per fix date
            rates_cmt = select_rates(data_rates['rates_cmt'], data_rates['schedule'], fix_dates, None)
            fixgrid = np.array([calendar.yearFraction(calc_date, d) for d in fix_dates])

            undl = np.array([self.compute_cmt_from_rates(rates_cmt[i], t, tenor1, option)
                            for i, t in enumerate(fixgrid)])

            result = {'undl': undl, 'nbsimu': nb_simu}
            if include_rates:
                rates = select_rates(data_rates['rates'], data_rates['schedule'], fix_dates, None)
                result['rates'] = rates
            return result
        else:
            # Depth case: subdivide periods
            rates_cmt = select_rates(data_rates['rates_cmt'], data_rates['schedule'], fix_dates, nb_sub_fix_points)
            n_periods = len(fix_dates) - 1

            # Pre-compute all sub-schedules and fixgrids

            # Compute rates for each period
            res=np.zeros((n_periods, nb_sub_fix_points, nb_simu))
            for i in range(n_periods):
                sub_fixgrids=(calendar.yearFraction(calc_date, d)
                                        for d in Dates.ql_linspace(fix_dates[i], fix_dates[i+1], nb_sub_fix_points))
                res[i] = np.array([self.compute_cmt_from_rates(rates_cmt[i][j], t, tenor1,option)
                            for j, t in enumerate(sub_fixgrids)])

            result = {'undl': res, 'nbsimu': nb_simu}
            if include_rates:
                rates = select_rates(data_rates['rates'], data_rates['schedule'], fix_dates, nb_sub_fix_points)
                result['rates'] = rates[:, -1, :]
            return result

    def compute_spread_undl_from_rates(self, data_rates:dict, fix_dates:list[ql.Date],
                                        undl1:str, undl2:str, nb_sub_fix_points:int|None=None,
                                        include_rates=True,option='adjusted') -> dict:
        """
        Compute CMT-CMS spread from simulated short rates.

        Args:
            nb_sub_fix_points: If None, returns shape (len(fix_dates), nb_simu)
                                If int, returns shape (len(fix_dates)-1, nb_sub_fix_points, nb_simu)
        """
        tenor1 = DIC_UNDL[undl1]['tenor']
        cur2, _, tenor2 = undl2.split()
        nb_simu=data_rates['rates'].shape[0]
        delta_fix = _DIC_FREQ_SWAPTION[cur2]["delta_fix"]
        delta_float = _DIC_FREQ_SWAPTION[cur2]["delta_float"]

        calendar = self.curve.calendar
        calc_date = self.curve.calc_date
        model_rate = self.model_rate

        if nb_sub_fix_points is None:
            # Simple case: one rate per fix date
            rates_cmt = select_rates(data_rates['rates_cmt'], data_rates['schedule'], fix_dates, None)
            rates = select_rates(data_rates['rates'], data_rates['schedule'], fix_dates, None)
            fixgrid = np.array([calendar.yearFraction(calc_date, d) for d in fix_dates])

            undl = np.array([
                self.compute_cmt_from_rates(rates_cmt[i], t, tenor1, option) -
                model_rate.compute_cms_from_rates(rates[i], t, tenor2, delta_fix, delta_float)
                for i, t in enumerate(fixgrid)])

            result = {'undl': undl, 'nbsimu': nb_simu}
            if include_rates:
                result['rates'] = rates
            return result
        else:
            # Depth case: subdivide periods
            rates_cmt = select_rates(data_rates['rates_cmt'], data_rates['schedule'], fix_dates, nb_sub_fix_points)
            rates = select_rates(data_rates['rates'], data_rates['schedule'], fix_dates, nb_sub_fix_points)
            n_periods = len(fix_dates) - 1

            # Compute spread for each period
            res=np.zeros((n_periods, nb_sub_fix_points, nb_simu))
            for i in range(n_periods):
                sub_fixgrids=(calendar.yearFraction(calc_date, d)
                                        for d in Dates.ql_linspace(fix_dates[i], fix_dates[i+1], nb_sub_fix_points))
                res[i]=np.array([
                    self.compute_cmt_from_rates(rates_cmt[i][j], t, tenor1,option) -
                    model_rate.compute_cms_from_rates(rates[i][j], t, tenor2, delta_fix, delta_float)
                    for j, t in enumerate(sub_fixgrids)])

            result = {'undl': res, 'nbsimu': nb_simu}
            if include_rates:
                result['rates'] = rates[:, -1, :]
            return result
