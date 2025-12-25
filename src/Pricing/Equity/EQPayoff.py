
import numpy as np

freq_dic = {"At maturity": "At maturity", "Monthly":1/12, "Quarterly": 0.25, "Semi-annually": 0.5, "Annually": 1}
final_opt_dic = {"No Option": "full_KG", "Put Down & In": "PDI", "Put Spread": "partial_KG"}
knock_lvl_dic = {"No Option" : "100%", "Put Down & In": "60%", "Put Spread": "80%"}
dt, Nsim, sd = 1/365, 10000, 10

def price_reverse(c, T, S0, D, Xt, forward, df, freq = "At maturity", type = "Final"):

    T_grid = T if freq == "At maturity" else np.arange(freq, T + freq, freq)
    price = c*np.sum(df(T_grid), axis = 0)

    if type == "Final":
        ST = float(forward(T))*Xt[:,-1]
        price += np.mean((ST >= D*S0) + ST/S0*(ST < D*S0))*df(T)

    elif type == "Daily":
        Nsim, N = Xt.shape
        dt = T/(N - 1)
        T_fix = np.arange(dt, T + dt, dt)
        S_fix = forward(T_fix)*Xt[:,1:]
        idx_nopdi = np.where(np.min(S_fix, axis = 1) >= D*S0)[0]
        idx_nopdi = np.union1d(idx_nopdi, np.where(S_fix[:,-1] >= S0)[0])
        idx_pdi = np.setdiff1d(np.arange(Nsim), idx_nopdi)
        price += (len(idx_nopdi) + np.sum(S_fix[idx_pdi, -1])/S0)/Nsim*df(T)
        
    return price

def coupon_reverse(uf, T, S0, D, Xt, forward, df, freq = "At maturity", typo = "Final"):

    T_grid = T if freq == "At maturity" else np.arange(freq, T + freq, freq)
    term1 = np.sum(df(T_grid), axis = 0)

    if typo == "Final":
        ST = float(forward(T))*Xt[:,-1]
        term2 = np.mean((ST >= D*S0) + ST/S0*(ST < D*S0))*df(T)

    elif typo == "Daily":
        Nsim, N = Xt.shape
        dt = T/(N - 1)
        T_fix = np.arange(dt, T + dt, dt)
        S_fix = forward(T_fix)*Xt[:,1:]
        idx_nopdi = np.where(np.min(S_fix, axis = 1) >= D*S0)[0]
        idx_nopdi = np.union1d(idx_nopdi, np.where(S_fix[:,-1] >= S0)[0])
        idx_pdi = np.setdiff1d(np.arange(Nsim), idx_nopdi)
        term2 = (len(idx_nopdi) + np.sum(S_fix[idx_pdi, -1])/S0)/Nsim*df(T)

    c = (1 - uf - term2)/term1
    return c

def price_autocall(c, T, S0, call_lvl, cpn_lvl, knock_lvl, freq, mem, per_nocall, final_option, Xt, forward, df):
    
    Nsim, N = Xt.shape
    dt = T/(N - 1)
    
    T_fix = np.arange(freq, T + freq, freq)
    index_fix = np.array(T_fix/dt, dtype = int)
    S_fix = forward(T_fix)*Xt[:,index_fix]
    Mem = np.zeros(Nsim)
    Redeemed = []
    price = 0

    T1_fix = T_fix[:per_nocall]
    T2_fix = T_fix[per_nocall:]

    if cpn_lvl < call_lvl:
        for i in range(T1_fix.shape[0]):

            idx_coupon = np.where(S_fix[:,i] >= cpn_lvl*S0)[0]
            price += np.sum((1 + Mem[idx_coupon]))*c/Nsim*df(T1_fix[i])
            Mem[idx_coupon] = 0

            idx_mem = np.setdiff1d(np.arange(Nsim), idx_coupon)
            Mem[idx_mem] += mem
    else:
        Mem += per_nocall*mem
        
    for i in range(T2_fix.shape[0]):

        idx_call = np.where(S_fix[:,i + per_nocall] >= call_lvl*S0)[0]
        idx_call = np.setdiff1d(idx_call, Redeemed)
        price +=  np.sum(1 + (1 + Mem[idx_call])*c)/Nsim*df(T2_fix[i])
        Redeemed = np.union1d(Redeemed, idx_call)

        idx_coupon = np.where(S_fix[:,i + per_nocall] >= cpn_lvl*S0)[0]
        idx_coupon = np.setdiff1d(idx_coupon, Redeemed)
        price += np.sum((1 + Mem[idx_coupon]))*c/Nsim*df(T2_fix[i])
        Mem[idx_coupon] = 0
        
        idx_mem = np.setdiff1d(np.arange(Nsim), idx_coupon)
        Mem[idx_mem] += mem

    NoRedeemed = np.setdiff1d(np.arange(Nsim), Redeemed)
    
    if final_option == "full_KG":

        price += len(NoRedeemed)/Nsim*df(T_fix[-1])

    elif final_option == "PDI":

        idx_nopdi = np.where(S_fix[:,-1] >= knock_lvl*S0)[0]
        idx_nopdi = np.intersect1d(idx_nopdi, NoRedeemed)
        price += len(idx_nopdi)/Nsim*df(T_fix[-1])
        idx_pdi = np.setdiff1d(NoRedeemed, idx_nopdi)  
        price += (np.sum(S_fix[idx_pdi, -1])/S0)/Nsim*df(T_fix[-1])

    elif final_option == "partial_KG":

        price += np.sum(np.maximum(S_fix[NoRedeemed, -1]/S0, knock_lvl))/Nsim*df(T_fix[-1])

    return price

def coupon_autocall(uf, T, S0, call_lvl, cpn_lvl, knock_lvl, freq, mem, per_nocall, final_option, Xt, forward, df):
    
    Nsim, N = Xt.shape
    dt = T/(N - 1)
    c = 0.05
    T_fix = np.arange(freq, T + freq, freq)
    index_fix = np.array(T_fix/dt, dtype = int)
    S_fix = forward(T_fix)*Xt[:,index_fix]
    Mem = np.zeros(Nsim)
    Redeemed = []
    price = 0

    T1_fix = T_fix[:per_nocall]
    T2_fix = T_fix[per_nocall:]
    Probas = np.zeros(T2_fix.shape[0])

    if cpn_lvl < call_lvl:
        for i in range(T1_fix.shape[0]):

            idx_coupon = np.where(S_fix[:,i] >= cpn_lvl*S0)[0]
            price += np.sum((1 + Mem[idx_coupon]))*c/Nsim*df(T1_fix[i])
            Mem[idx_coupon] = 0

            idx_mem = np.setdiff1d(np.arange(Nsim), idx_coupon)
            Mem[idx_mem] += mem
    else:
        Mem += per_nocall*mem

    for i in range(T2_fix.shape[0]):

        idx_call = np.where(S_fix[:,i + per_nocall] >= call_lvl*S0)[0]
        idx_call = np.setdiff1d(idx_call, Redeemed)
        Probas[i] = len(idx_call)/Nsim
        price +=  np.sum(1 + (1 + Mem[idx_call])*c)/Nsim*df(T2_fix[i])
        Redeemed = np.union1d(Redeemed, idx_call)

        idx_coupon = np.where(S_fix[:,i + per_nocall] >= cpn_lvl*S0)[0]
        idx_coupon = np.setdiff1d(idx_coupon, Redeemed)
        price += np.sum((1 + Mem[idx_coupon]))*c/Nsim*df(T2_fix[i])
        Mem[idx_coupon] = 0
        
        idx_mem = np.setdiff1d(np.arange(Nsim), idx_coupon)
        Mem[idx_mem] += mem
    
    NoRedeemed = np.setdiff1d(np.arange(Nsim), Redeemed)
    
    if final_option == "full_KG":

        prob_kg = len(NoRedeemed)/Nsim
        Probas[-1] += prob_kg
        price += prob_kg*df(T_fix[-1])
        p3 = 0 

    elif final_option == "PDI":

        idx_nopdi = np.where(S_fix[:,-1] >= knock_lvl*S0)[0]
        idx_nopdi = np.intersect1d(idx_nopdi, NoRedeemed)
        prob_nopdi = len(idx_nopdi)/Nsim
        Probas[-1] += prob_nopdi
        price += prob_nopdi*df(T_fix[-1])
        
        idx_pdi = np.setdiff1d(NoRedeemed, idx_nopdi) 
        p3 = (np.sum(S_fix[idx_pdi, -1])/S0)/Nsim*df(T_fix[-1])
        price += p3

    elif final_option == "partial_KG":

        p3 = np.sum(np.maximum(S_fix[NoRedeemed, -1]/S0, knock_lvl))/Nsim*df(T_fix[-1])
        price += p3

    p1 = np.sum(Probas*df(T2_fix), axis = 0)
    p2 = (price - (p1 + p3))/c
    
    return (1 - uf - (p1 + p3))/p2

def price_digitale(c, T, S0, cpn_lvl, knock_lvl, freq, mem, final_option, Xt, forward, df):

    Nsim, N = Xt.shape
    dt = T/(N - 1)
    T_fix = np.arange(freq, T + freq, freq)
    index_fix = np.array(T_fix/dt, dtype = int)
    S_fix = forward(T_fix)*Xt[:, index_fix]
    Mem = np.zeros(Nsim)
    price = 0

    for i in range(T_fix.shape[0]):

        idx_c = np.where(S_fix[:,i] >= cpn_lvl*S0)[0]
        price += np.sum((1 + Mem[idx_c]))/Nsim*c*df(T_fix[i])
        Mem[idx_c] = 0

        idx_noc = np.setdiff1d(np.arange(Nsim), idx_c)
        Mem[idx_noc] += mem

    if final_option == "full_KG":
        price += df(T_fix[-1])
    elif final_option == "PDI":
        idx_nopdi = np.where(S_fix[:,-1] >= knock_lvl*S0)[0]
        price += len(idx_nopdi)/Nsim*df(T_fix[-1])
        idx_pdi = np.setdiff1d(np.arange(Nsim), idx_nopdi)
        price += (np.sum(S_fix[idx_pdi, -1])/S0)/Nsim*df(T_fix[-1])
    elif final_option == "partial_KG":
        price += np.mean(np.minimum(np.maximum(S_fix[:,-1]/S0, knock_lvl), 1))*df(T_fix[-1])
        
    return price

def coupon_digitale(uf, T, S0, cpn_lvl, knock_lvl, freq, mem, final_option, Xt, forward, df):

    Nsim, N = Xt.shape
    dt = T/(N - 1)
    T_fix = np.arange(freq, T + freq, freq)
    n = T_fix.shape[0]
    index_fix = np.array(T_fix/dt, dtype = int)
    S_fix = forward(T_fix)*Xt[:, index_fix]

    index_c = [np.arange(Nsim)]
    for i in range(n):
        index_c.append(np.where(S_fix[:,i] >= cpn_lvl*S0)[0])

    probs = np.zeros((n, n))
    Mem = np.zeros((n, n))
    for j in range(n):
        Mem[j, j:] = j + 1 if mem == 1 else 1
        idx_j = index_c[j + 1]
        for i in range(j + 1):
            idx_i = index_c[j - i]
            idx_between = index_c[j - i + 1:j + 1]
            idx_between = np.unique(np.concatenate(idx_between)) if len(idx_between) != 0 else idx_between
            probs[i, j] = len(np.setdiff1d(np.intersect1d(idx_j, idx_i), idx_between))/Nsim
    
    p2 = np.sum(Mem*probs*df(T_fix))

    if final_option == "full_KG":
        p1 = df(T_fix[-1])
        p3 = 0
    elif final_option == "PDI":
        idx_nopdi = np.where(S_fix[:,-1] >= knock_lvl*S0)[0]
        p1 = len(idx_nopdi)/Nsim*df(T_fix[-1])
        idx_pdi = np.setdiff1d(np.arange(Nsim), idx_nopdi)
        p3 = (np.sum(S_fix[idx_pdi, -1])/S0)/Nsim*df(T_fix[-1])
    elif final_option == "partial_KG":
        idx_noknock = np.where(S_fix[:,-1] >= S0)[0]
        p1 = len(idx_noknock)/Nsim*df(T_fix[-1])
        idx_knock = np.setdiff1d(np.arange(Nsim), idx_noknock)
        p3 = np.sum(np.maximum(S_fix[idx_knock, -1]/S0, knock_lvl))/Nsim*df(T_fix[-1])
        
    c = (1 - uf - (p1 + p3))/p2
    return c