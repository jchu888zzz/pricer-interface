import numpy as np
import scipy as sp
import datetime 
import Pricing.Equity.Model.ConstrNMPy as cNM 

def TimeLeft(date1, date2, conv = 365):
    
    date1 = datetime.datetime.strptime(date1, "%Y-%m-%d").date()
    date2 = datetime.datetime.strptime(date2, "%Y-%m-%d").date()
    return (date2 - date1).days/conv

def BS(F,Div,P_ZC,K,vol,T):
    Strike = (K - Div)/(F - Div)
    d1 = (-np.log(Strike)+0.5*vol**2*T)/(vol*np.sqrt(T))
    d2 = d1 - vol*np.sqrt(T)
    Phi1 = sp.stats.norm.cdf(d1)
    Phi2 = sp.stats.norm.cdf(d2)
    return (F - Div)*P_ZC*(Phi1 - Strike*Phi2)

def Vega(F,Div,P_ZC,K,vol,T):
    Strike = (K - Div)/(F - Div)
    d1 = (-np.log(Strike) + 0.5*vol**2*T)/(vol*np.sqrt(T))
    return (F - Div)*P_ZC*np.sqrt(T)*np.exp(-0.5*d1**2)/np.sqrt(2*np.pi)


def d(ksi,rho,eta,kappa):
    i = complex(0,1)
    return np.sqrt((i*rho*eta*ksi-kappa)**2 + eta**2*ksi*(i+ksi))

def A(ksi,rho,eta,kappa,w):
    i = complex(0,1)
    d0 = kappa - i*rho*eta*ksi 
    return d0 + w*d(ksi, rho, eta, kappa)
    
def g(ksi,rho,eta,kappa):
    return A(ksi,rho,eta,kappa,-1)/A(ksi,rho,eta,kappa,1)

def Psi_bs(ksi,sigma,T):
    i = complex(0,1)
    return np.exp(-0.5*sigma**2*T*ksi*(i+ksi))

def Psi_h(ksi,rho,eta,kappa,theta,T,vol_init):
    return np.exp(kappa*theta/eta**2*(A(ksi,rho,eta,kappa,-1)*T - 2*np.log((1-g(ksi,rho,eta,kappa)*np.exp(-d(ksi,rho,eta,kappa)*T))/(1-g(ksi,rho,eta,kappa))))+vol_init*A(ksi,rho,eta,kappa,-1)*(1-np.exp(-d(ksi,rho,eta,kappa)*T))/(eta**2*(1-g(ksi,rho,eta,kappa)*np.exp(-d(ksi,rho,eta,kappa)*T))))

def Phi0(ksi,rho,eta,kappa,theta,T,vol_init,sigma):
    i = complex(0,1)
    return (Psi_bs(ksi-i,sigma,T) - Psi_h(ksi-i,rho,eta,kappa,theta,T,vol_init))/(ksi*(ksi-i))

def volcarre(rho,eta,kappa,theta,T,vol_init,delta = 0.001):
    i = complex(0,1)
    Psi2 = Psi_h(-i + delta,rho,eta,kappa,theta,T,vol_init)
    Psi1 = Psi_h(-i - delta,rho,eta,kappa,theta,T,vol_init)
    PsiPrime = abs(((Psi2 - Psi1)/(2*delta)).imag)
    return PsiPrime*2/T

def Heston(vol_init,kappa,theta,eta,rho,T,K,F,Div,P_ZC):
    i = complex(0,1)
    vol = np.sqrt(volcarre(rho,eta,kappa,theta,T,vol_init,delta = 0.001))
    Black = BS(F,Div,P_ZC,K,vol,T)
    Integrand = lambda ksi: np.exp(ksi)*(np.exp(-i*ksi*np.log(K/F)) * Phi0(ksi,rho,eta,kappa,theta,T,vol_init,vol)).real
    Xi,Wi = np.polynomial.laguerre.laggauss(10)
    if type(F) == np.ndarray:
        Xi = np.array([Xi]*len(F)).transpose()
        Wi = np.array([Wi]*len(F)).transpose()
    IntgQuad =  np.sum(Wi*Integrand(Xi),axis=0)
    price = Black + P_ZC*F*IntgQuad/np.pi
    return np.maximum(price,0)


def WT(t, t0 = 1, gamma = 5):
    return (np.exp(gamma*t) - 1)/(np.exp(gamma*t0) - 1) * (t <= t0) + (t > t0)

def WK(k,p=0.1):
    return k**p * np.exp(p*(1-k))

def WTK(k,t, t0 = 1, p = 0.1, gamma = 5):
    return WT(t,t0,gamma)*WK(k,p)

def CF(vol_init,kappa,theta,eta,rho,T,Div,Fwd,DF,Strike,Vol,t0=1,p=0.1,gamma=5,conv=365):
    W_TK = WTK(Strike/Fwd,T,t0,p,gamma)
    heston = Heston(vol_init,kappa,theta,eta,rho,T,Strike,Fwd,Div,DF)
    bs = BS(Fwd,Div,DF,Strike,Vol,T)
    vega = Vega(Fwd,Div,DF,Strike,Vol,T) 
    return np.sum((W_TK*(heston-bs)*vega)**2)

def calib_heston(ValuationDate, market_data, t0 = 1, p = 0.1, gamma = 5, conv = 365):
    
    n = market_data.shape[0]
    Maturities = market_data["Expiration Date"]
    T = np.array([TimeLeft(ValuationDate, Mat, conv) for Mat in Maturities])
    K = np.array(market_data["Strike Price"])
    F = np.array(market_data["Forward Price"])
    DF = np.array(market_data["Discount Factor"])
    Vol = np.array(market_data["Volatility"])
    
    OptCF = lambda params : CF(params[0], params[1], params[2], params[3], params[4], T, 0, F, DF, K, Vol, t0, p, gamma, conv)
    
    
    x0 = [0.04,  1,  0.04 ,  0.4, -0.5]
    bounds_min = [1e-6, 1e-4, 1e-6, 1e-4, -1]
    bounds_max = [1, 4, 1, 2, 0]
    bounds = sp.optimize.Bounds(bounds_min, bounds_max)
    
    
    res = cNM.constrNM(OptCF, [0.04,  1,  0.04 ,  0.4, -0.5], bounds_min, bounds_max, full_output = True)
    opt = res["xopt"]

    #res = minimize(OptCF, x0 = x0, bounds = bounds, method = "SLSQP")
    #opt = res.x
    
    return opt


def sample_vol(V_0,kappa,theta,eta,Psi_c,dt,U):
    
    Vt = V_0.copy()
    
    mt = theta + (V_0 - theta)*np.exp(-kappa*dt)
    st = V_0*eta**2*np.exp(-kappa*dt)*(1 - np.exp(-kappa*dt))/kappa + theta*eta**2*(1 - np.exp(-kappa*dt))**2/(2*kappa)
    Psi_t = st/mt**2
    
    idx1 = (Psi_t <= Psi_c)
    idx2 = (Psi_t > Psi_c)
    
    term1 = 2/Psi_t[idx1]
    bt = term1 - 1 + np.sqrt(term1*(term1 - 1))
    at = mt[idx1]/(1 + bt)
    Zv = sp.stats.norm.ppf(U[idx1])
    Vt[idx1] = at*(np.sqrt(bt) + Zv)**2
    
    term2 = Psi_t[idx2]
    pt = (term2 - 1)/(term2 + 1)
    beta_t = (1 - pt)/mt[idx2]
    Uv = U[idx2]  
    Vt[idx2] = 1/beta_t*np.log((1 - pt)/(1 - Uv))*((pt < Uv) & (Uv <= 1))    
    
    return Vt


def diffuse_heston_1D(params, dt, Nsim, T, seed, psi_c = 1.5):
    
    vol_init, kappa, theta, eta, rho = params

    N = int(T/dt)
    Xt = np.zeros((Nsim,N+1))
    Xt[:,0] = 1 
    Vs = np.ones(Nsim)*vol_init
    
    np.random.seed(seed)
    
    for i in range(N):
        
        U = np.random.uniform(size = Nsim)
        Vt = sample_vol(Vs,kappa,theta,eta,psi_c,dt,U)
        
        Zx = np.random.normal(size = Nsim)
        logXt = np.log(Xt[:,i]) + rho/eta*(Vt - Vs - kappa*theta*dt) + (kappa*rho/eta - 0.5)*(Vt + Vs)/2*dt + np.sqrt((Vt + Vs)/2*dt*(1 - rho**2))*Zx
        Xt[:,i+1] = np.exp(logXt)
    
        Vs = np.maximum(Vt,0)
    
    return Xt
            
def generate_correl_heston(dim,correls_list):
    
    cov = np.zeros((2*dim, 2*dim))

    Bloc1 = np.ones((dim, dim))
    Bloc1[np.where(Bloc1 - np.eye(dim) == np.tril(Bloc1))] = correls_list[: - dim]
    Bloc1 -= np.triu(Bloc1)
    cov[:dim, :dim] = Bloc1
    

    Bloc2 = np.ones((dim , dim))
    Bloc2[np.where(Bloc2 - np.eye(dim) == np.tril(Bloc2))] = correls_list[: - dim]
    Bloc2 -= np.triu(Bloc2)
    Bloc2 += Bloc2.T
    np.fill_diagonal(Bloc2, np.ones(dim))
    cov[dim: , : dim] = (correls_list[-dim:]*Bloc2).T

    impcorr_prod = correls_list[-dim:].reshape(dim,1)*correls_list[-dim:].reshape(1,dim)
    impcorr_prod -= np.triu(impcorr_prod)
    cov[dim: , dim:] = impcorr_prod*Bloc1
    
    
    cov += cov.T
    np.fill_diagonal(cov, np.ones(2*dim))
    
    return cov    
    
    
def diffuse_heston_MultiCase(params, dim, cov, dt, Nsim, T, seed, psi_c = 1.5):
    
    N = int(T/dt)
    Xt = np.zeros((dim,Nsim,N+1))
    Xt[:,:,0] = 1
    Vs = np.ones((Nsim,dim))*params[:,0]
    
    np.random.seed(seed)
    
    for i in range(N):
        
        Z = sp.stats.multivariate_normal.rvs(cov = cov, size = Nsim)
        
        logXt = np.log(Xt[:,:,i]) - 0.5*Vs.T*dt + np.sqrt(Vs.T*dt)*Z[:,:dim].T
        Xt[:,:,i+1] = np.exp(logXt)
    
        Zv = Z[:,dim:]
        Uv = sp.stats.norm.cdf(Zv)
        
        for j in range(dim):
            
            kappa_j, theta_j, eta_j = params[j,1:-1]
            Vs[:,j] = sample_vol(Vs[:,j],kappa_j,theta_j,eta_j,psi_c,dt,Uv[:,j])
            
    return Xt
