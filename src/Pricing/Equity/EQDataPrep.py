import numpy as np
import pandas as pd
import scipy as sp
import datetime
import os

ice_path = "//Umilp-p2.cdm.cm-cic.fr/cic-lai-lae-cigogne$/1_Structuration/13bis_ICE"
spread_path = "//Umilp-p2.cdm.cm-cic.fr/cic-lai-lae-cigogne$/1_Structuration/6_Lexifi/Credit spread CIC"
markit_path = "//Umilp-p2.cdm.cm-cic.fr/cic-lai-lae-cigogne$/1_Structuration/13_Markit/Market_Data"

def TimeLeft(date1, date2, conv = 365):
    
    date1 = datetime.datetime.strptime(date1, "%Y-%m-%d").date()
    date2 = datetime.datetime.strptime(date2, "%Y-%m-%d").date()
    return (date2 - date1).days/conv

def prep_data_markit(ValuationDate, MarkitPath, stock, ref_table):
                    
    markit_name, excel_name = ref_table[ref_table.Underlyings.eq(stock)].iloc[0,[1,2]]
    market_data = pd.read_csv(os.path.join(MarkitPath, excel_name + "".join(ValuationDate.split("-")) + ".csv"))
    market_data = market_data[market_data["Stock Name"].eq(markit_name)]
    market_data.index = range(market_data.shape[0])
    market_data["Expiration Date"] = market_data["Expiration Date"].apply(lambda u : "-".join(u.split("/")))
    T = np.array([TimeLeft(ValuationDate, date) for date in market_data["Expiration Date"].unique()])
    forward = sp.interpolate.interp1d(T, market_data["Forward Price"].unique(), fill_value = "extrapolate")
    df = sp.interpolate.interp1d(T, market_data["Discount Factor"].unique(), fill_value = "extrapolate")
    
    return market_data, forward, df

def ice_prep(ValuationDate, ice_data, stock, map_table):
    
    ice_stock_name = map_table[map_table["SHORT NAME LEXIFI"].eq(stock)]["Name ICE"].iloc[-1]
    ice_data = ice_data[ice_data.ASSETNAME.eq(ice_stock_name)]
    ice_data.index = ice_data.SDKey.to_list()
    
    forward_curve = ice_data.filter(like = "EQ_FORWARDCURVE_", axis = 0)
    forward_curve.index = range(forward_curve.shape[0])
    Spot = forward_curve.REFSPOT.unique()[0]
    forward_curve_expiry = forward_curve.EXPIRYDATE.apply(lambda u : "-".join([u.split("/")[-1], u.split("/")[0], u.split("/")[1]])).to_list()
    dico_forward = dict((T, F) for T, F in zip(forward_curve_expiry, forward_curve.FORWARDPRICE))
    forward = sp.interpolate.interp1d(np.array([TimeLeft(ValuationDate, T) for T in forward_curve_expiry]), forward_curve.FORWARDPRICE.to_list(), fill_value = "extrapolate")
    
    df = sp.interpolate.interp1d(np.array([TimeLeft(ValuationDate, T) for T in forward_curve_expiry]), forward_curve.DISCOUNTFACTOR.to_list(), fill_value = "extrapolate")
    
    vol_surface = ice_data.filter(like = "EQ_VOLSURFACE_", axis = 0)
    vol_surface.index = range(vol_surface.shape[0])
    vol_surface_expiry = vol_surface.EXPIRYDATE.apply(lambda u : "-".join([u.split("/")[-1], u.split("/")[0], u.split("/")[1]])).to_list()
    Strike = vol_surface.STRIKE.apply(lambda u : u*0.01*Spot).to_list()
    Volatility = vol_surface.MID*0.01
    
    market_data = pd.DataFrame({"Expiration Date" : vol_surface_expiry, "Strike Price" : Strike, "Volatility" : Volatility, "Forward Price" : [dico_forward[T] for T in vol_surface_expiry]})
    market_data.loc[:, "Discount Factor"] = list(map(lambda u : df(TimeLeft(ValuationDate, u)), vol_surface_expiry))
    
    return market_data, forward, df, Spot


def spread_prep(spread_data):
    Mat = spread_data.iloc[:,1].apply(lambda u: u.split(" ")[-1]).to_list()
    Mat = np.array([float(mat[:-1])/12 if mat[-1] == "M" else float(mat[:-1]) for mat in Mat])
    Spreads =spread_data.iloc[:,2].to_numpy() 
    return lambda x: np.interp(x,Mat,Spreads,left=0,right=Spreads[-1])
