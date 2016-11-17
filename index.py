import pandas as pd
import talib


def _sort(sdf):
    tdf = sdf.sort_values(by="date")
    tdf = tdf.reset_index()
    tdf = tdf.drop('index', axis=1)
    return tdf


def ma(sdf, ma_num=(5, 10, 20, 30, 60)):
    """
    ma(sdf, ma_num=[5, 10, 20, 30, 60]):

    Compute MA for stock's DataFrame

    Input:
        sdf: (DataFrame): DataFrame of stock

        ma_num: (list or tuple): MA lines to compute

    Output:
        (DataFrame): DataFrame with columns' names ['MA_'+num, ...]
    """
    tdf = _sort(sdf)
    name_list = list()
    for n in ma_num:
        ma_n = tdf['close'].rolling(center=False, window=n).mean()
        name = 'MA_' + str(n)
        name_list.append(name)
        tdf[name] = round(ma_n, 2)
    rdf = tdf[name_list]
    return rdf


def expma(sdf, expma_num=(5, 10, 20, 30, 60)):
    """
    expma(sdf, expma_num=[5, 10, 20, 30, 60]):

    Compute EXPMA for stock's DataFrame

    Input:
        sdf: (DataFrame): DataFrame of stock

        expma_num: (list or tuple): EXPMA lines to compute

    Output:
        (DataFrame): DataFrame with columns' names ['EXPMA_'+num, ...]
    """
    tdf =_sort(sdf)
    name_list = list()
    for n in expma_num:
        expma_n = tdf['close'].ewm(span=n, adjust=False).mean()
        name = 'EXPMA_' + str(n)
        name_list.append(name)
        tdf[name] = round(expma_n, 2)
    rdf = tdf[name_list]
    return rdf


def bbi(sdf):
    """
    bbi(sdf):

    Compute BBI for stock's DataFrame

    Input:
        sdf: (DataFrame): DataFrame of stock

    Output:
        (DataFrame): DataFrame with column's name ['BBI']
    """
    tdf = _sort(sdf)
    name_list = list()
    rdf=pd.DataFrame()
    for n in (3, 6, 12, 24):
        ma_n = tdf['close'].rolling(center=False, window=n).mean()
        name = 'MA_' + str(n)
        name_list.append(name)
        tdf[name] = round(ma_n, 2)
    rdf['BBI'] = tdf[name_list].apply(sum, axis=1)/4
    return rdf


def macd(sdf, fast=12, slow=26, avg=9):
    """
    macd(sdf, fast=12, slow=26, avg=9):

    Compute MACD for stock's DataFrame

    Input:
        sdf: (DataFrame): DataFrame of stock

        fast: (int): number of days for fast line of MACD_DIFF

        slow: (int): number of days for slow line of MACD_DIFF

        avg: (int): number for smoothing MACD_DEA

    Output:
        (DataFrame): DataFrame with column's name ['MACD_DIFF', 'MACD_DEA', 'MACD_BAR']
    """
    tdf = _sort(sdf)
    ema_fast = round(tdf['close'].ewm(span=fast, adjust=False).mean(), 2)
    ema_slow = round(tdf['close'].ewm(span=slow, adjust=False).mean(), 2)
    diff = ema_fast - ema_slow
    dea = round(diff.ewm(span=avg, adjust=False).mean(), 2)
    bar = 2 * (diff - dea)
    tdf['MACD_BAR'] = bar
    tdf['MACD_DIFF'] = diff
    tdf['MACD_DEA'] = dea
    rdf = tdf[['MACD_BAR', 'MACD_DIFF', 'MACD_DEA']]
    return rdf


def kdj(sdf, n=9, m=3):
    """
    kdj(sdf, n=9, m=3):

    Compute KDJ for stock's DataFrame

    Input:
        sdf: (DataFrame): DataFrame of stock

        n: (int): window number for RSV

        m: (int): expma days for K and D

    Output:
        (DataFrame): DataFrame with column's name ['KDJ_K', 'KDJ_D', 'KDJ_J']
    """
    tdf = _sort(sdf)
    low_list = tdf['low'].rolling(window=n, center=False).min()
    low_list = low_list.fillna(value=tdf['low'].expanding(min_periods=1).min())
    high_list = tdf['high'].rolling(window=n, center=False).max()
    high_list=high_list.fillna(value=tdf['high'].expanding(min_periods=1).max())
    rsv = round((tdf['close'] - low_list) / (high_list - low_list) * 100, 1)
    if high_list[len(high_list) - 1] - low_list[len(low_list) - 1] == 0:
        rsv[len(rsv) - 1] = 0
    else:
        rsv[len(rsv) - 1] = 33.3
    kdj_k = round(rsv.ewm(com=m - 1, adjust=False).mean(), 1)
    if not high_list[len(high_list) - 1] - low_list[len(low_list) - 1] == 0:
        kdj_k[len(kdj_k) - 1] = 11.1
    kdj_d = round(kdj_k.ewm(com=m - 1, adjust=False).mean(), 1)
    kdj_j = 3 * kdj_k - 2 * kdj_d
    tdf['KDJ_K'] = kdj_k
    tdf['KJD_D'] = kdj_d
    tdf['KDJ_J'] = kdj_j
    rdf = tdf[['KDJ_K', 'KDJ_D', 'KDJ_J']]
    return rdf


def rsi(sdf, n=(6, 12, 24)):
    """
    rsi(sdf, n=[6, 12, 24]):

    Compute RSI for stock's DataFrame

    Input:
        sdf: (DataFrame): DataFrame of stock

        n: (list or tuple): RSI lines to compute

    Output:
        (DataFrame): DataFrame with columns' names ['RSI_'+num, ...]
    """
    tdf = _sort(sdf)
    name_list = list()
    for num in n:
        rsi_n = talib.RSI(tdf['close'].values, timeperiod=num)
        name = 'RSI_' + str(num)
        name_list.append(name)
        tdf[name] = round(pd.Series(rsi_n).fillna(0), 1)

    rdf = tdf[name_list]
    return rdf


def wr(sdf, period=14):
    """
    wr(sdf, period=14):

    Compute WILLR for stock's DataFrame

    Input:
        sdf: (DataFrame): DataFrame of stock

        period: (int): period for WILLR

    Output:
        (DataFrame): DataFrame with column's name ['WR']
    """
    tdf = _sort(sdf)
    high = tdf['high'].values
    low = tdf['low'].values
    close = tdf['close'].values
    wr = talib.WILLR(high , low, close, period) * -1
    dic = ('WR': wr)
    rdf = pd.DataFrame(dic)
    return rdf


def boll(sdf,n=20, std=2):
    """
    boll(sdf,n=20, std=2):

    Compute BOLL for stock's DataFrame

    Input:
        sdf: (DataFrame): DataFrame of stock

        n: (int): number of days to compute standard

        std: (int): times of standard

    Output:
        (DataFrame): DataFrame with column's name ['BOLL_MD', 'BOLL_UPPER', 'BOLL_LOWER']
    """
    tdf = _sort(sdf)
    mid = tdf['close'].rolling(center=False, window=n).mean()
    std_line = tdf['close'].rolling(center=False, window=n).std()
    ub = mid + std * std_line
    lb = mid - std * std_line
    dic = {'BOLL_MD': mid, 'BOLL_UPPER': ub, 'BOLL_LOWER': lb}
    rdf = pd.DataFrame(dic)
    return rdf
