import pandas as pd


def _cross(df, judg):
    sign = df[judg].map(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
    cross = sign.diff()
    sign = sign * cross
    sign = sign.map(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
    return sign


def factor_macd_cross_1(sdf, fast=12, slow=26, avg=9):
    """
    factor_macd_cross(sdf, fast=12, slow=26, avg=9):

    计算MACD交叉信号，MACD出现金叉当天1,出现死叉当天为-1,其他为0

    Input:
        sdf: (DataFrame): 股票数据，至少包括['close']

        fast: (int): DIFF中的快线时间窗

        slow: (int): DIFF中的慢线时间窗

        avg: (int): DEA中的平滑时间窗

    Output:
        (DataFrame): MACD交叉信号，包含['MACD_CROSS']
    """
    rdf = macd(sdf, fast, slow, avg)
    sign = _cross(sdf, 'MACD_BAR')
    dic = {'MACD_CROSS': sign}
    rdf = pd.DataFrame(dic)
    return rdf


def factor_bbi_cross_1(sdf):
    """
    factor_bbi_cross_1(sdf):

    计算BBI交叉信号，BBI出现金叉当天1,出现死叉当天为-1,其他为0

    Input:
        sdf: (DataFrame): 股票数据，至少包括['close']

    Output:
        (DataFrame): BBI交叉信号，包含['BBI_CROSS']
    """
    rdf = bbi(sdf)
    rdf['BAR'] = rdf['BBI']-sdf['close']
    sign = _cross(sdf, 'BAR')
    dic = {'BBI_CROSS': sign}
    rdf = pd.DataFrame(dic)
    return rdf
