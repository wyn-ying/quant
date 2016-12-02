#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
from qntstock.index import *
from qntstock.factor_base_func import *


def factor_macd_cross(sdf, fast=12, slow=26, avg=9, strict=True):
    """
    factor_macd_cross(sdf, fast=12, slow=26, avg=9, strict=True):

    计算MACD交叉信号，MACD出现金叉当天1,出现死叉当天为-1,其他为0

    Input:
        sdf: (DataFrame): 股票数据，至少包括['close']

        fast: (int): DIFF中的快线时间窗

        slow: (int): DIFF中的慢线时间窗

        avg: (int): DEA中的平滑时间窗

        strict: (boolean): True时，MACD严格出现交叉才会发出信号；False时，若前一天MACD快线回踩慢线并重合，第二天再拉开，也认为出现交叉

    Output:
        (Series): MACD交叉信号
    """
    rdf = macd(sdf, fast, slow, avg)
    signal = cross(rdf['MACD_BAR'], strict)
    return signal


def factor_macd_fast_inflection(sdf, fast=12, slow=26, avg=9):
    """
    factor_macd_fast_inflection(sdf, fast=12, slow=26, avg=9):

    计算MACD快线拐头信号，向上拐头当天1,向下拐头当天为-1,其他为0

    Input:
        sdf: (DataFrame): 股票数据，至少包括['close']

        fast: (int): DIFF中的快线时间窗

        slow: (int): DIFF中的慢线时间窗

        avg: (int): DEA中的平滑时间窗

        strict: (boolean): True时，MACD快线出现拐头才会发出信号，连续上升或下降途中的走平不会发出信号；False时，若前一天MACD快线走平，当天发生变化，则认为当天会发出拐头信号

    Output:
        (Series): MACD快线拐头信号
    """
    rdf = macd(sdf, fast, slow, avg)
    signal = inflection(rdf['MACD_DIFF'], strict=True)
    return signal


def factor_macd_slow_inflection(sdf, fast=12, slow=26, avg=9):
    """
    factor_macd_slow_inflection(sdf, fast=12, slow=26, avg=9):

    计算MACD慢线拐头信号，向上拐头当天1,向下拐头当天为-1,其他为0

    Input:
        sdf: (DataFrame): 股票数据，至少包括['close']

        fast: (int): DIFF中的快线时间窗

        slow: (int): DIFF中的慢线时间窗

        avg: (int): DEA中的平滑时间窗

        strict: (boolean): True时，MACD慢线出现拐头才会发出信号，连续上升或下降途中的走平不会发出信号；False时，若前一天MACD慢线走平，当天发生变化，则认为当天会发出拐头信号

    Output:
        (Series): MACD慢线拐头信号
    """
    rdf = macd(sdf, fast, slow, avg)
    signal = inflection(rdf['MACD_DIFF'], strict=True)
    return signal


def factor_bbi_cross(sdf, strict=True):
    """
    factor_bbi_cross(sdf, strict=True):

    计算BBI交叉信号，BBI出现金叉当天1,出现死叉当天为-1,其他为0

    Input:
        sdf: (DataFrame): 股票数据，至少包括['close']

        strict: (boolean): True时，严格出现交叉才会发出信号；False时，若前一天日线回踩BBI线并重合，第二天再拉开，也认为出现交叉

    Output:
        (Series): BBI交叉信号
    """
    rdf = bbi(sdf)
    rdf['BAR'] = rdf['close']-sdf['BBI']
    signal = cross(rdf['BAR'], strict)
    return signal


def factor_rsi_cross(sdf, n=(6, 12), strict=True):
    """
    factor_rsi_cross(sdf, n=[6, 12], strict=True):

    计算给定两条RSI交叉信号，MACD出现金叉当天1,出现死叉当天为-1,其他为0

    Input:
        sdf: (DataFrame): 按时间升序排好到股票数据，至少包括['close']

        n: (list of int): 参与计算的两条RSI的时间窗，如[6,12]，以第一条线上穿第二条线为金叉

        strict: (boolean): True时，RSI严格出现交叉才会发出信号；False时，若前一天第一条线回踩第二条线并重合，第二天再拉开，也认为出现交叉

    Output:
        (Series): RSI交叉信号
    """
    rdf = rsi(sdf, n)
    rdf['BAR'] = rdf['RSI_'+str(n[0])]-rdf['RSI_'+str(n[1])]
    signal = cross(rdf['BAR'], strict)
    return signal


def factor_ma_raise(sdf, n=(5,10,20,30,60), strict=True):
    """
    factor_ma_raise(sdf, n=[5,10,20,30,60], strict=True):

    计算给定一组天数到均线是否都呈上升趋势，若都上升，则为1，反之则为0

    Input:
        sdf: (DataFrame): 按时间升序排好到股票数据，至少包括['close']

        n: (list of int): 要计算的移动平均线的时间窗，如[5,10,...]

        strict: (boolean): True为严格上升，走平为0；Flase为宽松条件，上升途中走平也认为在上升

    Output:
        (series): MA线组上升信号
    """
    rdf = ma(sdf, n)
    name_list = []
    for ma_n in n:
        col_name = 'MA_'+str(ma_n)+'_RAISE'
        rdf[col_name] = wave(rdf['MA_'+str(ma_n)], strict)
        rdf[col_name] = rdf[col_name].map(lambda x: True if x > 0 else False)
        name_list.append(col_name)
    signal = rdf[name_list].all(axis=1).map(lambda x: 1 if x is True else 0)
    return signal


def factor_ma_fall(sdf, n=(5,10,20,30,60), strict=True):
    """
    factor_ma_fall(sdf, n=[5,10,20,30,60], strict=True):

    计算给定一组天数到均线是否都呈下降趋势，若都下降，则为1，反之则为0

    Input:
        sdf: (DataFrame): 按时间升序排好到股票数据，至少包括['close']

        n: (list of int): 要计算的移动平均线的时间窗，如[5,10,...]

        strict: (boolean): True为严格下降，走平为0；Flase为宽松条件，下降途中走平也认为在下降

    Output:
        (series): MA线组下降信号
    """
    rdf = ma(sdf, n)
    name_list = []
    for ma_n in n:
        col_name = 'MA_'+str(ma_n)+'_FALL'
        rdf[col_name] = wave(rdf['MA_'+str(ma_n)], strict)
        rdf[col_name] = rdf[col_name].map(lambda x: True if x < 0 else False)
        name_list.append(col_name)
    signal = rdf[name_list].all(axis=1).map(lambda x: 1 if x is True else 0)
    return signal


def factor_ma_long_position(sdf, n=(5,10,20,30,60), strict=True):
    """
    factor_ma_long_position(sdf, n=[5,10,20,30,60], strict=True):

    计算给定一组天数到均线是否多头排列，若完全多头，则为1，反之则为0

    Input:
        sdf: (DataFrame): 按时间升序排好到股票数据，至少包括['close']

        n: (list of int): 严格按升序排列的移动平均线的时间窗，如[5,10,...]

        strict: (boolean): True为严格多头，两条线相等为0；Flase为宽松条件，两条线相等时看之前是不是多头，保持前一天的状态

    Output:
        (series): MA线组多头信号
    """
    num = sorted(n)
    ma_name=['MA_'+str(x) for x in num]
    rdf = ma(sdf, n)
    name_list=[]
    for i in range(1,len(num)):
        name = 'MA_LONG_POSITION_'+str(i)
        rdf[name] = upper_than(rdf[ma_name[i-1]], rdf[ma_name[i]], strict)
        rdf[name] = rdf[name].map(lambda x: True if x == 1 else False)
        name_list.append(name)
    signal = rdf[name_list].all(axis=1).map(lambda x: 1 if x is True else 0)
    return signal


def factor_ma_short_position(sdf, n=(5,10,20,30,60), strict=True):
    """
    factor_ma_short_position(sdf, n=[5,10,20,30,60], strict=True):

    计算给定一组天数到均线是否空头排列，若完全空头，则为1，反之则为0

    Input:
        sdf: (DataFrame): 按时间升序排好到股票数据，至少包括['close']

        n: (list of int): 严格按升序排列的移动平均线的时间窗，如[5,10,...]

        strict: (boolean): True为严格空头，两条线相等为0；Flase为宽松条件，两条线相等时看之前是不是空头，保持前一天的状态

    Output:
        (series): MA线组空头信号
    """
    num = sorted(n)
    ma_name=['MA_'+str(x) for x in num]
    rdf = ma(sdf, n)
    name_list=[]
    for i in range(1,len(num)):
        name = 'MA_SHORT_POSITION_'+str(i)
        rdf[name] = lower_than(rdf[ma_name[i-1]], rdf[ma_name[i]], strict)
        rdf[name] = rdf[name].map(lambda x: True if x == 1 else False)
        name_list.append(name)
    signal = rdf[name_list].all(axis=1).map(lambda x: 1 if x is True else 0)
    return signal
