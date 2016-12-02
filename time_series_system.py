#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import pandas as np
from qntstock.factor_i import *
from qntstock.index import *
from qntstock.factor_base_func import *

def combine_backward(df, order, period=30, strict=[]):
    """
    combine_backward(df, order, period=30, strict=[]):

    将一组信号按order中的顺序进行组合，在order[i+1]序列的两个1的区间内，匹配order[i]中最靠前的1。
    即时间靠前的下层信号会覆盖后面的下层信号。
    得到一个集合，集合中每条记录代表一个可行的信号组合。

    Input:
        df: (DataFrame): 一组信号序列，至少包含order中涉及的列

        order: (list of string): 按序排好的一组信号的名称，至少两列

        period: (int): 从末位信号开始算，一组信号的最长周期。超过该周期仍未凑够一组信号则放弃该条记录。

        strict: (list of boolean): 长度为len(order) - 1，strict[i]指示order[i]的信号出现时间是否必须早于order[i+1]的时间。
                                True必须严格早于；False可以同一天连续触发。
                                参数默认值为[True]*(len(order)-1)，若传入列表长度不足，则尾部用True填充

    Output:
        (DataFrame[order]):返回DataFrame，每条记录（一行）代表一个可行的信号组合，每列中的值为该信号在df中的索引

    """
    rdf=pd.DataFrame()
    #将df最后一列所有信号的位置全部放进rdf中
    rdf[order[-1]] = pd.Series(np.where(df[order[-1]] == 1)[0])
    if len(order) > 1:
        if len(strict) < len(order) - 1:
            rest_len = len(order) - 1 - len(strict)
            strict.extend([True] * rest_len)
        last_col = rdf[order[-1]]
        #每个信号的范围只与last_col-period有关，前面信号不对后面信号的范围造成影响
        if period is None:
            range_sr = pd.Series([(0, last_col[i]) if i == 0 \
                        else (last_col[i-1], last_col[i]) \
                        for i in range(len(rdf))])
        else:
            range_sr = pd.Series([(max(0,last_col[i]-period),last_col[i]) if i == 0\
                        else (last_col[i]-period, last_col[i])\
                        for i in range(len(rdf))])
        odr = order.copy()
        odr.reverse()
        st = strict.copy()
        st.reverse()
        #对每一列信号操作
        for i in range(1, len(odr)):
            #找该行的信号位置
            signal_loc = np.where(df[odr[i]] == 1)[0]
            tmp_sr = range_sr.map(lambda x: _backward(x, signal_loc, st[i-1]), \
                                  na_action='ignore')
            rdf[odr[i]] = tmp_sr.map(lambda x: x[1])
            range_sr = tmp_sr
    #去掉rdf中包含np.NAN的记录（行）
    rdf = rdf.drop(np.where(np.isnan(rdf))[0])
    rdf = rdf[order].reset_index(drop=True)

    return rdf


def combine_forward(df, order, period=None, strict=[]):
    """
    combine_forward(df, order, period=None, strict=[]):

    将一组信号按order中的顺序进行组合，在order[i+1]序列的两个1的区间内，匹配order[i]中最靠前的1。
    即时间靠前的下层信号会覆盖后面的下层信号。
    得到一个集合，集合中每条记录代表一个可行的信号组合。

    Input:
        df: (DataFrame): 一组信号序列，至少包含order中涉及的列

        order: (list of string): 按序排好的一组信号的名称，至少两列

        period: (int): 从末位信号开始算，一组信号的最长周期。若某条记录时间跨度超过该周期则放弃该条记录。

        strict: (list of boolean): 长度为len(order) - 1，strict[i]指示order[i]的信号出现时间是否必须早于order[i+1]的时间。
                                True必须严格早于；False可以同一天连续触发。
                                参数默认值为[True]*(len(order)-1)，若传入列表长度不足，则尾部用True填充

    Output:
        (DataFrame[order]):返回DataFrame，每条记录（一行）代表一个可行的信号组合，每列中的值为该信号在df中的索引

    """
    rdf=pd.DataFrame()
    #将df最后一列所有信号的位置全部放进rdf中
    rdf[order[-1]] = pd.Series(np.where(df[order[-1]] == 1)[0])
    if len(order) > 1:
        if len(strict) < len(order) - 1:
            rest_len = len(order) - 1 - len(strict)
            strict.extend([True] * rest_len)
        last_col = rdf[order[-1]]
        range_sr = pd.Series([(0, last_col[i]) if i == 0 \
                      else (last_col[i-1], last_col[i]) \
                      for i in range(len(rdf))])
        odr = order.copy()
        odr.reverse()
        st = strict.copy()
        st.reverse()
        #对每一列信号操作
        for i in range(1, len(odr)):
            #找该行的信号位置
            signal_loc = np.where(df[odr[i]] == 1)[0]
            tmp_sr = range_sr.map(lambda x: _forward(x, signal_loc, st[i-1]), \
                                  na_action='ignore')
            rdf[odr[i]] = tmp_sr.map(lambda x: x[1])
            range_sr = tmp_sr
    #若period不为None，则整个周期超过period的记录，把开始的信号标记np.nan
    if period is not None:
        rdf = rdf.apply(lambda x:_constrant(x,order[0],order[-1],period),axis=1)
    #去掉rdf中包含np.NAN的记录（行）
    rdf = rdf.drop(np.where(np.isnan(rdf))[0])
    rdf = rdf.reset_index(drop=True)
    return rdf[order]


def convert_record_to_signal(rdf, date):
    """
    convert_record_to_signal(rdf, date):

    将信号组合索引数据的结果转化成信号序列组合。
    combine和convert配合使用，即按一定方式去除无用或重复信号。

    Input:
        rdf: (DataFrame): 待转换到信号组合索引数据的集合

        date: (Series): 信号组合索引对应的日期

    Output：
        (DataFrame): 返回信号序列组合
    """
    df = pd.DataFrame(date)
    for col in list(rdf.columns):
        df[col] = 0
        for raw in rdf[col]:
            df.loc[raw, col] = 1
    return df


def convert_record_to_date(rdf, date):
    """
    convert_record_to_date(rdf, date):

    将信号组合索引数据的结果转化成Timestamp类型的日期

    Input:
        rdf: (DataFrame): 待转换到信号组合索引数据的集合

        date: (Series): 信号组合索引对应的日期

    Output:
        (DataFrame): 返回记录集合，其中每条记录是对应信号组合的日期
    """
    df = rdf.applymap(lambda x: pd.Timestamp(date[x]))
    return df


def _constrant(x, first, last, period):
    if x[last] - x[first] > period:
        x[first] = np.nan
    return x


def _forward(x, signal_loc, strict):
    #若strict=False即宽松条件，当进入的tuple为(0,0)时，需要让left为-1才能套公式
    left = -1 if (x[0]==0 and not strict) else x[0]
    right = x[1]
    if strict:
        loc = np.where((signal_loc >= left) & (signal_loc < right))[0]
    else:
        loc = np.where((signal_loc > left) & (signal_loc <= right))[0]
    right = signal_loc[loc.min()] if len(loc) > 0 else np.NAN
    left = signal_loc[loc.min()-1] if (len(loc) > 0 and loc.min() > 0) else 0
    result = (left, right) if (not np.isnan(right)) else (np.NAN, np.NAN)
    return result


def _backward(x, signal_loc, strict):
    #若strict=False即宽松条件，当进入的tuple为(0,0)时，需要让left为-1才能套公式
    left = -1 if (x[0]==0 and not strict) else x[0]
    right = x[1]
    if strict:
        loc = np.where((signal_loc >= left) & (signal_loc < right))[0]
    else:
        loc = np.where((signal_loc >= left) & (signal_loc <= right))[0]
    right = signal_loc[loc.max()] if len(loc) > 0 else np.NAN
    result = (left, right) if (not np.isnan(right)) else (np.NAN, np.NAN)
    return result


if __name__ == '__main__':
    import tushare as ts
    df = ts.get_k_data('002335',autype=None)
    ma_raise = factor_ma_raise(df)
    ma_long = factor_ma_long_position(df)
    cond_df = pd.DataFrame()
    ma_line=ma(df)
    cond_df[['date','close']] = df[['date','close']]
    cond_df['MA_5'] = ma_line['MA_5']
    cond_df['MA_10'] = ma_line['MA_10']
    cond_df['5r'] = inflection(ma_line['MA_5'], strict=True, signal_type='raise')
    cond_df['5rc10'] = cross(ma_line['MA_5'], ma_line['MA_10'], strict=True, signal_type='raise')
    cond_df['5f'] = inflection(ma_line['MA_5'], strict=True, signal_type='fall')
    order =['5r', '5rc10', '5f']
    print(cond_df)
    rdf = combine_backward(cond_df, order=order, period=20)
#    print(rdf)
    print(convert_record_to_date(rdf, df['date']))
#    print(convert_record_to_signal(rdf, df['date']))
