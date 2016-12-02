#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np


def sign(line, strict=True):
    """
    判断line符号。line每个元素为正数则为1，为负数则为-1
    strict=True时，0点为0
    strict=False时，0点的状态与前天状态保持一致
    """
    signal = line.map(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
    if strict is False:
        idx = sorted(np.where(signal == 0)[0])
        for i in range(1,len(idx)):
            signal[idx[i]] = signal[idx[i]-1]
    return signal


def upper_than(line1, line2, strict=True):
    """
    line1高于line2为1，低于则为0
    strict=True时，line1 = line2则为0
    strict=False时，line1 = line2的状态与前天状态保持一致
    """
    diff = line1 - line2
    signal = sign(diff, strict).map(lambda x: 1 if x > 0 else 0)

    return signal


def lower_than(line1, line2, strict=True):
    """
    line1低于line2为1，高于则为0
    strict=True时，line1 = line2则为0
    strict=False时，line1 = line2的状态与前天状态保持一致
    """
    diff = line1 - line2
    signal = sign(diff, strict).map(lambda x: 1 if x < 0 else 0)
    return signal


def wave(line, strict=True):
    """
    判断line走势。line向上行则为1，下行则为-1
    strict=True时，走平为0
    strict=False时，走平当天的状态与前天状态保持一致
    """
    w = line.diff()
    signal = sign(w, strict)
    return signal


def cross(judge_line, base_line=None, strict=False, signal_type='both'):
    """
    判断judge_line是否穿过base_line，上穿当天1,下穿当天为-1,其他为0。
    两条线相交的点不算穿轴。
    strict=False，若judge_line下踩base_line再向上视为上穿，反之亦然。
    strict=True，若judge_line下踩base_line再向上不视为上穿，反之亦然。
    signal_type可选'raise'或者'fall'，为单方向信号，选'both'为双方向信号。'fall'时返回值为1则表明下穿。
    """
    if base_line is not None:
        judge_line = judge_line - base_line
    signal = judge_line.map(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
    mark = signal.diff().map(lambda x: -x if x < 0 else x)
    signal = signal * mark
    signal = signal.map(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
    if strict:
        idx = np.where(signal != 0)[0]
        j = 0
        for i in range(1,len(idx)):
            if signal[idx[i]] * signal[idx[j]] > 0:
                signal[idx[i]] = 0
            else:
                j = i
    if signal_type == 'fall':
        signal = signal.map(lambda x: 1 if x < 0 else 0)
    elif signal_type == 'raise':
        signal = signal.map(lambda x: 1 if x > 0 else 0)
    return signal


def inflection(line, strict=False, signal_type='both'):
    """
    判断line拐头，向上拐头当天1,向下拐头当天为-1,其他为0。
    strict=False时，前一天走平，当天向上或向下也视为拐头。
    strict=True时，向上拐头和向下拐头严格交叉出现，连续上升或下降途中的走平不认为是拐头。
    signal_type可选'raise'或者'fall'，为单方向信号，选'both'为双方向信号。'fall'时返回值为1则表明下穿。
    """
    judge_line = line.diff()
    if len(judge_line) > 2:
        judge_line[0] = judge_line[1]
    signal = cross(judge_line, strict=strict)
    if signal_type == 'raise':
        signal = signal.map(lambda x: 1 if x > 0 else 0)
    elif signal_type == 'fall':
        signal = signal.map(lambda x: 1 if x < 0 else 0)
    return signal


if __name__ == '__main__':
    df = pd.DataFrame({'a':[0,1,3,5,6,7,7,8,8,7,7,8,9,10,9,8,7,6,7,8]})
    df['diff1'] = df['a'].diff()
    df['inflection'] = inflection(df['a'])
    df['inflection_c'] = inflection(df['a'], strict=True)
    df['testcross'] = df['a']-7
    df['cross'] = cross(df['testcross'])
    df['cross_c'] = cross(df['testcross'], strict=True)

    print(df)
