#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np


def _upper_than(line1, line2):
    """
    line1高于line2为1，不高于则为0
    """
    diff = line1 - line2
    sign = diff.map(lambda x: 1 if x > 0 else 0)
    return sign


def _lower_than(line1, line2):
    """
    line1低于line2为1，不低于则为0
    """
    diff = line1 - line2
    sign = diff.map(lambda x: 1 if x < 0 else 0)
    return sign


def _sign(line):
    """
    line每个元素为正数则为1，为负数则为-1
    """
    sign = line.map(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
    return sign


def _cross(judge_line, base_line=None):
    """
    判断judge_line是否穿过base_line，上穿当天1,下穿当天为-1,其他为0。
    两条线相交不算穿轴，若judge_line下踩base_line再向上视为上穿，反之亦然。
    """
    if base_line is not None:
        judge_line = judge_line - base_line
    sign = judge_line.map(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
    mark = sign.diff().map(lambda x: -x if x < 0 else x)
    sign = sign * mark
    sign = sign.map(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
    return sign


def _cross_c(judge_line, base_line=None):
    """
    判断judge_line是否穿过base_line，上穿当天1,下穿当天为-1,其他为0。
    修正版本，保持上穿下穿的一致性。
    两条线相交不算穿轴，若judge_line下踩base_line再向上不视为上穿，反之亦然。
    """
    if base_line is not None:
        judge_line = judge_line - base_line
    sign = _cross(judge_line)
    idx = np.where(sign != 0)[0]
    j = 0
    for i in range(1,len(idx)):
        if sign[idx[i]] * sign[idx[j]] > 0:
            sign[idx[i]] = 0
        else:
            j = i
    return sign


def _inflection(line):
    """
    判断line拐头，向上拐头当天1,向下拐头当天为-1,其他为0。
    由于只判断当天和前一天和前两天的线型，前一天走平，当天向上或向下也视为拐头。
    """
    judge_line = line.diff()
    if len(judge_line) > 2:
        judge_line[0] = judge_line[1]
    sign = _cross(judge_line)
    return sign


def _inflection_c(line):
    """
    判断line拐头，向上拐头当天1,向下拐头当天为-1,其他为0。
    修正版本，保持拐头的一致性，即向上拐头和向下拐头严格交叉出现。
    若之前走平，当天向上或向下视为拐头。
    """
    judge_line = line.diff()
    if len(judge_line) > 2:
        judge_line[0] = judge_line[1]
    sign = _cross_c(judge_line)
    return sign


if __name__ == '__main__':
    df = pd.DataFrame({'a':[0,1,3,5,6,7,7,8,8,7,7,8,9,10,9,8,7,6,7,8]})
    df['diff1'] = df['a'].diff()
    df['inflection'] = _inflection(df['a'])
    df['inflection_c'] = _inflection_c(df['a'])
    df['testcross'] = df['a']-7
    df['cross'] = _cross(df['testcross'])
    df['cross_c'] = _cross_c(df['testcross'])

    print(df)
