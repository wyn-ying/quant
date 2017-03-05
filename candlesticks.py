#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy
from qntstock.utils import PATH
import matplotlib.pyplot as pyplot
from matplotlib.ticker import NullLocator, NullFormatter


def plot(df, path, useexpo=False, dpi=600, part=['p', 'v'], price_part=0.7, xsize=10, ysize=10):
    '''
    plot(df, path, useexpo=False, dpi=600, part=['p', 'v'], price_part=0.7, xsize=10, ysize=10):

    根据数据绘制png格式的量价k线图，文件名'股票代码_日期_天数_pv.png'，如'600000_20161129_60_pv.png'

    Input:
        df: (DataFrame): 股票数据

        path: (string): 图片文件存储路径

        useexpo: (boolean): 是否使用对数坐标，True为使用对数坐标，False为线性坐标

        dpi: (int): 图片分辨率

        part: (list of string): 图片包含的内容，接受['p'], ['v'], ['p', 'v']，即单独价格k线图，成交量图，和量价k线图

        price_part: (float): 价格k线部分所占比例，仅当part为['p', 'v']时生效

        xsize: (int): 图片宽度

        ysize: (int): 图片高度
    '''
    if set(part) == set(['p', 'v']):
        rect_1 = (0, 1-price_part, 1, price_part)    # K线图部分
        rect_2 = (0, 0, 1, 1-price_part)   # 成交量部分
    elif part == ['p']:
        rect_1 = (0, 0, 1, 1)
    elif part == ['v']:
        rect_2 = (0, 0, 1, 1)
    else:
        print('part name invalid')
        return

    df = df.reset_index(drop=True)
    length = len(df['date'])
    highest_price, lowest_price = df['high'].max(), df['low'].min()
    raise_color, fall_color, keep_color = 'red', 'green', 'yellow'
    bg_color = 'black'

    if useexpo:
        expbase= 1.1

    xlen_fig = length * 0.05   #0.055是经验数值
    ylen_fig = 2.7 #2.7是经验数值

    xshrink = xsize / xlen_fig
    yshrink = ysize / ylen_fig

    #   建立 Figure 对象
    figobj= pyplot.figure(figsize=(xsize, ysize), dpi=dpi)

    xindex= numpy.arange(length)    # X 轴上的 index，一个辅助数据
    zipoc= zip(df['open'], df['close'])
    up=   numpy.array(df.apply(lambda x: True if x['open'] < x['close'] and x['open'] != None else False, axis=1))        # 标示出该天股价日内上涨的一个序列
    down= numpy.array(df.apply(lambda x: True if x['open'] > x['close'] and x['open'] != None else False, axis=1))        # 标示出该天股价日内下跌的一个序列
    side= numpy.array(df.apply(lambda x: True if x['open'] ==x['close'] and x['open'] != None else False, axis=1))      # 标示出该天股价日内走平的一个序列
    for i in range(len(df)):
        if df.loc[i,'open'] == df.loc[i, 'close']:
            var = min(round(df.loc[i,'open']+1/2000, 5), 0.005)
            df.loc[i,'open'] -= var
            df.loc[i,'close'] += var

    #======    成交量
    if 'v' in part:
        axes_2= figobj.add_axes(rect_2, axis_bgcolor=bg_color)

        volume= df['volume']
        rarray_vol= numpy.array(volume)
        volzeros= numpy.zeros(length)   # 辅助数据
        if True in up:
            axes_2.vlines(xindex[up], volzeros[up], rarray_vol[up], color=raise_color, linewidth=3.0 * xshrink, label='_nolegend_')
        if True in down:
            axes_2.vlines(xindex[down], volzeros[down], rarray_vol[down], color=fall_color, linewidth=3.0 * xshrink, label='_nolegend_')
        if True in side:
            axes_2.vlines(xindex[side], volzeros[side], rarray_vol[side], color=keep_color, linewidth=3.0 * xshrink, label='_nolegend_')
        #    设定x轴坐标范围
        axes_2.set_xlim(-1, length)
        axes_2.xaxis.set_major_locator(NullLocator())
        axes_2.xaxis.set_major_formatter(NullFormatter())
        #    设定 Y 轴坐标的范围
        maxvol= max(volume)
        axes_2.set_ylim(0, maxvol*1.01)
        axes_2.yaxis.set_major_locator(NullLocator())
        axes_2.yaxis.set_major_formatter(NullFormatter())

    #=======    K 线图
    if set(part) == set(['p', 'v']):
        axes_1= figobj.add_axes(rect_1, axis_bgcolor=bg_color, sharex=axes_2)
    elif part == ['p']:
        axes_1= figobj.add_axes(rect_1, axis_bgcolor=bg_color)
    if 'p' in part:
        if useexpo:
            axes_1.set_yscale('log', basey=expbase) # 使用对数坐标

        rarray_open= numpy.array(df['open'])
        rarray_close= numpy.array(df['close'])
        rarray_high= numpy.array(df['high'])
        rarray_low= numpy.array(df['low'])
        if True in up:
            axes_1.vlines(xindex[up], rarray_low[up], rarray_high[up], color=raise_color, linewidth=1.5 * xshrink, label='_nolegend_')
            axes_1.vlines(xindex[up], rarray_open[up], rarray_close[up], color=raise_color, linewidth=3.0 * xshrink, label='_nolegend_')
        if True in down:
            axes_1.vlines(xindex[down], rarray_low[down], rarray_high[down], color=fall_color, linewidth=1.5 * xshrink, label='_nolegend_')
            axes_1.vlines(xindex[down], rarray_open[down], rarray_close[down], color=fall_color, linewidth=3.0 * xshrink, label='_nolegend_')
        if True in side:
            axes_1.vlines(xindex[side], rarray_low[side], rarray_high[side], color=keep_color, linewidth=1.5 * xshrink, label='_nolegend_')
            axes_1.vlines(xindex[side], rarray_open[side], rarray_close[side], color=keep_color, linewidth=3.0 * xshrink, label='_nolegend_')

        #   在k线上面叠加绘制均线
        '''
        rarray_5dayave= numpy.array(df['close'].rolling(center=False, window=5).mean())
        rarray_30dayave= numpy.array(df['close'].rolling(center=False, window=30).mean())

        axes_1.plot(xindex, rarray_5dayave, 'o-', color='yellow', linewidth=0.1, markersize=0.7, markeredgecolor='yellow', markeredgewidth=0.1) # 5日均线
        axes_1.plot(xindex, rarray_30dayave, 'o-', color='green', linewidth=0.1, markersize=0.7, markeredgecolor='green', markeredgewidth=0.1)  # 30日均线
        '''
        #   设定 X 轴坐标的范围
        axes_1.set_xlim(-1, length)
        axes_1.xaxis.set_major_locator(NullLocator())
        axes_1.xaxis.set_major_formatter(NullFormatter())
        #   设定 Y 轴坐标的范围
        yhighlim_price, ylowlim_price = highest_price*1.005, lowest_price*0.995
        axes_1.set_ylim(ylowlim_price, yhighlim_price)
        axes_1.yaxis.set_major_locator(NullLocator())
        axes_1.yaxis.set_major_formatter(NullFormatter())

    date = str(df.tail(1)['date'].values[0])
    date = date[0:4] + date[5:7] + date[8:10]
    filetype = '_' + ('pv' if set(part)==set(['p','v']) else part[0]) + '.jpg'
    figpath = path + '_' + date + '_' + str(length) + filetype
    figobj.savefig(figpath, dpi=dpi)


if __name__ == '__main__':
    import tushare as ts
    code = '600000'
    df = ts.get_k_data(code)
    path = PATH + '/data/ml/' + code
    df = df.tail(60)
    plot(df, dpi=150, xsize=1.28, ysize=1.28, path=path, part=['p', 'v'])
