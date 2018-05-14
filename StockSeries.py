#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
from pandas import Series
from qntstock import factor_base_func

class StockSeries(Series):
    def __init__(self, s, stat=None):
        Series.__init__(self, s)
        self.s = s
        self.stat = stat


    def __call__(self):
        return self.s


    def sign(self, strict=True, stype='all'):
        s = factor_base_func.sign(self.s, strict, stype)
        return StockSeries(pd.Series(s, index=self.s.index))


    def upper_than(self, line, strict=True):
        s = factor_base_func.upper_than(self.s, line, strict)
        return StockSeries(pd.Series(s, index=self.s.index))


    def lower_than(self, line, strict=True):
        s = factor_base_func.lower_than(self.s, line, strict)
        return StockSeries(pd.Series(s, index=self.s.index))


    def wave(self, strict=True, stype='all'):
        s = factor_base_func.wave(self.s, strict, stype)
        return StockSeries(pd.Series(s, index=self.s.index))


    def increase(self, strict=True):
        s = factor_base_func.wave(self.s, strict, 'pos')
        return StockSeries(pd.Series(s, index=self.s.index))


    def decrease(self, strict=True):
        s = factor_base_func.wave(self.s, strict, 'neg')
        return StockSeries(pd.Series(s, index=self.s.index))


    def keep(self, n, strict=True):
        s = factor_base_func.keep(self.s, n, strict)
        return StockSeries(pd.Series(s, index=self.s.index))

    def cross(self, base_line, strict=False, signal_type='both'):
        s = factor_base_func.cross(self.s, base_line, strict, signal_type)
        return StockSeries(pd.Series(s, index=self.s.index))


    def inflection(self, strict=False, signal_type='both'):
        s = factor_base_func.inflection(self.s, strict, signal_type)
        return StockSeries(pd.Series(s, index=self.s.index))
