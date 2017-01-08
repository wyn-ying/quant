#!/usr/bin/python
# -*- coding: utf-8 -*-
from qntstock.local import DB_PATH, PATH

class ProgressBar:
    def __init__(self, count=0, total=0, width=50):
        self.count = count
        self.total = total
        self.width = width

    def move(self):
        self.count += 1

    def log(self, s=None):
        sys.stdout.write(' ' * (self.width + 9) + '\r')
        sys.stdout.flush()
        progress = round(self.width * self.count / self.total)
        sys.stdout.write('{0:4} {1:4}/{2:4}: '.format(s, self.count, self.total))
        sys.stdout.write('#' * progress + '-' * (self.width - progress) + '\r')
        if progress == self.width:
            sys.stdout.write('\n')
        sys.stdout.flush()


def _sort(sdf):
    tdf = sdf.sort_values(by='date')
    tdf = tdf.reset_index(drop=True)
    return tdf
