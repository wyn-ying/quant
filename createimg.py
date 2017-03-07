#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from qntstock.utils import PATH
from qntstock.database import get_df
def translate_ticks(df, xsize, ysize):
    num=df[['open','high','low','close']].values
    minim = np.amin(num)
    maxim = np.amax(num)
    # mark which row should be red (long)
    maskyr=np.ceil(num[:,3]-num[:,0]>=0)
    # mark which row should be green (short)
    maskyg=np.ceil(num[:,0]-num[:,3]>=0)
    # map (rows of data record) to (columns of image)
    maskx = sorted([num.shape[0]-1-i%num.shape[0] for i in range(xsize)]) if num.shape[0]<=xsize \
                                else [i for i in range(num.shape[0]-xsize, num.shape[0])]
    # normalize to [0, ysize]
    num = (num-minim)*(ysize-1)//(maxim-minim)+1
    # adjust coordinate
    num[:,2]=num[:,2]-1
    num[:,0]=num[:,0]-maskyr
    num[:,3]=num[:,3]-maskyg
    # reverse along y axis
    num = ysize - num[:,:]
    img = np.zeros((ysize, xsize,3), dtype=np.int8)
    for i in range(xsize):
        if maskyr[maskx[i]]:
            # maskx[] map from i-th y-axis in num to maskx[i]-th x-axis in img
            # num[maskx[i],1] map from (coordinate of high price in num) to (pixel position in img)
            img[int(num[maskx[i],1]):int(num[maskx[i],2]),i,0]=1
            img[int(num[maskx[i],3]):int(num[maskx[i],0]),i,0]=2
            img[int(num[maskx[i],1]):int(num[maskx[i],2]),i,2]=1
            img[int(num[maskx[i],3]):int(num[maskx[i],0]),i,2]=2
        if maskyg[maskx[i]]:
            img[int(num[maskx[i],1]):int(num[maskx[i],2]),i,1]=1
            img[int(num[maskx[i],0]):int(num[maskx[i],3]),i,1]=2
            img[int(num[maskx[i],1]):int(num[maskx[i],2]),i,2]=1
            img[int(num[maskx[i],0]):int(num[maskx[i],3]),i,2]=2
    return img

def translate_volumes(df, xsize, ysize):
    num=df[['open','close','volume']].values
    # mark which row should be red (long)
    maskyr=np.ceil(num[:,1]-num[:,0]>=0)
    # mark which row should be green (short)
    maskyg=np.ceil(num[:,0]-num[:,1]>=0)
    # map (rows of data record) to (columns of image)
    maskx = sorted([num.shape[0]-1-i%num.shape[0] for i in range(xsize)]) if num.shape[0]<=xsize \
                                else [i for i in range(num.shape[0]-xsize, num.shape[0])]
    num = num[:,2]
    maxim = np.amax(num)
    # normalize to [0, ysize]
    num = num*ysize//maxim
    # reverse along y axis
    num = ysize - num[:]
    img = np.zeros((ysize, xsize,3), dtype=np.int8)
    for i in range(xsize):
        if maskyr[maskx[i]]:
            # maskx[] map from i-th y-axis in num to maskx[i]-th x-axis in img
            # num[maskx[i],1] map from (coordinate of high price in num) to (pixel position in img)
            img[int(num[maskx[i]]):,i,0]=1
            img[int(num[maskx[i]]):,i,2]=1
        if maskyg[maskx[i]]:
            img[int(num[maskx[i]]):,i,1]=1
            img[int(num[maskx[i]]):,i,2]=1
            #img[int(num[maskx[i],1]):int(num[maskx[i],2]),i,1]=0.5
            #img[int(num[maskx[i],0]):int(num[maskx[i],3]),i,1]=1
            #img[int(num[maskx[i],1]):int(num[maskx[i],2]),i,2]=0.5
            #img[int(num[maskx[i],0]):int(num[maskx[i],3]),i,2]=1
    return img

if __name__ == '__main__':
    df=get_df('sh600230')
    df=df.tail(10)
    img = translate_volumes(df,10,15)
    print(img[:,:,0])
    print(img[:,:,1])
    print(img[:,:,2])
    print(df)
