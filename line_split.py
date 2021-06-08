# coding:utf-8
import numpy as np
from PIL import Image
from scipy import ndimage

THRESHOLD = 200  # for white background
TABLE = [1]*THRESHOLD + [0]*(256-THRESHOLD)

def line_split(image, y,x,table=TABLE):
    h,w = image.shape[0:2]
    image = Image.fromarray(image)
    image_ = image.convert('L')
    bn = image_.point(table, '1')
    bn_mat = np.array(bn)
    h, pic_len = bn_mat.shape

    bn_mat_diff = np.diff(bn_mat)
    project = np.sum(bn_mat_diff, 1)
    #print(project)

    signal = ndimage.median_filter(project,size=3)
    min_val = np.min(signal)
    signal[0] = 0
    signal[-1] = 0
    pos = np.where(signal == min_val)[0]
    #print(pos)

    diff = np.diff(pos)
    #print(diff)
    
    breakpoints = np.where(diff>=3)[0]
    #print(breakpoints)
    
    if not len(breakpoints):
        return [[x,y,w+x,h+y]]

    lines = []
    for breakpoint in breakpoints:
        lines.append([x,pos[breakpoint]+y,w+x,pos[breakpoint+1]+y])
    #print(lines)
    return(lines)
