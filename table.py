# -*- coding: utf-8 -*-
import time
from utils import letterbox_image,exp,minAreaLine,draw_lines,minAreaRectBox,draw_boxes,line_to_line,sqrt,rotate_bound,timer,is_in
from line_split import line_split
import numpy as np
import cv2
from PIL import Image
from skimage import measure
import json

# crnn
from crnn.crnn_torch import crnnOcr, crnnOcr2

tableNetPath = 'UNet/table.weights'
SIZE = 512,512
tableNet = cv2.dnn.readNetFromDarknet(tableNetPath.replace('.weights','.cfg'),tableNetPath)
     
def dnn_table_predict(img,prob=0.5):   
    imgResize,fx,fy,dx,dy = letterbox_image(img,SIZE)
    imgResize = np.array(imgResize)
    imgW,imgH = SIZE
    image = cv2.dnn.blobFromImage(imgResize,1,size=(imgW,imgH),swapRB=False)
    image = np.array(image)/255
    tableNet.setInput(image)
    out=tableNet.forward()
    out = exp(out[0]) # shape(2,512,512) , 2指的是横纵线两个类对应的map
    out = out[:,dy:,dx:] # 虽然左上点对上了，但是右方或下方的padding没去掉？
    return out,fx,fy,dx,dy

def get_seg_table(img,prob,row=10,col=10):
    out,fx,fy,dx,dy = dnn_table_predict(img,prob)
        
    rows = out[0]
    cols = out[1]

    labels=measure.label(cols>prob,connectivity=2)  
    regions = measure.regionprops(labels)
    ColsLines = [minAreaLine(line.coords) for line in regions if line.bbox[2]-line.bbox[0]>col ]
    # if debug:
    #     cv2.imwrite('_cols.jpg',labels*255)
    
    labels=measure.label(rows>prob,connectivity=2)
    regions = measure.regionprops(labels)
    RowsLines = [minAreaLine(line.coords) for line in regions if line.bbox[3]-line.bbox[1]>row ]
    # RowsLines[0] = [xmin,ymin,xmax,ymax]注x指横向上，y指纵向上
    # if debug:
    #     cv2.imwrite('_rows.jpg',labels*255)

    imgW,imgH = SIZE
    tmp =np.zeros((imgH-2*dy,imgW-2*dx),dtype='uint8')
    tmp = draw_lines(tmp,ColsLines+RowsLines,color=255, lineW=1)

    # 闭运算：先膨胀后腐蚀，用来连接被误分为许多小块的对象
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    tmp = cv2.morphologyEx(tmp, cv2.MORPH_CLOSE, kernel,iterations=1)
    seg_table = cv2.resize(tmp,None,fx=1.0/fx,fy=1.0/fy,interpolation=cv2.INTER_CUBIC)

    degree = 0.0
    if len(RowsLines) >= 3:
        degree = np.array([np.arctan2(bbox[3]-bbox[1],bbox[2]-bbox[0]) for bbox in  RowsLines])
        degree = np.mean(-degree*180.0/np.pi)
    return seg_table,degree

def find_tables(img_seg):
    # from the seg image, detect big bounding box and decide how many tables in the picture
    tables = []
    h,w = img_seg.shape
    _,contours, hierarchy = cv2.findContours(img_seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        table_flag = True
        contourArea = cv2.contourArea(contour)
        if contourArea < h * w  * 0.05:
            table_flag = False
        if not table_flag:
            continue
        contour = contour.reshape((-1, 2))
        xmin,ymin = np.min(contour,axis=0)
        xmax,ymax = np.max(contour,axis=0)
        tables.append([xmin,ymin,xmax,ymax])
    tables = sorted(tables,key=lambda x : x[1])
    return np.array(tables)

def find_cells(img_seg,tables):
    if not len(tables):
        return []
    h,w = img_seg.shape
    tabelLabels=measure.label(img_seg==0,connectivity=2)  
    regions=measure.regionprops(tabelLabels)
    rboxes= []
    for table in tables:
        tmp = []
        for i,region in enumerate(regions):
            if h*w*0.0001 < region.bbox_area <h*w*0.5:
                rbox = np.array(map(int,region.bbox))[[1,0,3,2]]
                if is_in(rbox,table):
                    tmp.append(rbox)
        rboxes.append(np.array(tmp))
    return np.array(rboxes)


def annotate_cell(img,cells):
    # now cells is a ndarray with shape (n,4)
    res = np.array([{'text':''} for cell in cells])
    # start col
    sc = 0
    idx = cells[:, 0].argsort()
    cells = cells[idx]
    res = res[idx]
    eps = np.diff(cells,axis=0)[:,0]
    mean = np.mean(eps)
    breakpoints = np.where(eps >= mean)[0]
    for i,item in enumerate(res):
        item['start_col'] = sc
        if i in breakpoints:
            sc += 1
    # end col
    ec = 0
    idx = cells[:, 2].argsort()
    cells = cells[idx]
    res = res[idx]
    eps = np.diff(cells,axis=0)[:,2]
    #print(eps)
    mean = np.mean(eps)
    breakpoints = np.where(eps >= mean)[0]
    for i,item in enumerate(res):
        item['end_col'] = ec
        if i in breakpoints:
            ec += 1
    # start row
    sr = 0
    idx = cells[:, 1].argsort()
    cells = cells[idx]
    res = res[idx]
    eps = np.diff(cells,axis=0)[:,1]
    mean = np.mean(eps)
    breakpoints = np.where(eps >= mean)[0]
    for i,item in enumerate(res):
        item['start_row'] = sr
        if i in breakpoints:
            sr += 1
    # end row
    er = 0
    idx = cells[:, 3].argsort()
    cells = cells[idx]
    res = res[idx]
    eps = np.diff(cells,axis=0)[:,3]
    mean = np.mean(eps)
    breakpoints = np.where(eps >= mean)[0]
    for i,item in enumerate(res):
        item['end_row'] = er
        if i in breakpoints:
            er += 1

    batch_list_text = []
    for i,([xmin,ymin,xmax,ymax],info) in enumerate(zip(cells,res)):
        lines = line_split(img[ymin:ymax,xmin:xmax],y=ymin,x=xmin)
        for [_xmin,_ymin,_xmax,_ymax] in lines:
            #cv2.imwrite('./part/'+str(i)+'_'+str(_ymax)+'.jpg',img[_ymin:_ymax,_xmin:_xmax])
            partImg = img[_ymin:_ymax,_xmin:_xmax]
            partImg = Image.fromarray(partImg).convert('L')
            batch_list_text.append((i, partImg.convert('L')))

    try:
        i_value, batch_text = crnnOcr2(batch_list_text)
    except:
        print("!"*20)
        print('CUDA OUT OF MEMORY, SPLIT BATCH')
        print("!"*20)
        pt = int(len(batch_list_text)/4)
        i_value1, batch_text1 = crnnOcr2(batch_list_text[:pt])
        i_value2, batch_text2 = crnnOcr2(batch_list_text[pt:2*pt])
        i_value3, batch_text3 = crnnOcr2(batch_list_text[2*pt:3*pt])
        i_value4, batch_text4 = crnnOcr2(batch_list_text[3*pt:])
        i_value = i_value1 + i_value2 + i_value3 + i_value4
        batch_text = batch_text1 + batch_text2 + batch_text3 + batch_text4

    for i,text in zip(i_value,batch_text):
        res[i]['text'] += text.encode("UTF-8")+ '\n'
    
    res = res.tolist()
    res = sorted(res,key=lambda x: (x['start_row'], x['start_col']))
    return res,er+1,ec+1


def find_text(tables,w,h):
    #find the non-table area for PSENet detection
    if not len(tables):
        return np.array([[0,0,w,h]])
    Y1 = tables[:,[1,3]]
    Y2 = []
    for i in range(len(Y1)):
        if i+1 == len(Y1):
            Y2.append(Y1[i])
            break
        if Y1[i][1] >= Y1[i+1][0]: # ymax1 >= ymin2
            Y1[i+1][0] = Y1[i][0]
            Y1[i+1][1] = max(Y1[i][1],Y1[i+1][1])
            continue
        else:
            Y2.append(Y1[i])
    Y2 = np.array(Y2).reshape(-1,)
    Y2 = np.append(0,Y2)
    Y2 = np.append(Y2,h)
    Y2 = Y2.reshape(-1,2)
    return np.array([[0,y[0],w,y[1]] for y in Y2])


################################################################CORE######################################################
################################################################CORE######################################################
################################################################CORE######################################################
@timer
def tableXH(img,prob=0.5,row=30,col=10,alph=50):
    start_time = time.time()
    # use Unet to recognize tabel lines, decide how many degrees to rotate
    # also, create a seg image for easy-tablebox-recongisize.
    img_seg,degree=get_seg_table(img,prob,row,col)
    img = cv2.cvtColor(np.array(img),cv2.COLOR_RGB2BGR)
    if degree > 0.5:
        print('Rotating...')
        img = rotate_bound(img,degree)
        img_seg = rotate_bound(img_seg,degree)
    h,w = img_seg.shape
    tables = find_tables(img_seg)
    
    cells = find_cells(img_seg,tables)

    text_areas = find_text(tables,w,h)

    #############create json##############
    blocks = []
    for area in text_areas:
        blocks.append({
                "is_table": False,
                "cells": [],
                "position": area.tolist(),
                "text":"text"})
    for table,cell in zip(tables,cells):
        # {"position":int[],"start_row":int,"end_row":int,"start_column":int,"end_column":int,"text":str}
        cell,nrow,ncol= annotate_cell(img,cell)
        blocks.append({
                "is_table": True,
                "cells": cell,
                "columns": ncol,
                "rows": nrow,
                "position":table.tolist(),
                "text":""})
    blocks.sort(key=lambda x: x['position'][1])
    end_time = time.time()
    
    return {
            "cost_time": end_time - start_time,
            "result": {
                "rotated_image_width": w,
                "rotated_image_height": h,
                "result_word":blocks
                #"blocks":blocks  #according to hehe-AI 
                }
            }
################################################################CORE######################################################
################################################################CORE######################################################
################################################################CORE######################################################

if __name__ == '__main__':
    img =Image.open('test_pics/0.jpg').convert('RGB')
    res = tableXH(img)
    #print(res)
    #json.dumps(res, ensure_ascii=False)

