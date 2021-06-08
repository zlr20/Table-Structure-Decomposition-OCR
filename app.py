# -*- coding: utf-8 -*-
import numpy as np
import time
import sys

import threading            # 加锁机制，防止并发
lock = threading.Lock()     # 生成锁对象，全局唯一

# angel
from angel import angel

# table
import table
from table import tableXH

# web
import json
import base64
from enum import Enum

class OCR_TYPE(Enum):  # 后续可新增单一场景下的表格识别功能
    ORIGIN_TABLE_OCR = 0

ocr_types = ['Table OCR']

# from scenes.utils import cvt_boxes, mg_boxes, cvt_boxes2

import cv2
from PIL import Image
import os
import time
import web

web.config.debug = False
from web import template

demo = template.render('templates', base='layout')
# dog
# from load_cdll import *

class OCR:
    def GET(self):
        post = {}
        post['types'] = ocr_types
        return demo.ocr(post)

    def POST(self):
        data = web.data()
        data = json.loads(data.decode('utf-8'))

        # base64
        lock.acquire()          # 获取锁。未获取到会阻塞程序，直到获取到锁才会往下执行
        img_bstr = data['image'].encode().split(b';base64,')[-1]
        img_bstr = base64.b64decode(img_bstr)
        ocr_type = OCR_TYPE(int(data['type']))
        buf = np.fromstring(img_bstr, dtype=np.uint8)
        img = cv2.imdecode(buf, 1)  # BGR
        # end
        # print 'decode time: %.3fs' % (dectime-taketime)
        angel(Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)))
        lock.release()          # 释放锁
        return json.dumps({}, ensure_ascii=False)
        feedback = tableXH(Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)))#RECOGNIZE(img)
        result = feedback['result']
        cost_time = feedback['cost_time']
        #print(result)
        print('Cost Time : {} s'.format(cost_time))

        lock.release()          # 释放锁
        return json.dumps(result, ensure_ascii=False)


urls = ('/ocr', 'OCR')

if __name__ == "__main__":
    app = web.application(urls, globals())
    app.run()
