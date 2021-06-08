#coding:utf-8
from crnn.utils import strLabelConverter,resizeNormalize
import threading
import time
from crnn.network_keras import keras_crnn as CRNN
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.6
sess = tf.Session(config=config)
set_session(sess)

graph = tf.get_default_graph()##解决web.py 相关报错问题

import crnn.keys  as keys
import os#from config import ocrModelKeras
import numpy as np
def crnnSource():
    alphabet = keys.alphabetChinese##中英文模型
    converter = strLabelConverter(alphabet)
    model = CRNN(32, 1, len(alphabet)+1, 256, 1,lstmFlag=True)
    model.load_weights(os.path.join(os.getcwd(),"crnn/model","ocr-dense-keras.h5"))
    return model,converter

##加载模型
model,converter = crnnSource()

def crnnOcr(image):
       """
       crnn模型，ocr识别
       image:PIL.Image.convert("L")
       """
       scale = image.size[1]*1.0 / 32
       w = image.size[0] / scale
       w = int(w)
       transformer = resizeNormalize((w, 32))
       image = transformer(image)
       image = image.astype(np.float32)
       image = np.array([[image]])
       global graph
       with graph.as_default():
           begin_time = time.time()
           preds       = model.predict(image)
           print("time" + str(time.time() - begin_time))
       preds = preds[0]
       preds = np.argmax(preds,axis=2).reshape((-1,))
       sim_pred  = converter.decode(preds)
       return sim_pred

'''
       
class CrnnOcrPool:
    def __init__(self, inst_num):
        self.get_lock = threading.Lock()
        self.release_lock = threading.Lock()
        self.model_lock = threading.Lock()
        self.models = []
        self.use_models = []
        self.converters = []
        self.use_converters = []
        alphabet = keys.alphabetChinese##中英文模型
        for i in range(inst_num):
            c = strLabelConverter(alphabet)
            m = CRNN(32, 1, len(alphabet)+1, 256, 1,lstmFlag=True)
            m.load_weights(os.path.join(os.getcwd(),"crnn/model","ocr-dense-keras.h5"))
            #m, c = crnnSource()
            print(id(m))
            self.models.append(m)
            self.converters.append(c)

    def get_model(self):
        while len(self.models) == 0:
            pass
        self.get_lock.acquire()
        m = self.models.pop()
        self.use_models.append(m)
        c = self.converters.pop()
        self.use_converters.append(c)
        self.get_lock.release()
        return m,c
    def release_model(self, model, converter):
        self.release_lock.acquire()
        self.models.append(model)
        self.use_models.remove(model)
        self.converters.append(converter)
        self.use_converters.remove(converter)
        self.release_lock.release()


    def ocr(self, image):
        model,converter = self.get_model()
        scale = image.size[1]*1.0 / 32
        w = image.size[0] / scale
        w = int(w)
        transformer = resizeNormalize((w, 32))
        image = transformer(image)
        image = image.astype(np.float32)
        image = np.array([[image]])
        global graph
        with graph.as_default():
            preds       = model.predict(image)
        preds = preds[0]
        preds = np.argmax(preds,axis=2).reshape((-1,))
        sim_pred  = converter.decode(preds)
        self.release_model(model, converter)
        return sim_pred

'''