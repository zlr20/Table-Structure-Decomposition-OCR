# coding:utf-8
import torch
torch.backends.cudnn.enabled = False    # to globally disable CuDNN
import numpy as np
from torch.autograd import Variable 
from crnn.utils import strLabelConverter, resizeNormalize, resizeNormalize2
from crnn.network_torch import CRNN
from crnn import keys
from collections import OrderedDict
import torch.backends.cudnn as cudnn
import os
#import pysnooper

def loadData(v, data):
    # v.data.resize_(data.size()).copy_(data)
    v.resize_(data.size()).copy_(data)

def softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x

def likelyhood(preds, mat):
    N = len(preds)
    raw = []
    for i in range(N):
        if preds[i] != 0 and (not (i > 0 and preds[i - 1] == preds[i])):
            prob = softmax(mat[i])
            raw.append(prob[preds[i]])
    return np.array(raw)

ocrModel = os.path.join(os.getcwd(), "crnn/model", "ocr-lstm.pth")
LSTMFLAG = True
GPU = True
chinsesModel = True

""" 加载模型 """
def crnnSource():
    chinsesModel = True
    if chinsesModel:
        alphabet = keys.alphabetChinese     # 中英文模型
    else:
        alphabet = keys.alphabetEnglish     # 英文模型

    converter = strLabelConverter(alphabet)
    if torch.cuda.is_available() and GPU:
        model = CRNN(32, 1, len(alphabet) + 1, 256, 1, lstmFlag=LSTMFLAG).cuda()    ## LSTMFLAG=True crnn 否则 dense ocr
    else:
        model = CRNN(32, 1, len(alphabet) + 1, 256, 1, lstmFlag=LSTMFLAG).cpu()

    trainWeights = torch.load(ocrModel, map_location=lambda storage, loc: storage)
    modelWeights = OrderedDict()
    for k, v in trainWeights.items():
        name = k.replace('module.', '')  # remove `module.`
        modelWeights[name] = v

    # load params
    model.load_state_dict(modelWeights)

    return model, converter

############ 加载模型
model, converter = crnnSource()
model.eval()

""" crnn模型，ocr识别 """
def crnnOcr(image):
    scale = image.size[1] * 1.0 / 32
    w = image.size[0] / scale
    w = int(w)
    transformer = resizeNormalize((w, 32))
    image = transformer(image)
    image = image.astype(np.float32)
    image = torch.from_numpy(image)

    if torch.cuda.is_available() and GPU:
        image = image.cuda()
    else:
        image = image.cpu()

    image = image.view(1, 1, *image.size())
    image = Variable(image)
    with torch.no_grad():
        preds = model(image)

    cudnn.benchmark = False
    pred_ = preds.cpu().numpy()
    # print('pred_: ', pred_.shape)
    mat_ = pred_.reshape((-1, 5530))
    # print('mat_.shape: ', mat_.shape)
    _, preds1 = preds.max(2)
    preds2 = preds1.transpose(1, 0).contiguous().view(-1)
    sim_pred = converter.decode(preds2)
    # print('groundtruth:', len(sim_pred))  # 真实的label和字符长度
    # print('preds2_shape: ', preds2.shape)
    certainty = likelyhood(preds2.cpu(), mat_)

    return sim_pred, certainty.mean()

""" crnn模型，batch处理 ocr识别 """
def crnnOcr2(batch_list):
    batch_size = len(batch_list)
    imgH = 32
    ratios = []
    for image in batch_list:        # 32*280 (这是我们数据集图片的大小)
        w, h = image[1].size        # image[0] 存放的是索引值
        ratios.append(w / float(h))
    ratios.sort()
    max_ratio = ratios[-1]                      # 最大的宽高比
    imgW = int(np.floor(max_ratio * imgH))      # 传来的宽度并没有影响（按照h：32同等的比例进行缩放）
    imgW = max(imgH, imgW)                      # 获取到最大的宽高信息
    transform = resizeNormalize2((1, imgH, imgW))
    images = [transform(image[1]) for image in batch_list]      # 通过对象(参数)调用__call__方法
    i_value = [image[0] for image in batch_list]

    # print('i_value = ', i_value)
    images = torch.cat([t.unsqueeze(0) for t in images], 0)     # 维度提升cat合并

    images = images.cuda()
    with torch.no_grad():
        preds = model(images)
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
    cudnn.benchmark = False

    # print('---------preds---------:', preds.shape)            # (41, 48, 5530)
    preds2 = preds.permute(1, 0, 2)
    # print('---------preds2---------:', preds2[0].shape)       # (41, 5530)

    #certaintys = []
    # for i in range(batch_size):
    #     pred2 = preds2[i]
    #     # print('-----------pred2-----------:', pred2.shape)    # (41, 5530)
    #     mat_ = pred2.cpu().numpy()
    #     pred2 = pred2.unsqueeze(1)
    #     # print('----------pred222----------:', pred2.shape)    # (41, 1, 5530)
    #     _, pred2 = pred2.max(2)
    #     pred2 = pred2.transpose(1, 0).contiguous().view(-1)
    #     certainty = likelyhood(pred2.cpu(), mat_)
    #     # print('----------certainty2------------:', np.mean(certainty))
    #     certainty = np.mean(certainty)
    #     certaintys.append(certainty)

    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)
    sim_preds = converter.decode2(preds.data, preds_size.data, raw=False)
    return i_value, sim_preds#, certaintys

'''
class CrnnOcrPool:
    def __init__(self, inst_num):
        ocrModel = os.path.join(os.getcwd(),"crnn/model","ocr-lstm.pth")        # 原作者模型
        # ocrModel = os.path.join(os.getcwd(), "crnn/model", "ocr-lstm_9.pth")
        LSTMFLAG = True
        GPU = True
        chinsesModel = True
        self.get_lock = threading.Lock()
        self.release_lock = threading.Lock()
        self.model_lock = threading.Lock()
        self.models = []
        self.use_models = []
        self.converters = []
        self.use_converters = []
        if chinsesModel:
            alphabet = keys.alphabetChinese     # 中英文模型
        else:
            alphabet = keys.alphabetEnglish     # 英文模型
        self.converter = strLabelConverter(alphabet)
        for i in range(inst_num):
            m, c = crnnSource()
            m.eval()
            # print(id(m))
            self.models.append(m)
            self.converters.append(c)

    @pysnooper.snoop()
    def get_model(self):
        while len(self.models) == 0:
            pass
        self.get_lock.acquire()
        m = self.models.pop()
        self.use_models.append(m)
        c = self.converters.pop()
        self.use_converters.append(c)
        self.get_lock.release()
        return m, c

    def release_model(self, model, converter):
        self.release_lock.acquire()
        self.models.append(model)
        self.use_models.remove(model)
        self.converters.append(converter)
        self.use_converters.remove(converter)
        self.release_lock.release()

    ##### 改：输入batch进行识别 #####
    def ocr2(self, batch_list):
        batch_size = len(batch_list)
        model, converter = self.get_model()
        imgH = 32
        ratios = []
        for image in batch_list:        # 32 * 280(这是我们数据集图片的大小)
            w, h = image[1].size
            ratios.append(w / float(h))
        ratios.sort()
        max_ratio = ratios[-1]          # 最大的宽高比
        imgW = int(np.floor(max_ratio * imgH))      # 传来的宽度并没有影响（按照h：32同等的比例进行缩放）
        imgW = max(imgH, imgW)                      # 获取到最大的宽高信息
        transform = resizeNormalize2((1, imgH, imgW))
        images = [transform(image[1]) for image in batch_list]  # 通过对象(参数)调用__call__方法
        i_value = [image[0] for image in batch_list]
        # print('i_value = ', i_value)
        images = torch.cat([t.unsqueeze(0) for t in images], 0)  # 维度提升cat合并

        images = images.cuda()
        with torch.no_grad():
            preds = model(images)
            preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
        cudnn.benchmark = False

        # print('---------preds---------:', preds.shape)            # (41, 48, 5530)
        preds2 = preds.permute(1, 0, 2)
        # print('---------preds2---------:', preds2[0].shape)       # (41, 5530)

        certaintys = []
        for i in range(batch_size):
            pred2 = preds2[i]
            # print('-----------pred2-----------:', pred2.shape)    # (41, 5530)
            mat_ = pred2.cpu().numpy()
            pred2 = pred2.unsqueeze(1)
            # print('----------pred222----------:', pred2.shape)    # (41, 1, 5530)
            _, pred2 = pred2.max(2)
            pred2 = pred2.transpose(1, 0).contiguous().view(-1)
            certainty = likelyhood(pred2.cpu(), mat_)
            # print('----------certainty2------------:', np.mean(certainty))
            certainty = np.mean(certainty)
            certaintys.append(certainty)

        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode2(preds.data, preds_size.data, raw=False)
        return i_value, sim_preds, certaintys

    def ocr(self, image):
        model,converter = self.get_model()
        # print('size:', image.size)
        scale = image.size[1]*1.0 / 32
        w = image.size[0] / scale
        w = int(w)
        transformer = resizeNormalize((w, 32))
        image = transformer(image)
        image = image.astype(np.float32)
        image = torch.from_numpy(image)
        image = image.cuda()
        image = image.view(1, 1, *image.size())
        image = Variable(image)
        with torch.no_grad():
            preds = model(image)
            # print(preds)
        cudnn.benchmark = False
        pred_ = preds.cpu().numpy()
        # print('pred_: ', pred_.shape)
        mat_ = pred_.reshape((-1, 5530))
        # print('mat_.shape: ', mat_.shape)
        _, preds1 = preds.max(2)
        preds2 = preds1.transpose(1, 0).contiguous().view(-1)
        sim_pred = converter.decode(preds2)
        # print('groundtruth:', len(sim_pred))  # 真实的label和字符长度
        # print('preds2_shape: ', preds2.shape)
        certainty = likelyhood(preds2.cpu(), mat_)
        self.release_model(model, converter)

        return sim_pred, certainty.mean()
'''
