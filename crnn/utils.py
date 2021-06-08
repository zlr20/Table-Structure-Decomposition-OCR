#!/usr/bin/python
# encoding: utf-8
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms

class strLabelConverter(object):

    def __init__(self, alphabet):
        self.alphabet = alphabet + u'ç'  # for `-1` index
        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1
            
    def decode(self,res):
        N = len(res)
        raw = []
        for i in range(N):
            if res[i] != 0 and (not (i > 0 and res[i - 1] == res[i])):
                raw.append(self.alphabet[res[i] - 1])
        return ''.join(raw)

    ### 利用 batch 批处理表格
    def decode2(self, t, length, raw=False):  # t : [64 71]  length:71
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])       # 根据索引值获取标签label
            else:                                                       # 经过变换之后的真实的识别结果
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            # print('---------------------------:', t.numel())          # 64 * 71 总字符元素的个数
            # print('===========================:', length.numel())     # 64 总字符组的个数
            texts = []
            index = 0
            for i in range(length.numel()):                             # 对每个字符组进行处理
                l = length[i]
                texts.append(self.decode2(t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts
 

class resizeNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        size = self.size
        imgW,imgH = size
        scale = img.size[1]*1.0 / imgH
        w = img.size[0] / scale
        w = int(w)

        if w == 0:
            w = 1

        # print('w = ', w)
        img = img.resize((w, imgH), self.interpolation)
        w, h = img.size
        img = (np.array(img) / 255.0 - 0.5) / 0.5

        return img


### for batch
class resizeNormalize2(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        scale = img.size[1] * 1.0 / 32
        w0 = img.size[0] / scale
        w0 = int(w0)

        if w0 == 0:
            w0 = 1
        # print('w0 = ', w0)

        img = img.resize((w0, 32), self.interpolation)

        img = transforms.ToTensor()(img)
        img.sub_(0.5).div_(0.5)
        c, h, w = img.size()
        Pad_img = torch.FloatTensor(*self.size).fill_(0)
        Pad_img[:, :, :w] = img
        if self.size[2] != w:
            Pad_img[:, :, w:] = img[:, :, w - 1].unsqueeze(2).expand(c, h, self.size[2] - w)

        return Pad_img