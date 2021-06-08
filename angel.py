import numpy as np
import onnxruntime as rt
import time
from PIL import Image

def _preprocess_input(x, data_format='channels_last'):
    if data_format == 'channels_first':
        # 'RGB'->'BGR'
        if x.ndim == 3:
            x = x[::-1, ...]
        else:
            x = x[:, ::-1, ...]
    else:
        # 'RGB'->'BGR'
        x = x[..., ::-1]
    mean = [103.939, 116.779, 123.68]
    std = None
    # Zero-center by mean pixel
    if data_format == 'channels_first':
        if x.ndim == 3:
            x[0, :, :] -= mean[0]
            x[1, :, :] -= mean[1]
            x[2, :, :] -= mean[2]
            if std is not None:
                x[0, :, :] /= std[0]
                x[1, :, :] /= std[1]
                x[2, :, :] /= std[2]
        else:
            x[:, 0, :, :] -= mean[0]
            x[:, 1, :, :] -= mean[1]
            x[:, 2, :, :] -= mean[2]
            if std is not None:
                x[:, 0, :, :] /= std[0]
                x[:, 1, :, :] /= std[1]
                x[:, 2, :, :] /= std[2]
    else:
        x[..., 0] -= mean[0]
        x[..., 1] -= mean[1]
        x[..., 2] -= mean[2]
        if std is not None:
            x[..., 0] /= std[0]
            x[..., 1] /= std[1]
            x[..., 2] /= std[2]
    return x

sess = rt.InferenceSession("onnx/model.onnx")

def angel(im):
    ROTATE = [0, 90, 180, 270]
    w, h = im.size
    # 对图像进行剪裁
    # 左上角(int(0.1 * w), int(0.1 * h))
    # 右下角(w - int(0.1 * w), h - int(0.1 * h))
    #xmin, ymin, xmax, ymax = int(0.1 * w), int(0.1 * h), w - int(0.1 * w), h - int(0.1 * h)
    #im = im.crop((xmin, ymin, xmax, ymax))  # 剪切图片边缘，清除边缘噪声
    # 对图片进行剪裁之后进行resize成(224,224)
    im = im.resize((224, 224))
    # 将图像转化成数组形式
    img = np.array(im)
    img = _preprocess_input(img.astype(np.float32))
    img = np.expand_dims(img,axis=0)
    input_name = sess.get_inputs()[0].name
    pred_onx = sess.run(None, {input_name: img})[0]
    print(pred_onx)
