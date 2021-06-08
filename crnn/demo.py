import glob
import cv2
import time
from PIL import Image
from crnn_keras import crnnOcr
paths = glob.glob('./test/*.*')
for path in sorted(paths):
    img = cv2.imread(path)
    timeTake = time.time()
    partImg = Image.fromarray(img)
    text = crnnOcr(partImg.convert('L'))
    print('cost time: %.3fs' % (time.time()-timeTake))
    print(text)
