import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import cv2

x_l=np.load=("C:\\Users\\ismail\\Desktop\\yapay zeka çalışma\\X.npy")
y_l=np.load=("C:\\Users\\ismail\\Desktop\\ANN\\isaretdili\\Y.npy")
print(len(x_l))

plt.imshow(x_l[260])
plt.show()

img_size=64
plt.subplot(1,2,1)
plt.imshow(x_l[260].reshape(img_size,img_size))
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(x_l[900].reshape(img_size,img_size))
plt.axis("off")











