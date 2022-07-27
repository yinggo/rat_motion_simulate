import numpy as np
import matplotlib.pyplot as plt
import os
from deblurgan.utils import loadmat_c

rootdir = '/home/user/data1/yalei/motion_num_24'
list = os.listdir(rootdir)

motiondata_num24_17_22 = np.zeros((3600, 256, 256),'float64')
for i in range(0, len(list)):
    path = os.path.join(rootdir,list[i])
    motion_img = loadmat_c(path)
    motiondata_num24_17_22[i*600:(i+1)*600, :, : ] = motion_img
