import numpy as np
import matplotlib.pyplot as plt
img = np.load('E:/chenyalei/mocogan_v2/scripts/test_pre_l1_simu_0927/pred_test_Channel3_1.npy')
plt.figure(num=2)
plt.imshow(img[6,:,:,2], cmap='gray')
plt.axis("off")
plt.show()