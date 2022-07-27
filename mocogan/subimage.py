from PIL import Image
import numpy as np
import cv2
import numpy as np
import matplotlib.pyplot as plt
import  matplotlib
from PIL import Image
import argparse
# img_my = Image.open('/home/user/data1/yalei/mocoGAN/moco-gan-master/scripts/test_mouse_0801/test_0_motion_fse.png')
# img_arr = np.array(img_my)
# img_arr = cv2.normalize(img_arr, None, 1.0, 0.0, norm_type=cv2.NORM_MINMAX)
#
# img_corre = img_arr[:,0:853,0]
# img_free = img_arr[:,853:1706,0]

# plt.imshow(img_corre,cmap='gray')
# plt.show()
gt_img = np.load('/home/user/data1/yalei/mocoGAN/moco-gan-master/scripts/unet_history/test_dncnn_0916/gt_test_dncnn_2.npy')
pred_img = np.load('/home/user/data1/yalei/mocoGAN/moco-gan-master/scripts/unet_history/test_dncnn_0916/motion_test_dncnn_2.npy')
dimg1 = gt_img[5, :, :, 0]
dimg2 = pred_img[5, :, :, 0]
d = abs(dimg1-dimg2)*1
# fig = plt.figure(dpi=200)
plt.figure(num=1)
cnorm = matplotlib.colors.Normalize(vmin=0, vmax=0.5)
m = matplotlib.cm.ScalarMappable(norm=cnorm, cmap=matplotlib.cm.jet)
# d = np.abs(np.abs(img) - np.abs(img_))
m.set_array(d)
plt.imshow(d, norm=cnorm, cmap="jet")
plt.axis("off")
plt.colorbar(m)
# plt.show(m)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
plt.margins(0, 0)
plt.savefig('/home/user/data1/yalei/mocoGAN/moco-gan-master/scripts/unet_history/test_dncnn_0916/%s_%d_%s.png' % ('sub_motion_1', 16,"dncnn"), bbox_inches='tight', dpi=400,
            pad_inches=0)