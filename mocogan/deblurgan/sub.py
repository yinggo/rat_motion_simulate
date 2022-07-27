import cv2
import numpy as np
import matplotlib.pyplot as plt
import  matplotlib
from PIL import Image
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('--sigma', default=5, type=int, help='noise level')
args = parser.parse_args()

# img= Image.open('E:/chenyalei/deblur-gan-master/images/test/B/35.png')
# img_ = Image.open('E:/chenyalei/deblur-gan-master/images/test/outputssim/035.png')
# # imgd = cv2.resize(img,(img.shape[1]/4,img.shape[0]/4))    #先下采样4倍再上采样恢复图像
# # imgu = cv2.resize(imgd,(img.shape[1],img.shape[0]))
# img = img.convert('L')
# img_= img_.convert('L')
# img = np.array(img)
# img = (img - 127.5) / 127.5
# img_ = np.array(img_)
img = np.load('E:/chenyalei/deblur-gan/deblur-gan-master/scripts/deblurgan/test_image/generated_image_epoch120.npy')
img_ = np.load('E:/chenyalei/deblur-gan/deblur-gan-master/scripts/deblurgan/test_image/test_gt_image.npy')
img__ = np.load('E:/chenyalei/deblur-gan/deblur-gan-master/scripts/deblurgan/test_image/test_motion_image.npy')
# def gasuss_noise(image, mean=0.05, var=0.05):
#     '''
#         添加高斯噪声
#         mean : 均值
#         var : 方差
#     '''
#     image = image
#     noise = np.random.normal(mean, var ** 0.5, image.shape)
#     out = image + noise
#     # if out.min() < 0:
#     #     low_clip = -1.
#     # else:
#     #     low_clip = 0.
#     # out = np.clip(out, low_clip, 1.0)
#     # out = np.uint8(out*255)
#     # cv2.imshow("gasuss", out)
#     return out
#
# noise_img = gasuss_noise(img)
# param = 25
# # 灰阶范围
# grayscale = 256
# w = img.shape[0]
# h = img.shape[1]
# newimg = np.zeros((h, w), np.float64)
#
# for x in range(0, h):
#     for y in range(0, w, 2):
#         r1 = np.random.random_sample()
#         r2 = np.random.random_sample()
#         z1 = param * np.cos(2 * np.pi * r2) * np.sqrt((-2) * np.log(r1))
#         z2 = param * np.sin(2 * np.pi * r2) * np.sqrt((-2) * np.log(r1))
#
#         fxy = int(img[x, y] + z1)
#         fxy1 = int(img[x, y + 1] + z2)
#         # f(x,y)
#         if fxy < 0:
#             fxy_val = 0
#         elif fxy > grayscale - 1:
#             fxy_val = grayscale - 1
#         else:
#             fxy_val = fxy
#         # f(x,y+1)
#         if fxy1 < 0:
#             fxy1_val = 0
#         elif fxy1 > grayscale - 1:
#             fxy1_val = grayscale - 1
#         else:
#             fxy1_val = fxy1
#         newimg[x, y] = fxy_val
#         newimg[x, y + 1] = fxy1_val
#
# plt.imshow(noise_img,cmap='gray')
# plt.show()
# cv2.imshow('preview', newimg)
# cv2.waitKey()
# cv2.destroyAllWindows()

# noise =  np.random.normal(0, args.sigma/255.0, img.shape)
# noise_img = img  + noise

# def add_gaussian_noise(image_in, noise_sigma):
#     """
#     给图片添加高斯噪声
#     image_in:输入图片
#     noise_sigma：
#     """
#
#
#     temp_image = np.float64(np.copy(image_in))
#
#     h, w, _ = temp_image.shape
#     # 标准正态分布*noise_sigma
#     noise = np.random.randn(h, w) * noise_sigma
#
#     noisy_image = np.zeros(temp_image.shape, np.float64)
#     if len(temp_image.shape) == 2:
#         noisy_image = temp_image + noise
#     else:
#         noisy_image[:, :, 0] = temp_image[:, :, 0] + noise
#         noisy_image[:, :, 1] = temp_image[:, :, 1] + noise
#         noisy_image[:, :, 2] = temp_image[:, :, 2] + noise
#
#     return noisy_image
#
#
# noise_sigma = 5
# noise_img = add_gaussian_noise(img, noise_sigma=noise_sigma)
img = img[2,:,:,0]
img_ = img_[2,:,:,0]
img__ = img__[2,:,:,0]
imgall = np.concatenate((img,img_,img__),axis=1)
plt.imshow(imgall,cmap='gray')
# plt.show()
plt.axis("off")
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
plt.margins(0, 0)
plt.savefig('C:/Users/Chenli/Desktop/%s_%d_%s.png' % ('bcr', 7, "f"),bbox_inches='tight', dpi=400,pad_inches=0)
# plt.show()
# plt.imshow(img_,cmap='gray')
# plt.show()
# plt.imshow(img__,cmap='gray')
# plt.show()
# cv2.imwrite('noise_{}.png'.format(noise_sigma), noise_img)

# img = img.convert('L')
# img_= img_.convert('L')
# img = np.array(img)
# img_ = np.array(img_)
d = cv2.absdiff(img,img_)
fig = plt.figure(dpi=200)
cnorm = matplotlib.colors.Normalize(vmin=0, vmax=0.65)
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
plt.savefig('C:/Users/Chenli/Desktop/%s_%d_%s.png' % ('bcr', 8, "f"),bbox_inches='tight', dpi=400,pad_inches=0)
# plt.savefig('./MEBC_RCAN/%s_%d_%s.png' % ('bcr', i, "w"), bbox_inches='tight',dpi=400,pad_inches=0)
