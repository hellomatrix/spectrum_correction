import rawpy
import numpy as np
import scipy.misc as sm
import matplotlib.pyplot as plt
import cv2
import os

testName = 'test'
path ='./'+testName
filelist = os.listdir(path)
pngdir = os.path.join(path, testName + '_pngfile')
if not os.path.exists(pngdir):
    os.mkdir(pngdir)

for filename in filelist:
    filepath = os.path.join(path, filename)
    if os.path.isdir(filepath):
        # dirlist(filepath, allfile)
        pass
    else:
        raw = rawpy.imread(filepath)
        img = raw.raw_image
        pngName = os.path.join(pngdir, filename + '.png')
        cv2.imwrite(pngName, img)

        # allfile.append(filepath)
        print(filename)

# def dirlist(path, allfile):
#     filelist =  os.listdir(path)
#     # print(filelist)
#     for filename in filelist:
#         filepath = os.path.join(path, filename)
#         if os.path.isdir(filepath):
#             dirlist(filepath, allfile)
#         else:
#
#             pngdir = os.path.join(path, 'pngfile')
#             raw = rawpy.imread(filepath)
#             img = raw.raw_image
#
#             pngName = os.path.join(pngdir,filename+'.png')
#             cv2.imwrite('do11g.png',img)
#
#
#             # allfile.append(filepath)
#             print(filename)
#     return allfile







# g = os.walk("./")
# for path,d,filelist in g:
#     print(d)
#     for filename in filelist:
#         print(os.path.join(path, filename))

# im = cv2.imread('./dog.png')
# im = cv2.imread('myImg.png')

# raw = rawpy.imread('A003_C036_20160924_R1_000314.dng')
# img = raw.raw_image
# cv2.imwrite('do11g.png',img)
# # img = raw.raw_image
#
# print(img.shape)
# fig = plt.figure()
# plt.imshow(img,cmap='gray')
#
#
# fig = plt.figure()
# im = cv2.imread('do11g.png')
# plt.imshow(img,cmap='gray')
#
#
# # cv2.imwrite('do11g.png',img)
# # sm.imsave('./dog.png',np.ndarray(img))
# # cv2.imwrite('dog.png',[img,img,img])
#
# myImg = np.random.randint(0,65535, size=(200, 400),dtype=np.uint16) # create a random image
# cv2.imwrite('myImg.png',myImg)
# fig = plt.figure()
# myImg = cv2.imread('myImg.png')
# plt.imshow(myImg,cmap='gray')
#
#
# plt.show()
# plt.figure(1)
# # plt.imshow(raw.raw_image,cmap='gray')
# cv2.axis("off")
# cv2.title("Input Image")
# cv2.imshow(img, cmap="gray")  #显示图片
# cv2.show()

# plt.figure(2)
# plt.imshow(raw.raw_image_visible)
#
# plt.show()


#
# Features:
#
# raw.black_level_per_channel  raw.raw_colors
# raw.camera_whitebalance      raw.raw_colors_visible
# raw.close(                   raw.raw_image
# raw.color_desc               raw.raw_image_visible
# raw.color_matrix             raw.raw_pattern
# raw.daylight_whitebalance    raw.raw_type
# raw.dcraw_make_mem_image(    raw.raw_value(
# raw.dcraw_process(           raw.raw_value_visible(
# raw.num_colors               raw.rgb_xyz_matrix
# raw.open_buffer(             raw.sizes
# raw.open_file(               raw.tone_curve
# raw.postprocess(             raw.unpack(
# raw.raw_color(


# if __name__ == '__main__':
#
#     allfile = []
#     allfile = dirlist('./', allfile)