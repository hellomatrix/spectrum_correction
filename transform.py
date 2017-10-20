import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

imglist=[]
imglist.append(cv2.imread('./dj_1/dim0_8.png',cv2.CV_8U))
imglist.append(cv2.imread('./dj_1/dim10_8.png',cv2.CV_8U))
imglist.append(cv2.imread('./dj_1/dim19_8.png',cv2.CV_8U))

cv2.namedWindow("image", cv2.WINDOW_NORMAL)
cv2.imshow("image", imglist[1])
cv2.waitKey(0)

# # # input and output windows
# cv2.namedWindow('input', cv2.WINDOW_NORMAL)
# # cv2.resizeWindow('input', 300, 400)
# cv2.imshow('input', imglist[1])
#
# # cv2.namedWindow('one', cv2.WINDOW_NORMAL)
# # cv2.resizeWindow('one', 200, 300)
#
# # cv2.imshow('one', imglist[0])
#
# cv2.waitKey(0)





# for i in range(len(imglist)):
#     fig = mpp.figure()
#     mpp.imshow(np.uint8(imglist[i]),cmap='gray')
# mpp.show()
    # cor_xy = []

# img = cv2.imread('lena.jpg', 1)
# rows, cols, channel = img.shape

# rows,cols = imglist[1].shape
#
# pts1 = np.float32([[7234, 1250], [14012, 1246], [10102, 922]])
# pts2 = np.float32([[5147, 1217], [11924, 1188], [11886, 891]])
# # pts2 = np.float32([[5080, 1217], [11860, 1188], [11886, 891]])
# pts3 = np.float32([[3349, 1241],[10122, 1207],[10106, 921]])

# M = cv2.getAffineTransform(pts3, pts2)
# # M = cv2.getAffineTransform(pts3, pts1)
#
# dst = cv2.warpAffine(imglist[2], M, (cols, rows))
#
# cv2.imwrite('dst.png',dst)

# plt.subplot(311), plt.imshow(imglist[0]), plt.title('Input')
# plt.subplot(312), plt.imshow(dst), plt.title('Output')
# plt.subplot(313), plt.imshow(imglist[1]), plt.title('Output')
# plt.show()
#
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax1 = fig.figimage(imglist[0])





# # def tranceform(path):
# #     filelist = os.listdir(path)
# #
# #     img_list=[]
# #     for filename in filelist:
# #         filepath = os.path.join(path, filename)
# #         if os.path.isdir(filepath):
# #             pass
# #         else:
# #             img_list.append(cv2.imread(filepath))
# #
# #     return img_list
# # imglist = tranceform('./dj_1')[9:12]
# # img_gold = imglist[len(imglist)//2]
#
# imglist=[]
# imglist.append(cv2.imread('./dj_1/dim0_8.png',cv2.CV_8U))
# imglist.append(cv2.imread('./dj_1/dim10_8.png',cv2.CV_8U))
# imglist.append(cv2.imread('./dj_1/dim19_8.png',cv2.CV_8U))
#
#
# # for i in range(len(imglist)):
# #     fig = mpp.figure()
# #     mpp.imshow(np.uint8(imglist[i]),cmap='gray')
# # mpp.show()
#     # cor_xy = []
#
# # img = cv2.imread('lena.jpg', 1)
# # rows, cols, channel = img.shape
#
# rows,cols = imglist[1].shape
#
# pts1 = np.float32([[7234, 1250], [14012, 1246], [10102, 922]])
# pts2 = np.float32([[5147, 1217], [11924, 1188], [11886, 891]])
# # pts2 = np.float32([[5080, 1217], [11860, 1188], [11886, 891]])
# pts3 = np.float32([[3349, 1241],[10122, 1207],[10106, 921]])
#
# M = cv2.getAffineTransform(pts3, pts2)
# # M = cv2.getAffineTransform(pts3, pts1)
#
# dst = cv2.warpAffine(imglist[2], M, (cols, rows))
#
# cv2.imwrite('dst.png',dst)
#
# # plt.subplot(311), plt.imshow(imglist[0]), plt.title('Input')
# # plt.subplot(312), plt.imshow(dst), plt.title('Output')
# # plt.subplot(313), plt.imshow(imglist[1]), plt.title('Output')
# # plt.show()





# imglist = tranceform(path)
#
# img_gold = imglist[len(imglist)/2]
# img2 = img_gold.copy()  # a copy of original image
#
# gold = np.zeros(img.shape, np.uint8)
# output = np.zeros(img.shape, np.uint8)  # output image to be shown
#
# # input and output windows
# cv2.namedWindow('output', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('output', 700, 500)
#
# cv2.namedWindow('input', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('input', 700, 500)
# cv2.setMouseCallback('input', onmouse)
#
# while (True):
#
#     cv2.imshow('output', output)
#     cv2.imshow('input',img)
#     k = 0xFF & cv2.waitKey(1)
#     # key bindings
#     if k == 27:  # esc to exit
#         break
#     elif k == ord('s'):  # BG drawing
#          print(" mark background regions with right mouse button \n")
#          value = DRAW_BG
#
#     elif k == ord('r'):  # reset everything
#         totalTime = 0
#         showResult = 0
#         print('ok')
#         print(rects[0][0],rects[0][1],rects[0][2],rects[0][3])
#         np.mean(np.mean(light1[rects[0][0]:rects[0][1],rects[0][2]:rects[0][3],:], 0), 0)
#
#         x_value = range(light1.shape[2])
#         # y_value = np.mean(np.mean(light1[rects[0][0]:rects[0][1],rects[0][2]:rects[0][3],:], 0), 0)
#
#
#         for k in range(len(avg_reflect_spectrum)-1):
#
#             plt.figure(k+100)
#             plt.plot(x_value,avg_reflect_spectrum[k][0],label='light1')
#             plt.plot(x_value,avg_reflect_spectrum[k][1],label='light2')
#             plt.plot(x_value,avg_reflect_spectrum[k][2],label='light3')
#
#             plt.legend(loc='upper right')
#
#         plt.show()
#
#         # plt.imshow(img, cmap='gray', interpolation='bicubic')
#         # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
#         # plt.show()
#
#         # plt.figure(1)
#         # ax1 = plt.subplot(141)
#         # plt.sca(ax1)
#         # plt.title('light')
#         # plt.plot(range(light1.shape[2]),
#         #          np.mean(np.mean(light1[rects[0][0]:rects[0][1], rects[0][2]:rects[0][3], :], 0), 0))
#
#         # for i in range(len(rects)):
#         #     plt.figure(i)
#         #     ax1 = plt.subplot(141)
#         #     ax2 = plt.subplot(142)
#         #     ax3 = plt.subplot(143)
#         #     ax3 = plt.subplot(144)
#         #
#         #     plt.sca(ax1)
#         #     plt.title('light')
#         #     plt.plot(range(light1.shape[2]),
#         #              np.mean(np.mean(light1[rects[0][0]:rects[0][1],rects[0][2]:rects[0][3],:],1),2),label='light')
#         #
#         # plt.show()
#
#
#         # rect = (0, 0, 1, 1)
#         # drawing = False
#         # rectangle = False
#         # rect_or_mask = 100
#         # rect_over = False
#         # value = DRAW_FG
#         # img = img2.copy()
#         # mask = np.zeros(img.shape[:2], dtype=np.uint8)  # mask initialized to PR_BG
#         # output = np.zeros(img.shape, np.uint8)  # output image to be shown
#     # elif k == ord('n'):  # segment the image
#     #     print
#     #     """ For finer touchups, mark foreground and background after pressing keys 0-3
#     #     and again press 'n' \n"""
#     #
#     #     showResult = 1
#     #
#     #     # 求出运行时间
#     #     A = cv2.getTickCount()
#     #
#     #     if (rect_or_mask == 0):  # grabcut with rect
#     #         bgdmodel = np.zeros((1, 65), np.float64)
#     #         fgdmodel = np.zeros((1, 65), np.float64)
#     #         cv2.grabCut(img2, mask, rect, bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_RECT)
#     #         rect_or_mask = 1
#     #     elif rect_or_mask == 1:  # grabcut with mask
#     #         bgdmodel = np.zeros((1, 65), np.float64)
#     #         fgdmodel = np.zeros((1, 65), np.float64)
#     #         cv2.grabCut(img2, mask, rect, bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_MASK)
#     #
#     #     B = cv2.getTickCount()
#     #     C = B - A
#     #     time_period = 1 / cv2.getTickFrequency()
#     #     time = time_period * C
#     #
#     #     totalTime = totalTime + time
#     #
#     #     # print
#     #     # 'this process time'
#     #     # print
#     #     # time
#     #     # print
#     #     # 'total time'
#     #     # print
#     #     # totalTime
#     #
#     # if (showResult == 1):
#     #     mask2 = np.where((mask == 1) + (mask == 3), 255, 0).astype('uint8')
#     #
#     #     # print
#     #     # 'mask pixels number'
#     #     print(mask2 > 0).sum()
#     #
#     #     output = cv2.bitwise_and(img2, img2, mask=mask2)
#     #
#     #     showResult = 0
# cv2.destroyAllWindows()

