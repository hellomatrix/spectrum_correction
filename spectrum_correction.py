
import matplotlib
matplotlib.use('TkAgg')

import scipy.io as sio
import cv2
import numpy as np

import matplotlib.pyplot as plt

# import Image
# import matplotlib.widgets as widgets
#
# from matplotlib.figure import Figure
# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# path = '../spectrum_correction/data/0920_01_3'
# data = sio.loadmat(path)
# sio.savemat('light3.mat',{'light3':data['0920_01_3']})
#
# path = '../spectrum_correction/data/0920_01_2'
# data = sio.loadmat(path)
# sio.savemat('light2.mat',{'light2':data['0920_01_2']})
#
# path = '../spectrum_correction/data/0920_01_1'
# data = sio.loadmat(path)
# sio.savemat('light1.mat',{'light1':data['0920_01_1']})

light1 = sio.loadmat('light1/light1.mat')['light1']
light2 = sio.loadmat('light2/light2.mat')['light2']
light3 = sio.loadmat('light3/light3.mat')['light3']

data=[light1,light2,light3]

BLUE = [255, 0, 0]  # rectangle color
RED = [0, 0, 255]  # PR BG
GREEN = [0, 255, 0]  # PR FG
BLACK = [0, 0, 0]  # sure BG
WHITE = [255, 255, 255]  # sure FG

DRAW_BG = {'color': BLACK, 'val': 0}
DRAW_FG = {'color': WHITE, 'val': 1}
DRAW_PR_FG = {'color': GREEN, 'val': 3}
DRAW_PR_BG = {'color': RED, 'val': 2}

# setting up flags
rect = (0, 0, 1, 1)
drawing = False  # flag for drawing curves
rectangle = False  # flag for drawing rect
rect_over = False  # flag to check if rect drawn
rect_or_mask = 100  # flag for selecting rect or mask mode
value = DRAW_FG  # drawing initialized to FG
thickness = 13  # brush thickness

light = True

totalTime = 0
showResult = 0

substance = True
rects = list()
avg_spectrum=list()
point_spectrum=list()
avg_reflect_spectrum=[]
point_reflect_spectrum=[]
draw = False
i = 0

def onmouse(event, x, y, flags, param):
    global img, img2, drawing, value, mask, rectangle, rect, rect_or_mask,\
        ix, iy, rect_over,light,substance,draw,rects,i,x_range,value,avg_spectrum,\
        point_spectrum,avg_reflect_spectrum,point_reflect_spectrum

    # Draw Rectangle for light
    if event == cv2.EVENT_LBUTTONDOWN:
        if light == True:
            ix, iy = x, y
            print('background')
        else:
            ix, iy = x, y
            print('class')
        draw = True

    elif event == cv2.EVENT_MOUSEMOVE and draw == True:
        img = img2.copy()
        if light == True:
            cv2.rectangle(img, (ix, iy), (x, y), WHITE, 2)
            rect = (ix, iy, abs(ix - x), abs(iy - y))
            print('move')
        elif substance==True :
            cv2.rectangle(img, (ix, iy), (x, y), RED, 2)
            rect = (ix, iy, abs(ix - x), abs(iy - y))
            print('move')

    elif event == cv2.EVENT_RBUTTONDOWN:
        if light == True:
            cv2.rectangle(img, (ix, iy), (x, y), WHITE, 2)
            rect = (ix, iy, abs(ix - x), abs(iy - y))
            light = False
            print('up')
            cv2.putText(img, 'light', (x+5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE)
        else:
            cv2.rectangle(img, (ix, iy), (x, y), RED, 2)
            rect = (ix, iy, abs(ix - x), abs(iy - y))
            light = False
            print('up')
            cv2.putText(img, 'class_'+str(i), (x+5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED)

            # relectance
            for j in range(3):
                avg_y_value = np.mean(np.mean(data[j][iy:y, ix:x, :], 0), 0)
                point_value = data[j][y, x, :]
                #
                # avg_temp.append(avg_y_value)
                # point_temp.append(point_value)

                avg_reflect_spectrum[i-1].append(avg_y_value/avg_spectrum[0][j])
                point_reflect_spectrum[i-1].append(point_value/point_spectrum[0][j])

        print('ok')
        print(iy,y,ix,x)
        avg_spectrum.append([])
        point_spectrum.append([])
        avg_reflect_spectrum.append([])
        point_reflect_spectrum.append([])

        for j in range(3):
            avg_y_value = np.mean(np.mean(data[j][iy:y,ix:x,:], 0), 0)
            point_value = data[j][y,x,:]
            #
            # avg_temp.append(avg_y_value)
            # point_temp.append(point_value)

            avg_spectrum[i].append(avg_y_value)
            point_spectrum[i].append(point_value)

        i=i+1
        # plt.figure()
        # plt.plot(x_value,y_value)
        # plt.show()

        rects.append([iy,y,ix,x])
        img2=img.copy()
        draw = False

        # plt.plot(x_value, y_value)
        # plt.show()

        # plt.figure()
        # ax1 = plt.subplot(111)
        # plt.sca(ax1)
        # plt.title('light')

img = light1[:,:,1]
img2 = img.copy()  # a copy of original image
# mask = np.zeros(img.shape[:2], dtype=np.uint8)  # mask initialized to PR_BG
# mask2 = np.zeros(img.shape[:2], dtype=np.uint8)

output = np.zeros(img.shape, np.uint8)  # output image to be shown

# input and output windows
cv2.namedWindow('output', cv2.WINDOW_NORMAL)
cv2.resizeWindow('output', 700, 500)

cv2.namedWindow('input', cv2.WINDOW_NORMAL)
cv2.resizeWindow('input', 700, 500)
cv2.setMouseCallback('input', onmouse)

while (True):
    cv2.imshow('output', output)
    cv2.imshow('input',img)

    k = 0xFF & cv2.waitKey(1)
    # key bindings
    if k == 27:  # esc to exit
        break
    # elif k == ord('0'):  # BG drawing
    #     print(" mark background regions with right mouse button \n")
    #     value = DRAW_BG
    # elif k == ord('1'):  # FG drawing
    #     # print
    #     # " mark foreground regions with left mouse button \n"
    #     value = DRAW_FG
    # elif k == ord('2'):  # PR_BG drawing
    #     value = DRAW_PR_BG
    # elif k == ord('3'):  # PR_FG drawing
    #     value = DRAW_PR_FG
    # elif k == ord('s'):  # save image
    #     bar = np.zeros((img.shape[0], 5, 3), np.uint8)
    #     res = np.hstack((img2, bar, img, bar, output))
        # cv2.imwrite('grabcut_output.png',res)

        # cv2.imwrite(maskName, mask2)
        # cv2.imwrite(operation, img)
        # print
        # " Result saved as image \n"
    elif k == ord('r'):  # reset everything
        # print
        # "resetting \n"

        totalTime = 0
        showResult = 0
        print('ok')
        print(rects[0][0],rects[0][1],rects[0][2],rects[0][3])
        np.mean(np.mean(light1[rects[0][0]:rects[0][1],rects[0][2]:rects[0][3],:], 0), 0)

        x_value = range(light1.shape[2])
        # y_value = np.mean(np.mean(light1[rects[0][0]:rects[0][1],rects[0][2]:rects[0][3],:], 0), 0)


        for k in range(len(avg_reflect_spectrum)-1):

            plt.figure(k+100)
            plt.plot(x_value,avg_reflect_spectrum[k][0],label='light1')
            plt.plot(x_value,avg_reflect_spectrum[k][1],label='light2')
            plt.plot(x_value,avg_reflect_spectrum[k][2],label='light3')

            plt.legend(loc='upper right')

        plt.show()

        # plt.imshow(img, cmap='gray', interpolation='bicubic')
        # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        # plt.show()




        # plt.figure(1)
        # ax1 = plt.subplot(141)
        # plt.sca(ax1)
        # plt.title('light')
        # plt.plot(range(light1.shape[2]),
        #          np.mean(np.mean(light1[rects[0][0]:rects[0][1], rects[0][2]:rects[0][3], :], 0), 0))

        # for i in range(len(rects)):
        #     plt.figure(i)
        #     ax1 = plt.subplot(141)
        #     ax2 = plt.subplot(142)
        #     ax3 = plt.subplot(143)
        #     ax3 = plt.subplot(144)
        #
        #     plt.sca(ax1)
        #     plt.title('light')
        #     plt.plot(range(light1.shape[2]),
        #              np.mean(np.mean(light1[rects[0][0]:rects[0][1],rects[0][2]:rects[0][3],:],1),2),label='light')
        #
        # plt.show()


        # rect = (0, 0, 1, 1)
        # drawing = False
        # rectangle = False
        # rect_or_mask = 100
        # rect_over = False
        # value = DRAW_FG
        # img = img2.copy()
        # mask = np.zeros(img.shape[:2], dtype=np.uint8)  # mask initialized to PR_BG
        # output = np.zeros(img.shape, np.uint8)  # output image to be shown
    # elif k == ord('n'):  # segment the image
    #     print
    #     """ For finer touchups, mark foreground and background after pressing keys 0-3
    #     and again press 'n' \n"""
    #
    #     showResult = 1
    #
    #     # 求出运行时间
    #     A = cv2.getTickCount()
    #
    #     if (rect_or_mask == 0):  # grabcut with rect
    #         bgdmodel = np.zeros((1, 65), np.float64)
    #         fgdmodel = np.zeros((1, 65), np.float64)
    #         cv2.grabCut(img2, mask, rect, bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_RECT)
    #         rect_or_mask = 1
    #     elif rect_or_mask == 1:  # grabcut with mask
    #         bgdmodel = np.zeros((1, 65), np.float64)
    #         fgdmodel = np.zeros((1, 65), np.float64)
    #         cv2.grabCut(img2, mask, rect, bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_MASK)
    #
    #     B = cv2.getTickCount()
    #     C = B - A
    #     time_period = 1 / cv2.getTickFrequency()
    #     time = time_period * C
    #
    #     totalTime = totalTime + time
    #
    #     # print
    #     # 'this process time'
    #     # print
    #     # time
    #     # print
    #     # 'total time'
    #     # print
    #     # totalTime
    #
    # if (showResult == 1):
    #     mask2 = np.where((mask == 1) + (mask == 3), 255, 0).astype('uint8')
    #
    #     # print
    #     # 'mask pixels number'
    #     print(mask2 > 0).sum()
    #
    #     output = cv2.bitwise_and(img2, img2, mask=mask2)
    #
    #     showResult = 0
cv2.destroyAllWindows()