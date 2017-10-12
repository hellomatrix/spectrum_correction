# -*- coding: utf-8 -*-

# ===============================================================================
#
#  svm with the original opencv
#  features: b, g, r, distance, x, y
#  distance = (point1,point2)间的欧式距离 * 255/(point1,point2)间的对角线长度
#  big data success
#  09/04/2015
#  excellent
#
#
#  系统问题，就是跑不起来，图像太大，所以占内存比较多.
#  图片进行切割
#  20/04/2015
#  Final version1.0
# ===============================================================================


'''
@author: Daniel(hua)
===============================================================================

Key '0' - To select areas of sure background
Key '1' - To select areas of sure foreground
Key '2' - To select areas of probable background
Key '3' - To select areas of probable foreground

Key 'n' - To update the segmentation
Key 'r' - To reset the setup
Key 's' - To save the results
===============================================================================
'''



def onselect(eclick, erelease):
    if eclick.ydata>erelease.ydata:
        eclick.ydata,erelease.ydata=erelease.ydata,eclick.ydata
    if eclick.xdata>erelease.xdata:
        eclick.xdata,erelease.xdata=erelease.xdata,eclick.xdata
    ax2.set_ylim(erelease.ydata,eclick.ydata)
    ax2.set_xlim(eclick.xdata,erelease.xdata)
    fig2.canvas.draw()

arr = np.asarray(light1['light1'][:,:,1])

# fig= Figure(1)
# canvas = FigureCanvas(fig)
# ax1 = fig.add_axes()

fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111)
plt.imshow(arr,cmap='gray')

fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111)
plt_image=plt.imshow(arr,cmap='gray')

rs=widgets.RectangleSelector(
    ax1, onselect, drawtype='box',
    rectprops = dict(facecolor='red', edgecolor = 'black', alpha=0.5, fill=True))

fig3 = plt.figure(3)
ax=fig3.add_subplot(111)

plt.plot()

plt.show()



import numpy as np
import cv2
import sys

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

totalTime = 0
showResult = 0


def onmouse(event, x, y, flags, param):
    global img, img2, drawing, value, mask, rectangle, rect, rect_or_mask, ix, iy, rect_over

    # Draw Rectangle
    if event == cv2.EVENT_RBUTTONDOWN:
        rectangle = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if rectangle == True:
            img = img2.copy()
            cv2.rectangle(img, (ix, iy), (x, y), BLUE, 12)
            rect = (ix, iy, abs(ix - x), abs(iy - y))
            rect_or_mask = 0

    elif event == cv2.EVENT_RBUTTONUP:
        rectangle = False
        rect_over = True
        cv2.rectangle(img, (ix, iy), (x, y), BLUE, 12)
        rect = (ix, iy, abs(ix - x), abs(iy - y))
        rect_or_mask = 0
        print
        " Now press the key 'n' a few times until no further change \n"

    # draw touchup curves

    if event == cv2.EVENT_LBUTTONDOWN:
        if rect_over == False:
            print
            "first draw rectangle \n"
        else:
            drawing = True
            cv2.circle(img, (x, y), thickness, value['color'], -1)
            cv2.circle(mask, (x, y), thickness, value['val'], -1)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.circle(img, (x, y), thickness, value['color'], -1)
            cv2.circle(mask, (x, y), thickness, value['val'], -1)

    elif event == cv2.EVENT_LBUTTONUP:
        if drawing == True:
            drawing = False
            cv2.circle(img, (x, y), thickness, value['color'], -1)
            cv2.circle(mask, (x, y), thickness, value['val'], -1)


# print documentation
print
__doc__

# Loading images
if len(sys.argv) == 2:
    filename = sys.argv[1]  # for drawing purposes
else:
    print
    "No input image given, so loading default image, lena.jpg \n"
    print
    "Correct Usage : python grabcut.py <filename> \n"

# ===========================================================================
# filename = "L://visualstudiospace//big image//1 Arterial dit peu_small.png"
# maskName = "L://visualstudiospace//big image//result//1 Arterial dit peu//1 Arterial dit peu_grabcut.png"
# operation ="L://visualstudiospace//big image//result//1 Arterial dit peu//1 Arterial dit peu_grabcut_operation.png"
# ===========================================================================


img = cv2.imread(filename)
img2 = img.copy()  # a copy of original image
mask = np.zeros(img.shape[:2], dtype=np.uint8)  # mask initialized to PR_BG
mask2 = np.zeros(img.shape[:2], dtype=np.uint8)
output = np.zeros(img.shape, np.uint8)  # output image to be shown

# input and output windows
cv2.namedWindow('output', cv2.WINDOW_NORMAL)
cv2.resizeWindow('output', 700, 500)

cv2.namedWindow('input', cv2.WINDOW_NORMAL)
cv2.resizeWindow('input', 700, 500)

cv2.setMouseCallback('input', onmouse)

print
" Instructions : \n"
print
" Draw a rectangle around the object using right mouse button \n"

while (1):

    cv2.imshow('output', output)
    cv2.imshow('input', img)
    k = 0xFF & cv2.waitKey(1)

    # key bindings
    if k == 27:  # esc to exit
        break
    elif k == ord('0'):  # BG drawing
        print
        " mark background regions with left mouse button \n"
        value = DRAW_BG
    elif k == ord('1'):  # FG drawing
        print
        " mark foreground regions with left mouse button \n"
        value = DRAW_FG
    elif k == ord('2'):  # PR_BG drawing
        value = DRAW_PR_BG
    elif k == ord('3'):  # PR_FG drawing
        value = DRAW_PR_FG
    elif k == ord('s'):  # save image
        bar = np.zeros((img.shape[0], 5, 3), np.uint8)
        res = np.hstack((img2, bar, img, bar, output))
        # cv2.imwrite('grabcut_output.png',res)

        cv2.imwrite(maskName, mask2)
        cv2.imwrite(operation, img)
        print
        " Result saved as image \n"
    elif k == ord('r'):  # reset everything
        print
        "resetting \n"

        totalTime = 0
        showResult = 0

        rect = (0, 0, 1, 1)
        drawing = False
        rectangle = False
        rect_or_mask = 100
        rect_over = False
        value = DRAW_FG
        img = img2.copy()
        mask = np.zeros(img.shape[:2], dtype=np.uint8)  # mask initialized to PR_BG
        output = np.zeros(img.shape, np.uint8)  # output image to be shown
    elif k == ord('n'):  # segment the image
        print
        """ For finer touchups, mark foreground and background after pressing keys 0-3
        and again press 'n' \n"""

        showResult = 1

        # 求出运行时间
        A = cv2.getTickCount()

        if (rect_or_mask == 0):  # grabcut with rect
            bgdmodel = np.zeros((1, 65), np.float64)
            fgdmodel = np.zeros((1, 65), np.float64)
            cv2.grabCut(img2, mask, rect, bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_RECT)
            rect_or_mask = 1
        elif rect_or_mask == 1:  # grabcut with mask
            bgdmodel = np.zeros((1, 65), np.float64)
            fgdmodel = np.zeros((1, 65), np.float64)
            cv2.grabCut(img2, mask, rect, bgdmodel, fgdmodel, 1, cv2.GC_INIT_WITH_MASK)

        B = cv2.getTickCount()
        C = B - A
        time_period = 1 / cv2.getTickFrequency()
        time = time_period * C

        totalTime = totalTime + time

        print
        'this process time'
        print
        time
        print
        'total time'
        print
        totalTime

    if (showResult == 1):
        mask2 = np.where((mask == 1) + (mask == 3), 255, 0).astype('uint8')

        print
        'mask pixels number'
        print(mask2 > 0).sum()

        output = cv2.bitwise_and(img2, img2, mask=mask2)

        showResult = 0

cv2.destroyAllWindows()
