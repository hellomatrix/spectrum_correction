import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

# path='../data/dj_1'
path='../dj_1'
path_pkl = '../dj_1_m/'
path_bac='../dj_1_bac/'
dirlist = os.listdir(path)
dirlist.sort()

files = []
for filename in dirlist:
    file = os.path.join(path,filename)

    if os.path.isdir(file):
        pass
    else:
        files.append(file)

idx_file = 19
img1 = cv2.imread(path + '/' + 'dim%d_8.png' % idx_file)
img_gold = cv2.imread(path + '/' + 'dim10_8.png')
output = img1.copy()


RED = [0,0,255]
GREEN = [0,255,0]
BLUE = [255,0,0]
flag = 100
rectangle = False
rect1 = []
rect2 = []

font = 10

def onmouse(event,x,y,flags,param):
    global img,img_gold,flag,ix,iy,rect1,rect2,rectangle,font

    if flag == 0:
        color = RED
        if event == cv2.EVENT_LBUTTONDOWN:
            rectangle = True
            ix, iy = x, y
            print(ix, iy)

        elif event == cv2.EVENT_MOUSEMOVE:
            if rectangle == True:
                # img = img2.copy()
                # cv2.rectangle(img2,(ix,iy),(x,y),color,font)
                # rect = (ix,iy,abs(ix-x),abs(iy-y))
                print((ix, iy, abs(ix - x), abs(iy - y)))

        elif event == cv2.EVENT_LBUTTONUP:
            rectangle = False
            # rect_over = True
            cv2.rectangle(img_gold, (ix, iy), (x, y), color, font)
            rect1.append([ix,x,iy,y])
            print(rect1)
    elif flag == 1:
        color = BLUE
        if event == cv2.EVENT_LBUTTONDOWN:
            rectangle = True
            ix, iy = x, y
            print(ix, iy)

        elif event == cv2.EVENT_MOUSEMOVE:
            if rectangle == True:
                # img = img2.copy()
                # cv2.rectangle(img2,(ix,iy),(x,y),color,font)
                # rect = (ix,iy,abs(ix-x),abs(iy-y))
                print((ix, iy, abs(ix - x), abs(iy - y)))

        elif event == cv2.EVENT_LBUTTONUP:
            rectangle = False
            # rect_over = True
            cv2.rectangle(img1, (ix, iy), (x, y), color, font)
            rect2.append([ix,x,iy,y])
            print(rect2)


cv2.namedWindow('gold',cv2.WINDOW_NORMAL)
# cv2.moveWindow("gold", 0, 0)
# cv2.setWindowProperty("gold", cv2.WINDOW_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.setMouseCallback('gold',onmouse)
cv2.resizeWindow('gold',700,900)

# cv2.namedWindow('1',cv2.WINDOW_OPENGL)
# cv2.namedWindow('1',cv2.WINDOW_GUI_NORMAL)

cv2.namedWindow('other',cv2.WINDOW_NORMAL)
# cv2.moveWindow("other", 0, 0)
# cv2.setWindowProperty("other", cv2.WINDOW_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
cv2.resizeWindow('other',700,900)
cv2.setMouseCallback('other',onmouse)

cv2.namedWindow('output',cv2.WINDOW_NORMAL)
cv2.resizeWindow('output',700,900)
# cv2.setMouseCallback('other',onmouse)

jj = 0
while(True):
    cv2.imshow('gold',img_gold)
    cv2.imshow('other',img1)
    cv2.imshow('output', output)

    k= 0xFF & cv2.waitKey(1)

    if k == 27:
        break
    elif k == ord('g'):
        print('Mark gold area')
        flag = 0
    elif k == ord('o'):
        print('Mark other area')
        flag = 1

    elif k == ord('n'):
        print('Change image')

        if idx_file != len(files)-1:
            idx_file = idx_file + 1

            if idx_file == len(files)//2:
                idx_file = idx_file + 1

            img1 = cv2.imread(files[idx_file])
            output = img1.copy()
        else:
            print('The last one image')

    elif k == ord('c'):
        print('Cal the transeform matrix')
        flag = 9
        pos_gold=[]
        pos_other=[]

        img_c= img1.copy()
        for i in range(len(rect2)):

            img_big = img_c[rect2[i][2]:rect2[i][3],rect2[i][0]:rect2[i][1],1]
            img_template = img_gold[rect1[i][2]:rect1[i][3],rect1[i][0]:rect1[i][1],1]

            res = cv2.matchTemplate(img_big,img_template,cv2.TM_CCORR_NORMED)
            min_val,max_val,min_loc,max_loc = cv2.minMaxLoc(res)

            # cv2.rectangle(img2,top_le)
            print(min_loc,max_loc)

            top_left = (rect2[i][0]+max_loc[0],rect2[i][2]+max_loc[1])
            bot_right = (top_left[0]+abs(rect1[i][0]-rect1[i][1]),top_left[1]+abs(rect1[i][2]-rect1[i][3]))

            cv2.rectangle(img1,top_left,bot_right,GREEN,10)

            pos_gold.append([rect1[i][0],rect1[i][2]])
            pos_other.append([top_left[0],top_left[1]])

            fig = plt.figure()
            plt.subplot(121),plt.imshow(img_big,cmap='gray')
            plt.subplot(122),plt.imshow(img_template, cmap='gray')

            #
            # gold = [[2761, 722], [8732, 572], [8713, 2355]]
            #
            # other = [[4913, 860], [10816, 658], [10781, 2313]]

        M = cv2.getAffineTransform(np.float32(np.array(pos_other)), np.float32(np.array(pos_gold)))
        pickle.dump(M, open(path_pkl + 'm_%d.pkl' % idx_file, 'wb'))
        output = cv2.warpAffine(output,M,(output.shape[1], output.shape[0]))
        cv2.imwrite(path_bac +'%d_Affine.png' % idx_file, output)


    plt.show()


    # cv2.waitKey(0)

cv2.destroyAllWindows()
