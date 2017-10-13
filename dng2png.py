import rawpy
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import sys
import getopt

def dng2png(folderName):

    testName = folderName
    path ='./'+testName
    filelist = os.listdir(path)
    pngdir = os.path.join(path, testName + '_pngfile')
    if not os.path.exists(pngdir):
        os.mkdir(pngdir)

    for filename in filelist:
        filepath = os.path.join(path, filename)
        if os.path.isdir(filepath):
            pass
        else:
            raw = rawpy.imread(filepath)
            raw_img = raw.raw_image
            img = np.uint16(raw_img)

            pngName = os.path.join(pngdir, filename + '.png')
            cv2.imwrite(pngName, img)
    print('%s is ok'%folderName)

def show(dng_name):
    raw = rawpy.imread(dng_name)
    raw_img = raw.raw_image

    fig = plt.figure()
    plt.imshow(raw_img,cmap='gray')
    plt.show()

if __name__=="__main__":

    if sys.argv[1] == '-show':
        show(sys.argv[2])
    else:
        for i in range(len(sys.argv)-1):
            dng2png(sys.argv[i+1])