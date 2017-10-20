import cv2
import stitcher

# def stitch2one(folderName):
#
#     testName = folderName
#     path ='./'+testName
#     filelist = os.listdir(path)
#     pngdir = os.path.join(path, testName + '_pngfile')
#     if not os.path.exists(pngdir):
#         os.mkdir(pngdir)
#
#     for filename in filelist:
#         filepath = os.path.join(path, filename)
#         if os.path.isdir(filepath):
#             pass
#         else:
#             raw = rawpy.imread(filepath)
#             raw_img = raw.raw_image
#             img = np.uint16(raw_img)
#
#             pngName = os.path.join(pngdir, filename + '.png')
#             cv2.imwrite(pngName, img)
#     print('%s is ok'%folderName)


# stitcher = cv2.createStitcher(False)

foo1 = cv2.imread("./stitch/2015_0117_010013_036.JPG")
foo2 = cv2.imread("./stitch/2015_0117_010013_038.JPG")
result = stitcher.stitch((foo1,foo2))

cv2.imshow("stitch_image",result)

if __name__ == '__main__':
    print()