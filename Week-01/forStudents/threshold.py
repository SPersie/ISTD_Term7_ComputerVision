img = cv2.imread('imgs/sudoku-original.jpg',0)
img = cv2.medianBlur(img,5)
img_mean = np.mean(img)               # find mean of image
img_median = np.median(img)           # find median of image
img_thres = 40

ret,th1 = cv2.threshold(img, img_thres, 255, cv2.THRESH_BINARY)
#############################################################################
# TODO:                                                                     #
# Trying several values of constant 'img_thres', for example: 40, 60, 80,   #
# 100, 120, 140, 160, img_mean, img_median; and observing how the output    #
# thresholded images change. Report the value you think provide the best    #
# output image.                                                             #
#############################################################################

C = 2
th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,C)
th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,C)

#############################################################################
# TODO:                                                                     #
# Trying several values of constant C1, C2, for example: 0, 1, 2, 3, 4, 5,  #
# 10, 20, 50 100; and observing how the output thresholded images change.   #
# Report the value you think provide the best output image.                 #
#############################################################################

titles = ['Original Image', 'Global Thresholding (v = {:.2f})'.format(img_thres),
          'Adaptive Mean Thresholding (C = {:.2f})'.format(C),
          'Adaptive Gaussian Thresholding (C = {:.2f})'.format(C)]
images = [img, th1, th2, th3]

fig = plt.figure(figsize=(10, 10))
for i in xrange(4):
    plt.subplot(2,2,i+1)
    plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])
plt.show()

#############################################################################
# QUESTION:                                                                 #
# In comparison between cv2.threshold and cv2.adaptiveThreshold, how        #
# sensitive is the output images to changes of the hyper-parameters:        #
# 'img_thres' and 'C', respectively?                                        #
#############################################################################

#############################################################################
# QUESTION:                                                                 #
# As you have observed, the (adaptive) threshold function requires tuning   #
# hyper-parameter to achieve good outputs. Instead, Otsu binarization       #
# algorithm automatically calculates a threshold value. A tutorial about    #
# Otsu algorithm is at this link:                                           #
# https://docs.opencv.org/3.3.1/d7/d4d/tutorial_py_thresholding.html        #
#                                                                           # 
# What is the main, general idea of the Otsu binarization algorithm?        #
#############################################################################