import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('',0)
img2 = img.copy()
template = cv.imread('',0)
w,h = template.shape[::-1]

methods = ['cv.TM_CCOEFF','cv.TM_CCOEFF_NORMED','cv.TM_CCORR','cv.TM_CCORRNORMED','cv.SQDIFF','cv.SQDIff_NORMED']

for meth in methods:
	img = img2.copy()
	method=eval(meth)

	res = cv.matchTemplate[img,template,method]
	minv, maxv, minl,maxl= cv.minMaxLoc(res)
	
	if method in [cv.TM_SQDIFF, cv.TM_SQDIFFF_NORMED]
		topleft=minl
	else:
		topleft=maxl
	bottomright = (topleft[0]+w,topleft[1] + h)

	cv.rectangle(img, topleft,bottomright,255,2)

	 plt.subplot(121),plt.imshow(res,cmap = 'gray')
	 plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
	 plt.subplot(122),plt.imshow(img,cmap = 'gray')
	 plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
	 plt.suptitle(meth)

	plt.show()
