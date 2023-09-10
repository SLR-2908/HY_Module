'''
#Histogram Code 9B :


import cv2

from matplotlib import pyplot as plt

img = cv2.imread('C:/Users/yuvasri/Desktop/S.Y/Resources/Photos/cats 2.jpg',0)
 
histr = cv2.calcHist([img],[0],None,[256],[0,256])
  
plt.plot(histr)
plt.show()
'''

'''
#reading an image Code 9B :
import cv2 as cv
img = cv.imread('Photos/cat.jpg')
cv.imshow("cats",img)
cv.waitKey(0)
'''

'''
#reading an video Code 9B:
import cv2 as cv
capture =cv.VideoCapture('C:/Users/yuvasri/Desktop/S.Y/Resources/Videos/kitten.mp4')

while True :
    isTrue ,frame=capture.read()
    cv.imshow("video",frame)
    if cv.waitKey(20) & 0xFF == ord('d'):
        break
capture.release()
cv.destroyAllWindows()

'''
'''
'''
'''
#image reflection Code 9B:
import numpy as np
import cv2 as cv
img = cv.imread('C:/Users/yuvasri/Desktop/S.Y/Resources/Photos/cat.jpg',0)
rows , cols = img.shape
M = np.float32([[1,  0, 0],
                [0, -1, rows],
                [0,  0, 1]])
reflected_img = cv.warpPerspective(img, M,
                                   (int(cols),
                                    int(rows)))
cv.imshow('img', reflected_img)
cv.imwrite('reflection_out.jpg', reflected_img)
cv.waitKey(0)
cv.destroyAllWindows()
'''
'''
#image scaling Code 9B:
import numpy as np
import cv2 as cv 
img = cv.imread('C:/Usersyuvasri/Desktop/S.Y/Resources/Photos/park.jpg')
rows,cols=img.shape
img_shrinked=cv.resize(img,(250,200),interpolation=cv.INTER_AREA)
cv.imshow("img",img_shrinked)
img.enlarge=cv.resize(img_shrinked,None,fx=1.5,fy=1.5,interpolation=cv.INTER_CUBIC)
cv.imshow("img",img.enlarge)
cv.waitKey(0)
cv.destroyAllWindows()
'''
'''
'''
# colour spaces Code 9B:
''''
import cv2 
img = cv2.imread('C:/Users/yuvasri/Desktop/S.Y/Resources/Photos/cat.jpg')
B,G,R = cv2.split(img)
cv2.imshow("original",img)

cv2.waitKey(0)
cv2.imshow('blue',B)
cv2.waitKey(0)
cv2.imshow('red',R)
cv2.waitKey(0)
cv2.imshow('green',G)

cv2.waitKey(0)

cv2.destroyAllWindows()
'''
#Method 2 of Blurring Code :
#average blurring Code 9B :
'''
import numpy as np
import cv2  
img = cv2.imread('C:/Users/yuvasri/Desktop/S.Y/Resources/Photos/group 1.jpg')
averageBlur= cv2.blur(img,(5,5))
cv2.imshow('original',img)
cv2.imshow("average blurring",averageBlur)
cv2.waitKey(0)

cv2.destroyAllWindows()

'''
'''
#gaussian blurring Code 9B:
import numpy as np
import cv2  
img = cv2.imread("Photos/cat.jpg")
gaussianBlur= cv2.GaussianBlur(img,(3,3),0)
cv2.imshow('original',img)
cv2.imshow("gausian blur",gaussianBlur)
cv2.waitKey(0)

cv2.destroyAllWindows()

#resizeing image Code 9B:

import cv2 as cv
img = cv.imread("Photos/cat.jpg")
cv.imshow('cat',img)
def rescaleframe(frame,scale=0.5)
    width=int(frame.shape[1]*scale)
    height=int(frame.shape[0]*scale)
    dimensions=(width,height)
    return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)
resized_img=rescaleframe(img)
cv.imshow('image',resized_img)
cv.waitKey(0)

#resizeing video Code 9B:
import cv2 as cv
capture = cv.videocapture("Videos/kitten.mp4")

while True :
    is True , frame = capture.read()
    frame_resized=rescaleframe(frame)

    cv.imshow("video",frame)
    cv.imshow("video_resized",frame_resized)
    if cv.waitKey(20)& 0*FF ==ord("d"):
        break
capture.release()
cv.destroyAllWindows()

#Blurring method 1 Code 9B:
import cv2
import numpy as np

image = cv2.imread('Photos/cat.jpg')

kernel2 = np.ones((5, 5), np.float32)/25

img = cv2.filter2D(src=image, ddepth=-1, kernel=kernel2)

cv2.imshow('Original', image)
cv2.imshow('Kernel Blur', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
'''
# mediam Blurring Code 9B:
import cv2
import numpy as np

image = cv2.imread('C:/Users/yuvasri/Desktop/S.Y/Resources/Photos/lady.jpg')

medianBlur =cv2.medianBlur(image, 13)

cv2.imshow('Original', image)
cv2.imshow('Median blur',medianBlur)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
'''
# bilateral blurring Code 9B:
import cv2
import numpy as np

image = cv2.imread('Photos/cat.jpg')

bilateral = cv2.bilateralFilter(image,9, 65, 75)

cv2.imshow('Original', image)
cv2.imshow('Bilateral blur', bilateral)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Bitwise opperations:
#Bitwise opperations_and Code 9B:
'''
import cv2
import numpy as np

img1 = cv2.imread('C:/Users/yuvasri/Desktop/S.Y/Resources/Photos/cats.jpg')
img2 = cv2.imread('C:/Users/yuvasri/Desktop/S.Y/Resources/Photos/cats.jpg')

dest_and = cv2.bitwise_and(img2, img1, mask = None)

cv2.imshow('Bitwise And', dest_and)

if cv2.waitKey(0) & 0xff == 27:
cv2.destroyAllWindows()
'''
# #Bitwise opperations_or Code 9B:

import cv2
import numpy as np

img1 = cv2.imread('Photos/cat.jpg')
img2 = cv2.imread('Photos/cat.jpg')

dest_or = cv2.bitwise_or(img2, img1, mask = None)

cv2.imshow('Bitwise OR', dest_or)

if cv2.waitKey(0) & 0xff == 27:
cv2.destroyAllWindows()

 #Bitwise opperations_xor Code 9B:

import cv2
import numpy as np

img1 = cv2.imread('Photos/cat.jpg')
img2 = cv2.imread('Photos/cat.jpg')

dest_xor = cv2.bitwise_xor(img1, img2, mask = None)

cv2.imshow('Bitwise XOR', dest_xor)

if cv2.waitKey(0) & 0xff == 27:
cv2.destroyAllWindows()

#Bitwise opperations_not Code 9B:

import cv2
import numpy as np

img1 = cv2.imread('C:/Users/yuvasri/Desktop/S.Y/Resources/Photos/cats.jpg',0)
img2 = cv2.imread('C:/Users/yuvasri/Desktop/S.Y/Resources/Photos/cats.jpg',0)

dest_not1 = cv2.bitwise_not(img1, mask = None)
dest_not2 = cv2.bitwise_not(img2, mask = None)

cv2.imshow('Bitwise NOT on image 1', dest_not1)
cv2.imshow('Bitwise NOT on image 2', dest_not2)

if cv2.waitKey(0) & 0xff == 27:
cv2.destroyAllWindows()
'''
#alpha blending Code 9B:
import cv2
img1 = cv2.imread('C:/Users/yuvasri/Desktop/S.Y/Resources/Photos/cats.jpg',0)
cv.imshow('Original image', img)
img2 = cv2.imread('Photos/cat.jpg')
img2 = cv2.resize(img2, img1.shape[1::-1])
cv2.imshow("img 1",img1)
cv2.waitKey(0)
cv2.imshow("img 2",img2)
cv2.waitKey(0)
choice = 1

while (choice) :alpha= float(input("Enter alpha value"))
dst = cv2.addWeighted(img1, alpha , img2,
1-alpha, 0)
cv2.imwrite('alpha_mask_.png', dst)
img3 = cv2.imread('alpha_mask_.png')
cv2.imshow("alpha blending 1",img3)
cv2.waitKey(0)
choice = int(input("Enter 1 to continue and 0 toexit"))
'''
#masking Code 9B:
import cv2 as cv
import numpy as np
img = cv.imread('C:/Users/yuvasri/Desktop/S.Y/Resources/Photos/cats.jpg',0)
cv.imshow('Original image', img)
blank = np.zeros(img.shape[:2], dtype='uint8')
cv.imshow('Blank Image', blank)
circle = cv.circle(blank,
(img.shape[1]//2,img.shape[0]//2),200,255, -1)
cv.imshow('Mask',circle)
masked = cv.bitwise_and(img,img,mask=circle)
cv.imshow('Masked Image', masked)
cv.waitKey(0)


#image translation Code 9B:
import numpy as np
import cv2 as cv
img = cv.imread('C:/Users/yuvasri/Desktop/S.Y/Resources/Photos/cats.jpg', 0)
rows, cols = img.shape
M = np.float32([[1, 0, 100], [0, 1, 50]])
dst = cv.warpAffine(img, M, (cols, rows))
cv.imshow('img', dst)
cv.waitKey(0)
cv.destroyAllWindows()
'''
'''
#image cropping Code 9B:


import numpy as np
import cv2 as cv
img = cv.imread('C:/Users/yuvasri/Desktop/S.Y/Resources/Photos/group 2.jpg', 0)
cropped_img = img[100:300, 100:300]
cv.imwrite('cropped_out.jpg', cropped_img)
cv.waitKey(0)
cv.destroyAllWindows()


#image rotation Code 9B:
import numpy as np
import cv2 as cv
img = cv.imread('C:/Users/yuvasri/Desktop/S.Y/Resources/Photos/group 2.jpg',0)
rows, cols = img.shape
M = np.float32([[1,  0, 0], [0, -1, rows], [0,  0, 1]])
img_rotation = cv.warpAffine(img,
                             cv.getRotationMatrix2D((cols/2, rows/2),
                                                    30, 0.6),
                             (cols, rows))
cv.imshow('img', img_rotation)
cv.imwrite('rotation_out.jpg', img_rotation)
cv.waitKey(0)
cv.destroyAllWindows()

'''
'''
'''
'''
#shearing on x - axis Code 9B:
import numpy as np
import cv2 as cv
img = cv.imread('C:/Users/yuvasri/Desktop/S.Y/Resources/Photos/group 2.jpg',0)
rows, cols = img.shape
M = np.float32([[1, 0.5, 0], [0, 1, 0], [0, 0, 1]])
sheared_img = cv.warpPerspective(img, M, (int(cols*1.5), int(rows*1.5)))
cv.imshow('img', sheared_img)
cv.waitKey(0)
cv.destroyAllWindows()

'''
''''
#shearing on y - axis Code 9B:
import numpy as np
import cv2 as cv
img = cv.imread('C:/Users/yuvasri/Desktop/S.Y/Resources/Photos/park.jpg',0)
rows, cols = img.shape
M = np.float32([[1,   0, 0], [0.5, 1, 0], [0,   0, 1]])
sheared_img = cv.warpPerspective(img, M, (int(cols*1.5), int(rows*1.5)))
cv.imshow('sheared_y-axis_out.jpg', sheared_img)
cv.waitKey(0)
cv.destroyAllWindows()
'''
'''
'''
#drawing shape Code 9B:
'''
import cv2 as cv
import numpy as np


blank=np.zeros((500,500,3),dtype='unit8')
cv.imshow("blank image",blank)
blank[:]=0,255,0
cv.waitKey(0)


#drawing a rectangle:
import numpy as np
import cv2
img = np.zeros((400, 400, 3), dtype = "uint8")
cv2.rectangle(img, (30, 30), (300, 200), (0, 255, 0), 5)
cv2.imshow('dark', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
'''
#drawing a circle:
import numpy as np
import cv2
img = np.zeros((400, 400, 3), dtype = "uint8")
  

cv2.circle(img, (200, 200), 80, (255, 0, 0), 3)
  
cv2.imshow('dark', img)
  

cv2.waitKey(0)
cv2.destroyAllWindows()

import numpy as np
import cv2
  

img = np.zeros((400, 400, 3), dtype = "uint8")
  

cv2.line(img, (20, 160), (100, 160), (0, 0, 255), 10)
  
cv2.imshow('dark', img)
  


cv2.waitKey(0)
cv2.destroyAllWindows()

'''
#putting a text in a shape in 9B
'''
import numpy as np
import cv2
img = np.zeros((400, 400, 3), dtype = "uint8")
 
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, 'CS', (50, 50),
            font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
  
cv2.imshow('dark', img)
  
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
'''
#Countours detection Code 9B:
import cv2
import numpy as np
  

image = cv2.imread('C:/Users/yuvasri/Desktop/S.Y/Resources/Photos/park.jpg')
cv2.waitKey(0)
  

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  

edged = cv2.Canny(gray, 30, 200)
cv2.waitKey(0)
  

contours, hierarchy = cv2.findContours(edged, 
    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
  
cv2.imshow('Canny Edges After Contouring', edged)
cv2.waitKey(0)

print("Number of Contours found = " + str(len(contours)))

cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
  
cv2.imshow('Contours', image)
cv2.waitKey(0)
cv2.destroyAllWindows()



# THANK YOU
'''









        