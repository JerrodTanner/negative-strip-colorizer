'''
CSC 372 Computer Vision
Homework 1: Restoring the Russian Empire
This is the starter code. Replace this with your better comments.
Author: Jerrod Tanner
Date: 20 February 2020
'''
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys

xTranslation = 0
yTranslation = 0


def translate(I, x, y):
    ''' Translate the given image by the given offset in the x and y directions.
    '''
    rows, cols = I.shape[:2]
    
    M = np.array([[1, 0, x],
                  [0, 1, -y]], np.float)
    img_translated = cv2.warpAffine(I, M, (cols, rows))
    
    return img_translated
    

def compute_ssd(I1, I2):
    ''' Compute the sum-of-squared differences between two images.

    Find the difference between the images (subtract), square the difference,
    and then sum up the squared differences over all the pixels. This should
    require no explicit loops. Potentially helpful method: np.sum().

    Think carefully about math and data types.
    '''
    I1 = I1.astype(np.float)
    I2 = I2.astype(np.float)
    rows, cols = I1.shape[:2]


    return np.sum((I1[2*rows//10:8*rows//10,2*cols//10:8*cols//10]-I2[2*rows//10:8*rows//10,2*cols//10:8*cols//10])**2,dtype= np.float)


def align_images(I1, I2):
    ''' Compute the best offset to align the second image with the first.

     Loop over a range of offset values in the x and y directions. (Use nested
     for loops.) For each possible (x, y) offset, translate the second image
     and then check to see how well it lines up with the first image using the
     SSD. 
     
     Return the aligned second image.
    '''
    first_diff = sys.maxsize
    x = 0
    y = 0
    rows, cols = I2.shape[:2]
    
    rows = int((rows//5)/2)
    cols = int((cols//5)/2)

    for i in range(-rows,rows):
        for j in range(-cols,cols):
           im = translate(I2, i, j)
           #plt.imshow(cv2.cvtColor(im, cv2.COLOR_GRAY2RGB))
           #plt.show()
           if compute_ssd(I1,im) <= first_diff:
                first_diff = compute_ssd(I1,im)
                x = i
                # print("x = ")
                # print(x)
                y = j
                # print("y = ")
                # print(y)
                # print("________________")


    #print("done")
    global xTranslation
    global yTranslation
    if(abs(x) > abs(xTranslation)):
        xTranslation = x
    if(abs(y) > abs(yTranslation)):
        yTranslation = y
    new_img = translate(I2, x, y)
    return new_img


# --------------------------------------------------------------------------

# this is the image to be processed
image_name = 'tobolsk.jpg'
img = cv2.imread(image_name)

def convert_gray(img): #this function convert an image to grayscale in order to make make the picel matrix two dimensional
    new_img = img.astype(np.float)
    b,g,r = cv2.split(new_img)
    
    b = b*.07
    g = g*.72
    r = r*.21
    
    grey = b+g+r
    
    grey = grey.astype(np.uint8)

    
    return grey

def crop(img): #this function crops an images based off of the larges absolute value translations x and y
    rows, cols = img.shape[:2]

    cropped = img[abs(yTranslation)+15:rows-15,abs(xTranslation)+25:cols-25]
    return cropped


def colorize_image(imageName): #this function converts an image to grey, cuts it into 1/3s, assignes each to be an r, g, or b value,
                               #aligns r and g to b using the align images function, then merges them back together
    img_grey = convert_gray(img)
    rows, cols = img_grey.shape[:2]
    b,g,r = img_grey[0:rows//3,0:cols], img_grey[rows//3:2*rows//3,0:cols], img_grey[2*rows//3:rows,0:cols]

    rows, cols = b.shape[:2]
    g = g[:rows,:cols]
    r = r[:rows,:cols]

    unaligned = cv2.merge((r,g,b))

    g = align_images(b,g)
    gAligned = cv2.merge((r,g,b))

    r = align_images(b,r)

    color_img = cv2.merge((r,g,b))
    
    plt.imshow(unaligned)
    plt.show()
    plt.imshow(gAligned)
    plt.show()
    plt.imshow(color_img)
    plt.show()
    return color_img

cleaned = crop(colorize_image(img))
plt.imshow(cleaned)
plt.show()
new = cv2.cvtColor(cleaned, cv2.COLOR_RGB2BGR)
cv2.imwrite('colored_image.jpg',new)