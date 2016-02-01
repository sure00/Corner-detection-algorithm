__author__ = 'shuo'

import cv2
import sys
import numpy as np
import math

global image1
global image2

error = -1
DEBUG  = 1

def nothing(x):
    pass

def launch_camera():
    print sys._getframe().f_code.co_name
    cap = cv2.VideoCapture(0)

    if(cap.isOpened()):
        print("Camera open successful")
        return cap
    else :
        print("Camera open failed")
        return error


def magnitude_gradient(img):
    if DEBUG == 1 :
        print sys._getframe().f_code.co_name

    #print("debug",img)
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)

    out_img = np.sqrt(np.power(sobelx,2) + np.power(sobely,2))

    return out_img

def compute_harris(frame, sigma, k) :
    if DEBUG == 1 :
        print sys._getframe().f_code.co_name

    rows = frame.shape[0]
    cols = frame.shape[1]

    cov = np.zeros((rows,cols * 3), dtype = np.float32)
    dst = np.zeros((rows,cols), dtype = np.float32)

    dx = cv2.Sobel(frame,cv2.CV_64F,1,0,ksize=5)
    dy = cv2.Sobel(frame,cv2.CV_64F,0,1,ksize=5)

    ksize = max(5,  5 * sigma)
    if (ksize %2 == 0 ) :
        ksize = ksize + 1

    if DEBUG == 1 :
        print('ksize', ksize)
        print('k',k)
        print('rows',rows)
        print('cols', cols)
        print('sigma',sigma)

    #if sigma ==0 :
     #   ksize = 5
    #else :
    #h    ksize = (int)sigma * 5

    Ixx = cv2.GaussianBlur(dx*dx,(ksize,ksize),sigma)
    Iyy = cv2.GaussianBlur(dy*dy,(ksize,ksize),sigma)
    Ixy = cv2.GaussianBlur(dx*dy,(ksize,ksize),sigma)

    #cv2.imshow('dx',dx)
    #cv2.imshow('dy',dy)

    for i in xrange(0, rows,1) :
            for j in xrange(0, cols,1) :
                a = cov[i, j*3] =   Ixx[i,j]
                b = cov[i, j*3+1] = Ixy[i,j]
                c = cov[i, j*3+2] = Iyy[i,j]
                dst[i,j] = a*c - b*b - k*(a+c)*(a+c)

    return dst

def load_image(src_type) :
    if DEBUG == 1 :
        print sys._getframe().f_code.co_name
    if src_type == 1:
        global image1
        global image2

        file1 = "../data/default1.jpg"
        file2 = "../data/default2.jpg"
        image1 = cv2.imread(file1)
        image2 = cv2.imread(file2)
        #image1 = cv2.resize(image1, (image1.shape[1]/3, image1.shape[0]/3))
        #image2 = cv2.resize(image2, (image2.shape[1]/3, image2.shape[0]/3))

    else :
        file = sys.argv[1]
    return

def normalization(src_img) :
    #print("src normalization", src_img)
    if DEBUG == 1 :
        print sys._getframe().f_code.co_name
    out_img = ((src_img - np.amin(src_img)) / (np.amax(src_img) - np.amin(src_img)))

    return out_img

def draw_rectangle(src_img, threshold) :
    rows = src_img.shape[0]
    cols = src_img.shape[1]

    cv2.imshow('rect',src_img)

def process_video(cam):
    if DEBUG == 1 :
        print sys._getframe().f_code.co_name

    cv2.createTrackbar('Variance of Gaussian','Harris corner',0,1,nothing)
    cv2.createTrackbar('Neighborhood Size','Harris corner',2,50,nothing)
    cv2.createTrackbar('Weight of trace','Harris corner',2,50,nothing)
    cv2.createTrackbar('Threshold value','Harris corner',1,50,nothing)

    while (True) :
            # Capture frame-by-frame
            ret, frame = cam.read()
            rows = frame.shape[0]
            cols = frame.shape[1]

            sigma = cv2.getTrackbarPos('Variance of Gaussian','Harris corner')
            block_size = cv2.getTrackbarPos('Neighborhood Size','Harris corner')
            apertureSize = cv2.getTrackbarPos('Weight of trace','Harris corner')
            threshold_value = cv2.getTrackbarPos('Threshold value','Harris corner')

            threshold_value = 1

            sigma = 0
            blockSize = 2
            k = 0.02

            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            gray = np.float32(gray)
            harrisim = compute_harris(gray,  sigma, k)

            # Threshold for an optimal value, it may vary depending on the image.
            frame[harrisim>0.01*harrisim.max()]=[0,0,255]
            cv2.imshow('hw2',frame)

            if cv2.waitKey(5) == 27 :
                cv2.destroyAllWindows()
                break

def process_cmdline(argv) :
    if DEBUG == 1 :
        print sys._getframe().f_code.co_name

    # Load Default Image
    if len(argv) < 2:
            src_type = 1
            print("No ordered source of image and load default image")

    # capture image from camera
    elif 'camera' == argv [1] :
            src_type = 0
            print("load camera")

    # Read the ordered image
    else :
            src_type = 2
            print("Read the image from parameter image")
    return src_type

def cornerSubPix(src, centroids, block_size):
    if DEBUG == 1 :
        print sys._getframe().f_code.co_name

    out = np.zeros((centroids.shape[0]-1,centroids.shape[1],1), dtype=np.float32 )

    win_w = block_size * 2 + 1
    win_h = block_size * 2 + 1

    if DEBUG == 1 :
        print('win w',win_h)
        print('win h', win_w)

    # pixel position in the rectangle
    x = np.arange(-block_size, block_size +1, dtype=np.int)
    y = np.arange(-block_size, block_size +1, dtype=np.int)

    length = len(centroids)

    # do optimization loop for all centroids
    for i in range(1,len(centroids)) :
        im = cv2.getRectSubPix(src,(win_w, win_h), (centroids[i][0],centroids[i][1]))

        # 1st derivative of image
        dx = cv2.Sobel(im,cv2.CV_64F,1,0,ksize=5)
        dy = cv2.Sobel(im,cv2.CV_64F,0,1,ksize=5)

        # dx2,dy2, dxy
        Ixx = cv2.GaussianBlur(dx*dx,(5,5),0)
        Iyy = cv2.GaussianBlur(dy*dy,(5,5),0)
        Ixy = cv2.GaussianBlur(dx*dy,(5,5),0)

        #sum of the dx, dy, dxy
        a = np.sum(Ixx,None)
        b = np.sum(Ixy,None)
        c = np.sum(Iyy,None)

        #sum I(xi)I(xi)t*xi
        bb1 = Ixx * x + Ixy * y
        bb2 = Ixy * x + Iyy * y

        #c-1 * sum I(xi)I(xi)t*xi
        det = a*c - b*b
        scale =  1/det

        out[i-1][0] = centroids[i][0] + c*scale* np.sum(bb1,None) - b*scale * np.sum(bb2,None)
        out[i-1][1] = centroids[i][1] - b*scale* np.sum(bb1,None) + a*scale * np.sum(bb2,None)

    return  out

def corner_harris_and_localization(image,sigma,k,block_size,threshold_value ) :
        # convert image1 to gray
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)

        #using harris corner detection to find the corners. Notice that there would be lots of corners find out.

        #dst = cv2.cornerHarris(gray,2,3,0.04)
        dst = compute_harris(gray, sigma, k)

        print('threshold value', threshold_value)
        print('dst.max', dst.max())
        print('threds total', 0.001 * threshold_value *dst.max())
        ret, dst1 = cv2.threshold(dst,0.001 * threshold_value *dst.max(),255,0)

        # find centroids of the points around the area
        dst2 = np.uint8(dst1)
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst2)

        #localization
        #corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)
        corners = cornerSubPix(gray, np.float32(centroids), block_size)

        return corners

def corner_featureVector(corners, image, block_size):
    if DEBUG == 1 :
        print sys._getframe().f_code.co_name

    # convert image1 to gray
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    win_w = block_size * 2 + 1
    win_h = block_size * 2 + 1

    # column means: -180 -135 -90 -45 0 45 90 135 180
    # row means: corner

    degree_histogram = np.zeros((len(corners),9,1), dtype=np.uint8)

    for i in range(0,len(corners)) :
        im = cv2.getRectSubPix(gray,(win_w, win_h), (corners[i][0],corners[i][1]))

        # 1st derivative of image
        dx = cv2.Sobel(im,cv2.CV_64F,1,0,ksize=5)
        dy = cv2.Sobel(im,cv2.CV_64F,0,1,ksize=5)

        angle = np.arctan2(dy,dx)

        # range from -180 to 180
        # -180 <= angle < -135
        # -135 <= angle < -90
        # -90 <= angle < -45
        # ''''''

        for j in range(0, 9) :
            degree_histogram[i][j] = ((( math.pi/4 * (j-4))<= angle) & (angle < (math.pi/4 *(j-3)))).sum()

    return degree_histogram

def ShowCornerAndFeaturePoints(image1, image2, corners1,corners2,histogram1,histogram2) :

    # draw rectangle of the corner
    corners1 = np.int0(corners1)
    corners2 = np.int0(corners2)

    # draw a rectangle around the corner
    for i in range(0,corners1.shape[0]) :
          cv2.rectangle(image1, (corners1[i,0] - 18, corners1[i,1]-18), (corners1[i,0]+18, corners1[i,1]+18), [0,0,0])
    for j in range(0,corners2.shape[0]) :
          cv2.rectangle(image2, (corners2[j,0] - 18, corners2[j,1]-18), (corners2[j,0]+18, corners2[j,1]+18), [0,0,0])

    cno = 1
    min_distance = 0
    min_index = 0

    for i in range(0, len(corners1)):
        for j in range(0, len(corners2)) :
                # compare histogram for each corner,find the smallest distance between corner in image 1 and image 2.
                distance = np.sum( (histogram1[i] - histogram2[j] ) * (histogram1[i] - histogram2[j] ),None)
                if (j ==0 ) :
                    min_distance = distance
                    min_index = j
                else  :
                    if distance < min_distance :
                        min_distance = distance
                        min_index = j

        cv2.putText(image1,str(cno), (corners1[i,0],corners1[i,1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255),2,cv2.LINE_AA)
        cv2.putText(image2,str(cno), (corners2[min_index,0],corners2[min_index,1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255),2,cv2.LINE_AA)
        cno = cno +1

    cv2.imshow('image1',image1)
    cv2.imshow('image2',image2)


def process_image(orgimg1, orgimg2) :
    if DEBUG == 1 :
        print sys._getframe().f_code.co_name

    while True:
        key = cv2.waitKey(33)

        if key == ord('h') :
            cv2.namedWindow('corner')

            #Track Bar
            trace_bar = np.zeros((120,320,1), dtype=np.float32 )
            cv2.createTrackbar('Variance of Gaussian','corner',0,20,nothing)
            cv2.createTrackbar('Neighborhood Size','corner',2,4,nothing)
            cv2.createTrackbar('Weight of trace','corner',4,15,nothing)
            cv2.createTrackbar('Threshold value','corner',10,50,nothing)

            cv2.imshow('corner',trace_bar)

            while (True) :
                image1=orgimg1.copy()
                image2=orgimg2.copy()

                sigma = cv2.getTrackbarPos('Variance of Gaussian','corner')
                block_size = cv2.getTrackbarPos('Neighborhood Size','corner')
                k = cv2.getTrackbarPos('Weight of trace','corner')
                threshold_value = cv2.getTrackbarPos('Threshold value','corner')

                #threshold_value = 1

                sigma = 0.01 * sigma

                k = 0.01 * k

                corners1 = corner_harris_and_localization(image1,sigma,k,block_size,threshold_value)
                histogram1 = corner_featureVector(corners1, image1, block_size)

                corners2 = corner_harris_and_localization(image2,sigma,k,block_size,threshold_value)
                histogram2 = corner_featureVector(corners2, image2, block_size)

                ShowCornerAndFeaturePoints(image1,image2,corners1,corners2,histogram1,histogram2)

                if cv2.waitKey(5) == 27 :
                    cv2.destroyAllWindows()
                    break

def main():
    if DEBUG == 1 :
        print sys._getframe().f_code.co_name

    src = process_cmdline(sys.argv)

    if src == 0 :                   # source from camera
            camera = launch_camera()
            process_video(camera)
    else:                           # source from image
            load_image(src)
            cv2.imshow('image1',image1)
            cv2.imshow('image2',image2)

            orgimg1=image1.copy()
            orgima2=image2.copy()

            process_image(orgimg1,orgima2)

if __name__ == '__main__':
    main()

