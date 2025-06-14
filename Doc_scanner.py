import cv2 as cv
import numpy as np

# Draw outlines to document
def draw_outline(img,pnts):
    for i in pnts:
        img = cv.circle(img,i[0],4,(255,5,0),-1)
    img = cv.polylines(img,[pnts],1,(255,0,0),2)
        # print('yes',i[0])
    return img

#make image like scanned copy

def scanned(img):
    grey = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    _,thres = cv.threshold(grey,125,250,cv.THRESH_BINARY)
    return thres

# getting points in particular order

def reorder_pnts(pts):
    pts = np.reshape(pts,(4,2))
    coord = np.zeros((4, 2), dtype="float32")

    # Sum and diff of points
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1) 

    # Order: top-left, top-right, bottom-right, bottom-left
    coord[0] = pts[np.argmin(s)]     
    coord[1] = pts[np.argmin(diff)]  
    coord[2] = pts[np.argmax(s)]     
    coord[3] = pts[np.argmax(diff)]  
    # print(coord)
    return coord

# get boundaries of document

def get_contours(img):
    img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(img,(3,3),1)
    canny = cv.Canny(blur,120,200)
    contours , _ = cv.findContours(canny,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    big_points = np.array([])
    maxarea = 0
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > 700:
            perimeter = cv.arcLength(cnt,True)
            sides_c = cv.approxPolyDP(cnt , 0.03 * perimeter ,True)
            if len(sides_c) > 3 and area > maxarea:
                maxarea = area
                big_points = sides_c
    return big_points if big_points.size else None

img = cv.imread('Photo/Doc1.jpg')

points = get_contours(img)

p=reorder_pnts(points)

img_out = draw_outline(img.copy(),points)

scanned = scanned(img_out)
cv.imshow('Original image',img)
cv.imshow("Scanned",scanned)

cv.imshow("Bounded Points",img_out)
cv.waitKey(0)
cv.destroyAllWindows()