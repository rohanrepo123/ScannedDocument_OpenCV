import cv2 as cv
import numpy as np
from flask import Flask, render_template , request, redirect, url_for , session
import os 
from werkzeug.utils import secure_filename 
from io import BytesIO
import base64
app = Flask(__name__)

app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif','webp'}

def image_to_base64(img):
    _,buffer = cv.imencode('.png',img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return img_base64

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
    _,thres = cv.threshold(grey,200,220,cv.THRESH_BINARY)
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
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray,(3,3),1)
    canny = cv.Canny(blur,140,210)
    contours , _ = cv.findContours(canny,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    big_points = np.array([])
    maxarea = 0
    canny_filter = canny[4:796,4:676]
    lines = cv.HoughLinesP(canny_filter, rho=1, theta = np.pi/180, threshold=80, minLineLength=70, maxLineGap=10)
    
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > 9000:
            perimeter = cv.arcLength(cnt,True)
            sides_c = cv.approxPolyDP(cnt , 0.02 * perimeter ,True)
            if len(sides_c) == 4 and area > maxarea:
                maxarea = area
                big_points = sides_c
    return big_points if big_points.size else None ,lines

# img = cv.imread('Photo/Doc1.jpg') 

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/scan",methods=["POST"])
def scan():
        if 'profile_image' not in request.files:
            return "No file Uploaded"
        
        file = request.files['profile_image']
        if file.filename == '':
            return "No selected file."
        
        img_file = np.frombuffer(file.read(),np.uint8)
        img = cv.imdecode(img_file,cv.IMREAD_COLOR)
        
        img = cv.resize(img.copy(),(680,800),interpolation=cv.INTER_AREA)
        
        if img is None:
            return "Could not read image, might be unsupported or corrupted."
        points,lines = get_contours(img)
        print(points)
        joint = []
        lines = np.array(lines)
    
        print("Ye hai perpendicular Joints :",lines)
        imgx = img.copy()   
        for i in lines:
            x1, y1, x2, y2 = i[0]  # i[0] is [x1, y1, x2, y2]
            pts = np.array([[x1, y1], [x2, y2]], dtype=np.int32).reshape((-1, 1, 2))
            imgx = cv.polylines(imgx, [pts], isClosed=1  , color=(0, 255, 0), thickness=2)
        print(i)
        cv.imshow("Scanned X",imgx)

        img_out = draw_outline(img.copy(),points)
        grayy = cv.cvtColor(img_out.copy(),cv.COLOR_BGR2GRAY)
        canny = cv.Canny(grayy,100,240)
        print(canny)
        original_img_b64 = image_to_base64(img)
        outlined_img_b64 = image_to_base64(img_out)
        canny_img_b64 = image_to_base64(canny)
        scanned1 = scanned(img_out)
        # cv.imshow('Original image',img)

        # cv.imshow("Bounded Points",img_out)
        scanned_img_b64 = image_to_base64(scanned1)

        cv.waitKey(0)
        cv.destroyAllWindows()
        return render_template('home.html',joint=joint,lines=lines,canny_img =canny_img_b64,scanned_image = scanned_img_b64,org_img = original_img_b64,out_img = outlined_img_b64)

if __name__ == '__main__':
    app.run(debug=True)