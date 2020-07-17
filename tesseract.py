import cv2
import pytesseract
import matplotlib.pyplot as plt
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r'Z:\HALOTEC\License Plate Detector\Tesseract-OCR\tesseract.exe'

class tesseract():
    # get grayscale image
    def get_grayscale(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    # noise removal
    def remove_noise(image):
        return cv2.medianBlur(image, 5)
    
    #thresholding
    def thresholding(image):
        # threshold the image, setting all foreground pixels to
        # 255 and all background pixels to 0
        return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    #dilation
    def dilate(image):
        kernel = np.ones((5,5),np.uint8)
        return cv2.dilate(image, kernel, iterations = 2)
        
    #erosion
    def erode(image):
        kernel = np.ones((5,5),np.uint8)
        return cv2.erode(image, kernel, iterations = 1)

    #opening - erosion followed by dilation
    def opening(image):
        kernel = np.ones((5,5),np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    #canny edge detection
    def canny(image):
        return cv2.Canny(image, 100, 200)

    #skew correction
    def deskew(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(gray)
        thresh = cv2.threshold(gray, 0, 255,
            cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        coords = np.column_stack(np.where(thresh > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h),
            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)    
        return rotated

    #template matching
    def match_template(image, template):
        return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED) 
        
    # Get OCR result
    def tesseratOcr(img):
        custom_config = r'--oem 2 --psm 1'
        result = pytesseract.image_to_string(img, lang='plate_v1')
        h, w = img.shape
        boxes = pytesseract.image_to_boxes(img) 
        for b in boxes.splitlines():
            b = b.split(' ')
            img = cv2.rectangle(img, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 1)
        cv2.imshow('Result', img)
        return result

def processOcr(img):
    if img.shape[1] <= 484 :
        W = 484
        height, width, depth = img.shape
        imgScale = W/width
        newX,newY = img.shape[1]*imgScale, img.shape[0]*imgScale
        img = cv2.resize(img,(int(newX),int(newY)))
    
    deskew = tesseract.deskew(img)
    gray = tesseract.get_grayscale(deskew)
    thresh = tesseract.thresholding(gray)
    rnoise = tesseract.remove_noise(thresh)
    dilate = tesseract.dilate(rnoise)
    erode = tesseract.erode(dilate)
    opening = tesseract.opening(erode)
    licensePlate = tesseract.tesseratOcr(opening)

    cv2.imshow('opening', opening)
    cv2.imshow('New Scale', img)

    return licensePlate