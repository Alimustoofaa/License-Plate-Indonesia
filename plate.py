from darkflow.net.build import TFNet
import tensorflow as tf
import numpy as np
import cv2
import sys
import imutils
import os
import argparse
import perspective
import tesseract

path = os.getcwd()

# Load pbLoad and Models Plate
option = {'pbLoad': 'models/yolo-plate.pb', 'metaLoad': 'models/yolo-plate.meta', 'gpu': 0.9}
yoloPate = TFNet(option)

# Crop image based on prediction
def cropImage(img, prediction):
    lenPred = len(prediction)
    prediction.sort(key = lambda x: x.get('confidence'))
    xtop = prediction[-1].get('topleft').get('x')
    ytop = prediction[-1].get('topleft').get('y')
    xbottom = prediction[-1].get('bottomright').get('x')
    ybottom = prediction[-1].get('bottomright').get('y')
    imageCrop = img[ytop : ybottom, xtop : xbottom]
    
    return imageCrop


# Perspective  correction
def perspectiveCorrection(img):
    image = img.copy()
    filter_image = perspective.apply_filter(image)
    threshold_image = perspective.apply_threshold(filter_image)
    cnv, largest_contour = perspective.detect_contour(threshold_image, image.shape)
    corners = perspective.detect_corners_from_contour(cnv, largest_contour)
    destination_points, h, w = perspective.get_destination_points(corners)
    un_warped = perspective.unwarp(image, np.float32(corners), destination_points)

    # Cropped Image Perspective
    cropped = un_warped[1:h, 1:w]
    filtered_image_crop = perspective.apply_filter(cropped)
    threshold_image_crop = perspective.apply_threshold(filtered_image_crop)

    return cropped


# Image to text with tesseract
def ocrTesseract(img):
    LicensePlate = tesseract.processOcr(img)
    print('licensePlate :', LicensePlate)
    return LicensePlate


def main(frame):
    prediction = yoloPate.return_predict(frame)
    imageCrop = cropImage(frame, prediction)
    imagePerspective = perspectiveCorrection(imageCrop)
    imagePerspectiveCrop = imagePerspective.copy()
    licensePlate = ocrTesseract(imagePerspectiveCrop)

    # Results show
    cv2.imshow('Image Crop', imageCrop)
    cv2.imshow('Image Perspective', imagePerspective)
    
    

if __name__ == '__main__':
    # Argumen Parser
    parser = argparse.ArgumentParser(description='Plate Recognition')
    parser.add_argument('--image', help='Path to image file')
    parser.add_argument('--video', help='Path to video file')
    args = parser.parse_args()

    if args.image:
        if not os.path.isfile(args.image):
            print('Input image file ', args.image, 'doesnt exit')
            sys.exit(1)
        frame = cv2.imread(args.image)
    elif args.video:
        if not os.path.isfile(args.video):
            print('Input video file ', args.video, 'doesnt exit')
            sys.exit(1)
    else:
        cv2.VideoCapture(0)

    if args.image:
        main(frame)
        cv2.waitKey(5000)
    else:
        while cv2.waitKey(1) < 0:
            hasFrame, frame = cap.read()
            if not hasFrame:
                print('Done Processing')
                cv2.waitKey(3000)
                break
            h, w, l = frame.shape
            frame = imutils.rotate(frame, 270)
            main(frame)

            if 0xFF == ord('q'):
                break
        cap.release()

    cv2.waitKey(0)