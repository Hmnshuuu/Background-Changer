import cv2
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os
import numpy as np
import time

# Initialize video capture from webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height
cap.set(cv2.CAP_PROP_FPS, 60)

# Initialize the SelfiSegmentation module
segmentor = SelfiSegmentation()


# Initialize the FPS variables
fps_start_time = time.time()
fps_frame_count = 0
fps = 0  # Initialize the fps variable

# imgBG = cv2.imread("Images/img1.jpg")

listImg = os.listdir("Images")
print(listImg)

imgList = []
for imgPath in listImg:
    img = cv2.imread(f'Images/{imgPath}')
    imgList.append(img)
print(len(imgList))


imgIndex = 0

while True:
    success, img = cap.read()
    if not success:
        break

    # Remove the background from the image
    # imgOut = segmentor.removeBG(img, (255, 0, 0), cutThreshold=0.98)
    # imgOut = segmentor.removeBG(img, imgBG, cutThreshold=0.98)
    imgOut = segmentor.removeBG(img, imgList[imgIndex], cutThreshold=0.98)


    # Stack the original and the processed images side by side
    stackedImg = np.concatenate((img, imgOut), axis=1)

    # Update FPS counter
    fps_frame_count += 1
    if (time.time() - fps_start_time) >= 1:
        fps = fps_frame_count
        fps_frame_count = 0
        fps_start_time = time.time()

    # Add FPS value to the stacked image
    cv2.putText(stackedImg, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    print(imgIndex)
    # Display the stacked image
    cv2.imshow("Image", stackedImg)
    key = cv2.waitKey(1)
    if key == ord('a'):
        if imgIndex>0:
            imgIndex -= 1
    elif key == ord('d'):
        if imgIndex< len(imgList)-1:
            imgIndex += 1
    elif key == ord('q'):
        break


# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
