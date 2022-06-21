import cv2
import numpy as np

img = cv2.imread('resources/sizeReference.jpg')

while True:

    diameter = ((320, 247), (380, 395))
    cv2.circle(img, diameter[0], 5, (255, 255, 255), 2, 0)
    cv2.circle(img, diameter[1], 5, (255, 255, 255), 2, 0)
    cv2.line(img, diameter[0], diameter[1], (255, 255, 255), thickness=1)

    cv2.imshow('Image', img)

    if cv2.waitKey(1) == ord('q'):
        print(np.math.sqrt(((diameter[0][0] - diameter[1][0]) ** 2) + ((diameter[0][1] - diameter[1][1]) ** 2)))
        break
