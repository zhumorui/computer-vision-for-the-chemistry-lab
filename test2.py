import cv2
from Exp_v3 import Exp
import time
import numpy as np

def cal_color_change(img1,img2):
    assert img1.shape == img2.shape, "Image Size doesn't match"
    
    # Image should use RGB channel not BGR
    # Convert int type to float
    img1 = img1.astype(float)
    img2 = img2.astype(float)

    rmean = (img1[:,:,0] + img2[:,:,0])/2
    r = img1[:,:,0] - img2[:,:,0]
    g = img1[:,:,1] - img2[:,:,1]
    b = img1[:,:,2] - img2[:,:,2]
    distance = np.sqrt(((2 + rmean / 256) * r ** 2 + 4 * g ** 2 + (2 + (255 - rmean) / 256) * b ** 2))
    mean_distance = np.mean(distance)

    return mean_distance


cap = cv2.VideoCapture("../Videos_for_test/short_20210607_152134.mp4")
#cap2 = cv2.VideoCapture("Videos_for_test/short_20210607_152134.mp4")
camera1 = Exp("test1")

# round image entropy decimal places: RIED
# Difference between max and min of 5 entropy clip:DBMM
# interval time to detect vessel: ITDV (Unit: minutes)
" Test 15 description: ITDV:10;DBMM >= 0.08;RIED:4;"
" Test 16 description: ITDV:30; DBMM>= 0.05;RIED:4"
while True:
    ret,frame = cap.read()
    if frame is not None:
        image_with_mask = camera1.get_vessel_image_with_mask(frame)
        cv2.imshow('image_with_mask',image_with_mask)
    else:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()   