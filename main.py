import cv2
from Exp_v3 import Exp
import time
import numpy as np
import matplotlib.pyplot as plt


cap = cv2.VideoCapture("Videos_Resources/Four Colour Change Reaction (Chameleon Chemical Reaction).mp4")
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

        # time.sleep is only used in the demo for a better visualization.
        # time.sleep(0.04) 

        image_with_mask,main_colors, color_distance = camera1.get_vessel_image_with_mask(frame)
        color_distance_list = []
        color_distance_list.append(color_distance)

        main_colors = str(main_colors)

        if color_distance >= 30 and color_distance <= 100:
            color_change_info = "detect color change! color distance = " + str(color_distance)
        elif color_distance > 100:
            color_change_info = "detect abnormal color change!" + str(color_distance)
        else:
            color_change_info = "no color change detect" + str(color_distance)

        font = cv2.FONT_HERSHEY_SIMPLEX

        # main colors analysis text on frame
        cv2.putText(image_with_mask, 
                main_colors, 
                (50, 50), 
                font, 0.6, 
                (0, 255, 255), 
                2, 
                cv2.LINE_4)

        # color change information text on frame
        cv2.putText(image_with_mask, 
                color_change_info, 
                (50, 100), 
                font, 0.6, 
                (0, 255, 255), 
                2, 
                cv2.LINE_4)

        cv2.imshow('image_with_mask',image_with_mask)
    else:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 

# plot color change
x = []
for i in range(len(color_distance_list)):
    x.append(i)



plt.plot(x, color_distance_list)
plt.savefig("color_change_figure.png")
plt.show()
