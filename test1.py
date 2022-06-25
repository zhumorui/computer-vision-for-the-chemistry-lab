import cv2
from Exp_v3 import Exp
import time

cap = cv2.VideoCapture("../Videos_for_test/20210607_152134.mp4")
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
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_with_mask = camera1.get_vessel_image_with_mask(rgb)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image_with_mask, 
                'TEXT ON VIDEO', 
                (50, 50), 
                font, 1, 
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
