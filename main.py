import cv2
from cv2 import exp
from Exp_v3 import Exp
import time


# Average processing time for each frame: 0.005 s (Open all functions)



# Create video stream, support rtsp stream, support multiple streams
camera1_stream = cv2.VideoCapture("Videos_Resources/BZ01.mp4")
#camera1_stream = cv2.VideoCapture(0)
stream_fps = camera1_stream.get(5)
print("stream_fps: {}".format(stream_fps))
 
# Create camera object and set configuration, support multiple streams
camera1_instance = Exp("BZ01",detect_liquid_separation_mode = False,
                detect_color_change_mode = True,
                main_colors_analysis_mode = False,
                video_stream_mode = 'offline',
                video_stream_fps = int(stream_fps),
                default_save_data_format = 'xlsx', # optional format:csv
                interval_time_detect_vessel = 120, # Unit: Second 
                interval_time_detect_vessel_while_no_vessel_detect = 1, # Unit: Second
                interval_time_calculate_image_entropy = 1, # Unit: Second
                interval_time_calculate_color_change = 1, # Unit: Second
                interval_time_main_colors_analysis = 5, # Unit: Second
                interval_time_saving_color_change_figure = 300, # Unit: Second
                color_change_detect_threshold = 10 # color change detection threshold
                )



# Start video processing
while True:
    ret,camera1_frame = camera1_stream.read()

    if camera1_frame is not None:
        # slow down the video play
        #time.sleep(0.04)

        camera1_output_frame = camera1_instance.get_output_frame(camera1_frame)
        # cv2.imshow('image_with_mask',camera1_output_frame)
        
    else:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera1_stream.release()
cv2.destroyAllWindows() 
camera1_instance.save_color_distance_array()