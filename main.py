import cv2
from cv2 import exp
from Exp_v3 import Exp

# Average processing time for each frame: 0.005 s (Open all functions)


# Create video stream, support rtsp stream, support multiple streams
camera1_stream = cv2.VideoCapture("Videos_Resources/Four Colour Change Reaction (Chameleon Chemical Reaction).mp4")

# Create camera object and set configuration, support multiple streams
camera1_instance = Exp("test1",detect_liquid_separation_mode = True,
                detect_color_change_mode = True,
                main_colors_analysis_mode = True,
                video_stream_fps = 25,
                default_save_data_format = 'xlsx', # optional format:csv
                interval_time_detect_vessel = 120, # Unit: Second 
                interval_time_detect_vessel_while_no_vessel_detect = 1, # Unit: Second
                interval_time_calculate_image_entropy = 1, # Unit: Second
                interval_time_calculate_color_change = 1, # Unit: Second
                interval_time_main_colors_analysis = 5 # Unit: Second
                )

# Start video processing
while True:
    ret,camera1_frame = camera1_stream.read()

    if camera1_frame is not None:
        camera1_output_frame = camera1_instance.get_output_frame(camera1_frame)
        cv2.imshow('image_with_mask',camera1_output_frame)
        
    else:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera1_stream.release()
cv2.destroyAllWindows() 



"""
# Offline Videos Batch Processing
class Video_processing():

    def __init__(self,
                camera_id,
                camera_stream_link):

        self.camera_stream = cv2.VideoCapture(camera_stream_link)
        self.camera_instance = Exp(camera_id,detect_liquid_separation_mode = True,
                detect_color_change_mode = True,
                main_colors_analysis_mode = True,
                video_stream_fps = 25,
                default_save_data_format = 'xlsx', # optional format:csv
                interval_time_detect_vessel = 120, # Unit: Second 
                interval_time_detect_vessel_while_no_vessel_detect = 1, # Unit: Second
                interval_time_calculate_image_entropy = 1, # Unit: Second
                interval_time_calculate_color_change = 1, # Unit: Second
                interval_time_main_colors_analysis = 5 # Unit: Second
                )

        while True:
            ret,camera1_frame = self.camera_stream.read()

            if camera1_frame is not None:

                start = time.time()

                camera_output_frame = self.camera_instance.get_output_frame(camera1_frame)
                cv2.imshow('image_with_mask',camera_output_frame)
                
                end = time.time()
                try:
                    average_time
                except NameError:
                    average_time = end - start
                average_time = (average_time + (end - start)) / 2

                print("average processing time for each frame: {}".format(average_time))

            else:
                break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.camera_stream.release()
        cv2.destroyAllWindows()

video1 = Video_processing(camera_id="stream1",camera_stream_link="Videos_Resources/Four Colour Change Reaction (Chameleon Chemical Reaction).mp4")
video2 = Video_processing(camera_id="stream2",camera_stream_link="Videos_Resources/Four Colour Change Reaction (Chameleon Chemical Reaction).mp4")

"""


