#........................Imports..............................
import os
import cv2
import time
import numpy as np
import pandas as pd
from tkinter.constants import N
from utils.Vessel_detect import get_frame_OutAnnMap
from utils.color_analysis import cal_color_change

#........................Description.........................
# Including liquid separation detect(finished) and color change detect function(not finished).
# Including data save function.
# Add Parameters Description.
# The Speed of detect vessel is 2X than v1.
# Default mode is CPU, the mode can be changed in the file(Exp_detect/Vessel_detect.py)

# round image entropy decimal places: RIED
# Difference between max and min of 5 entropy clip:DBMM
# interval time to detect vessel: ITDV (Unit: minutes)

" Important parameters description: ITDV:30; DBMM>= 0.05;RIED:4"

class Exp():
    
    def __init__(self,webcamera_id:str,
                detect_liquid_separation_mode = True,
                detect_color_change_mode = False,
                main_colors_analysis_mode = False,
                video_stream_fps = 25,
                default_save_data_format = 'xlsx', # optional format:csv
                interval_time_detect_vessel = 1200, # Unit: Second 
                interval_time_detect_vessel_while_no_vessel_detect = 180, # Unit: Second
                interval_time_calculate_image_entropy = 1, # Unit: Second
                interval_time_calculate_color_change = 1 # Unit: Second
                #...............Parameters Description...............................#
                # webcamer_id: webcamer_id should be unique for each wecamer stream. And the type of id is string.
                # detect_liquid_separation_mode: image_entropy will be calculated and liquid_separation_process will be detected when it's True.
                # detect_color_change_mode: color_change will be detected when it's True.
                # video_stream_fps: the actual FPS of webcamer video stream should be the same as the given video_stream_fps.
                # default_save_data_format: save image entropy of video clip in the file which default format is xlsx, and the optional format is csv.
                # interval_time_detect_vessel: the interval time to detect vessel if the program detect a vessel in the video.
                # interval_time_detect_vessel_while_no_vessel_detect: the interval time to detect vessel again if the program didn't detect a vessel in the video.
                # interval_time_calculate_image_entropy: interval time to calculate image entropy.
                # interval_time_calculate_color_change: interval time to calculate color change.
                ):
        "get webcamer id and save data in the dir named with id"
        self.id = webcamera_id

        "decide the specific task for experiment phenomenon detect."
        self.detect_liquid_separation_mode = detect_liquid_separation_mode
        self.detect_color_change_mode = detect_color_change_mode
        self.main_colors_analysis_mode = main_colors_analysis_mode

        "get the interval time to calculate or detect"
        self.inteval_time_detect_vessel = interval_time_detect_vessel * video_stream_fps
        self.interval_time_calculate_image_entropy = interval_time_calculate_image_entropy * video_stream_fps
        self.interval_time_calculate_color_change = interval_time_calculate_color_change * video_stream_fps
        self.interval_time_detect_vessel_while_no_vessel_detect = interval_time_detect_vessel_while_no_vessel_detect * video_stream_fps

        "get fps of video stream."
        self.video_stream_fps = video_stream_fps

        "get default data file format for image entropy calculation."
        self.default_save_data_format = default_save_data_format

        "initialization count-number for detection and calculation."
        self.count_for_detect_vessel = 0
        self.count_for_detect_vessel_while_no_vessel_detect = 0
        self.count_for_calculate_image_entropy = 0
        self.count_for_calculate_color_change = 0
        
        "creat new dir to save output data"
        self.output_dir = 'output/'+ webcamera_id +'/'
        if not os.path.exists(self.output_dir): os.makedirs(self.output_dir)

        "creat empty sets for video&entropy clip."
        self.liquid_sep_video_clip = []
        self.liquid_sep_entropy_clip = []      
        self.color_change_video_clip = []

    def liquid_separation_detect(self,img,mask):
        "Create output dir for liquid_separation_detect"

        self.liquid_sep_output_dir = self.output_dir + 'liquid_separation_detect/'
        if not os.path.exists(self.liquid_sep_output_dir): os.makedirs(self.liquid_sep_output_dir)

        # Get frame/entropy data, and put them into a video/entropy clip.
        entropy = round(self.cal_1D_entropy(img,mask),4)
        if len(self.liquid_sep_entropy_clip) < 40:
            self.liquid_sep_video_clip.append(img)
            self.liquid_sep_entropy_clip.append(entropy)
        else:
            clip = self.liquid_sep_entropy_clip[15:20]
            m = max(self.liquid_sep_entropy_clip) - min(self.liquid_sep_entropy_clip)

            if all(clip[i]<clip[i+1] for i in range(len(clip)-1)) and m > 0.05:
                print("Detect Liquid Separation Process! Save Data in the output dir!")
                self.save_liquid_separation_results(self.liquid_sep_video_clip,self.liquid_sep_entropy_clip)
                self.liquid_sep_entropy_clip = []
                self.liquid_sep_video_clip = []

            else:
                self.liquid_sep_video_clip.append(img)
                self.liquid_sep_entropy_clip.append(entropy)
                del self.liquid_sep_video_clip[0]
                del self.liquid_sep_entropy_clip[0]

    def color_change_detect(self,img,mask):
        "Create output dir for color_change_detect"

        self.color_change_output_dir = self.output_dir + 'color_change_detect/'
        if not os.path.exists(self.color_change_output_dir): os.makedirs(self.color_change_output_dir)

        # Get color data, and put them into a video clip.
        color = self.cal_color(img,mask)
        if len(self.color_change_video_clip) < 40:
            self.color_change_video_clip.append(img)
        else:
            if color is True:
                print("Detect Color Change Process! Save Data in the output dir!")
                self.save_color_change_results(self.color_change_video_clip)
                self.color_change_video_clip = []

            else:
                self.color_change_video_clip.append(img)
                del self.color_change_video_clip[0]

    def main_colors_analysis(self, img, mask):
        "Create output dir for main_colors_analysis"

        self.main_colors_analysis_output_dir = self.output_dir + 'main_colors_analysis/'
        if not os.path.exists(self.main_colors_analysis_output_dir): os.makedirs(self.main_colors_analysis_output_dir)




    def get_vessel_image_with_mask(self,frame):
        "Input vessel image and return image with mask(trigger liquid_separation_detect)"

        global value

        # If program detects vessel, the second_condition is not satified. Program try to detect vessel again decided by the first condition.
        # If program doesn't detect vessel, the first_condition is not satified. Program try to detect vessel again decided by the second condition.
        first_condition = self.count_for_detect_vessel % self.inteval_time_detect_vessel
        second_condition = self.count_for_detect_vessel_while_no_vessel_detect % self.interval_time_detect_vessel_while_no_vessel_detect

        
        if first_condition == 0 or second_condition == 0:
            value,mask = get_frame_OutAnnMap(frame)
            self.mask = mask.copy()
            self.liquid_sep_video_clip = [] # if program detect vessel again, the video_clip will be emptied.
            self.liquid_sep_entropy_clip = [] # if program detect vessel again, the entropy_clip will be emptied.
            self.count_for_detect_vessel_while_no_vessel_detect = 0 

        h,w = np.shape(self.mask)
        resized_frame = cv2.resize(frame,(w,h),interpolation= cv2.INTER_AREA)
        self.resized_frame = resized_frame.copy() # save resized_frame as the input of entropy calculation
        
        if value is False:
            if  self.count_for_detect_vessel_while_no_vessel_detect % (5 * self.video_stream_fps) == 0: # display remaining time before detect vessel again per 1 second.
                print("No vessel detect! Program starts to try again after %d seconds." \
                    %((self.interval_time_detect_vessel_while_no_vessel_detect - \
                        self.count_for_detect_vessel_while_no_vessel_detect)/self.video_stream_fps))
            image_with_mask = frame
            self.count_for_detect_vessel = 0 # When value is False, which means on vessel detect. 
                                             # Set count_for_detect_vessel is 0, it will plus 1 at the end of the loop, 
                                             # At the beginning of the new loop, count_for_detect_vessel is 1.
                                             # The first condition is not satisfied.
                                             # So we just decide when program try to detect vessel again by the second condition.
        else:
            img1 = self.mask.copy()
            img2 = self.resized_frame.copy()
            
            img2_bg = cv2.bitwise_and(img2,img2,mask = img1)
            img_composed = cv2.addWeighted(img2_bg,0.7,img2,0.3,0)
            ##################
            image_with_mask = img_composed
        
            if self.detect_liquid_separation_mode is True:
                if self.count_for_calculate_image_entropy % self.interval_time_calculate_image_entropy == 0:
                    self.liquid_separation_detect(self.resized_frame,self.mask)
            
            if self.detect_color_change_mode is True:
                if self.count_for_calculate_color_change % self.interval_time_calculate_color_change == 0:
                    self.color_change_detect(self.resized_frame,self.mask)
            self.count_for_calculate_image_entropy += 1
            self.count_for_calculate_color_change += 1
            self.count_for_detect_vessel_while_no_vessel_detect = 0 # When value is not False, which means a vessel is detected.
                                                                    # Set count_for_detect_vessel_while_no_vessel_detect is 0, it will plus 1 at the end of the loop, 
                                                                    # At the beginning of the new loop, count_for_detect_vessel_while_no_vessel_detect is 1.
                                                                    # The second condition is not satisfied.
                                                                    # So we just decide when program try to detect vessel again by the first condition.
        self.count_for_detect_vessel += 1
        self.count_for_detect_vessel_while_no_vessel_detect += 1

        return image_with_mask
        
    def save_liquid_separation_results(self,video_clip,entropy_clip):
        """Save original frames into output dirs"""

        now_time = time.strftime("%Y-%m-%d-%H-%M-%S",time.localtime())
        output_dir = self.liquid_sep_output_dir + now_time[:-9] + '/' + now_time[-8:] + '/'
        if not os.path.exists(output_dir): os.makedirs(output_dir)


        #....................Save Entropy Clip For Liquid Separation Detect......
        if self.default_save_data_format == 'xlsx':
            entropy_clip_df = pd.DataFrame(entropy_clip) # entropy_clip shape is (data_num,2); entropy_clip[1][0]:time_serise; entropy_clip[1][1]:entropy
            
            writer = pd.ExcelWriter(output_dir + now_time + '.xlsx')
            entropy_clip_df.to_excel(writer,'page_1',float_format='%.5f')
            writer.save()
        elif self.default_save_data_format == 'csv':
            np.savetxt('data.csv',entropy_clip,delimiter = '')
        else:
            print("Save entropy_clip failed!\nPlease check data file format")


        #....................Save Video Clip For Liquid Separation Detect......
        frames_num,h,w,color_spaces_num = np.shape(video_clip) # video_clip shape [frames_num,height,width,color_spaces_num]
        out = cv2.VideoWriter(output_dir + now_time + '.mp4',\
            cv2.VideoWriter_fourcc(*'mp4v'),self.video_stream_fps,(w,h),True)

        for i in range(len(video_clip)):
            out.write(video_clip[i])
        out.release()

    def save_color_change_results(self,video_clip):
        """Save original frames into output dirs"""
        
        now_time = time.strftime("%Y-%m-%d-%H-%M-%S",time.localtime())
        output_dir = self.color_change_output_dir + now_time[:-9] + '/' + now_time[-8:] + '/'
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        

        #....................Save Video Clip For Color Change Detect......
        frames_num,h,w,color_spaces_num = np.shape(video_clip) # video_clip shape [frames_num,height,width,color_spaces_num]
        out = cv2.VideoWriter(output_dir + now_time + '.mp4',\
            cv2.VideoWriter_fourcc(*'mp4v'),self.video_stream_fps,(w,h),True)
        for i in range(len(video_clip)):
            out.write(video_clip[i])
        out.release()

    def cal_1D_entropy(self,img,mask):
        hist_cv = cv2.calcHist([img],[0],mask,[256],[0,256])
        P = hist_cv/(len(img)*len(img[0])) 
        E = -np.sum([p *np.log2(p + 1e-5) for p in P])
        return E

    def cal_color(self,img,mask):
        color = 1
        return color