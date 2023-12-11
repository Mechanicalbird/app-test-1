import numpy as np
import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode

import numpy as np
from ultralytics import YOLO
import ultralytics
import torch
from PIL import Image, ImageDraw
from numpy import asarray
import math
import pandas as pd 
import time

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
                                                                                    
def size_messurment_func(counter,calculation_frames,frame,name,input_value,width,height,fps,User_hight,model,model_pose,width_mes_1,width_mes_2,width_mes_3,width_mes_4,inseem_mes_5,inseem_mes_6,arm_mes_7,arm_mes_8,length_man_9,cm_value_factor_10,User_hight_array_11):
    data_arr_names = ['sholder_level','Chest','Waist','Hipps','Inseem_1','Inseem_2','arm_1','arm_2','human_length','cm_value_factor_10','User_hight']
    
    if counter % calculation_frames == 0:
                    checks_number = 0
                    
                    # seg estimation
                    try:
                        istance_results = model.predict(source = frame, save=False, classes = [0])
                    
                        istance_result = istance_results[0]
                        masks = istance_result.masks
                        
                        print(len(masks))
                        mask1 = masks[0]
                        mask1 = mask1.cpu()
                        mask = mask1.data[0].numpy()
                        polygon = mask1.xy[0]

                        #print(polygon)
                        pts=  np.array(polygon, np.int32)
                        #pts = pts.reshape((-1, 1, 2))
                        isClosed = True 
                        # red color in BGR
                        color = (0, 0, 255)
                        # Line thickness of 8 px
                        thickness = 2
                        cv2.polylines(frame, [pts],isClosed, color, thickness)
                    except:
                        print("seg not detected")
                        
                    
                    # pose estimation
                    try:
                        results_pose = model_pose.predict(source = frame, save=False, classes = [0])
                        masks = results_pose[0].keypoints
                        masks = masks.cpu()
                        mask_np = masks.data[0].numpy()
                        print("mask_np",mask.shape)
                    except:
                        print("pos not detected")
                    
                    
                    
                    # write messurment estimation
                    try:
                        font = cv2.FONT_HERSHEY_SIMPLEX

                        mask_img = Image.fromarray(mask, "I")
                        print(mask_img.size)
                        newsize = (width, height)
                        mask_img = mask_img.resize(newsize)

                        numpydata = asarray(mask_img)
                        image_mask_width = numpydata.shape[1]
                        print(numpydata.shape)
                        
                        
                        
                        ### location of sholder level #
                        messurment_location = int((int(mask_np[5][1])+int(mask_np[6][1]))/2)

                        #print(mask_np[5][1])
                        cv2.line(frame,(0, int(messurment_location)),(width, int(messurment_location)),(0,0,0),2)

                        print("messurment_location",messurment_location)

                        start_condition = 0
                        for i in range(0,image_mask_width):
                            current_location_value = abs(numpydata[messurment_location][i])
                            #print(current_location_value)
                            if current_location_value > 0:
                                if start_condition == 0:
                                    start_mes = i
                                    start_condition = 1
                                    print("start_mes",start_mes)


                            if current_location_value < 1:
                                if start_condition == 1:
                                    end_mes = i
                                    print("end_mes",end_mes)
                                    start_condition = start_condition+1

                        print("image_mask_width",image_mask_width)

                        messurment_value = end_mes-start_mes
                        
                        width_mes_1_value = messurment_value
                        print("messurment_value",width_mes_1_value)
                        checks_number += 1
                        

                        input_text = str(messurment_value)
                        cv2.putText(frame,input_text,(0, int(messurment_location)), font,1,(0,0,0),2)
                        cv2.line(frame,(int(start_mes), 0),(int(start_mes), height),(0,0,0),2)
                        cv2.line(frame,(int(end_mes), 0),(int(end_mes), height),(0,0,0),2)

                        #Chest #

                        persentage = 0.3

                        upper_level = (int(mask_np[5][1])+int(mask_np[6][1]))/2
                        lower_level = (int(mask_np[11][1])+int(mask_np[12][1]))/2

                        messurment_location = int((abs(lower_level-upper_level)*persentage)+upper_level)

                        #print(mask_np[5][1])
                        cv2.line(frame,(0, int(messurment_location)),(width, int(messurment_location)),(255,0,0),2)

                        print("messurment_location",messurment_location)

                        start_condition = 0
                        for i in range(0,image_mask_width):
                            current_location_value = abs(numpydata[messurment_location][i])
                            #print(current_location_value)
                            if current_location_value > 0:
                                if start_condition == 0:
                                    start_mes = i
                                    start_condition = 1
                                    print("start_mes",start_mes)


                            if current_location_value < 1:
                                if start_condition == 1:
                                    end_mes = i
                                    print("end_mes",end_mes)
                                    start_condition = start_condition+1

                        print("image_mask_width",image_mask_width)

                        messurment_value = end_mes-start_mes
                        
                        width_mes_2_value = messurment_value
                        print("messurment_value",width_mes_2_value)
                        checks_number += 1

                        input_text = str(messurment_value)
                        cv2.putText(frame,input_text,(0, int(messurment_location)), font,1,(255,0,0),2)
                        cv2.line(frame,(int(start_mes), 0),(int(start_mes), height),(255,0,0),2)
                        cv2.line(frame,(int(end_mes), 0),(int(end_mes), height),(255,0,0),2)




                        #Waist

                        persentage = 0.6

                        upper_level = (int(mask_np[5][1])+int(mask_np[6][1]))/2
                        lower_level = (int(mask_np[11][1])+int(mask_np[12][1]))/2

                        messurment_location = int((abs(lower_level-upper_level)*persentage)+upper_level)

                        #print(mask_np[5][1])
                        cv2.line(frame,(0, int(messurment_location)),(width, int(messurment_location)),(0,255,0),2)

                        print("messurment_location",messurment_location)

                        start_condition = 0
                        for i in range(0,image_mask_width):
                            current_location_value = abs(numpydata[messurment_location][i])
                            #print(current_location_value)
                            if current_location_value > 0:
                                if start_condition == 0:
                                    start_mes = i
                                    start_condition = 1
                                    print("start_mes",start_mes)


                            if current_location_value < 1:
                                if start_condition == 1:
                                    end_mes = i
                                    print("end_mes",end_mes)
                                    start_condition = start_condition+1

                        print("image_mask_width",image_mask_width)

                        messurment_value = end_mes-start_mes
                        width_mes_3_value = messurment_value
                        print("messurment_value",width_mes_3_value)
                        checks_number += 1

                        

                        input_text = str(messurment_value)
                        cv2.putText(frame,input_text,(0, int(messurment_location)), font,1,(0,255,0),2)
                        cv2.line(frame,(int(start_mes), 0),(int(start_mes), height),(0,255,0),2)
                        cv2.line(frame,(int(end_mes), 0),(int(end_mes), height),(0,255,0),2)


                        #Hips


                        lower_level = (int(mask_np[11][1])+int(mask_np[12][1]))/2

                        messurment_location = int(lower_level)

                        #print(mask_np[5][1])
                        cv2.line(frame,(0, int(messurment_location)),(width, int(messurment_location)),(0,0,255),2)

                        print("messurment_location",messurment_location)

                        start_condition = 0
                        for i in range(0,image_mask_width):
                            current_location_value = abs(numpydata[messurment_location][i])
                            #print(current_location_value)
                            if current_location_value > 0:
                                if start_condition == 0:
                                    start_mes = i
                                    start_condition = 1
                                    print("start_mes",start_mes)


                            if current_location_value < 1:
                                if start_condition == 1:
                                    end_mes = i
                                    print("end_mes",end_mes)
                                    start_condition = start_condition+1

                        print("image_mask_width",image_mask_width)

                        messurment_value = end_mes-start_mes
                        
                        width_mes_4_value = messurment_value
                        print("messurment_value",width_mes_4_value)
                        checks_number += 1

                        input_text = str(messurment_value)
                        cv2.putText(frame,input_text,(0, int(messurment_location)), font,1,(0,0,255),2)
                        cv2.line(frame,(int(start_mes), 0),(int(start_mes), height),(0,0,255),2)
                        cv2.line(frame,(int(end_mes), 0),(int(end_mes), height),(0,0,255),2)


                        #Inseem
                        print("Inseem_length_1_start")
                        hipps_knee_length_1 =math.sqrt((((mask_np[11][1])-(mask_np[13][1]))**2)+(((mask_np[11][0])-(mask_np[13][0]))**2))
                        knee_ankle_length_1 =math.sqrt((((mask_np[15][1])-(mask_np[13][1]))**2)+(((mask_np[15][0])-(mask_np[13][0]))**2))
                        Inseem_length_1 = hipps_knee_length_1+knee_ankle_length_1

                        cv2.line(frame,(int((mask_np[11][0])), int((mask_np[11][1]))),(int((mask_np[13][0])), int((mask_np[13][1]))),(0,255,255),5)
                        cv2.line(frame,(int((mask_np[13][0])), int((mask_np[13][1]))),(int((mask_np[15][0])), int((mask_np[15][1]))),(0,255,255),5)

                        inseem_mes_5_value = Inseem_length_1
                        print("Inseem_length_1",inseem_mes_5_value)
                        checks_number += 1
                        
                        

                        hipps_knee_length_2 =math.sqrt((((mask_np[12][1])-(mask_np[14][1]))**2)+(((mask_np[12][0])-(mask_np[14][0]))**2))
                        knee_ankle_length_2 =math.sqrt((((mask_np[16][1])-(mask_np[14][1]))**2)+(((mask_np[16][0])-(mask_np[14][0]))**2))
                        Inseem_length_2 = hipps_knee_length_2+knee_ankle_length_2

                        cv2.line(frame,(int((mask_np[12][0])), int((mask_np[12][1]))),(int((mask_np[14][0])), int((mask_np[14][1]))),(0,255,255),5)
                        cv2.line(frame,(int((mask_np[14][0])), int((mask_np[14][1]))),(int((mask_np[16][0])), int((mask_np[16][1]))),(0,255,255),5)

                        inseem_mes_6_value = Inseem_length_2
                        print("Inseem_length_2",inseem_mes_6_value)
                        checks_number += 1


                        #Arms lengths

                        sholder_elbow_length_1 =math.sqrt((((mask_np[5][1])-(mask_np[7][1]))**2)+(((mask_np[5][0])-(mask_np[7][0]))**2))
                        elbow_weist_length_1 =math.sqrt((((mask_np[9][1])-(mask_np[7][1]))**2)+(((mask_np[9][0])-(mask_np[7][0]))**2))
                        arm_length_1 = sholder_elbow_length_1+elbow_weist_length_1

                        cv2.line(frame,(int((mask_np[5][0])), int((mask_np[5][1]))),(int((mask_np[7][0])), int((mask_np[7][1]))),(0,255,255),5)
                        cv2.line(frame,(int((mask_np[7][0])), int((mask_np[7][1]))),(int((mask_np[9][0])), int((mask_np[9][1]))),(0,255,255),5)

                        arm_mes_7_value = arm_length_1
                        print("arm_length_1",arm_mes_7_value)
                        checks_number += 1
                        

                        sholder_elbow_length_2 =math.sqrt((((mask_np[6][1])-(mask_np[8][1]))**2)+(((mask_np[6][0])-(mask_np[8][0]))**2))
                        elbow_weist_length_2 =math.sqrt((((mask_np[10][1])-(mask_np[8][1]))**2)+(((mask_np[10][0])-(mask_np[8][0]))**2))
                        arm_length_2 = sholder_elbow_length_2+elbow_weist_length_2

                        cv2.line(frame,(int((mask_np[6][0])), int((mask_np[6][1]))),(int((mask_np[8][0])), int((mask_np[8][1]))),(0,255,255),5)
                        cv2.line(frame,(int((mask_np[8][0])), int((mask_np[8][1]))),(int((mask_np[10][0])), int((mask_np[10][1]))),(0,255,255),5)

                        arm_mes_8_value = arm_length_2
                        print("arm_length_1",arm_mes_8_value)
                        checks_number += 1
                        
                        
                        
                        # DETECT THE HIGHT (How tall in pixels) of the person 

                        nose_location_y = int(mask_np[0][1])

                        ankle_location_1_y = int(mask_np[15][1])
                        ankle_location_2_y = int(mask_np[16][1])

                        cv2.line(frame,(int((mask_np[0][0])), int((mask_np[0][1]))),(int((mask_np[15][0])), int((mask_np[16][1]))),(125,0,125),2)
                        cv2.line(frame,(int((mask_np[0][0])), int((mask_np[0][1]))),(int((mask_np[16][0])), int((mask_np[16][1]))),(125,0,125),2)

                        image_mask_hight = numpydata.shape[0]


                        nose_location_x = int(mask_np[0][0])
                        start_condition = 0
                        for i in range(0,image_mask_hight):
                            current_location_value = abs(numpydata[i][nose_location_x])
                            #print(current_location_value)
                            if current_location_value > 0:
                                if start_condition == 0:
                                    start_mes = i
                                    start_condition = 1
                                    head_tip_location = start_mes
                                    print("head_tip_location",head_tip_location)
                                    

                        ankle_location_1_x = int(mask_np[15][0])            
                        start_condition = 1
                        for i in range(ankle_location_1_y,image_mask_hight):
                            current_location_value = abs(numpydata[i][ankle_location_1_x])
                            #print(current_location_value)
                            if current_location_value < 1:
                                if start_condition == 1:
                                    end_mes = i
                                    #print("end_mes1",end_mes)
                                    legs_tip_location_1 = end_mes
                                    start_condition = 0

                                    
                        ankle_location_2_x = int(mask_np[16][0])            
                        start_condition = 1
                        for i in range(ankle_location_2_y,image_mask_hight):
                            current_location_value = abs(numpydata[i][ankle_location_2_x])
                            #print(current_location_value)
                            if current_location_value < 1:
                                if start_condition == 1:
                                    end_mes = i
                                    #print("end_mes2",end_mes)
                                    legs_tip_location_2 = end_mes
                                    start_condition = 0

                        legs_tip_location = max(legs_tip_location_1, legs_tip_location_2)#(legs_tip_location_1+legs_tip_location_2)/2
                        man_hight_calculated = abs(legs_tip_location-head_tip_location)
                        
                        length_man_9_value = man_hight_calculated
                        print("man_hight_calculated",length_man_9_value)
                        checks_number += 1
                        
                        
                        
                        
                        User_hight = float(User_hight)
                        cm_value_factor = User_hight/man_hight_calculated
                        
                        cm_value_factor_10_value = cm_value_factor
                        print("current Factor Value = ",cm_value_factor_10_value)
                        checks_number += 1
                        
                        
                        
                        

                        cv2.line(frame,(int(nose_location_x), int(head_tip_location)),(int(nose_location_x), int(legs_tip_location)),(125,0,125),2)

                        input_text = str(man_hight_calculated)
                        cv2.putText(frame,input_text,(int(nose_location_x), int(nose_location_y)), font,1,(125,0,125),2)
                        
                        number_of_columns = len(data_arr_names)
                        print("number_of_columns",number_of_columns)
                        print("checks_number", checks_number)
                        
                        
                        if int(checks_number) == int(number_of_columns-1) : 
                            cm_value_factor_10.append(cm_value_factor_10_value)
                            length_man_9.append(length_man_9_value)
                            arm_mes_8.append(arm_mes_8_value)
                            arm_mes_7.append(arm_mes_7_value)
                            inseem_mes_6.append(inseem_mes_6_value)
                            inseem_mes_5.append(inseem_mes_5_value)
                            width_mes_4.append(width_mes_4_value)
                            width_mes_3.append(width_mes_3_value)
                            width_mes_2.append(width_mes_2_value)
                            width_mes_1.append(width_mes_1_value)
                            User_hight_array_11.append(User_hight)
                        
                    except:
                        print("messurment not detected")
                        # Record the values:
                        # Mes_1 : sholder level
                        width_mes_1.append(0)
                        # Mes_2 : Chest
                        width_mes_2.append(0)
                        # Mes_3 : Waist 
                        width_mes_3.append(0)
                        # Mes_4 : Hipps
                        width_mes_4.append(0)
                        
                        # Mes_5 : Inseem_1
                        inseem_mes_5.append(0)
                        # Mes_6 : Inseem_2
                        inseem_mes_6.append(0)
                        
                        # Mes_7 : arm_1
                        arm_mes_7.append(0)
                        # Mes_8 : arm_2
                        arm_mes_8.append(0)
                        
                        # human _ length
                        length_man_9.append(0)
                        
                        # cm_value_factor_10
                        cm_value_factor_10.append(0)
                        
                        # User_hight_array
                        User_hight_array_11.append(0)
                    
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    input_text_2 = str(width) + "X" + str(height)
                    cv2.putText(frame,input_text_2,(int(width*0.05), int(height*0.99)), font,1,(0,255,255),2)
                    
                    
                    
                    
                    
                    
                    ## Draw a diagonal blue line with thickness of 5 px
                    start_line1= int(width/2) + int(0.3*width/2) 
                    start_line2= int(width/2) - int(0.3*width/2) 
                    cv2.line(frame,(start_line1,0),(width,height),(255,0,255),3)
                    cv2.line(frame,(start_line2,0),(0,height),(255,0,255),3)
                    
                    cv2.ellipse(frame,(int(width/2),int(height*0.8)),(int(width*0.45),50),0,0,180,(255,0,255),3)
                        
                    #cv2.imshow('frame', frame)
                    
                     
                    
                    
                    print("FRAME_E___XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
                    
                    # Draw input name and value on the frame
                    text_name = f"Name: {name}"
                    cv2.putText(frame, text_name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    duble_input_value=input_value*2
                    text_hight = f"Value: {duble_input_value}"
                    cv2.putText(frame, text_hight, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    

                    
                    
   
    return frame,width_mes_1,width_mes_2,width_mes_3,width_mes_4,inseem_mes_5,inseem_mes_6,arm_mes_7,arm_mes_8,length_man_9,cm_value_factor_10,User_hight_array_11




class Faceemotion(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        cap = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
        width  = int(cap.get(3))
        height = int(cap.get(4)) 
      
        User_hight = input_value
        if User_hight <=0:
            User_hight =10
        
        model = YOLO("yolov8m-seg.pt")
    
        model_pose = YOLO("yolov8m-pose.pt")
    

        return img

def main():
    ########################## USER INPUTS #############################
    st.title("The Size")
    
    name = st.text_input('Enter your name: ')
    st.write('Your name is ', name)

    input_value = st.number_input("Insert your hight:")
    st.write("You entered:", input_value)
  
    webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION,
                        video_processor_factory=Faceemotion)



if __name__ == "__main__":
    main()
