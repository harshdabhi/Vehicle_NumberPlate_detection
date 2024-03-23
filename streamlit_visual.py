import streamlit as st  
from tracking.sort import *
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from utils import get_car,read_license_plate,write_csv
import numpy as np
import tempfile
from model_training.train_model import model_training
import pandas as pd

st.set_page_config(
    page_title="Object Detection using YOLOv8",  # Setting page title
    page_icon="🤖",     # Setting page icon
    layout="wide",      # Setting layout to wide
    initial_sidebar_state="expanded",
        # Expanding sidebar by default
)

def task():
    '''
	This function is responsible for running the main program for automatic license plate detection. 
    It starts by configuring the Streamlit interface, including the title and settings sidebar. 
    It allows the user to adjust the confidence thresholds for vehicle detection and license plate detection, enable GPU usage, specify the vehicles to track, and upload a video file for tracking. 
    It then displays buttons to start, stop, and download the video, as well as an empty image container for displaying the frames.

	The function then installs the necessary libraries and dependencies.
    It loads the trained models for vehicle detection, license plate detection, and vehicle tracking.
    It opens the video file specified by the user or uses the default video file.
    It initializes variables for storing the detection results and the current frame number.
    It loops through each frame of the video, performing the following steps:

	1. Detecting vehicles and obtaining their bounding boxes.
	2. Filtering out the vehicles based on the specified vehicle classes.
	3. Tracking the vehicles using the SORT algorithm.
	4. Detecting license plates and obtaining their bounding boxes.
	5. Processing the license plates and extracting the license plate numbers.
	6. Visualizing the detection results by drawing bounding boxes on the frame.
	7. Storing the detection results in a dictionary.

	The function continues to process each frame until the end of the video or until the user clicks the stop button. Finally, it releases the video and cleans up any remaining resources.

	Parameters:
	None

	Returns:
	None
	'''

############################# Streamlit configuration starts here ##########################################################

    
    st.title('Automatic license plate detection ')
    st.markdown('_____')   

###################### setting up sidebar in streamlit ######################

    st.sidebar.title('Settings')

    st.sidebar.markdown('_____')

    confidence_vehicle_input=st.sidebar.slider('Confidence vehicle',min_value=0.1,max_value=0.9,step=0.1,)

    st.sidebar.markdown('_____')

    confidence_plate_input=st.sidebar.slider('Confidence license plate',min_value=0.1,max_value=0.9,step=0.1,)

    st.sidebar.markdown('_____')


    gpu=st.sidebar.checkbox('Enable GPU')

###################### getting values for vehicle class id to track ######################

 
    class_enable=st.sidebar.checkbox('Specify Vehicle to track')
<<<<<<< HEAD
    name=['car','truck','bus','motorcycle']
    classes_id=[]
=======

    name=['auto rickshaw','bus','car','motorbike','truck']

    classes_input=[]
>>>>>>> 2c87c62688394a581690193c917d9f41ba10c2ab
    if class_enable:
        classes=st.sidebar.multiselect('Select the custom names',name,default=['car'])
        for each in classes:
<<<<<<< HEAD
            classes_id.append(name.index(each))

    st.sidebar.markdown('_____')    
=======
            classes_input.append(name.index(each))
>>>>>>> 2c87c62688394a581690193c917d9f41ba10c2ab

    st.sidebar.markdown('_____')
    
###################### getting video inputs from user ######################

    video_input=st.sidebar.file_uploader('Upload file with video for tracking', type=['mp4'])
    if video_input is not None:
    # Save the uploaded video to a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(video_input.read())
            video_path = temp_file.name
            st.sidebar.video(video_input)



    a,b,c=st.columns([0.3,0.3,0.4])
    video_inputs=('./videos/sample2.mp4')

    Training=st.button('Train Model')

    
    with a:
        start=st.button('Start')
    with b:
        stop=st.button('Stop')
    with c:
        st.download_button('Download',video_inputs)

    show_img=st.empty()

############################# Streamlit configuration ends here ##########################################################



########################### main program starts from here ################################################################

    # Installing necessary libraries and dependencies
    if Training:
        p=model_training()
        p.train_model()
        os.rename('./model_training/yolov8.pt', './vehicle_detection/models/yolov8n.pt')

    #loading Trained Model for vehicle detection/plate detection/vehicle tracking
    if start:
        st.cache_data.clear()
        vehicle_detect_model=YOLO('./vehicle_detection/models/car_model.pt')

        number_plate_detect_model=YOLO('./number_plate_detection/models/best.pt')

        vehicle_tracking_model=Sort()

        #loading video file (mention path) / real time mention (0 in cv2.VideoCapture)
        time.sleep(5)
        # video=cv2.VideoCapture('./videos/sample3.mp4')
        video=cv2.VideoCapture(video_path)
        results = {}
        ret=True
        frame_no=0
        #vehicle_classid=classes_input
        vehicle_classid=[0,1,2,3,4]
        #vehicle_classid=classes_id


        while ret:
            ret,frame=video.read()
            if ret:
                results[frame_no] = {}
                # first detecting vehicles and getting bounding boxes
                vehicle_detection=vehicle_detect_model(frame,conf=confidence_vehicle_input,classes=vehicle_classid,device=0)[0]
                detection_=[]
                for detection in vehicle_detection.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = detection

                    if class_id in vehicle_classid:
                        detection_.append([x1,y1,x2,y2,score])
                

    ############################ To visualise the detection of vehicles from various frame ####################

                    # if score>0.3:
                    #     detection_.append(((int(x1),int(y1)),(int(x2),int(y2))))

                # for i in detection_:
                #     cv2.rectangle(frame,i[0],i[1],(0,255,0))
                # cv2.imshow('test',frame)


    ###########################################################################################################

                ###  tracking vehicles ####
                
                vehicle_tracking=vehicle_tracking_model.update(np.asarray(detection_))

                ## second detecting number plates and getting bounding boxes


                number_plate_detection=number_plate_detect_model(frame,conf=confidence_plate_input,device=0)[0]
                for license_plate in number_plate_detection.boxes.data.tolist():    
                    x1, y1, x2, y2, score, class_id = license_plate
                

                #### To visualise the detection of number plates from various frame ######
                #     cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0))
                # cv2.imshow('test',frame)
                # cv2.waitKey(1)
                # cv2.destroyAllWindows()
                # break

    #######################################################################################################################
            


                    xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, vehicle_tracking)
                    if car_id != -1:

                        # crop license plate
                        license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                        # process license plate
                        license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                        _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 120, 255, cv2.THRESH_BINARY_INV)
                        
                        

                        # read license plate number
                        license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)


                        ####### To visualise the detection of number plates and vehicles from various frame ######

                        cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(100,255,100),thickness=5)
                        cv2.rectangle(frame,(int(xcar1),int(ycar1)),(int(xcar2),int(ycar2)),(100,255,100),thickness=5)
                        #cv2.imshow('test',frame)
                        frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                        show_img.image(frame)
        

                        # if license_plate_text_score !=[]:
                        results[frame_no][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                        'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                        'text': license_plate_text,
                                                                        'bbox_score': score,
                                                                        'text_score': license_plate_text_score}}

            
                cv2.waitKey(1)
                

                frame_no+=1
<<<<<<< HEAD
        write_csv(results, './number_plate_file/test.csv')
=======
    ############# To stop the program ###################
            if stop:
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                video.release()
                break

                    
    ###############################################################################################################
                        

>>>>>>> 2c87c62688394a581690193c917d9f41ba10c2ab
        cv2.destroyAllWindows()
        video.release()

########################## main programs ends here ####################################################################
        
st.markdown('_____')   

        


if __name__ == "__main__":
    try:
        task()
    
    except Exception as e:
        st.write(f'Error: {e} has occured')