# Installing necessary libraries and dependencies

from tracking.sort import *
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from utils import get_car,read_license_plate



#loading Trained Model for vehicle detection/plate detection/vehicle tracking

vehicle_detect_model=YOLO('./vehicle_detection/models/yolov8n.pt')

number_plate_detect_model=YOLO('./number_plate_detection/models/best.pt')

vehicle_tracking_model=Sort()

#loading video file (mention path) / real time mention (0 in cv2.VideoCapture)

video=cv2.VideoCapture('./videos/sample2.mp4')
results = {}
ret=True
frame_no=0
vehicle_classid=[0,1,2,3,4]

while ret:
    ret,frame=video.read()
    if ret:
        results[frame_no] = {}
        # first detecting vehicles and getting bounding boxes
        vehicle_detection=vehicle_detect_model(frame)[0]
        detection_=[]
        for detection in vehicle_detection.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection

            if class_id in vehicle_classid:
                detection_.append([x1,y1,x2,y2,score])
        

        ###### To visualise the detection of vehicles from various frame ######

            # if score>0.3:
            #     detection_.append(((int(x1),int(y1)),(int(x2),int(y2))))

        # for i in detection_:
        #     cv2.rectangle(frame,i[0],i[1],(0,255,0))
        # cv2.imshow('test',frame)


        #######################################################################

        ###  tracking vehicles ####
        
        vehicle_tracking=vehicle_tracking_model.update(np.asarray(detection_))

        ## second detecting number plates and getting bounding boxes


        number_plate_detection=number_plate_detect_model(frame)[0]
        for license_plate in number_plate_detection.boxes.data.tolist():    
            x1, y1, x2, y2, score, class_id = license_plate
        

        #### To visualise the detection of number plates from various frame ######
        #     cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0))
        # cv2.imshow('test',frame)
        # cv2.waitKey(1)
        # cv2.destroyAllWindows()
        # break

        ######################################################################
    


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
                cv2.imshow('test',frame)
            
                ###########################################################################################


                # if license_plate_text_score !=[]:
                results[frame_no][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                'text': license_plate_text,
                                                                'bbox_score': score,
                                                                'text_score': license_plate_text_score}}
                
    
        cv2.waitKey(1)
        

        frame_no+=1

cv2.destroyAllWindows()
video.release()



        

    

        



