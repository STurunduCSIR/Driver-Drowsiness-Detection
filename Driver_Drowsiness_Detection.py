#!/usr/bin/env python
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from datetime import datetime
import warnings
import argparse
import imutils
import time
import dlib
import math
import cv2
import datetime
#from cv2 import cv2
import numpy as np
from EAR import eye_aspect_ratio
from MAR import mouth_aspect_ratio
from HeadPose import getHeadTiltAndCoords
import psycopg2
import configparser

# initialize dlib's face detector (HOG-based) and then create the
# facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    './dlib_shape_predictor/shape_predictor_68_face_landmarks.dat')

# initialize the video stream and sleep for a bit, allowing the
# camera sensor to warm up
print("[INFO] initializing camera...")

vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start() # Raspberry Pi
time.sleep(2.0)
start_time = time.time()

# Date of Execution
start_localcurrentdateandtime = datetime.datetime.now() # Get the local date and time

# Video-stream pop-up dimension settings in pixels
FRAME_WIDTH = 1024
FRAME_HEIGHT = 576
## FRAME_WIDTH = 400
## FRAME_HEIGHT = 225

# Read the configuration file
config = configparser.ConfigParser()
config.read('config.ini')

# Constants from config file
TILT_THRESH = config.getint('headtilt', 'angle_thresh')
TILT_FRAMES = config.getint('headtilt', 'tilt_frames')
MOUTH_AR_THRESH = config.getfloat('yawncount', 'yawnthreshold')
EYE_AR_THRESH = config.getfloat('eyeclosure', 'eyethreshold')
EYE_AR_CONSEC_FRAMES = config.getint('eyeclosure', 'eyeclosure_frames') 
DROWSY_THRESH = config.getint('drowsyalert', 'drowsy_threshold') # initially drowsy_threshold = 80 frames - equivalent to 5 seconds
YAWNING =  config.getint('yawncount', 'yawn_frames') #
FULL_YAWN = config.getint('yawncount', 'yawn_complete') 

# loop over the frames from the video stream
# 2D image points. If you change the image, you need to change vector
image_points = np.array([
    (359, 391),     # Nose tip 34
    (399, 561),     # Chin 9
    (337, 297),     # Left eye left corner 37
    (513, 301),     # Right eye right corne 46
    (345, 465),     # Left Mouth corner 49
    (453, 469)      # Right mouth corner 55
], dtype="double")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
                 

# Counter variables for eye and mouth closure
COUNTER_E = 0           # blink frame counter
COUNTER_Y = 0           # yawn frame counter
eye_counter = 0         # number of times eyes closed
yawn_counter = 0        # displays how many times person yawned*

# Storage of data to be transfered to database
LIST_EYE_COUNTER = []       # storage of eye close event count
LIST_EYE_TIMESTAMP = []     # store timestamp for eye closure
DURATION_LIST = []          # stores length of time eyes were closed at a time
LIST_YAWN_COUNTER = []      # storage of yawn event count
LIST_YAWN_TIMESTAMP = []    # store timestamp for eye closure
time_tracker = []           # assists to keep track of start and end of eye closure
DROWSY = []                 # Keeps track of when drowsiness approaches
DROWSY_TIME = []            # time stamp storage for drowsy periods
HEAD_TILT = []              # head tilt time stamp capture for those greater than 18 degrees
HEAD_TILT_TRACKER = []      # keeps track of when head tilt period becomes dangerously long
Dtemp = []    # temporary arrays for keeping length of time of each symptom
Etemp = []    # temporary arrays for keeping length of time of each symptom
Ytemp = []    # temporary arrays for keeping length of time of each symptom

# Timer variable
t = 0  


# grab the indexes of the facial landmarks for the mouth
(mStart, mEnd) = (49, 68)

# Initialize variables for FPS calculation
num_frames = 0
drowsy_frame_counter = 0
fps_start_time = time.time()


while True:
    # Timer to establish duration of video stream
    mins, secs = divmod(t, 15) 
    timer = '{:02d}:{:02d}'.format(mins, secs) 
    t += 1
 
    # grab the frame from the threaded video stream, resize it to
    # have a maximum width of 400 pixels, and convert it to
    # grayscale
    frame = vs.read()
    frame = imutils.resize(frame, width=FRAME_WIDTH, height=FRAME_HEIGHT)
    num_frames += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    size = gray.shape
    
    # FPS Display
    # Draw FPS on the frame
    fps_calculation = num_frames / (time.time() - fps_start_time)
    fps_text = "FPS: {:.2f}".format(fps_calculation)
    cv2.putText(frame, fps_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Print out total time elapsed
    cv2.putText(frame, timer, (900, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # detect faces in the grayscale frame
    rects = detector(gray, 0)

    # check to see if a face was detected, and if so, draw the total
    # number of faces on the frame
    if len(rects) > 0:
        text = "{} face(s) found".format(len(rects))
        cv2.putText(frame, text, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # loop over the face detections
    for rect in rects:
        # compute the bounding box of the face and draw it on the
        # frame
        (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
        cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH), (0, 255, 0), 1)
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        # average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0

        # compute the convex hull for the left and right eye, then
        # visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # check to see if the eye aspect ratio is below the blink
        # threshold, and if so, increment the blink frame counter
        eye_length = len(Etemp)
        if ear < EYE_AR_THRESH:
            COUNTER_E += 1
            # if the eyes were closed for a sufficient number of times
            # then show the warning
            if COUNTER_E >= EYE_AR_CONSEC_FRAMES:
                ##start = time.time()
                eye_counter = 0
                ##print(LIST_EYE_COUNTER) 
                current = datetime.datetime.now()  # eye closure timer starts at this point per event
                start_timestamp = current.strftime("%Y-%m-%d %H:%M:%S") 
                start_timestamp = datetime.datetime.strptime(start_timestamp, "%Y-%m-%d %H:%M:%S") # to be stored in database
                if time_tracker[-1]=="end":
                    LIST_EYE_TIMESTAMP.append(start_timestamp)
                time_tracker.append("start")
                ##print("start: ", start_timestamp)
                cv2.putText(frame, "Eyes Closed!", (500, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Monitor when 5 seconds have passed to alert drowsy
                
                
                # METHOD 1
            
                # Calculate average number of frames passed after 5 seconds
                sec_to_frames = int(fps_calculation * DROWSY_THRESH) # number of frames required for threshold of 5 seconds
                if num_frames >= sec_to_frames:
                    Dtemp.append(num_frames)
                    #print("5 seconds passed")
                    #print(num_frames)
                    num_frames = 0
                    fps_start_time = time.time()
                average = np.mean(Dtemp)
                #print("The threshold: ", average)

                # Drowsy alert thrown if total no. of frames after 5 seconds have passed
                DROWSY.append("1")
                drowsy_length = len(DROWSY)
                ##print(drowsy_length)
                if drowsy_length >= average: 
                    cv2.putText(frame, "DROWSY ALERT", (500, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                if drowsy_length == average:
                    drowsy_start = current.strftime("%Y-%m-%d %H:%M:%S")   
                    print("Drowsy at ", drowsy_start) 
                    DROWSY_TIME.append(drowsy_start)

                ##METHOD 2
                #DROWSY.append("1")
                #drowsy_length = len(DROWSY)
                ###print(drowsy_length)
                #if drowsy_length >= DROWSY_THRESH: 
                #    cv2.putText(frame, "DROWSY ALERT", (500, 50),
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                #if drowsy_length == DROWSY_THRESH:
                #    drowsy_start = current.strftime("%Y-%m-%d %H:%M:%S")   
                #    print("Drowsy at ", drowsy_start) 
                #    DROWSY_TIME.append(drowsy_start)
       

            # otherwise, the eye aspect ratio is not below the blink
            # threshold, so reset the counter and alarm

            else: 
                # Track number of times eyes closed
                ##print("eyes Open")
                DROWSY = []
                eye_counter +=1  
                if eye_counter == 1:
                    end_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    end_timestamp = datetime.datetime.strptime(end_timestamp, "%Y-%m-%d %H:%M:%S")
                    ##print("end: ", end_timestamp)
                    time_tracker.append("end")
                    LIST_EYE_TIMESTAMP.append(end_timestamp)
                    LIST_EYE_COUNTER.append(eye_counter)
                    ##print(LIST_EYE_COUNTER)         

        else:
            COUNTER_E = 0
            time_tracker.append("")
            ##print(time_tracker)
        
        
        mouth = shape[mStart:mEnd]
        mouthMAR = mouth_aspect_ratio(mouth)
        mar = mouthMAR
        # compute the convex hull for the mouth, then
        # visualize the mouth
        mouthHull = cv2.convexHull(mouth)

        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
        cv2.putText(frame, "MAR: {:.2f}".format(mar), (650, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Write text if mouth is open
        yawn_length = len(Ytemp)
        if mar > MOUTH_AR_THRESH:
            print(yawn_length)
            COUNTER_Y += 1
            #yawn_counter = 0
            Ytemp.append("yawn")
            #print(yawn_length) 
            if yawn_length >= (2*fps_calculation):  #next, replace YAWNING with 2*fps_calculation 
                cv2.putText(frame, "Yawning!", (800, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if yawn_length == (FULL_YAWN): #next, replace FULL_YAWN with 4*fps_calculation
                #yawn_counter+= 1
                LIST_YAWN_COUNTER.append("1")
                Ytemp = []

                            
        
        # Tracker number of yawn events
        else:
            COUNTER_Y = 0
            
        ##else:
          ##  #Ytemp = []
            ##COUNTER_Y = 0
            ##yawn_counter +=1  
            ##if yawn_counter == 1:
              ##  LIST_YAWN_COUNTER.append(yawn_counter)
                ####print(LIST_YAWN_COUNTER)  


        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw each of them
        for (i, (x, y)) in enumerate(shape):
            if i == 33:
                # something to our key landmarks
                # save to our new key point list
                # i.e. keypoints = [(i,(x,y))]
                image_points[0] = np.array([x, y], dtype='double')
                # write on frame in Green
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            elif i == 8:
                # something to our key landmarks
                # save to our new key point list
                # i.e. keypoints = [(i,(x,y))]
                image_points[1] = np.array([x, y], dtype='double')
                # write on frame in Green
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            elif i == 36:
                # something to our key landmarks
                # save to our new key point list
                # i.e. keypoints = [(i,(x,y))]
                image_points[2] = np.array([x, y], dtype='double')
                # write on frame in Green
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            elif i == 45:
                # something to our key landmarks
                # save to our new key point list
                # i.e. keypoints = [(i,(x,y))]
                image_points[3] = np.array([x, y], dtype='double')
                # write on frame in Green
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            elif i == 48:
                # something to our key landmarks
                # save to our new key point list
                # i.e. keypoints = [(i,(x,y))]
                image_points[4] = np.array([x, y], dtype='double')
                # write on frame in Green
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            elif i == 54:
                # something to our key landmarks
                # save to our new key point list
                # i.e. keypoints = [(i,(x,y))]
                image_points[5] = np.array([x, y], dtype='double')
                # write on frame in Green
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            else:
                # everything to all other landmarks
                # write on frame in Red
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
                cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        #Draw the determinant image points onto the person's face
        for p in image_points:
            cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

        (head_tilt_degree, start_point, end_point, 
            end_point_alt) = getHeadTiltAndCoords(size, image_points, FRAME_HEIGHT)

        cv2.line(frame, start_point, end_point, (255, 0, 0), 2)
        cv2.line(frame, start_point, end_point_alt, (0, 0, 255), 2)

        if head_tilt_degree:
            cv2.putText(frame, 'Head Tilt Degree: ' + str(head_tilt_degree[0]), (170, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        if TILT_THRESH < head_tilt_degree:
            HEAD_TILT_TRACKER.append("1")
            head_length = len(HEAD_TILT_TRACKER)
            ##print(head_length)
            if head_length == TILT_FRAMES:
                current = datetime.datetime.now()  # event starts at this point
                head_tilt_marker = current.strftime("%Y-%m-%d %H:%M:%S") 
                HEAD_TILT.append(head_tilt_marker)
                ##print(HEAD_TILT)
            if head_length >= TILT_FRAMES:
                cv2.putText(frame, 'Low head tilt alert ' + str(head_tilt_degree[0]), (170, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        else:
            HEAD_TILT_TRACKER = []


        # extract the mouth coordinates, then use the
        # coordinates to compute the mouth aspect ratio
    
    # show the frame
    cv2.imshow("Fatigue Monitoring", frame)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# Configuration of first and last eye closure timestamps
if time_tracker[0]=="end":          # if videostream starts with eyes open, ignore first timestamp
    LIST_EYE_TIMESTAMP = LIST_EYE_TIMESTAMP[1:]
if time_tracker[-1]=="start":       # if videostream ends with eyes closed, add timestamp to represent ending
    last_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") 
    last_timestamp = datetime.datetime.strptime(last_timestamp, "%Y-%m-%d %H:%M:%S")
    LIST_EYE_TIMESTAMP.append(last_timestamp)
##print(len(time_tracker))##################################REMOVE


# Data for table 2 in fatigue database - drowsy/non-drowsy statistics
LIST_EYECLOSURE_TIME = [] # temporary array storing timestamps (close and open) of eye closures and its duration in seconds 
for n in range(1, len(LIST_EYE_TIMESTAMP), 2):
    Difference = LIST_EYE_TIMESTAMP[n] - LIST_EYE_TIMESTAMP[n-1]
    seconds = int(Difference.total_seconds())
    DURATION_LIST.append(seconds)
    LIST_EYECLOSURE_TIME.append(LIST_EYE_TIMESTAMP[n])
    LIST_EYECLOSURE_TIME.append(LIST_EYE_TIMESTAMP[n-1])
    LIST_EYECLOSURE_TIME.append(seconds)
    if seconds < 5:
        LIST_EYECLOSURE_TIME.append("NOT DROWSY")
    else:
        LIST_EYECLOSURE_TIME.append("DROWSY")


##print("NEW LIST ", LIST_EYECLOSURE_TIME)
##print("EYE CLOSE COUNTER ", DURATION_LIST)
##print("seconds ", seconds)



# End Date Record
end_localcurrentdateandtime = datetime.datetime.now() # Gets the local date and timestamp at termination of algorithm

# Timer calculation for whole algorithm
end_time = time.time()
total_time = end_time -  start_time


# Total number of fatigue symptom events
#total_eye_counter = len(LIST_EYE_COUNTER)
total_eye_counter = len(DURATION_LIST)
total_yawn_counter = len(LIST_YAWN_COUNTER)
total_drowsy_counter = len(DROWSY_TIME)
total_head_tilt_counter = len(HEAD_TILT)

# Results printed in console
print("Start date and time: ", start_localcurrentdateandtime)
print("End date and time: ", end_localcurrentdateandtime)
print(f'Total Time elapsed: {total_time} seconds')
print(f'Eyes closed: {total_eye_counter} times')
print("Total yawn count: ", total_yawn_counter)
print("Total number of drowsy periods: ", total_drowsy_counter)
print("Total number of drowsy head tilts: ", total_head_tilt_counter)


#print("TYPE =")
#print(type(Difference)) #type datetime.datetime

# Database connection and queries

# Get PostgreSQL database information from the configuration file
db_config = config['postgresql']

# Establish connection to the PostgreSQL database
conn = psycopg2.connect(
    dbname=db_config['database'],
    user=db_config['user'],
    password=db_config['password'],
    host=db_config['host'],
    port=db_config['port']
)

# Create a cursor object to execute PostgreSQL commands
cur = conn.cursor()

def create_table(connection, table_name, columns):
    # Create a cursor object to execute PostgreSQL commands
    cur = connection.cursor()

    # Define the SQL statement to create the table
    create_table_query = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns});"

    # Execute the SQL statement
    cur.execute(create_table_query)

    # Commit the transaction
    connection.commit()

    # Close the cursor
    cur.close()


for section in config.sections():
    if section == 'maintable' or section == 'drowsytable':
        # Get table information from the configuration file
        table_name = config[section]['name']
        columns = config[section]['columns']

        # Create the fataddigue stats table using information from the config file
        create_table(conn, table_name, columns)

# TABLE 1 Creations and Insertion
# Data to be logged into the maintable
data_to_insert1 = [
    (start_localcurrentdateandtime, end_localcurrentdateandtime, total_eye_counter, total_yawn_counter, total_drowsy_counter)
]

# Insert logged data into maintable
insert_query1 = '''
INSERT INTO Fatigue_Symptom_Stats (StartDate, EndDate, EyeClosureCount, YawnCount, DrowsyCount)
VALUES (%s, %s, %s, %s, %s);
'''

# Execute the SQL statement for each row of data
for row in data_to_insert1:
    cur.execute(insert_query1, row)
conn.commit()

# TABLE 2 Creations and Insertion
# Data to be logged into the drowsy stats table
try:
    for i in range(0, len(LIST_EYECLOSURE_TIME), 4):
        data_to_insert2 = [
            (LIST_EYECLOSURE_TIME[i], LIST_EYECLOSURE_TIME[i+1], LIST_EYECLOSURE_TIME[i+2], LIST_EYECLOSURE_TIME[i+3])
        ]
        

    # Insert logged data into drowsy stats table
    insert_query2 = '''
    INSERT INTO Drowsy_Stats (StartTime, EndTime, DurationSec, Drowsy)
    VALUES (%s, %s, %s, %s);
    '''

    # Execute the SQL statement for each row of data
    for row in data_to_insert2:
        cur.execute(insert_query2, row)
except NameError:
        print("No record to store in 'Drowsy Table' database")
# Commit the transaction
conn.commit()

# Close the cursor and connection
cur.close()
conn.close()

# print(image_points)

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
