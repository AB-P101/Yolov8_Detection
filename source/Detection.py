import cv2
from datetime import datetime
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import time
import numpy as np

def read_config(filename):
    config = {}
    current_section = None
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('[') and line.endswith(']'):
                current_section = line[1:-1]
                config[current_section] = {}
            else:
                if current_section is not None:
                    key, value = line.split('=', 1)
                    config[current_section][key.strip()] = value.strip()
    return config


def append_to_note(filename, content):
    with open(filename, 'a') as file:
        file.write(content + '\n')

config_txt = read_config('config.txt')
if config_txt['Setting_Model']['select'] == '1':
    model = YOLO(config_txt['File_Path_Model']['path_1'])
elif config_txt['Setting_Model']['select'] == '2':
    model = YOLO(config_txt['File_Path_Model']['path_2'])
elif config_txt['Setting_Model']['select'] == '3':
    model = YOLO(config_txt['File_Path_Model']['path_3'])
elif config_txt['Setting_Model']['select'] == '4':
    model = YOLO(config_txt['File_Path_Model']['path_4'])
fw = int(config_txt['Setting_Camera']['display_width'])
fh = int(config_txt['Setting_Camera']['display_height'])
color_rgb = config_txt['Setting_Model']['colorLine'].split(',')
filter_ = config_txt['Setting_Model']['filter']
filter_ = [int(cls) for cls in filter_.split(',')]
conf_ = float(config_txt['Setting_Model']['conf'])
class_list = config_txt['Setting_Model']['class_list'].split(',')
color_ = (int(color_rgb[0]), int(color_rgb[1]), int(color_rgb[2]))

'''cap_1 = cv2.VideoCapture(2) 
cap_2 = cv2.VideoCapture(0) 
cap_1.set(3, fw)
cap_1.set(4, fh)
cap_2.set(3, fw)
cap_2.set(4, fh)'''


object_tracker = DeepSort(max_age=5,max_cosine_distance= 0.4,nn_budget=None)
line_zone_1={}
line_zone_2={}
counter_1 = []
counter_2 = []

def CSI_camera_1(
    sensor_id=0,
    capture_width=fw,
    capture_height=fh,
    display_width=fw,
    display_height=fh,
    framerate=15,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )
def CSI_camera_2(
    sensor_id=1,
    capture_width=fw,
    capture_height=fh,
    display_width=fw,
    display_height=fh,
    framerate=15,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

cap_1 = cv2.VideoCapture(CSI_camera_1(flip_method=0), cv2.CAP_GSTREAMER)
cap_2 = cv2.VideoCapture(CSI_camera_2(flip_method=0), cv2.CAP_GSTREAMER)

if cap_1.isOpened() & cap_2.isOpened():
    while True:
    
        ret1, frame1 = cap_1.read()
        ret2, frame2 = cap_2.read()

        img = np.concatenate((frame1,frame2), axis=1)

        detections = []
        results = model(source=img,classes=filter_,conf=conf_)[0]

        for result in results:
                boxes = result.boxes
                for r in result.boxes.data.tolist():
                    x1, y1, x2, y2 = r[:4]
                    w, h = x2 - x1, y2 - y1
                    coordinates = list((int(x1), int(y1), int(w), int(h)))
                    conf = r[4]
                    
                    clsId = int(r[5])
                    print(clsId)
                    cls = class_list[clsId]
                    detections.append((coordinates, conf, cls))

        tracks = object_tracker.update_tracks(detections, frame=img)

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            bbox = track.to_ltrb()
            name_class = track.get_det_class()
            x3,y3,x4,y4 = int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])
            cx = (x3+x4)//2
            cy = (y3+y4)//2

            cv2.rectangle(img,(x3, y3),(x4, y4),color=color_,thickness=4,)
            cv2.putText(img,str(track_id)+":"+name_class,(int(bbox[0]), int(bbox[1]) - 10),cv2.FONT_HERSHEY_SIMPLEX,2,(0, 255, 0),2,)

            now = datetime.today()
            Floder_Log = now.strftime("_%d_%m_%Y")
            now = now.strftime("Date : %d/%m/%Y Time : %H:%M:%S")
            if 0 < cx and 550 > cx:
                print("Zone 1")
                line_zone_1[track_id] = time.time()
                if track_id in line_zone_1:
                    elapsed_time = time.time() - line_zone_1[track_id]
                    if counter_1.count(track_id)==0:
                        counter_1.append(track_id)
                        append_to_note("Log/Log"+str(Floder_Log)+".txt", "error >> Camara1 >> NG, " + str(now))

            if 740 < cx and 1200 > cx:
                print("Zone 2")
                line_zone_2[track_id] = time.time()
                if track_id in line_zone_2:
                    elapsed_time = time.time() - line_zone_2[track_id]
                    if counter_2.count(track_id)==0:
                        counter_2.append(track_id)
                        append_to_note("Log/Log"+str(Floder_Log)+".txt", "error >> Camara2 >> NG, " + str(now))

        cv2.rectangle(img,(0, 0),(fw+230, fh+200),color=color_,thickness=4)
        cv2.rectangle(img,(fw+250, 0),((fw+240)*2, fh+200),color=color_,thickness=4)

        cv2.putText(img,str(len(counter_1)),(50,50),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
        cv2.putText(img,str(len(counter_2)),(fw+300,50),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)

        cv2.imshow("Program", img)

        k = cv2.waitKey(70) & 0xFF 
        if k == 27:  # 
            break
else:
    print("Error: Unable to open camera")

cap_1.release()
cap_2.release()
cv2.destroyAllWindows()