import cv2
from ultralytics import YOLO
import os
from cluster import cluster_images
def detect_objects_yolo(image,folder):
    model = YOLO('best.pt')  
    results = model.predict(image,conf=0.55,show_labels=False,show_conf=True)  
    boxes = results[0].boxes.xyxy.tolist()

    if len(boxes)>0:
        os.makedirs(f"static/result/{folder}", exist_ok=True)
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            crop = image[int(y1):int(y2), int(x1):int(x2)]
            cv2.imwrite(f'static/result/{folder}/' + str(i) + '.jpg', crop)

        cluster_images(f'static/result/{folder}/')
        return results[0].plot(labels=False,conf=False)
    return None

def get_file_paths(filename,url):
    data={}
    folder_path=f"{filename}/clustered_images/"
    for cluster in os.listdir(folder_path):
        data[cluster]=[]
        for img in os.listdir(os.path.join(folder_path,cluster)):
            data[cluster].append(url+folder_path+cluster+"/"+img) 
    return data
