import cv2
import torch
from PIL import Image
import numpy as np
import pika
from datetime import datetime
import json

connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()

channel.queue_declare(queue='cars')

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

video_path = 'your_video.avi'
video = cv2.VideoCapture(video_path)

width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)

previous_box = None

while True:
    ret, frame = video.read()

    if not ret:
        break

    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    results = model(image)

    boxes = results.xyxy[0].numpy()
    class_labels = results.xyxy[0][:, -1].numpy().astype(int)

    for detection in results.xyxy[0]:
        xmin, ymin, xmax, ymax, confidence, class_label = detection.tolist()

        class_label = int(class_label)
        class_name = results.names[class_label]

        if class_name == 'car':
            current_box = (int(xmin), int(ymin), int(xmax), int(ymax))
            if previous_box is not None and current_box == previous_box:
                continue
            previous_box = current_box

            cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
            cv2.putText(frame, class_name, (int(xmin), int(ymin - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            car_region = frame[int(ymin):int(ymax), int(xmin):int(xmax)]

            mean_color_car = np.mean(car_region, axis=(0, 1)).astype(int)
            color_hex_car = '#{:02x}{:02x}{:02x}'.format(mean_color_car[2], mean_color_car[1], mean_color_car[0])

            cv2.putText(frame, color_hex_car, (int(xmin), int(ymin - 40)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0),
                        2)
            date_car = datetime.now().timestamp()
            path = f'/Volumes/RAMDisk/cars/car_{date_car}.png'
            cv2.imwrite(path, car_region)
            car_data = {
                "RAM_Path": str(path),
                "date_car": str(date_car),
                "color_hex": str(color_hex_car),
                "fps": str(fps),
                "width": str(width),
                "height": str(height),
                "type": "car",
            }
            channel.basic_publish(exchange='', routing_key='cars', body=json.dumps(car_data))

        elif class_name == 'truck':
            current_box_truck = (int(xmin), int(ymin), int(xmax), int(ymax))
            if previous_box is not None and current_box_truck == previous_box:
                continue
            previous_box = current_box_truck
            cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
            cv2.putText(frame, class_name, (int(xmin), int(ymin - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            truck_region = frame[int(ymin):int(ymax), int(xmin):int(xmax)]

            mean_color_truck = np.mean(truck_region, axis=(0, 1)).astype(int)
            color_hex_truck = '#{:02x}{:02x}{:02x}'.format(mean_color_truck[2], mean_color_truck[1],
                                                           mean_color_truck[0])

            cv2.putText(frame, color_hex_truck, (int(xmin), int(ymin - 40)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0),
                        2)
            date_truck = datetime.now().timestamp()
            path = f'/Volumes/RAMDisk/cars/truck_{date_truck}.png'
            cv2.imwrite(path, truck_region)
            truck_data = {
                "RAM_Path": str(path),
                "date_truck": str(date_truck),
                "color_hex": str(color_hex_truck),
                "fps": str(fps),
                "width": str(width),
                "height": str(height),
                "type": "truck",
            }

            channel.basic_publish(exchange='', routing_key='cars', body=json.dumps(truck_data))

        elif class_name == 'motorcycle':
            current_box_moto = (int(xmin), int(ymin), int(xmax), int(ymax))
            if previous_box is not None and current_box_moto == previous_box:
                continue
            previous_box = current_box_moto
            cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
            cv2.putText(frame, class_name, (int(xmin), int(ymin - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            moto_region = frame[int(ymin):int(ymax), int(xmin):int(xmax)]

            mean_color_moto = np.mean(moto_region, axis=(0, 1)).astype(int)
            color_hex_moto = '#{:02x}{:02x}{:02x}'.format(mean_color_moto[2], mean_color_moto[1],
                                                          mean_color_moto[0])

            cv2.putText(frame, color_hex_moto, (int(xmin), int(ymin - 40)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0),
                        2)

            date_moto = datetime.now().timestamp()
            path = f'/Volumes/RAMDisk/cars/moto_{date_moto}.png'
            cv2.imwrite(path, moto_region)
            moto_data = {
                "RAM_Path": str(path),
                "date_moto": str(date_moto),
                "color_hex": str(color_hex_moto),
                "fps": str(fps),
                "width": str(width),
                "height": str(height),
                "type": "motorcycle",
            }

            channel.basic_publish(exchange='', routing_key='cars', body=json.dumps(moto_data))

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
