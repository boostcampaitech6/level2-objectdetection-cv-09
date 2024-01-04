from ultralytics import YOLO
from tqdm import tqdm
from glob import glob
import cv2
import pandas as pd

model = YOLO('../runs/detect/train/weights/best.pt')

BATCH_SIZE = 64


def get_test_image_paths(test_image_paths):
    for i in range(0, len(test_image_paths), BATCH_SIZE):
        yield test_image_paths[i:i+BATCH_SIZE]


test_image_paths = sorted(glob("../../dataset/test/*.jpg"))
for i, image in enumerate(get_test_image_paths(test_image_paths)):
    model.predict(image, imgsz=(1024, 1024), save_conf=True, save=False, save_txt=True)


def yolo_to_format(line, image_width, image_height):
    class_id, x, y, width, height, confidence = [float(temp) for temp in line.split()]

    x_min = int((x - width / 2) * image_width)
    x_max = int((x + width / 2) * image_width)
    y_min = int((y - height / 2) * image_height)
    y_max = int((y + height / 2) * image_height)

    return str(int(class_id)) + ' ' + str(confidence) + ' ' + str(x_min) + ' ' + str(y_min) + ' ' + str(x_max) + ' ' + str(y_max) + ' '


infer_txts = sorted(glob("../runs/detect/predict/labels/*.txt"))
prediction_strings = []
file_names = []
for infer_txt in tqdm(infer_txts):
    file_path = f'../../dataset/test/{infer_txt.split("/")[-1].split(".")[0]}.jpg'
    imgage_height, imgage_width = cv2.imread(file_path).shape[:2]
    prediction_string = ''
    with open(infer_txt, "r") as reader:
        lines = reader.readlines()
        for line in lines:
            prediction_string += yolo_to_format(line, imgage_width, imgage_height)
    image_ids = str(file_path).replace('../../dataset/', '')
    prediction_strings.append(prediction_string)
    file_names.append(image_ids)

df_submission = pd.DataFrame()
df_submission['PredictionString'] = prediction_strings
df_submission['image_id'] = file_names
df_submission.to_csv("../results/yolov8.csv", index=False)
