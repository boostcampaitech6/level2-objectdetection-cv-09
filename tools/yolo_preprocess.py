import json
import os
import shutil
from tqdm import tqdm


input_path = '../../dataset/train'
output_path = '../../dataset/yolo_train'

if not os.path.exists(output_path):
    os.makedirs(output_path)
    os.makedirs(os.path.join(output_path, 'images'))
    os.makedirs(os.path.join(output_path, 'labels'))

file = '../../dataset/train.json'
with open(file) as f:
    data = json.load(f)

file_names = []


def load_images_from_folder(folder):
    count = 0
    for filename in tqdm(sorted(os.listdir(folder)), desc='file copying...'):
        source = os.path.join(folder, filename)
        destination = f"{output_path}/images/{str(count).zfill(4)}.jpg"
        shutil.copy(source, destination)
        file_names.append(filename)
        count += 1


def get_img(filename):
    for img in data['images']:
        if img['file_name'] == 'train/'+filename:
            return img


def get_img_ann(image_id):
    img_ann = []
    isFound = False
    for ann in data['annotations']:
        if ann['image_id'] == image_id:
            img_ann.append(ann)
            isFound = True
    if isFound:
        return img_ann
    else:
        return print(image_id)


load_images_from_folder(input_path)

count = 0

for filename in tqdm(sorted(file_names), desc='make ann...'):
    img = get_img(filename)
    img_id = img['id']
    img_w = img['width']
    img_h = img['height']

    img_ann = get_img_ann(img_id)

    if img_ann:
        file_object = open(
            f"{output_path}/labels/{str(count).zfill(4)}.txt", "a"
        )

        for ann in img_ann:
            current_category = ann['category_id']
            current_bbox = ann['bbox']
            x = current_bbox[0]
            y = current_bbox[1]
            w = current_bbox[2]
            h = current_bbox[3]

            x_centre = (x + (x+w))/2
            y_centre = (y + (y+h))/2

            x_centre = x_centre / img_w
            y_centre = y_centre / img_h
            w = w / img_w
            h = h / img_h

            x_centre = format(x_centre, '.6f')
            y_centre = format(y_centre, '.6f')
            w = format(w, '.6f')
            h = format(h, '.6f')

            file_object.write(
                f"{current_category} {x_centre} {y_centre} {w} {h}\n"
            )

        file_object.close()
        count += 1
