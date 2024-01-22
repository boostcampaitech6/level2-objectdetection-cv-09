import pandas as pd
import numpy as np
from tqdm import tqdm
from glob import glob

from ensemble_boxes import weighted_boxes_fusion as wbf
from pycocotools.coco import COCO

import click


@click.command()
@click.option('--submissions', required=True, help='Files for ensemble')
@click.option('--annotation_path', required=True, help='test.json path')
@click.option('--save_pth', required=True, help='Output path')
def main(submissions, annotation_path, save_pth):
    sub_files = sorted(glob(submissions+'/*'))
    sub_dfs = [pd.read_csv(sub) for sub in sub_files]
    image_ids = sub_dfs[0]['image_id'].tolist()
    coco = COCO(annotation_path)

    prediction_strings = []
    file_names = []

    for i, image_id in tqdm(enumerate(image_ids), total=len(image_ids)):
        prediction_string = ''
        boxes_list = []
        scores_list = []
        labels_list = []
        image_info = coco.loadImgs(i)[0]

        for df in sub_dfs:
            predict_string = \
                df[df['image_id'] == image_id]['PredictionString'].tolist()[0]
            predict_list = str(predict_string).split()

            if len(predict_list) == 0 or len(predict_list) == 1:
                continue

            predict_list = np.reshape(predict_list, (-1, 6))
            box_list = []

            for box in predict_list[:, 2:6].tolist():
                box[0] = float(box[0]) / image_info['width']
                box[1] = float(box[1]) / image_info['height']
                box[2] = float(box[2]) / image_info['width']
                box[3] = float(box[3]) / image_info['height']
                box_list.append(box)

            boxes_list.append(box_list)
            scores_list.append(list(map(float, predict_list[:, 1].tolist())))
            labels_list.append(list(map(int, predict_list[:, 0].tolist())))

        if len(boxes_list):
            boxes, scores, labels = wbf(
                boxes_list, scores_list, labels_list,
                weights=None, iou_thr=0.55, skip_box_thr=0.001
            )

            for box, score, label in zip(boxes, scores, labels):
                prediction_string += str(int(label)) + ' ' + str(score) + \
                    ' ' + str(box[0] * image_info['width']) + ' ' + \
                    str(box[1] * image_info['height']) + ' ' + \
                    str(box[2] * image_info['width']) + ' ' + \
                    str(box[3] * image_info['height']) + ' '

        prediction_strings.append(prediction_string)
        file_names.append(image_id)

    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = file_names
    submission.to_csv(save_pth, index=False)


if __name__ == '__main__':
    main()
