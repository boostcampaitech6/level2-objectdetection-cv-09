import os
from typing import Sequence

from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.utils import mkdir_or_exist

from mmdet.registry import HOOKS
from mmdet.structures import DetDataSample
import pandas as pd


@HOOKS.register_module()
class SubmissionHook(Hook):
    """
    Hook for submitting results. Saves test process prediction results.

    In the testing phase:

    1. Receives labels, scores, and bboxes from outputs and stores them.
    2. Get the img_path of outputs and save it in file_names.

    Args:
        prediction_strings (list): [label + ' ' + scores + ' ' + x1 + ' '
            + y1 + ' ' + x2 + ' ' + y2]를 추가한 list
        file_names (list): img_path를 추가한 list
        test_out_dir (str) : 저장할 경로
    """

    def __init__(self, test_out_dir='submit'):
        self.prediction_strings = []
        self.file_names = []
        self.test_out_dir = test_out_dir

    def after_test_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                        outputs: Sequence[DetDataSample]) -> None:
        """Run after every testing iterations.

        Args:
            runner (:obj:`Runner`): The runner of the testing process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`DetDataSample`]): A batch of data samples
                that contain annotations and predictions.
        """
        assert len(outputs) == 1, \
            'only batch_size=1 is supported while testing.'

        for output in outputs:
            prediction_string = ''
            labels = output.pred_instances.labels
            scores = output.pred_instances.scores
            bboxes = output.pred_instances.bboxes
            for label, score, box in zip(labels, scores, bboxes):
                box = box.cpu().numpy()
                prediction_string += str(int(label.cpu())) + ' ' + \
                    str(float(score.cpu())) + ' ' + str(box[0]) + ' ' + \
                    str(box[1]) + ' ' + str(box[2]) + ' ' + str(box[3]) + ' '
            self.prediction_strings.append(prediction_string)
            self.file_names.append(output.img_path[13:])

    def after_test(self, runner: Runner):
        """
        Run after testing

        Args:
            runner (:obj:`Runner`): The runner of the testing process.
        """
        if self.test_out_dir is not None:
            self.test_out_dir = os.path.join(
                runner.work_dir, runner.timestamp, self.test_out_dir
            )
            mkdir_or_exist(self.test_out_dir)

        submission = pd.DataFrame()
        submission['PredictionString'] = self.prediction_strings
        submission['image_id'] = self.file_names
        submission.to_csv(os.path.join(
            self.test_out_dir, 'submission.csv'), index=None
        )
        print('submission saved to {}'.format(os.path.join(
            self.test_out_dir, 'submission.csv')))
