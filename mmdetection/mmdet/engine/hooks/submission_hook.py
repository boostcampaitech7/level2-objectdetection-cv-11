# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Sequence

from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.utils import mkdir_or_exist

from mmdet.registry import HOOKS
from mmdet.structures import DetDataSample
import pandas as pd

@HOOKS.register_module()
class SubmissionHook(Hook):


    def __init__(self, test_out_dir='submit'):
        self.outputs_data = []
        self.test_out_dir = test_out_dir

    def after_test_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                        outputs: Sequence[DetDataSample]) -> None:

        for output in outputs:
            prediction_string = ''
            labels = output.pred_instances.labels.cpu().numpy()
            scores = output.pred_instances.scores.cpu().numpy()
            bboxes = output.pred_instances.bboxes.cpu().numpy()

            for label, score, bbox in zip(labels, scores, bboxes):
                prediction_string += f'{int(label)} {float(score)} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]} '

            self.outputs_data.append([prediction_string, output.img_path[-13:]])

    def after_test(self, runner: Runner):
        if self.test_out_dir is not None:
            self.test_out_dir = osp.join(runner.work_dir, runner.timestamp,
                                         self.test_out_dir)
            mkdir_or_exist(self.test_out_dir)

        prediction_strings = []
        file_names = []
        for predict, file_name in self.outputs_data:
            prediction_strings.append(predict)
            file_names.append(file_name)

        submission = pd.DataFrame()
        submission['PredictionString'] = prediction_strings
        submission['image_id'] = file_names
        submission.to_csv(osp.join(self.test_out_dir, 'submission.csv'), index=None)
        print('submission saved to {}'.format(osp.join(self.test_out_dir, 'submission.csv')))