#
# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#
import logging
from pathlib import Path
from typing import Union, Optional, Callable, Tuple, List, Any, Dict

from oscar.data.video import VideoFolder
from oscar.data.video import StackingVideoLoader

logger = logging.getLogger(__name__)


class UCF101Dataset(VideoFolder):
    def __init__(
        self,
        frames_root: Union[str, Path],
        annotation_dir: Union[str, Path],
        file_patterns: List[str],
        train: bool = True,
        fold: int = 1,
        transform: Optional[Callable] = None,
        sampler: Optional[object] = None,
    ):
        super().__init__(frames_root,
                         video_loader=StackingVideoLoader(file_patterns, sampler=sampler),
                         transform=transform,
                         target_transform=UCF101Labeler(Path(annotation_dir) / "classInd.txt"),
                         is_valid_video=UCF101Filter(Path(annotation_dir), train=train, fold=fold))


class UCF101Labeler(object):
    def __init__(self, annotation_path: Union[str, Path]):
        with Path(annotation_path).open() as f:
            # 1 ApplyEyeMakeup
            lines = [line.strip().split(' ') for line in f]
            lines = map(lambda x: (x[1].lower(), int(x[0])), lines)

            self.label_to_idx = dict(lines)

    def __getitem__(self, label: str):
        return self.label_to_idx[label.lower()] - 1

    def __call__(self, label):
        return self.label_to_idx[label.lower()] - 1


class UCF101Filter(object):
    def __init__(self, annotation_dir: str, train: bool = True, fold: int = 1):
        annotation_path = Path(annotation_dir) / ("%slist%02d.txt" % ("train" if train else "test", fold))

        with annotation_path.open() as f:
            # train: ApplyEyeMakeup/v_ApplyEyeMakeup_g08_c01.avi 1
            # test: ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi
            self.filter_list = [line.strip().split(' ')[0].split('/')[1].split('.')[0].lower() for line in f]

    def __call__(self, path):
        # {root}/ApplyEyeMakeup/v_ApplyEyeMakeup_g08_c01
        name = path.split('/')[-1].lower()

        return name in self.filter_list
