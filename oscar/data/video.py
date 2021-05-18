#
# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#
import logging
import pickle
from pathlib import Path
from collections import defaultdict
from typing import Union, Optional, Callable, Tuple, List, Any, Dict

import torch
import torchvision
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import make_dataset as _make_dataset

logger = logging.getLogger(__name__)


class VideoFolder(ImageFolder):
    def __init__(
        self,
        root: Union[str, Path],
        video_loader: Callable[[str], Any],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        is_valid_video: Optional[Callable[[str], bool]] = None,
    ):
        # BEGIN: Monkey patch make_dataset
        torchvision.datasets.folder.make_dataset = VideoFolder.cached_make_dataset
        super().__init__(root,
                         transform=transform,
                         target_transform=target_transform,
                         loader=video_loader,
                         is_valid_file=is_valid_file)
        torchvision.datasets.folder.make_dataset = _make_dataset
        # END: Monkey patch make_dataset

        # Reduce individual samples (i.e., frames) to videos and make labels strings again
        self.samples = self.samples_to_videos(self.samples,
                                              is_valid_video=is_valid_video,
                                              idx_to_class=self.classes)
        self.targets = [s[1] for s in self.samples]

    @staticmethod
    def samples_to_videos(samples, is_valid_video=None, idx_to_class=None):
        # samples = [(path, label), ...]
        paths = defaultdict(list)
        labels = {}

        for path, label in samples:
            label = idx_to_class[label]

            # XXX: Would be nice to use Path.parent, but it's slow :(
            video = '/'.join(path.split('/')[:-1])
            paths[video].append(path)

            if video not in labels:
                labels[video] = label

            assert labels[video] == label

        # Construct output like samples but with many paths (i.e., frames) per label
        videos = []
        for video in paths:
            if is_valid_video is None or is_valid_video(video):
                videos.append((paths[video], labels[video]))

        return videos

    @staticmethod
    def cached_make_dataset(
        directory: str,
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        cache = Path(directory) / "make_dataset.cache"

        if cache.exists():
            logger.info("Loading dataset cache: %s", cache)
            instances = pickle.load(cache.open('rb'))
        else:
            logger.info("Making dataset cache: %s", cache)
            instances = _make_dataset(directory, class_to_idx, extensions, is_valid_file)
            logger.info("Saving dataset cache: %s", cache)
            pickle.dump(instances, cache.open('wb'))

        return instances


class StackingVideoLoader(object):
    def __init__(self, patterns: List[str], sampler: Optional[Callable[[int], int]] = None):
        self.patterns = patterns
        self.sampler = sampler
        if self.sampler is None:
            self.sampler = range

    def __call__(self, paths: List[str]) -> torch.Tensor:
        # XXX: Assumes all paths have the same parent path
        parent = Path(paths[0]).parent

        # XXX: This can be very slow. How can we cache these values?
        lengths = [self._find_pattern_length(parent, pattern, paths) for pattern in self.patterns]
        length = min(lengths)

        # Sample and read frame tensors
        indices = self.sampler(length)
        frames = [self._read_frame(parent, i) for i in indices]
        video = torch.stack(frames)

        # XXX: Make this configurable so it works with existing stuff (i.e., don't return a tuple, just a video)
        return { 'x': video, 'parent': parent, 'indices': indices }

    @staticmethod
    def _find_pattern_length(parent: Path, pattern: str, paths: List[str]) -> int:
        length_min = 0
        length_max = len(paths)

        # Binary search for video length
        while length_min + 1 != length_max:
            length = (length_max + length_min) // 2

            if str(parent / pattern.format(length)) in paths:
                length_min = length
            else:
                length_max = length

        # Make a choice between min and max
        length = length_min
        if str(parent / pattern.format(length_max)) in paths:
            length = length_max

        return length

    def _read_frame(self, parent: Path, i: int):
        channels = []

        # Read pattern for current frame
        for pattern in self.patterns:
            path = str(parent / pattern.format(i + 1)) # 1-indexed
            img = torchvision.io.read_image(path).float()
            channels.append(img)

        frame = torch.cat(channels, dim=0)

        return frame


class ClipSampler(object):
    def __init__(self, length: int = 16, step: int = 1):
        self.sample_length = (length - 1) * step + 1
        self.sample_step = step

    def __call__(self, length: int) -> int:
        assert length >= self.sample_length

        # XXX: Does this sample near the end of the video?
        sample_start = torch.randint(length - self.sample_length, size=(1,))[0].item()

        return range(sample_start, sample_start + self.sample_length, self.sample_step)


class MiddleClipSampler(object):
    def __init__(self, length: int = 16, step: int = 1):
        self.sample_length = (length - 1) * step + 1
        self.sample_step = step

    def __cal__(self, length: int) -> int:
        assert length >= self.sample_length

        sample_start = length // 2 - self.sample_length // 2

        return range(sample_start, sample_start + self.sample_length, self.sample_step)
