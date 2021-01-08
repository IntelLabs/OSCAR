#
# Copyright (C) 2020 Intel Corporation
#
# Licensed subject to the terms of the separately executed evaluation license
# agreement between Intel Corporation and you.
#

import os
import logging
import pkg_resources

import torch
import numpy as np

from art.classifiers.classifier import Classifier, ClassifierNeuralNetwork, ClassifierGradients

from oscar.utils import print_tensor, \
                        image_to_tensor, \
                        create_inputs, \
                        create_model, \
                        compute_bounding_boxes

from detectron2.model_zoo import get_config_file
from detectron2.utils.events import EventStorage
from detectron2.data import MetadataCatalog
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ROI_HEADS_REGISTRY
from detectron2.modeling.roi_heads import Res5ROIHeads

import big_transfer.bit_pytorch.models
from big_transfer import bit_pytorch

from armory.data.utils import maybe_download_weights_from_s3

logger = logging.getLogger(__name__)

@BACKBONE_REGISTRY.register()
class ResNetV2(Backbone):
    def __init__(self, cfg, input_shape):
        super().__init__()

        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        model = bit_pytorch.models.KNOWN_MODELS['BiT-M-R50x1'](self.num_classes)

        self.stem = model.root
        self.res2 = model.body.block1
        self.res3 = model.body.block2
        self.res4 = model.body.block3

        self._out_features = cfg.MODEL.RESNETS.OUT_FEATURES

        self._out_feature_channels = {}
        self._out_feature_channels['stem'] = self.stem.conv.out_channels
        self._out_feature_channels['res2'] = self.res2.unit01.downsample.out_channels
        self._out_feature_channels['res3'] = self.res3.unit01.downsample.out_channels
        self._out_feature_channels['res4'] = self.res4.unit01.downsample.out_channels

        self._out_feature_strides = {}
        self._out_feature_strides['stem'] = self.stem.conv.stride[0] * self.stem.pool.stride
        self._out_feature_strides['res2'] = self.res2.unit01.downsample.stride[0]*self._out_feature_strides['stem']
        self._out_feature_strides['res3'] = self.res3.unit01.downsample.stride[0]*self._out_feature_strides['res2']
        self._out_feature_strides['res4'] = self.res4.unit01.downsample.stride[0]*self._out_feature_strides['res3']

    def forward(self, x):
        outputs = {}

        x = self.stem(x)
        if 'stem' in self._out_features:
            outputs['stem'] = x

        x = self.res2(x)
        if 'res2' in self._out_features:
            outputs['res2'] = x

        x = self.res3(x)
        if 'res3' in self._out_features:
            outputs['res3'] = x

        x = self.res4(x)
        if 'res4' in self._out_features:
            outputs['res4'] = x

        return outputs

@ROI_HEADS_REGISTRY.register()
class Res5V2ROIHeads(Res5ROIHeads):
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)

    def _build_res5_block(self, cfg):
        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        model = bit_pytorch.models.KNOWN_MODELS['BiT-M-R50x1'](self.num_classes)

        res5 = model.body.block4
        out_channels = model.body.block4.unit01.downsample.out_channels

        return res5, out_channels

class Detectron2Classifier(Classifier, ClassifierNeuralNetwork, ClassifierGradients):
    def __init__(self, config_file, weights_file, clip_values=None, preprocessing_defences=None, postprocessing_defences=None, preprocessing=None, channel_index=None, device='gpu'):
        super(Detectron2Classifier, self).__init__(
            clip_values=clip_values,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing,
            channel_index=channel_index
        )

        if device == 'cpu' or not torch.cuda.is_available():
            self._device = 'cpu'
        else:
            self._device = 'cuda'

        # Register resisc45 metadata
        # XXX: The order of these classes matters, and, in particular, are in a different order
        #      than what Armory expects. Armory expects these to be sorted alphabetically.
        MetadataCatalog.get('resisc45_val').set(thing_classes=['medium_residential', 'intersection', 'cloud', 'commercial_area', 'industrial_area', 'airport', 'railway', 'lake', 'desert', 'ground_track_field', 'tennis_court', 'harbor', 'stadium', 'mobile_home_park', 'freeway', 'bridge', 'rectangular_farmland', 'wetland', 'sparse_residential', 'snowberg', 'airplane', 'railway_station', 'baseball_diamond', 'circular_farmland', 'mountain', 'terrace', 'beach', 'parking_lot', 'overpass', 'dense_residential', 'island', 'basketball_court', 'roundabout', 'ship', 'storage_tank', 'forest', 'golf_course', 'meadow', 'river', 'thermal_power_station', 'palace', 'runway', 'church', 'chaparral', 'sea_ice'])

        self._model, self._metadata = create_model(config_file, weights_file, device=self._device, score_thresh=0.0)
        self.preprocessing = None
        self.preprocessing_defences = None

    def nb_classes(self):
        return len(self._metadata.thing_classes)

    def save(self, filename, path=None):
        raise NotImplementedError

    def set_learning_phase(self, train):
        if train:
            self._model.train()
        else:
            self._model.eval()

    def _random_scale_and_pad(self, image, scale=(0.25, 3.0), size=(800, 800)):
        # Randomly scale image with uniformed sampled scale factor
        image = image[None]
        scale = (scale[1] - scale[0]) * np.random.sample() + scale[0]
        image = torch.nn.functional.interpolate(image, scale_factor=(scale, scale))
        image = image[0]

        # Randomly pad image to size
        pad_w = np.random.choice(range(size[1] - image.shape[2]))
        pad_h = np.random.choice(range(size[0] - image.shape[1]))
        padding = (pad_w, size[1] - image.shape[2] - pad_w, pad_h, size[0] - image.shape[1] - pad_h)
        image = torch.nn.functional.pad(image, padding, value=0.)

        return image

    def predict(self, x, batch_size=1, **kwargs):
        # TODO: Should _apply_preprocessing

        # Convert numpy.ndarrays to Tensors
        images = image_to_tensor(x).to(self._device)

        # Preprocess images using differentiable operations
        #preprocessed_images = [self._random_scale_and_pad(img) for img in images]
        preprocessed_images = images

        # Create inputs for Detectron2 model
        batched_inputs = create_inputs(preprocessed_images)

        # Run inference on examples
        batched_outputs = self._model.inference(batched_inputs)

        # Turn outputs into results
        results = np.zeros((x.shape[0], self.nb_classes()), dtype=np.float32)
        for i, outputs in enumerate(batched_outputs):
            instances = outputs['instances']

            instances.pred_classes = self.map_classes(instances.pred_classes)

            if len(instances) > 0:
                # Take most confident prediction
                results[i, instances.pred_classes[0]] = 1
            else:
                # XXX: What happens if there are no detections? Doesn't seem like there is a way
                #      to obstain from making a prediction.
                logger.warn('Nothing detected in image!')

        # TODO: Should _apply_postprocessing

        return results

    def map_classes(self, pred_classes):
        CLASS_MAP = [20,  5, 22, 31, 26, 15, 43, 42, 23,  2,  3, 29,  8, 35, 14, 36,  9,
                     11,  4,  1, 30,  7, 37,  0, 13, 24, 28, 40, 27,  6, 21, 16, 38, 32,
                     41, 44, 33, 19, 18, 12, 34, 10, 25, 39, 17]

        return [CLASS_MAP.index(pred_class) for pred_class in pred_classes]

    def loss_gradient(self, x, y, **kwargs):
        # TODO: Should _apply_preprocessing

        # Convert numpy.ndarrays to Tensors
        images = image_to_tensor(x).to(self._device)
        images.requires_grad_(True)

        # Preprocess images using differentiable operations
        #preprocessed_images = [self._random_scale_and_pad(img) for img in images]
        preprocessed_images = images

        # Create inputs for Detectron2 model
        #gt_bboxes = compute_bounding_boxes(preprocessed_images)
        gt_bboxes = [torch.Tensor([[0, 0, image.shape[1], image.shape[2]]]).long() for image in preprocessed_images]
        gt_classes = [torch.Tensor([self.map_classes([np.argmax(onehot)])[0]]).long() for onehot in y]

        batched_inputs = create_inputs(preprocessed_images, gt_bboxes=gt_bboxes, gt_classes=gt_classes)

        # Run foward on examples with gradients
        with EventStorage():
            self._model.train()
            batched_outputs = self._model.forward(batched_inputs)
            self._model.eval()

        # Compute gradients using the classification loss for now
        loss = batched_outputs['loss_cls']
        loss.backward()

        grads = images.grad.detach().cpu().numpy().transpose(0, 2, 3, 1)
        grads = self._apply_preprocessing_gradient(x, grads)

        assert grads.shape == x.shape

        return grads

def get_art_model(model_kwargs, wrapper_kwargs, weights_file):
    config_file = model_kwargs['config_file']

    if config_file.startswith('detectron2://'):
        config_file = config_file[len('detectron2://'):]
        config_file = get_config_file(config_file)

    if config_file.startswith('oscar://'):
        config_file = config_file[len('oscar://'):]
        config_file = pkg_resources.resource_filename('oscar.model_zoo', config_file)

    if weights_file.startswith('oscar://'):
        weights_file = weights_file[len('oscar://'):]
        weights_file = pkg_resources.resource_filename('oscar.model_zoo', weights_file)

    if weights_file.startswith('armory://'):
        weights_file = weights_file[len('armory://'):]
        weights_file = maybe_download_weights_from_s3(weights_file)

    logger.info('config_file = %s', config_file)
    logger.info('weights_file = %s', weights_file)

    classifier = Detectron2Classifier(config_file, weights_file)

    return classifier
