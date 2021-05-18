#
# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import cv2
import torch
import kornia
import numpy as np
import matplotlib.pyplot as plt

from detectron2.data import detection_utils as utils
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import Metadata
from detectron2.utils.visualizer import Visualizer
from detectron2.structures.instances import Instances
from detectron2.structures.boxes import Boxes

import pytorch_lightning as pl

from itertools import product
from collections.abc import Iterable

def read_image(path):
    """
        Read image from disk.

        Parameters:
            path (str): Path to the image to read

        Returns:
            image (numpy.ndarray): Image in HWC format with pixels in [0, 255] range.
    """
    image = utils.read_image(path)

    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)

    return image


def read_images(*paths):
    """
        See `read_image`.
    """
    # FIXME: It would be nice if read_image comprehended lists and this function went away.
    return [read_image(path) for path in paths]


def image_to_tensor(image, gamma=1.):
    """
        Convert image to tensor with specified gamma correction.

        Parameters:
            image (numpy.ndarray or list[numpy.ndarray]): Images or list of images in HWC format with pixels in [0, 255] range.
            gamma (float): Gamma correction factor (pixels**gamma). (default: 1.)

        Returns:
            torch.Tensor: Tensors in NCHW format with values in [0, 1] range.
    """
    if isinstance(image, list):
        # FIXME: This will only work when images have the same shape
        image = np.array(image)

    if image.min() < 0 or image.max() > 255:
        raise Exception("Unusual image range [", x.min(), ",", x.max(), "]; expected [0, 255].")
    if image.max() <= 1:
        raise Exception("Unusual image range max", x.max(), "; expected more image pixel values to be greater than 1.")

    image = image.astype('float32') / 255.
    image = np.power(image, gamma)

    return kornia.image_to_tensor(image)


def tensor_to_image(tensor, gamma=1.):
    """
        Convert tensor to image with specified gamma correction. Rounds to nearest pixel.

        Parameters:
            tensor (torch.Tensor): Tensor in NCHW format with values in [0, 1] range.
            gamma (float): Inverse gamma correction factor for pixels**(1/gamma). (default: 1.)

        Returns:
            list[numpy.ndarray]: Image in HWC format with pixels in [0, 255] range.
    """
    if tensor.min() < 0 or tensor.max() > 1:
        raise Exception("Unusual tensor range [", tensor.min(), ",", tensor.max(), "]; expected [0, 1].")

    tensor = torch.pow(tensor, 1 / gamma)
    image = kornia.tensor_to_image(tensor)

    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)

    if len(image.shape) == 3:
        image = np.expand_dims(image, axis=0)

    image = np.round(image * 255.).astype('uint8')

    return image


def create_model(config_path, weights_path, device='cuda', score_thresh=None, iou_thresh=None):
    """
        Creates detectron2 model with specified weights, turns off all parameter gradients, and retrives model metadata.

        Parameters:
            config_path (str): Path to model configuration.
            weights_path (str): A path or url to weights.

        Returns:
            nn.Module: Model loaded with specified weights.
            detectron2.data.catalog.Metadata: Metadata associated with the model.
    """
    # Get configuration
    cfg = get_cfg()
    cfg.merge_from_file(config_path)

    # FIXME: Unsure whether we should take a cfg object and merge it, but this works for now
    if score_thresh is not None:
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    if iou_thresh is not None:
        # FIXME: Why is this a list?
        cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS = [iou_thresh]
    cfg.MODEL.DEVICE = str(device)

    # Create model to attack...
    model = build_model(cfg)

    # ...and load weights...
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(weights_path)

    # ...and turn of gradients.
    for param in model.parameters():
        param.requires_grad = False

    # Get metadata
    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

    return model, metadata


def composite_renders_on_backgrounds(renders, backgrounds):
    """
        Composites each render onto each background using alpha channel from render.

        Parameters:
            renders (torch.Tensor or list[torch.Tensor]): List of renders with alpha channel.
            backgrounds (torch.Tensor or list[Torch.Tensor]): List of images.

        Returns:
            list[torch.Tensor]: N*M images with N renders composited on M backgrounds.
    """
    images = []

    for render in renders:
        render_rgb = render[:3, :, :]
        render_alpha = render[3:, :, :]

        for background in backgrounds:
            # Gaussian blur alpha image
            #render_alpha = kornia.filters.gaussian_blur2d(render_alpha, (3, 3), (sigma, sigma))
            # TODO: Use box blur?

            # Composite onto background
            composite = render_rgb * render_alpha + background * (1 - render_alpha)

            # Gaussian blur composite image
            #composite = kornia.filters.gaussian_blur2d(composite, (3, 3), (sigma, sigma))

            # Composite blurred object into non-blurred background
            #composite = composite*render_alpha + background*(1 - render_alpha)

            #images[start+j:start+j+1].copy_(composite, non_blocking=True)
            images.append(composite)

    return images


def compute_bounding_boxes(alphas, multiple=1):
    """
        Computes bounding boxes using alpha channel. Possible to repeat each bounding box multiples.
        Repeating is useful when a render as been composited on multiple backgrounds since they all share the same
        bounding box.

        Note: This runs of the CPU since torch.where cannot run on the GPU.

        Parameters:
            alphas (torch.Tensor or list[torch.Tensor]): List of alpha channels where any pixels > 0.5 are assumed to be the object.
            mutliple (int): Number of times to repeat each bounding box. (default: 1)

        Returns:
            list[torch.Tensor]: List of bounding boxes in [xmin, ymin, xmax, ymax] format.
    """
    bboxes = []

    for alpha in alphas:
        assert(alpha.shape[0] == 1)

        # Binarize transformed mask using threshold and collapse rows and columns
        # For example, a binary image like this:
        #   00000
        #   01010
        #   00110
        #   00000
        # Will be collapsed into vectors:
        #  row = 01110
        #  col = 0110
        xs = (alpha >= 0.5).any(dim=-2)
        ys = (alpha >= 0.5).any(dim=-1)

        # Compute 4-point convex hull of collapsed, binarized, transformed mask.
        xs = xs.to('cpu')
        ys = ys.to('cpu')

        xmin = torch.where(xs, torch.arange(xs.shape[-1]), torch.as_tensor( xs.shape[-1])).argmin(-1, keepdims=True)
        xmax = torch.where(xs, torch.arange(xs.shape[-1]), torch.as_tensor(-xs.shape[-1])).argmax(-1, keepdims=True)
        ymin = torch.where(ys, torch.arange(ys.shape[-1]), torch.as_tensor( ys.shape[-1])).argmin(-1, keepdims=True)
        ymax = torch.where(ys, torch.arange(ys.shape[-1]), torch.as_tensor(-ys.shape[-1])).argmax(-1, keepdims=True)

        bbox = torch.cat([xmin, ymin, xmax, ymax], -1)

        for _ in range(multiple):
            bboxes.append(bbox)

    return bboxes


def create_inputs(images, gt_bboxes=None, gt_classes=None, input_format='BGR', gamma=1., transforms=None):
    """
        Creates inputs for Detectron2 model from images with specified groundtruth bounding boxes and class.
        Will convert image-channel format to specified format and add gamma correction to images.

        Parameters:
            images (list[torch.Tensor]): List of images in CHW format and channels in RGB format.
            gt_bboxes (list[torch.Tensor(N, 4)]): Groundtruth bounding boxes for each image. (default: None)
            gt_classes (list[torch.Tensor(N)]): Groundtruth class for all images. (default: None)
            input_format (str): Image channel format. (default: 'BGR')
            gamma (float): Gamma correction factor for each pixel**gamma. (default: 1.)

        Returns:
            list[dict]: List of dictionaries with 'image', 'height', 'width', and 'instances' (if groundtruth supplied).
    """
    batched_inputs = []

    for i, image in enumerate(images):
        inputs = {}

        # Convert to tensor
        if not isinstance(image, torch.Tensor):
            image = image_to_tensor(image)

        if image.min() < 0 or image.max() > 1:
            raise Exception("Unusual tensor range [", image.min(), ",", image.max(), "]; expected [0, 1].")

        # Flip channels as necessary
        if input_format == 'BGR':
            image = image.flip(0)

        # Gamma correct
        image = image.pow(gamma)

        # Quantize image
        image = torch.fake_quantize_per_tensor_affine(image, scale=1 / 255, zero_point=0, quant_min=0, quant_max=255)

        # Model wants image between [0, 255]
        image *= 255

        # Set height and width to the size of the original image (before resizing), which is the desired output image size.
        inputs['height'] = image.shape[1]
        inputs['width'] = image.shape[2]

        if transforms is not None:
            # CHW -> HWC for resizing purpose
            original_image = image.permute(1, 2, 0).cpu().detach().numpy().astype('uint8')
            resized_image = transforms.get_transform(original_image).apply_image(original_image)
            # HWC -> CHW for detectron2 model
            resized_image = resized_image.transpose(2, 0, 1)
            image = torch.as_tensor(resized_image.astype("float32"))

        inputs['image'] = image

        if gt_bboxes is not None and gt_classes is not None:
            inputs['instances'] = Instances(image.shape[1:3])
            inputs['instances'].gt_boxes = Boxes(gt_bboxes[i])
            inputs['instances'].gt_classes = gt_classes[i]

        batched_inputs.append(inputs)

    return batched_inputs


def print_tensor(*args):
    """
        Like `print` but expands numpy.ndarrays and torch.Tensors with id/ptr, device, shape, dtype, min, and max information.
    """
    print_args = []
    for arg in args:
        if isinstance(arg, np.ndarray):
            print_args.append(id(arg))
            print_args.append(arg.shape)
            print_args.append(arg.dtype)
            print_args.append(arg.min())
            print_args.append(arg.max())
        elif isinstance(arg, torch.Tensor):
            with torch.no_grad():
                print_args.append(arg.data_ptr())
                print_args.append(arg.device)
                print_args.append(arg.shape)
                print_args.append(arg.dtype)
                print_args.append(arg.min().cpu().numpy())
                print_args.append(arg.max().cpu().numpy())
        else:
            print_args.append(arg)

    print(*print_args)


def plot_images(images, bboxes=None, model=None, metadata=None, col_wrap=5, scale=0.25, gamma=1.):
    """
        Plot images using matplotlib. Add bounding boxes, if specified, or computes them using the specified model.

        Parameters:
            images (list(torch.Tensor) or list[numpy.ndarray]): List of images. Uses `tensor_to_image` to convert to Tensors.
            bboxes (list[torch.Tensor(4)]): List of bounding boxes to overlay on image. (default: None)
            model (torch.nn.Module): Model used to create bounding boxes. (default: None)
            metadata (detectron2.data.catalog.Metadata): Metadata used to annotate bounding boxes. (default: None)
            col_wrap (int): Number of columns before wrapping to next row. (default: 5)
            scale (float): Image scaling factor. (default: 0.25)
            gamma (float): Gamma correction factor for Tensor images. (default: 1.)

        Returns:
            Nothing. But will display matplotlib Figure with each image and their respective detections or bounding boxes.

    """
    if metadata is None:
        metadata = Metadata()

    # Get predictions
    predictions = []
    if model is not None:
        with torch.no_grad():
            batched_inputs = create_inputs(images, input_format=model.input_format, gamma=gamma)
            # Models are usually large so we send one image in at a time.
            for inputs in batched_inputs:
                preds = model.inference([inputs], do_postprocess=False)[0]
                predictions.append({'instances': preds.to('cpu')})
    elif bboxes is not None:
        for image, boxes in zip(images, bboxes):
            instances = Instances(image.shape[2:])
            instances.pred_boxes = Boxes(boxes)
            predictions.append({'instances': instances.to('cpu')})
    else:
        for image in images:
            predictions.append(None)

    # Plot visualizations
    for i, (image, preds) in enumerate(zip(images, predictions)):
        if isinstance(image, torch.Tensor):
            image = tensor_to_image(image, gamma=gamma)[0]

        if preds is not None:
            # Visualizer does not support RGBA imges
            if image.shape[-1] == 4:
                image = image[..., :3]

            image_viz = Visualizer(image, metadata, scale=scale)
            image_viz.draw_instance_predictions(predictions=preds['instances'])
            image = image_viz.get_output().get_image()
        else:
            # Scale image
            width, height, _ = image.shape
            image = cv2.resize(image, (int(round(height * scale + 1e-2)), int(round(width * scale + 1e-2))))

        if i == 0:
            if isinstance(images, list):
                nimages = len(images)
            else:
                nimages = images.shape[0]

            rows = ((nimages + col_wrap - 1) // col_wrap)
            cols = col_wrap
            size = (cols * image.shape[1] / 72, rows * image.shape[0] / 72)

            plt.figure(figsize=size, facecolor=(0.8, 0.8, 0.8))
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0.01, hspace=0., wspace=0.01)

        plt.subplot(rows, cols, i + 1)
        if len(image.shape) == 2 or image.shape[2] == 1:
            plt.imshow(image, cmap='gray')
        else:
            plt.imshow(image)
        plt.axis('off')

    plt.show()

def batch_yield(x, dim=0, size=0):
    """
    Yields batches of x over the given dimension(s) and size(s).

    Args:
        x (numpy.ndarray): Array to yield batches from.
        dim (int, Iterable[int]): Dimension of x to yield batches of.
        size (int, Iterable[int]): Length of batches to yield.

    Returns:
        Yields a tuple containing a batch of x and an index of x

    Notes:
        If size is 0, this function yields individual samples of x.

    Examples:
        >>> list(batch_yield(np.array([1, 2, 3, 4]))) #doctest: +NORMALIZE_WHITESPACE
        [(1, (0,)),
         (2, (1,)),
         (3, (2,)),
         (4, (3,))]
        >>> list(batch_yield(np.array([1, 2, 3, 4]), size=1)) #doctest: +NORMALIZE_WHITESPACE
        [(array([1]), (slice(0, 1, None),)),
         (array([2]), (slice(1, 2, None),)),
         (array([3]), (slice(2, 3, None),)),
         (array([4]), (slice(3, 4, None),))]
        >>> list(batch_yield(np.array([1, 2, 3, 4]), size=2)) #doctest: +NORMALIZE_WHITESPACE
        [(array([1, 2]), (slice(0, 2, None),)),
         (array([3, 4]), (slice(2, 4, None),))]
    """
    if not isinstance(dim, Iterable):
        dim = [dim]
    if not isinstance(size, Iterable):
        size = [size]

    assert len(dim) == len(size)

    # Construct indices of x
    indices = []
    for d, ds in enumerate(x.shape):
        # If this is a dimension we want to batch over...
        if d in dim:
            s = size[dim.index(d)]

            if s > 0:
                index = [slice(i, i+s) for i in range(0, ds, s)]
            else:
                index = range(0, ds, 1)
        else:
            index = [slice(None)]
        indices.append(index)

    # Pull out each batch using contructed indices and yield
    for index in product(*indices):
        x_batch = x[index]
        yield x_batch, index

def bmap(map_fn, *args, batch_dim=0, batch_size=1):
    """
    Maps args using map_fn over the given batch dimension(s) and batch size(s).

    Args:
        map_fn (func): A function taking exactly the number of args to map.
        *args (numpy.ndarray, ...): An array to map
        batch_dim (int, Iterable[int]): The dimension of args to batch over
        batch_size (int, Iterable[int]): The size of each batch

    Returns:
        args mapped by map_fn.

    Notes:
        This version of map works differently than Python's `map` function because
        it is batching and variadic. This allows one to focus on the concerns of the
        `map_fn` without having to account for batching itself. The variadicity enables
        one to supply different inputs to the `map_fn` over the specified batch dimension.

        If `map_fn` does not return an equal number of numpy.ndarrays as args, the
        mapped output will be truncated. If None is returned by map_fn, the output
        will be mapped to None.
    """
    if not isinstance(args, tuple):
        args = (args,)
    if not isinstance(batch_dim, Iterable):
        batch_dim = (batch_dim,)
    if not isinstance(batch_size, Iterable):
        batch_size = (batch_size,)

    # Ensure 0 <= batch_dim < len(args.shape)
    for dim in batch_dim:
        for i, arg in enumerate(args):
            if not (0 <= dim < len(arg.shape)):
                raise ValueError(f"batch_dim {dim} exceeds dimensions of arg[{i}] (dims={len(arg.shape)})")

    # Ensure 0 < batch_size
    for size in batch_size:
        if not (0 < size):
            raise ValueError(f"batch_size must be greater than 0: {batch_size}")

    # Wait to create this dynamically by using first output to get proper shape
    args_mapped = None

    # Yield on each batch for each arg
    batch_yields = [batch_yield(x, dim=batch_dim, size=batch_size) for x in args]

    for list_of_args_batch_index in zip(*batch_yields):
        # [(batch0, index0), ..., (batchN, indexN)] ->
        #  [(batch0, ..., batchN), (index0, ..., indexN)]
        args_batch, args_index = zip(*list_of_args_batch_index)

        # Call map function
        args_batch_mapped = map_fn(*args_batch)

        # Convert to tuple in case fn returns single item
        if not isinstance(args_batch_mapped, tuple):
            args_batch_mapped = (args_batch_mapped,)

        # Construct output array using existing batch dimension sizes but new transformed sizes
        if args_mapped is None:
            args_mapped = []
            for arg, arg_mapped in zip(args, args_batch_mapped):
                if arg_mapped is not None:
                    # Combine batch dimension sizes with...
                    shape = list(arg_mapped.shape)
                    # non-batch dimension sizes to get total output size
                    for d in batch_dim:
                        shape[d] = arg.shape[d]

                    # Copy and resize first sample to total output size
                    arg_mapped = np.resize(arg_mapped, shape)

                args_mapped.append(arg_mapped)

        # Save mapped values
        for arg_mapped, arg_index, arg_batch_mapped in zip(args_mapped, args_index, args_batch_mapped):
            if arg_mapped is None:
                continue
            arg_mapped[arg_index] = arg_batch_mapped

    # Make sure we maintain dimension sizes we batched over
    for arg, arg_mapped in zip(args, args_mapped):
        if arg_mapped is None:
            continue

        assert np.all(np.array(arg.shape)[list(batch_dim)] ==
                      np.array(arg_mapped.shape)[list(batch_dim)])

    # If only single output returned from map_fn, then return only that
    if len(args_mapped) == 1:
        args_mapped = args_mapped[0]
    return args_mapped
