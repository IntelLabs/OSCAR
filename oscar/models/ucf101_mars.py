import numpy as np
import torch
from art.classifiers import PyTorchClassifier
from armory.baseline_models.pytorch.ucf101_mars import get_art_model, make_model, DEVICE, preprocessing_fn
from MARS.dataset.preprocess_data import get_mean as get_ucf101_mean

# Run predict() in one batch and no_grad().
# Note: The batch_size is ignored, so no need to change the scenario definition.
class EfficientPyTorchClassifier(PyTorchClassifier):
    def predict(self, x, batch_size=128, **kwargs):
        """
        Perform prediction for a batch of inputs.

        :param x: Test set.
        :type x: `np.ndarray`
        :param batch_size: Size of batches.
        :type batch_size: `int`
        :return: Array of predictions of shape `(nb_inputs, nb_classes)`.
        :rtype: `np.ndarray`
        """

        # Apply preprocessing
        x_preprocessed, _ = self._apply_preprocessing(x, y=None, fit=False)

        # Run prediction in one batch and no_grad().
        x_pth = torch.from_numpy(x_preprocessed)
        with torch.no_grad():
            x_cuda = x_pth.to(self._device)
            results_cuda = self._model(x_cuda)
            del x_cuda
            results = results_cuda[0].detach().cpu().numpy()
            del results_cuda[0]
            del results_cuda

        # Apply postprocessing
        predictions = self._apply_postprocessing(preds=results, fit=False)

        return predictions


# Make clip_values works with RGBA input.
# FIXME: Make a switch between 3 and 4 channels so that the model
#        also works with the official data pipeline which gives 3-channel input.
def get_art_model(model_kwargs, wrapper_kwargs, weights_file):
    # Note: The clipping parameters need to know the shape of input to preprocessing here.
    # FIXME: We may find a better place for clipping parameters outside this function.
    if "preprocess_input_channels" not in model_kwargs:
        preprocess_input_channels = 3
    else:
        preprocess_input_channels = model_kwargs["preprocess_input_channels"]
        # The following make_model() will complain unrecognized option "preprocess_input_channels".
        del model_kwargs["preprocess_input_channels"]

    model, optimizer = make_model(weights_file=weights_file, **model_kwargs)
    model.to(DEVICE)

    activity_means = get_ucf101_mean('activitynet')
    if preprocess_input_channels == 4:
        activity_means += [0.]
    activity_means = np.array(activity_means, dtype=np.float32)

    wrapped_model = EfficientPyTorchClassifier(
        model=model,
        loss=torch.nn.CrossEntropyLoss(),
        optimizer=optimizer,
        input_shape=(3, 16, 112, 112),
        nb_classes=101,
        **wrapper_kwargs,
        clip_values=(
            np.transpose(np.zeros((16, 112, 112, preprocess_input_channels)) - activity_means, (3, 0, 1, 2)),
            np.transpose(
                255.0 * np.ones((16, 112, 112, preprocess_input_channels)) - activity_means, (3, 0, 1, 2)
            ),
        ),
    )
    return wrapped_model
