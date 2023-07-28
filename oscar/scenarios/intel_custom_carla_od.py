#
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import numpy as np
from armory import metrics
from armory.instrument import GlobalMeter, Meter
from armory.instrument.instrument import ResultsLogWriter
from armory.logs import log
from armory.metrics.task import _intersection_over_union
from armory.scenarios.carla_object_detection import CarlaObjectDetectionTask

# Wrap tpr_fpr output in a list so Hub doesn't overwrite it at each sample
tpr_fpr = metrics.get("tpr_fpr")


def list_wrap_tpr_fpr(actual_conditions, predicted_conditions):
    return [tpr_fpr(actual_conditions, predicted_conditions)]


def mean_tpr_fpr(tpr_results):
    """Finalize the result of metrics.task.tpr_fpr on multiple examples by computing the metric on the cumulative data.  The computation is copied from that
    function.

    tpr_results is a list of dicts.  Each dict is the output of metrics.task.tpr_fpr
    """

    true_positives = sum(d["true_positives"] for d in tpr_results)
    true_negatives = sum(d["true_negatives"] for d in tpr_results)
    false_positives = sum(d["false_positives"] for d in tpr_results)
    false_negatives = sum(d["false_negatives"] for d in tpr_results)

    actual_positives = true_positives + false_negatives
    if actual_positives > 0:
        true_positive_rate = true_positives / actual_positives
        false_negative_rate = false_negatives / actual_positives
    else:
        true_positive_rate = false_negative_rate = float("nan")

    actual_negatives = true_negatives + false_positives
    if actual_negatives > 0:
        false_positive_rate = false_positives / actual_negatives
        true_negative_rate = true_negatives / actual_negatives
    else:
        false_positive_rate = true_negative_rate = float("nan")

    if true_positives or false_positives or false_negatives:
        f1_score = true_positives / (true_positives + 0.5 * (false_positives + false_negatives))
    else:
        f1_score = float("nan")

    return dict(
        true_positives=true_positives,
        true_negatives=true_negatives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        true_positive_rate=true_positive_rate,
        false_positive_rate=false_positive_rate,
        false_negative_rate=false_negative_rate,
        true_negative_rate=true_negative_rate,
        f1_score=f1_score,
    )


class CustomIntelCarlaObjectDetectionTask(CarlaObjectDetectionTask):
    """A custom scenario to accommodate a post-processor that filters the model's output boxes to remove hallucinations."""

    def load_metrics(self):
        super().load_metrics()

        meters = [
            # mAP on filtered predictions
            GlobalMeter(
                "adv_AP_per_class_filtered_predictions",
                metrics.get("object_detection_AP_per_class"),
                "scenario.y",
                "scenario.y_pred_adv_filtered",
            ),
            # True/False Positive Rate on post-process box filter
            Meter(
                "postprocess_box_filter",
                list_wrap_tpr_fpr,
                "scenario.true_hallucinations",
                "scenario.predicted_hallucinations",
                final=mean_tpr_fpr,
                final_name="postprocess_box_filter",
                record_final_only=True,
            ),
        ]
        for meter in meters:
            self.hub.connect_meter(meter)
        self.hub.connect_writer(ResultsLogWriter(format_string="{name}: {result}"), meters=meters, default=False)

    def hallucination_index_mask(self, y, y_pred, iou_threshold=0.5, score_threshold=0.5):
        # For each box in y_pred, mark it as hallucinated if there is no matching box in y.
        # Returns boolean array, size of y_pred, where True indicates the box is a hallucination.

        # Initialize array for tracking
        index = np.ones(len(y_pred["boxes_raw"]))

        # Loop over pred boxes, mark which ones overlap with any GT boxes
        for i, box in enumerate(y_pred["boxes_raw"]):
            overlaps = np.array([_intersection_over_union(box, b) > iou_threshold for b in y["boxes"]])
            if sum(overlaps) > 0:
                index[i] = 0

        return index.astype(bool)

    def run_attack(self):
        self._check_x("run_attack")
        if not hasattr(self, "y_patch_metadata"):
            raise AttributeError("y_patch_metadata attribute does not exist. Please set --skip-attack if using " "CARLA train set")
        self.hub.set_context(stage="attack")
        x, y = self.x, self.y

        with self.profiler.measure("Attack"):
            if self.use_label:
                y_target = y
            elif self.targeted:
                y_target = self.label_targeter.generate(y)
            else:
                y_target = None

            x_adv = self.attack.generate(
                x=x,
                y=y_target,
                y_patch_metadata=self.y_patch_metadata,
                **self.generate_kwargs,
            )

        # Ensure that input sample isn't overwritten by model
        self.hub.set_context(stage="adversarial")
        x_adv.flags.writeable = False
        y_pred_adv = self.model.predict(x_adv, **self.predict_kwargs)

        # Predict which boxes are hallucinations and filter them out
        assert len(y_pred_adv) == 1  # batch size is 1
        # If the detection key is not there, can't use this scenario
        assert "predicted_hallucinations" in y_pred_adv[0].keys(), "Custom scenario requires detection-capable model!"

        predicted_hallucinations = y_pred_adv[0]["predicted_hallucinations"]
        y_pred_adv_filtered = [
            {
                "boxes": y_["boxes_raw"][~predicted_hallucinations],
                "scores": y_["scores_raw"][~predicted_hallucinations],
                "labels": y_["labels_raw"][~predicted_hallucinations],
            }
            for y_ in y_pred_adv
        ]

        # Get the true hallucinations
        true_hallucinations = self.hallucination_index_mask(y[0], y_pred_adv[0])
        self.probe.update(
            x_adv=x_adv,
            y_pred_adv=y_pred_adv,
            y_pred_adv_filtered=y_pred_adv_filtered,
            predicted_hallucinations=predicted_hallucinations,
            true_hallucinations=true_hallucinations,
        )

        if self.targeted:
            self.probe.update(y_target=y_target)

        # If using multimodal input, add a warning if depth channels are perturbed
        if x.shape[-1] == 6:
            if (x[..., 3:] != x_adv[..., 3:]).sum() > 0:
                log.warning("Adversarial attack perturbed depth channels")

        self.x_adv, self.y_target, self.y_pred_adv = x_adv, y_target, y_pred_adv
