#
# Copyright (C) 2020 Intel Corporation
#
# Licensed subject to the terms of the separately executed evaluation license
# agreement between Intel Corporation and you.
#
# DO NOT DISTRIBUTE. FOR INTERNAL USE ONLY.
#

import logging
import itertools
from collections import defaultdict
from typing import Optional

import numpy as np
from tqdm import tqdm

# Use our version of load_dataset since it supports dataset.kwargs
from oscar.scenarios.video_ucf101_scenario import _load_dataset as load_dataset

from armory.utils.config_loading import (
    load_model,
    load_attack,
    load_adversarial_dataset,
    load_defense_internal,
)

from armory.utils import metrics
from armory.scenarios.base import Scenario
# Unclear how to extract label strings from TFDS, so we import this.
from armory.data.adversarial.ucf101_mars_perturbation_and_patch_adversarial_112x112 import _LABELS

from scipy.special import softmax

logger = logging.getLogger(__name__)


class Ucf101(Scenario):
    # Decouple these loading utils from self._evaluate() so that we can reuse them in Jupyter.
    def _load_defenses(self, config, classifier):
        # Add support to chaining defenses, which will be supported by ART 1.4.
        defenses_config = config.get("defenses") or {}
        if defenses_config == {}:
            # Backward compatibility with single defense.
            defense_config = config.get("defense") or {}
            defenses_config = [defense_config]

        for defense_config in defenses_config:
            defense_type = defense_config.get("type")
            if defense_type in ["Preprocessor", "Postprocessor"]:
                logger.info(f"Applying internal {defense_type} defense to classifier")
                classifier = load_defense_internal(defense_config, classifier)

        return classifier

    def _load_classifier_with_defense(self, config):
        model_config = config["model"]
        classifier, preprocessing_fn = load_model(model_config)
        classifier = self._load_defenses(config, classifier)
        classifier.set_learning_phase(False)
        return classifier, preprocessing_fn

    # Load multiple attacks.
    def _load_attacks(self, config, classifier):
        attack_config = config["attack"]
        attack_type = attack_config.get("type")
        targeted = bool(attack_config.get("kwargs", {}).get("targeted"))
        if targeted and attack_config.get("use_label"):
            raise ValueError("Targeted attacks cannot have 'use_label'")
        if attack_type == "preloaded":
            raise Exception("This scenario automatically supports preloaded adversaries. Please specifiy an adaptive attack.")

        attack_kwargs = attack_config['kwargs']

        # The number of attacks is the longest list
        attack_count = 1
        for attack_kwarg_value in attack_kwargs.values():
            if isinstance(attack_kwarg_value, list):
                attack_count = max(attack_count, len(attack_kwarg_value))

        # Repeat every non-list item
        for attack_kwarg_name, attack_kwarg_value in attack_kwargs.items():
            if not isinstance(attack_kwarg_value, list):
                attack_kwargs[attack_kwarg_name] = itertools.repeat(attack_kwarg_value, attack_count)

        # Create new attack for every attack_kwarg
        attacks = {}
        attacks_kwargs = [dict(zip(attack_kwargs.keys(), attack_kwargs_values)) for attack_kwargs_values in zip(*attack_kwargs.values())]
        for i, attack_kwargs in enumerate(attacks_kwargs):
            attack_config['kwargs'] = attack_kwargs
            attack = load_attack(attack_config, classifier)
            attack.set_params(batch_size=1)

            # Create attack name from attack parameters
            attack_name = attack_config['name']
            for attack_param in attack.attack_params:
                # FIXME: A weird behavior since ART 1.3.3. PGD defines 'minimal' but never uses it.
                if not hasattr(attack, attack_param):
                    continue
                attack_value = getattr(attack, attack_param)
                attack_name += f"-{attack_param}-{attack_value}"

            # This will overwrite any existing attack with the same attack params
            attacks[attack_name] = attack

        logger.info('Attacks')
        logger.info('-------')
        for attack_name in attacks:
            logger.info(attack_name)

        return attacks

    def _load_test_data(self, config, preprocessing_fn, num_eval_batches=None):
        # Copy dict because load_adversarial_dataset modifies the dict
        dataset_config = config['dataset'].copy()

        if dataset_config['batch_size'] != 1:
            raise ValueError(
                "batch_size must be 1 for evaluation, due to variable length inputs.\n"
                "    If training, set config['model']['fit_kwargs']['fit_batch_size']"
            )

        # Rewrite dataset_config because load_dataset and load_adversarial_dataset are not unified
        if 'kwargs' not in dataset_config:
            dataset_config['kwargs'] = {}

        if 'adversarial' in dataset_config['name']:
            test_data = load_adversarial_dataset(
                dataset_config,
                epochs=1,
                split_type='adversarial',
                num_batches=num_eval_batches,
                shuffle_files=False,
             )
        else:
            test_data = load_dataset(
                dataset_config,
                epochs=1,
                split='test',
                num_batches=num_eval_batches,
                shuffle_files=False,
            )

        return test_data

    def _evaluate(self, config: dict, num_eval_batches: Optional[int], skip_benign: Optional[bool], skip_attack: Optional[bool]) -> dict:
        """
        Evaluate the config and return a results dict
        """

        classifier, preprocessing_fn = self._load_classifier_with_defense(config)

        # Evaluate the ART classifier on adversarial test examples
        logger.info("Generating or loading / testing adversarial examples...")

        attacks = self._load_attacks(config, classifier)

        test_data = self._load_test_data(config, preprocessing_fn, num_eval_batches=num_eval_batches)

        # Multiple attacks.
        adv_metrics_logger = {}
        for attack_name in attacks:
            adv_metrics_logger[attack_name] = metrics.MetricsLogger.from_config(config["metric"],
                                                                                skip_benign=skip_benign,
                                                                                skip_attack=skip_attack,
                                                                                targeted=False)

        # Check if we can run fast security curve.
        epsilons = [attack.eps for attack in attacks.values()]
        if epsilons != sorted(epsilons):
            raise ValueError("Fast security curve requires monotonically increasing power of an adversary.")

        y_benign_labels = []
        y_benign_pred_labels = []
        y_benign_pred_scores = []
        y_benign_pred_logits = []

        y_adv_pred_labels = defaultdict(list)
        y_adv_pred_scores = defaultdict(list)
        y_adv_pred_logits = defaultdict(list)

        pbar = tqdm(test_data, desc="Science")
        for x_benign, y_benign in pbar:
            assert(len(x_benign) == 1)
            assert(len(y_benign) == 1)

            y_benign = y_benign[0]

            # Benign
            if not skip_benign:
                y_benign_logits = classifier.predict(x_benign)[0]
                y_benign_scores = softmax(y_benign_logits)

                # Update every metric logger with benign results
                for attack_name in attacks:
                    adv_metrics_logger[attack_name].update_task(y_benign, y_benign_logits, adversarial=False)

                y_benign_labels.append(_LABELS[y_benign])
                y_benign_pred_labels.append(_LABELS[y_benign_logits.argmax()])
                y_benign_pred_scores.append(y_benign_scores.max().tolist())
                y_benign_pred_logits.append(y_benign_logits.tolist())
            else:
                # Set these to None so the attack can work
                y_benign_logits = None
                y_benign_scores = None

            # Adaptive Attack on Benign
            if not skip_attack:
                # TODO: Support targeted attacks.
                x_adv = x_benign
                y_adv_logits = y_benign_logits
                y_adv_scores = y_benign_scores

                for attack_name, attack in attacks.items():
                    # Run an attack only if the weaker adversarial example correctly classifies, or we don't have logits
                    if (y_adv_logits is None) or (y_adv_logits.argmax() == y_benign):
                        x_adv = attack.generate(x=x_adv) # Begin attack search from existing adversarial example
                        y_adv_logits = classifier.predict(x_adv)[0]
                        y_adv_scores = softmax(y_adv_logits)

                    adv_metrics_logger[attack_name].update_task(y_benign, y_adv_logits, adversarial=True)
                    adv_metrics_logger[attack_name].update_perturbation([x_benign], [x_adv])

                    y_adv_pred_labels[attack_name].append(_LABELS[y_adv_logits.argmax()])
                    y_adv_pred_scores[attack_name].append(y_adv_scores.max().tolist())
                    y_adv_pred_logits[attack_name].append(y_adv_logits.tolist())

            preview_msg = ""

            # Preview top-1 benign accuracy
            if not skip_benign:
               for adv_metric_logger in adv_metrics_logger.values():
                   benign_acc = float(adv_metric_logger.results()['benign_mean_categorical_accuracy']) * 100

               preview_msg += "B=%d%% " % (benign_acc)

            # Preview top-1 robust accuracy for each adaptive attacks.
            if not skip_attack:
                get_metric_top1_percentage = lambda m: float(m.results()['adversarial_mean_categorical_accuracy']) * 100
                for i, (attack_name, adv_metric_logger) in enumerate(adv_metrics_logger.items()):
                    adv_acc = get_metric_top1_percentage(adv_metric_logger)
                    preview_msg += "A%d=%d%% " % (i, adv_acc)

            pbar.set_description(preview_msg)
            pbar.refresh()

        pbar.close()

        # Combine results from each logger into single results dict
        results = {}

        for attack_name in attacks:
            adv_results = adv_metrics_logger[attack_name].results()
            adv_results = dict((f"{attack_name}_{key}" if 'benign' not in key else key, value) for key, value in adv_results.items())
            results.update(adv_results)

        for key, value in results.items():
            task_type, metric_name = key.split('_', 1)

            if 'accuracy' in metric_name:
                logger.info(f"Average {metric_name} on {task_type} test examples: {value:.2%}")
            else:
                logger.info(f"Average {metric_name} on {task_type} test examples: {value}")

        # Add dataframe to results
        results['attacks'] = list(attacks.keys())
        results['groundtruth'] = y_benign_labels

        results['benign_predictions'] = y_benign_pred_labels
        results['benign_scores'] = y_benign_pred_scores
        #results['benign_logits'] = y_benign_pred_logits

        for attack_name, attack in attacks.items():
            results[f"{attack_name}_predictions"] = y_adv_pred_labels[attack_name]
            results[f"{attack_name}_scores"] = y_adv_pred_scores[attack_name]
            #results[f"{attack_name}_logits"] = y_adv_pred_logits[attack_name]

        return results
