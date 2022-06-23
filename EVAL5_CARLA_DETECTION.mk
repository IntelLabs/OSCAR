submission/%.pt: $(MODEL_ZOO)/%.pt | submission/
> cp $< $@

submission/INTL_carla_obj_det_rgb_adv_trained.json: $(ARMORY_SCENARIOS)/eval5/carla_object_detection/carla_obj_det_adversarialpatch_undefended.json \
                                               submission/INTL_carla_obj_det_rgb_adv_trained.pt | submission/
> cat $< | $(JQ) '.model.weights_file = "INTL_carla_obj_det_rgb_adv_trained.pt"' > $@

submission/INTL_carla_obj_det_mm_depth_only.json: $(ARMORY_SCENARIOS)/eval5/carla_object_detection/carla_obj_det_multimodal_adversarialpatch_undefended.json \
                                               submission/INTL_carla_obj_det_mm_depth_only.pt | submission/
> cat $< | $(JQ) '.model.weights_file = "INTL_carla_obj_det_mm_depth_only.pt"' \
         | $(JQ) '.model.model_kwargs.num_classes = 4' \
         | ${JQ} '.model.model_kwargs.min_size = 960' \
         | ${JQ} '.model.model_kwargs.max_size = 1280' \
         | $(JQ) '.model.module = "oscar.models.detection.carla_object_detection_frcnn"' \
         | $(JQ) '.model.name = "get_art_model"' \
         | $(JQ) '.model.model_kwargs.backbone.name = "resnet50_fpn"' \
         | $(JQ) '.model.model_kwargs.input_slice = [-3, null]' \
         | $(JQ) '.sysconfig.external_github_repo = ["IntelLabs/OSCAR@gard-eval5"]' > $@

submission/INTL_carla_obj_det_multimodal_distance_estimator.json: $(ARMORY_SCENARIOS)/eval5/carla_object_detection/carla_obj_det_multimodal_adversarialpatch_undefended.json | submission/
> cat $< | $(JQ) '.model.module = "oscar.models.detection.carla_multimodality_object_detection_frcnn"' \
         | $(JQ) '.model.name = "get_art_model_mm"' \
         | $(JQ) '.model.wrapper_kwargs.distance_estimator.intercept = [12.73797679731033657, 9.43340793824440716, 15.30049750594996933]' \
         | $(JQ) '.model.wrapper_kwargs.distance_estimator.slope = [-1.31948423473154029, -0.66098790523127123, -1.92653181473068357]' \
         | $(JQ) '.model.wrapper_kwargs.distance_estimator.penality_ratio = 0.8' \
         | $(JQ) '.model.wrapper_kwargs.distance_estimator.logarithmic = true' \
         | $(JQ) '.model.wrapper_kwargs.distance_estimator.reciprocal = false' \
         | $(JQ) '.model.wrapper_kwargs.distance_estimator.confidence = [0.7, 0.7, 0.7]' \
         | $(JQ) '.sysconfig.external_github_repo = ["IntelLabs/OSCAR@gard-eval5"]' > $@

.PHONY: carla_detection_submission
carla_detection_submission: submission/INTL_carla_obj_det_rgb_adv_trained.json submission/INTL_carla_obj_det_mm_depth_only.json submission/INTL_carla_obj_det_multimodal_distance_estimator.json
> @echo "Created CARLA Detection submission!"

run_carla_detection_submission: carla_detection_submission | .venv
> cp submission/INTL_carla_obj_det_rgb_adv_trained.pt $(ARMORY_SAVED_MODEL_DIR)
> $(POETRY) run armory run submission/INTL_carla_obj_det_rgb_adv_trained.json $(ARGS)
> cp submission/INTL_carla_obj_det_mm_depth_only.pt $(ARMORY_SAVED_MODEL_DIR)
> $(POETRY) run armory run submission/INTL_carla_obj_det_mm_depth_only.json $(ARGS)
> $(POETRY) run armory run submission/INTL_carla_obj_det_multimodal_distance_estimator.json $(ARGS)
