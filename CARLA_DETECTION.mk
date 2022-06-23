CARLA_OBJDET = $(DATASETS)/carla_obj_det/annotations/instances_train.json $(DATASETS)/carla_obj_det/annotations/instances_val.json

$(CARLA_OBJDET):
> $(info $(YELLOW)Couldn't find CARLA dataset in $(@D). Run the commands below, if you really want to generate this data!$(RESET))
> $(warning $(RED)Do not run these commands on spr-gpu*, unless you are absolutely sure what you are doing!$(RESET))
> $(info $(POETRY) run python3 -m oscar.tools.tfds_to_coco --name carla_obj_det_train --split train --output_dir $(DATASETS)/carla_obj_det/train --annotations_name instances_train.json)
> $(info $(POETRY) run python3 -m oscar.tools.tfds_to_coco --name carla_obj_det_dev --split dev --output_dir $(DATASETS)/carla_obj_det/val --annotations_name instances_val.json)
> $(info $(POETRY) run python3 -m oscar.tools.tfds_to_coco --name carla_obj_det_test --split small+medium+large --output_dir $(DATASETS)/carla_obj_det/test --annotations_name instances_test.json)
> $(error Missing dataset)

$(DT2_MODEL_ZOO)/CARLA-Armory: | $(DT2_MODEL_ZOO)
> mkdir -p $@
> touch $@

$(DT2_MODEL_ZOO)/carla_rgb_weights.pkl: $(MODEL_ZOO)/carla_rgb_weights.pt | $(DT2_MODEL_ZOO)
> $(POETRY) run python3 -m oscar.tools.convert-torchvision-to-d2 $< $@

$(DT2_MODEL_ZOO)/CARLA-Armory/baseline.yaml: $(DT2_MODEL_ZOO)/CARLA-Armory/faster_rcnn-imagenet_C2-R_50-FPN-3x.yaml $(DT2_MODEL_ZOO)/carla_rgb_weights.pkl
> cat $< | $(YQ) '.INPUT.FORMAT = "RGB"' \
         | $(YQ) '.MODEL.WEIGHTS = "$(DT2_MODEL_ZOO)/carla_rgb_weights.pkl"' \
         | $(YQ) '.MODEL.RESNETS.STRIDE_IN_1X1 = false' \
         | $(YQ) '.MODEL.PIXEL_MEAN = [123.675, 116.28, 103.53]' \
         | $(YQ) '.MODEL.PIXEL_STD = [58.395, 57.12, 57.375]' > $@


$(DT2_MODEL_ZOO)/CARLA-Armory/faster_rcnn-imagenet_C2-R_50-FPN-3x.yaml: $(DT2_CONFIGS)/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml $(DT2_MODEL_ZOO)/Base-RCNN-FPN.yaml $(DT2_MODEL_ZOO)/CARLA-Armory
> cat $< | $(YQ) '.TEST.EVAL_PERIOD = 2500' \
         | $(YQ) '.DATASETS.TRAIN = ["carla_obj_det_train"]' \
         | $(YQ) '.DATASETS.TEST = ["carla_obj_det_val"]' \
         | $(YQ) '.MODEL.ROI_HEADS.NUM_CLASSES = 3' > $@

$(DT2_MODEL_ZOO)/CARLA-Armory/faster_rcnn-imagenet_C2-R_50-FPN-1x.yaml: $(DT2_MODEL_ZOO)/CARLA-Armory/faster_rcnn-imagenet_C2-R_50-FPN-3x.yaml
> cat $< | $(YQ) '.SOLVER.BASE_LR = 0.02' \
         | $(YQ) '.SOLVER.STEPS = [60000, 80000]' \
         | $(YQ) '.SOLVER.GAMMA = 0.1' \
         | $(YQ) '.SOLVER.MAX_ITER = 90000'> $@

# This diverges because base_lr is too high (0.04)
$(DT2_MODEL_ZOO)/CARLA-Armory/faster_rcnn-imagenet_C2-R_50-FPN-1x-highbaselr.yaml: $(DT2_MODEL_ZOO)/CARLA-Armory/faster_rcnn-imagenet_C2-R_50-FPN-3x.yaml
> cat $< | $(YQ) '.SOLVER.BASE_LR = 0.04' \
         | $(YQ) '.SOLVER.STEPS = [60000, 80000]' \
         | $(YQ) '.SOLVER.GAMMA = 0.02' \
         | $(YQ) '.SOLVER.MAX_ITER = 90000'> $@

$(DT2_MODEL_ZOO)/CARLA-Armory/faster_rcnn-imagenet_C2-R_50-FPN-1x-lowbaselr.yaml: $(DT2_MODEL_ZOO)/CARLA-Armory/faster_rcnn-imagenet_C2-R_50-FPN-3x.yaml
> cat $< | $(YQ) '.SOLVER.BASE_LR = 0.01' \
         | $(YQ) '.SOLVER.STEPS = [60000, 80000]' \
         | $(YQ) '.SOLVER.GAMMA = 0.02' \
         | $(YQ) '.SOLVER.MAX_ITER = 90000'> $@

$(DT2_MODEL_ZOO)/CARLA-Armory/faster_rcnn-imagenet_C2-R_50-FPN-1x-highcosine.yaml: $(DT2_MODEL_ZOO)/CARLA-Armory/faster_rcnn-imagenet_C2-R_50-FPN-3x.yaml
> cat $< | $(YQ) '.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"' \
         | $(YQ) '.SOLVER.WARMUP_METHOD = "linear"' \
         | $(YQ) '.SOLVER.WARMUP_ITERS = 45000' \
         | $(YQ) '.SOLVER.WARMUP_FACTOR = 0.001' \
         | $(YQ) '.SOLVER.BASE_LR = 0.16' \
         | $(YQ) '.SOLVER.MAX_ITER = 90000'> $@

$(DT2_MODEL_ZOO)/CARLA-Armory/faster_rcnn-imagenet_C2-R_50-FPN-1x-cosine.yaml: $(DT2_MODEL_ZOO)/CARLA-Armory/faster_rcnn-imagenet_C2-R_50-FPN-3x.yaml
> cat $< | $(YQ) '.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"' \
         | $(YQ) '.SOLVER.WARMUP_METHOD = "linear"' \
         | $(YQ) '.SOLVER.WARMUP_ITERS = 45000' \
         | $(YQ) '.SOLVER.WARMUP_FACTOR = 0.001' \
         | $(YQ) '.SOLVER.BASE_LR = 0.08' \
         | $(YQ) '.SOLVER.MAX_ITER = 90000'> $@

$(DT2_MODEL_ZOO)/CARLA-Armory/faster_rcnn-imagenet_C2-R_50-FPN-1x-lowcosine.yaml: $(DT2_MODEL_ZOO)/CARLA-Armory/faster_rcnn-imagenet_C2-R_50-FPN-3x.yaml
> cat $< | $(YQ) '.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"' \
         | $(YQ) '.SOLVER.WARMUP_METHOD = "linear"' \
         | $(YQ) '.SOLVER.WARMUP_ITERS = 45000' \
         | $(YQ) '.SOLVER.WARMUP_FACTOR = 0.001' \
         | $(YQ) '.SOLVER.BASE_LR = 0.04' \
         | $(YQ) '.SOLVER.MAX_ITER = 90000'> $@

# This diverges
$(DT2_MODEL_ZOO)/CARLA-Armory/faster_rcnn-imagenet_C2-R_50-FPN-1x-earlycosine.yaml: $(DT2_MODEL_ZOO)/CARLA-Armory/faster_rcnn-imagenet_C2-R_50-FPN-3x.yaml
> cat $< | $(YQ) '.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"' \
         | $(YQ) '.SOLVER.WARMUP_METHOD = "linear"' \
         | $(YQ) '.SOLVER.WARMUP_ITERS = 20000' \
         | $(YQ) '.SOLVER.WARMUP_FACTOR = 0.001' \
         | $(YQ) '.SOLVER.BASE_LR = 0.08' \
         | $(YQ) '.SOLVER.MAX_ITER = 90000'> $@

$(DT2_MODEL_ZOO)/CARLA-Armory/faster_rcnn-imagenet_C2-R_50-FPN-0.5x.yaml: $(DT2_MODEL_ZOO)/CARLA-Armory/faster_rcnn-imagenet_C2-R_50-FPN-3x.yaml
> cat $< | $(YQ) '.SOLVER.BASE_LR = 0.02' \
         | $(YQ) '.SOLVER.STEPS = [30000, 40000]' \
         | $(YQ) '.SOLVER.GAMMA = 0.1' \
         | $(YQ) '.SOLVER.MAX_ITER = 45000'> $@

# This diverges
$(DT2_MODEL_ZOO)/CARLA-Armory/faster_rcnn-imagenet_C2-R_50-FPN-0.5x-highcosine.yaml: $(DT2_MODEL_ZOO)/CARLA-Armory/faster_rcnn-imagenet_C2-R_50-FPN-3x.yaml
> cat $< | $(YQ) '.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"' \
         | $(YQ) '.SOLVER.WARMUP_METHOD = "linear"' \
         | $(YQ) '.SOLVER.WARMUP_ITERS = 22500' \
         | $(YQ) '.SOLVER.WARMUP_FACTOR = 0.001' \
         | $(YQ) '.SOLVER.BASE_LR = 0.16' \
         | $(YQ) '.SOLVER.MAX_ITER = 45000'> $@

$(DT2_MODEL_ZOO)/CARLA-Armory/faster_rcnn-imagenet_C2-R_50-FPN-0.5x-cosine.yaml: $(DT2_MODEL_ZOO)/CARLA-Armory/faster_rcnn-imagenet_C2-R_50-FPN-3x.yaml
> cat $< | $(YQ) '.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"' \
         | $(YQ) '.SOLVER.WARMUP_METHOD = "linear"' \
         | $(YQ) '.SOLVER.WARMUP_ITERS = 22500' \
         | $(YQ) '.SOLVER.WARMUP_FACTOR = 0.001' \
         | $(YQ) '.SOLVER.BASE_LR = 0.08' \
         | $(YQ) '.SOLVER.MAX_ITER = 45000'> $@

$(DT2_MODEL_ZOO)/CARLA-Armory/faster_rcnn-imagenet_C2-R_50-FPN-0.5x-lowcosine.yaml: $(DT2_MODEL_ZOO)/CARLA-Armory/faster_rcnn-imagenet_C2-R_50-FPN-3x.yaml
> cat $< | $(YQ) '.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"' \
         | $(YQ) '.SOLVER.WARMUP_METHOD = "linear"' \
         | $(YQ) '.SOLVER.WARMUP_ITERS = 22500' \
         | $(YQ) '.SOLVER.WARMUP_FACTOR = 0.001' \
         | $(YQ) '.SOLVER.BASE_LR = 0.04' \
         | $(YQ) '.SOLVER.MAX_ITER = 45000'> $@

# This diverges
$(DT2_MODEL_ZOO)/CARLA-Armory/faster_rcnn-imagenet_C2-R_50-FPN-0.5x-earlycosine.yaml: $(DT2_MODEL_ZOO)/CARLA-Armory/faster_rcnn-imagenet_C2-R_50-FPN-3x.yaml
> cat $< | $(YQ) '.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"' \
         | $(YQ) '.SOLVER.WARMUP_METHOD = "linear"' \
         | $(YQ) '.SOLVER.WARMUP_ITERS = 20000' \
         | $(YQ) '.SOLVER.WARMUP_FACTOR = 0.001' \
         | $(YQ) '.SOLVER.BASE_LR = 0.08' \
         | $(YQ) '.SOLVER.MAX_ITER = 45000'> $@

$(DT2_MODEL_ZOO)/CARLA-Armory/faster_rcnn-robust_l2_eps5_imagenet_C2-R_50-FPN-3x.yaml: $(DT2_MODEL_ZOO)/CARLA-Armory/faster_rcnn-imagenet_C2-R_50-FPN-3x.yaml $(IMAGENET_MODEL_ZOO)/resnet50_l2_eps5.pkl
> cat $< | $(YQ) '.MODEL.WEIGHTS = "$(IMAGENET_MODEL_ZOO)/resnet50_l2_eps5.pkl"' \
         | $(YQ) '.MODEL.RESNETS.STRIDE_IN_1X1 = false' \
         | $(YQ) '.MODEL.PIXEL_MEAN = [123.675, 116.280, 103.530]' \
         | $(YQ) '.MODEL.PIXEL_STD= [58.395, 57.120, 57.375]' \
         | $(YQ) '.INPUT.FORMAT = "RGB"' > $@

$(DT2_MODEL_ZOO)/CARLA-Armory/faster_rcnn-robust_l2_eps5_imagenet_C3-R_50-FPN-3x.yaml: $(DT2_MODEL_ZOO)/CARLA-Armory/faster_rcnn-robust_l2_eps5_imagenet_C2-R_50-FPN-3x.yaml
> cat $< | $(YQ) '.MODEL.BACKBONE.FREEZE_AT = 3' > $@

$(DT2_MODEL_ZOO)/CARLA-Armory/faster_rcnn-robust_l2_eps5_imagenet_C4-R_50-FPN-3x.yaml: $(DT2_MODEL_ZOO)/CARLA-Armory/faster_rcnn-robust_l2_eps5_imagenet_C2-R_50-FPN-3x.yaml
> cat $< | $(YQ) '.MODEL.BACKBONE.FREEZE_AT = 4' > $@

$(DT2_MODEL_ZOO)/CARLA-Armory/faster_rcnn-robust_l2_eps5_imagenet_C5-R_50-FPN-3x.yaml: $(DT2_MODEL_ZOO)/CARLA-Armory/faster_rcnn-robust_l2_eps5_imagenet_C2-R_50-FPN-3x.yaml
> cat $< | $(YQ) '.MODEL.BACKBONE.FREEZE_AT = 5' > $@


$(DT2_MODEL_ZOO)/CARLA-Armory/faster_rcnn-robust_l2_eps5_imagenet_C2-R_50_W4-FPN-3x.yaml: $(DT2_MODEL_ZOO)/CARLA-Armory/faster_rcnn-robust_l2_eps5_imagenet_C2-R_50-FPN-3x.yaml $(IMAGENET_MODEL_ZOO)/wide_resnet50_4_l2_eps5.pkl
> cat $< | $(YQ) '.MODEL.WEIGHTS = "$(IMAGENET_MODEL_ZOO)/wide_resnet50_4_l2_eps5.pkl"' \
         | $(YQ) '.MODEL.RESNETS.WIDTH_PER_GROUP = 256' > $@

$(DT2_MODEL_ZOO)/CARLA-Armory/faster_rcnn-robust_l2_eps5_imagenet_C3-R_50_W4-FPN-3x.yaml: $(DT2_MODEL_ZOO)/CARLA-Armory/faster_rcnn-robust_l2_eps5_imagenet_C2-R_50_W4-FPN-3x.yaml
> cat $< | $(YQ) '.MODEL.BACKBONE.FREEZE_AT = 3' > $@

$(DT2_MODEL_ZOO)/CARLA-Armory/faster_rcnn-robust_l2_eps5_imagenet_C4-R_50_W4-FPN-3x.yaml: $(DT2_MODEL_ZOO)/CARLA-Armory/faster_rcnn-robust_l2_eps5_imagenet_C2-R_50_W4-FPN-3x.yaml
> cat $< | $(YQ) '.MODEL.BACKBONE.FREEZE_AT = 4' > $@

$(DT2_MODEL_ZOO)/CARLA-Armory/faster_rcnn-robust_l2_eps5_imagenet_C5-R_50_W4-FPN-3x.yaml: $(DT2_MODEL_ZOO)/CARLA-Armory/faster_rcnn-robust_l2_eps5_imagenet_C2-R_50_W4-FPN-3x.yaml
> cat $< | $(YQ) '.MODEL.BACKBONE.FREEZE_AT = 5' > $@


$(DT2_MODEL_ZOO)/CARLA-Armory/faster_rcnn-random-R_50_gn-FPN-9x.yaml: $(DT2_MODEL_ZOO)/CARLA-Armory/faster_rcnn-imagenet_C2-R_50-FPN-3x.yaml
> cat $< | $(YQ) '.MODEL.WEIGHTS = ""' \
         | $(YQ) '.MODEL.PIXEL_STD = [57.375, 57.12, 58.395]' \
         | $(YQ) '.MODEL.RESNETS.STRIDE_IN_1X1 = false' \
         | $(YQ) '.MODEL.RESNETS.NORM = "GN"' \
         | $(YQ) '.MODEL.FPN.NORM = "GN"' \
         | $(YQ) '.MODEL.ROI_BOX_HEAD.NAME = "FastRCNNConvFCHead"' \
         | $(YQ) '.MODEL.ROI_BOX_HEAD.NUM_CONV = 4' \
         | $(YQ) '.MODEL.ROI_BOX_HEAD.NUM_FC = 1' \
         | $(YQ) '.MODEL.ROI_BOX_HEAD.NORM = "GN"' \
         | $(YQ) '.MODEL.ROI_MASK_HEAD.NORM = "GN"' \
         | $(YQ) '.MODEL.BACKBONE.FREEZE_AT = 0' \
         | $(YQ) '.SOLVER.IMS_PER_BATCH = 64' \
         | $(YQ) '.SOLVER.STEPS = [187500, 197500]' \
         | $(YQ) '.SOLVER.MAX_ITER = 202500' \
         | $(YQ) '.SOLVER.BASE_LR = 0.08' > $@

# 4x base_lr
$(DT2_MODEL_ZOO)/CARLA-Armory/faster_rcnn-random-R_50_gn-FPN-9x-cosine.yaml: $(DT2_MODEL_ZOO)/CARLA-Armory/faster_rcnn-random-R_50_gn-FPN-9x.yaml
> cat $< | $(YQ) '.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"' \
         | $(YQ) '.SOLVER.WARMUP_METHOD = "linear"' \
         | $(YQ) '.SOLVER.WARMUP_ITERS = 45000' \
         | $(YQ) '.SOLVER.WARMUP_FACTOR = 0.001' \
         | $(YQ) '.SOLVER.BASE_LR = 0.32' \
         | $(YQ) '.SOLVER.MAX_ITER = 90000'> $@

# 8x base_lr
$(DT2_MODEL_ZOO)/CARLA-Armory/faster_rcnn-random-R_50_gn-FPN-9x-highcosine.yaml: $(DT2_MODEL_ZOO)/CARLA-Armory/faster_rcnn-random-R_50_gn-FPN-9x.yaml
> cat $< | $(YQ) '.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"' \
         | $(YQ) '.SOLVER.WARMUP_METHOD = "linear"' \
         | $(YQ) '.SOLVER.WARMUP_ITERS = 45000' \
         | $(YQ) '.SOLVER.WARMUP_FACTOR = 0.001' \
         | $(YQ) '.SOLVER.BASE_LR = 0.64' \
         | $(YQ) '.SOLVER.MAX_ITER = 90000'> $@

# 2x base_lr
$(DT2_MODEL_ZOO)/CARLA-Armory/faster_rcnn-random-R_50_gn-FPN-9x-lowcosine.yaml: $(DT2_MODEL_ZOO)/CARLA-Armory/faster_rcnn-random-R_50_gn-FPN-9x.yaml
> cat $< | $(YQ) '.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"' \
         | $(YQ) '.SOLVER.WARMUP_METHOD = "linear"' \
         | $(YQ) '.SOLVER.WARMUP_ITERS = 45000' \
         | $(YQ) '.SOLVER.WARMUP_FACTOR = 0.001' \
         | $(YQ) '.SOLVER.BASE_LR = 0.16' \
         | $(YQ) '.SOLVER.MAX_ITER = 90000'> $@

$(DT2_MODEL_ZOO)/CARLA-IntelMini:
> mkdir -p $@

$(DT2_MODEL_ZOO)/CARLA-IntelMini/%.yaml: $(DT2_MODEL_ZOO)/CARLA-Armory/%.yaml | $(DT2_MODEL_ZOO)/CARLA-IntelMini
> cat $< | $(YQ) '.DATASETS.TRAIN = ["intel_obj_det_minitrain"]' \
         | $(YQ) '.DATASETS.TEST = ["intel_obj_det_minival"]' > $@

$(DT2_MODEL_ZOO)/CARLA-Intel:
> mkdir -p $@

$(DT2_MODEL_ZOO)/CARLA-Intel/%.yaml: $(DT2_MODEL_ZOO)/CARLA-Armory/%.yaml | $(DT2_MODEL_ZOO)/CARLA-Intel
> cat $< | $(YQ) '.DATASETS.TRAIN = ["intel_obj_det_train"]' \
         | $(YQ) '.DATASETS.TEST = ["intel_obj_det_val"]' > $@

$(DT2_MODEL_ZOO)/CARLA-IntelTwoWheeledAsVehicle:
> mkdir -p $@

$(DT2_MODEL_ZOO)/CARLA-IntelTwoWheeledAsVehicle/%.yaml: $(DT2_MODEL_ZOO)/CARLA-Armory/%.yaml | $(DT2_MODEL_ZOO)/CARLA-IntelTwoWheeledAsVehicle
> cat $< | $(YQ) '.DATASETS.TRAIN = ["intel_twowheeled_as_vehicle_obj_det_train"]' \
         | $(YQ) '.DATASETS.TEST = ["intel_twowheeled_as_vehicle_obj_det_val"]' > $@

$(DT2_MODEL_ZOO)/CARLA-IntelTwoWheeledAsPedestrian:
> mkdir -p $@

$(DT2_MODEL_ZOO)/CARLA-IntelTwoWheeledAsPedestrian/%.yaml: $(DT2_MODEL_ZOO)/CARLA-Armory/%.yaml | $(DT2_MODEL_ZOO)/CARLA-IntelTwoWheeledAsPedestrian
> cat $< | $(YQ) '.DATASETS.TRAIN = ["intel_twowheeled_as_pedestrian_obj_det_train"]' \
         | $(YQ) '.DATASETS.TEST = ["intel_twowheeled_as_pedestrian_obj_det_val"]' > $@

$(DT2_MODEL_ZOO)/%/CARLA-Armory/metrics.json: $(DT2_MODEL_ZOO)/%/model_final.pth | .venv
> $(DT2_TRAIN) --resume --eval-only --config-file $(DT2_MODEL_ZOO)/$*/config.yaml $(ARGS) MODEL.WEIGHTS $< DATASETS.TEST \(\"carla_obj_det_val\",\) OUTPUT_DIR $(DT2_MODEL_ZOO)/$*/CARLA-Armory

$(DT2_MODEL_ZOO)/%/CARLA-Intel/metrics.json: $(DT2_MODEL_ZOO)/%/model_final.pth | .venv
> $(DT2_TRAIN) --resume --eval-only --config-file $(DT2_MODEL_ZOO)/$*/config.yaml $(ARGS) MODEL.WEIGHTS $< DATASETS.TEST \(\"full_intel_obj_det_val\",\) OUTPUT_DIR $(DT2_MODEL_ZOO)/$*/CARLA-Intel

$(DT2_MODEL_ZOO)/%/CARLA-IntelTwoWheeledAsVehicle/metrics.json: $(DT2_MODEL_ZOO)/%/model_final.pth | .venv
> $(DT2_TRAIN) --resume --eval-only --config-file $(DT2_MODEL_ZOO)/$*/config.yaml $(ARGS) MODEL.WEIGHTS $< DATASETS.TEST \(\"intel_twowheeled_as_vehicle_obj_det_val\",\) OUTPUT_DIR $(DT2_MODEL_ZOO)/$*/CARLA-IntelTwoWheeledAsVehicle

$(DT2_MODEL_ZOO)/%/CARLA-IntelTwoWheeledAsPedestrian/metrics.json: $(DT2_MODEL_ZOO)/%/model_final.pth | .venv
> $(DT2_TRAIN) --resume --eval-only --config-file $(DT2_MODEL_ZOO)/$*/config.yaml $(ARGS) MODEL.WEIGHTS $< DATASETS.TEST \(\"intel_twowheeled_as_pedestrian_obj_det_val\",\) OUTPUT_DIR $(DT2_MODEL_ZOO)/$*/CARLA-IntelTwoWheeledAsPedestrian


$(SCENARIOS)/carla_obj_det_dpatch_%.json: $(ARMORY_SCENARIOS)/carla_obj_det_dpatch_%.json | $(SCENARIOS)/
> cat $< | $(JQ) '.scenario.export_samples = 30' \
         | $(JQ) '.dataset.name = "carla_obj_det_test"' \
         | $(JQ) '.dataset.eval_split = "test"' > $@

$(SCENARIOS)/carla_obj_det_dpatch_fpn.json: $(SCENARIOS)/carla_obj_det_dpatch_undefended.json
> cat $< | $(JQ) '.model.module = "oscar.models.detection.carla_object_detection_frcnn"' \
         | $(JQ) '.model.name = "get_art_model"' \
         | $(JQ) '.model.model_kwargs.backbone.name = "resnet50gn_fpn"' \
         | $(JQ) '.model.model_kwargs.rpn_score_thresh = 0.5' \
         | $(JQ) '.model.model_kwargs.box_roi_pool = "RoIAlignV2"' \
         | $(JQ) '.model.model_kwargs.image_mean = [0.485, 0.456, 0.406]' \
         | $(JQ) '.model.model_kwargs.image_std = [0.229, 0.224, 0.225]' \
         | $(JQ) '.model.weights_file = "carla_rgb_fpn_weights.pt"' > $@

$(SCENARIOS)/carla_obj_det_dpatch_pfn.json: $(SCENARIOS)/carla_obj_det_dpatch_fpn.json
> cat $< | $(JQ) '.model.model_kwargs.backbone.name = "resnet50gn_pfn"' \
         | $(JQ) '.model.weights_file = "carla_rgb_pfn_weights.pt"' > $@

$(SCENARIOS)/carla_obj_det_multimodal_dpatch_%.json: $(ARMORY_SCENARIOS)/carla_obj_det_multimodal_dpatch_%.json | $(SCENARIOS)/
> cat $< | $(JQ) '.scenario.export_samples = 30' \
         | $(JQ) '.dataset.name = "carla_obj_det_test"' \
         | $(JQ) '.dataset.eval_split = "test"' > $@

$(SCENARIOS)/carla_obj_det_multimodal_dpatch_naive_undefended.json: $(SCENARIOS)/carla_obj_det_multimodal_dpatch_undefended.json
> cat $< | $(JQ) '.model.module = "oscar.models.detection.carla_object_detection_frcnn"' \
         | $(JQ) '.model.name = "get_art_model"' \
         | $(JQ) '.model.model_kwargs.backbone.name = "multimodal_naive"' \
         | $(JQ) '.model.model_kwargs.image_mean = [0.485, 0.456, 0.406, 0.485, 0.456, 0.406]' \
         | $(JQ) '.model.model_kwargs.image_std = [0.229, 0.224, 0.225, 0.229, 0.224, 0.225]' \
         | $(JQ) '.model.model_kwargs.weights_file = .model.weights_file' \
         | $(JQ) '.model.weights_file = null' > $@

$(SCENARIOS)/carla_obj_det_multimodal_dpatch_naive_defended.json: $(SCENARIOS)/carla_obj_det_multimodal_dpatch_defended.json
> cat $< | $(JQ) '.model.module = "oscar.models.detection.carla_object_detection_frcnn"' \
         | $(JQ) '.model.name = "get_art_model"' \
         | $(JQ) '.model.model_kwargs.backbone.name = "multimodal_robust"' \
         | $(JQ) '.model.model_kwargs.image_mean = [0.485, 0.456, 0.406, 0.485, 0.456, 0.406]' \
         | $(JQ) '.model.model_kwargs.image_std = [0.229, 0.224, 0.225, 0.229, 0.224, 0.225]' \
         | $(JQ) '.model.model_kwargs.weights_file = .model.weights_file' \
         | $(JQ) '.model.weights_file = null' > $@

$(SCENARIOS)/carla_obj_det_multimodal_dpatch_fpn.json: $(SCENARIOS)/carla_obj_det_multimodal_dpatch_undefended.json
> cat $< | $(JQ) '.model.module = "oscar.models.detection.carla_object_detection_frcnn"' \
         | $(JQ) '.model.name = "get_art_model"' \
         | $(JQ) '.model.model_kwargs.backbone.name = "resnet50gn_fpn"' \
         | $(JQ) '.model.model_kwargs.rpn_score_thresh = 0.5' \
         | $(JQ) '.model.model_kwargs.box_roi_pool = "RoIAlignV2"' \
         | $(JQ) '.model.model_kwargs.backbone.in_channels = 4' \
         | $(JQ) '.model.model_kwargs.image_mean = [0.485, 0.456, 0.406, 0.4498]' \
         | $(JQ) '.model.model_kwargs.image_std = [0.229, 0.224, 0.225, 0.3186]' \
         | $(JQ) '.model.model_kwargs.weights_file = "carla_rgbd_fpn_weights.pt"' \
         | $(JQ) '.model.weights_file = null' > $@

$(SCENARIOS)/carla_obj_det_multimodal_dpatch_pfn.json: $(SCENARIOS)/carla_obj_det_multimodal_dpatch_fpn.json
> cat $< | $(JQ) '.model.model_kwargs.backbone.name = "resnet50gn_pfn"' \
         | $(JQ) '.model.model_kwargs.weights_file = "carla_rgbd_pfn_weights.pt"' > $@

$(SCENARIOS)/carla_obj_det_multimodal_dpatch_%_depth_proposals_fpn.json: $(SCENARIOS)/carla_obj_det_multimodal_dpatch_%.json
> cat $< | $(JQ) '.model.model_kwargs.rpn.backbone.in_channels = 1' \
         | $(JQ) '.model.model_kwargs.rpn.backbone.name = "resnet50gn_fpn"' \
         | $(JQ) '.model.model_kwargs.rpn.rpn_score_thresh = 0.5' \
         | $(JQ) '.model.model_kwargs.rpn.box_roi_pool = "RoIAlignV2"' \
         | $(JQ) '.model.model_kwargs.rpn.image_mean = [0.4498]' \
         | $(JQ) '.model.model_kwargs.rpn.image_std = [0.3186]' \
         | $(JQ) '.model.model_kwargs.rpn.interpolation = "nearest"' \
         | $(JQ) '.model.model_kwargs.rpn.weights_file = "carla_depth_fpn_weights.pt"' > $@

$(SCENARIOS)/carla_obj_det_multimodal_dpatch_%_depth_proposals_pfn.json: $(SCENARIOS)/carla_obj_det_multimodal_dpatch_%_depth_proposals_fpn.json
> cat $< | $(JQ) '.model.model_kwargs.rpn.backbone.name = "resnet50gn_pfn"' \
         | $(JQ) '.model.model_kwargs.rpn.weights_file = "carla_depth_pfn_weights.pt"' > $@

$(SCENARIOS)/carla_obj_det_multimodal_dpatch_%_resize2x.json: $(SCENARIOS)/carla_obj_det_multimodal_dpatch_%.json
> cat $< | $(JQ) '.model.model_kwargs.min_size = 1200' \
         | $(JQ) '.model.model_kwargs.max_size = 1600' > $@

# Submission
submission/INTL_carla_rgb_fpn_weights.pt: $(MODEL_ZOO)/carla_rgb_fpn_weights.pt | submission/
> cp $< $@

submission/INTL_carla_obj_det_dpatch_fpn.json: $(SCENARIOS)/carla_obj_det_dpatch_fpn.json \
                                               submission/INTL_carla_rgb_fpn_weights.pt | submission/
> cat $< | $(JQ) '.model.weights_file = "INTL_carla_rgb_fpn_weights.pt"' \
         | $(JQ) '.sysconfig.external_github_repo = ["IntelLabs/OSCAR@gard-eval4", "colour-science/colour"]' > $@


submission/INTL_carla_rgb_pfn_weights.pt: $(MODEL_ZOO)/carla_rgb_pfn_weights.pt | submission/
> cp $< $@

submission/INTL_carla_obj_det_dpatch_pfn.json: $(SCENARIOS)/carla_obj_det_dpatch_pfn.json \
                                               submission/INTL_carla_rgb_pfn_weights.pt | submission/
> cat $< | $(JQ) '.model.weights_file = "INTL_carla_rgb_pfn_weights.pt"' \
         | $(JQ) '.sysconfig.external_github_repo = ["IntelLabs/OSCAR@gard-eval4", "colour-science/colour"]' > $@


submission/INTL_carla_rgbd_fpn_depth_proposals_weights.pt: $(MODEL_ZOO)/carla_rgbd_fpn_depth_proposals_weights.pt | submission/
> cp $< $@

submission/INTL_carla_obj_det_multimodal_dpatch_fpn_depth_proposals.json: $(SCENARIOS)/carla_obj_det_multimodal_dpatch_fpn_depth_proposals_fpn.json \
                                                                          submission/INTL_carla_rgbd_fpn_depth_proposals_weights.pt | submission/
> cat $< | $(JQ) 'del(.model.model_kwargs.weights_file)' \
         | $(JQ) 'del(.model.model_kwargs.rpn.weights_file)' \
         | $(JQ) '.model.weights_file = "INTL_carla_rgbd_fpn_depth_proposals_weights.pt"' \
         | $(JQ) '.sysconfig.external_github_repo = ["IntelLabs/OSCAR@gard-eval4", "colour-science/colour"]' > $@


submission/INTL_carla_rgbd_pfn_depth_proposals_weights.pt: $(MODEL_ZOO)/carla_rgbd_pfn_depth_proposals_weights.pt | submission/
> cp $< $@

submission/INTL_carla_obj_det_multimodal_dpatch_pfn_depth_proposals.json: $(SCENARIOS)/carla_obj_det_multimodal_dpatch_pfn_depth_proposals_pfn.json \
                                                                          submission/INTL_carla_rgbd_pfn_depth_proposals_weights.pt | submission/
> cat $< | $(JQ) 'del(.model.model_kwargs.weights_file)' \
         | $(JQ) 'del(.model.model_kwargs.rpn.weights_file)' \
         | $(JQ) '.model.weights_file = "INTL_carla_rgbd_pfn_depth_proposals_weights.pt"' \
         | $(JQ) '.sysconfig.external_github_repo = ["IntelLabs/OSCAR@gard-eval4", "colour-science/colour"]' > $@


.PHONY: carla_detection_submission
carla_detection_submission: submission/INTL_carla_obj_det_multimodal_dpatch_fpn_depth_proposals.json \
                            submission/INTL_carla_obj_det_multimodal_dpatch_pfn_depth_proposals.json \
                            submission/INTL_carla_obj_det_dpatch_fpn.json \
                            submission/INTL_carla_obj_det_dpatch_pfn.json
> @echo "Created CARLA Detection submission!"

run_carla_detection_submission: carla_detection_submission | .venv
> cp submission/INTL_carla_rgb_fpn_weights.pt $(ARMORY_SAVED_MODEL_DIR)
> cp submission/INTL_carla_rgb_pfn_weights.pt $(ARMORY_SAVED_MODEL_DIR)
> cp submission/INTL_carla_rgbd_fpn_depth_proposals_weights.pt $(ARMORY_SAVED_MODEL_DIR)
> cp submission/INTL_carla_rgbd_pfn_depth_proposals_weights.pt $(ARMORY_SAVED_MODEL_DIR)
> $(POETRY) run armory run submission/INTL_carla_obj_det_dpatch_fpn.json $(ARGS)
> $(POETRY) run armory run submission/INTL_carla_obj_det_dpatch_pfn.json $(ARGS)
> $(POETRY) run armory run submission/INTL_carla_obj_det_multimodal_dpatch_fpn_depth_proposals.json $(ARGS)
> $(POETRY) run armory run submission/INTL_carla_obj_det_multimodal_dpatch_pfn_depth_proposals.json $(ARGS)
