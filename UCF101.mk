UCF101 ?= $(DATASETS)/UCF-101
UCF101_VARIANTS ?= $(DATASETS)/ucf101_and_variants# Ideally this would be off the UCF101 directory, but such as it is.

# Nothing below here is settable
UCF101_FRAMES = $(UCF101_VARIANTS)/ucf101_frames
UCF101_FRAMES_SMOOTHED = $(UCF101_VARIANTS)/ucf101_frames_predictions/smoothed
UCF101_FRAMES_MASKS = $(UCF101_VARIANTS)/ucf101_frames_predictions/InstanceSegmentation_w_predictions_RGB
UCF101_ANNOT = $(UCF101_VARIANTS)/ucfTrainTestlist

# Old frames directory was not all frames in UCF-101. We leave this here just in case.
UCF101_TRUNCATED_FRAMES = $(UCF101_VARIANTS)/ucf101_frames_predictions/OpticalFlow


###################################### Armory Scenarios ######################################
######################################       Begin      ######################################

$(SCENARIOS)/ucf101_%.json: $(ARMORY_SCENARIOS)/ucf101_%.json $(SCENARIOS)/
> cat $< | $(JQ) '.sysconfig.external_github_repo = ""' > $@

# Convert PyTorch Lightning checkpoint .ckpt to .pth to be loaded by MARS.
# Handle normal conversion of ckpt to pth
$(MODEL_ZOO)/MARS/%.pth: $(MODEL_ZOO)/MARS/%.ckpt
> $(POETRY) run python ./bin/convert_lightning_ckpt_to_mars.py $< $@

# Handle special case when pth is '*'
$(MODEL_ZOO)/MARS/%/*.pth: $(MODEL_ZOO)/MARS/%/model_epoch*.ckpt
> $(if $(word 2, $^),$(error There are $(words $^) ckpt files in $(MODEL_ZOO)/$%. There should only be one to convert: $^),)
> $(POETRY) run python ./bin/convert_lightning_ckpt_to_mars.py $< $(patsubst %.ckpt, %.pth, $<)

$(SCENARIOS)/ucf101clean_baseline_pretrained.json: $(SCENARIOS)/ucf101_baseline_pretrained.json
> cat $< | $(JQ) '.dataset.name = "ucf101_clean"' > $@

# Baseline JPEG defense.
# This does not work due a bug: https://github.com/twosixlabs/armory/issues/838
$(SCENARIOS)/%_jpeg.json: $(SCENARIOS)/%.json
> cat $< | $(JQ) '.defense.module = "armory.art_experimental.defences.jpeg_compression_normalized"' \
         | $(JQ) '.defense.name = "JpegCompressionNormalized"' \
         | $(JQ) '.defense.type = "Preprocessor"' \
         | $(JQ) '.defense.kwargs.apply_fit = false' \
         | $(JQ) '.defense.kwargs.apply_predict = true' \
         | $(JQ) '.defense.kwargs.channel_index = 3' \
         | $(JQ) '.defense.kwargs.clip_values = [0, 1] ' \
         | $(JQ) '.defense.kwargs.quality = 50' > $@

# Baseline H264 defense.
$(SCENARIOS)/%_h264.json: $(SCENARIOS)/%.json
> cat $< | $(JQ) '.defense.module = "art.defences.preprocessor.video_compression"' \
         | $(JQ) '.defense.name = "VideoCompression"' \
         | $(JQ) '.defense.type = "Preprocessor"' \
         | $(JQ) '.defense.kwargs.channels_first = false' \
         | $(JQ) '.defense.kwargs.video_format = "mp4"' > $@

# Our depth defense.
$(SCENARIOS)/%_depth.json: $(SCENARIOS)/%.json
> cat $< | $(JQ) '.defense.module = "oscar.defences.preprocessor.monocular_depth"' \
         | $(JQ) '.defense.name = "MonocularDepth"' \
         | $(JQ) '.defense.type = "Preprocessor"' \
         | $(JQ) '.defense.kwargs = {}' \
         | $(JQ) '.dataset.batch_size = 1' > $@

# Randomized smoothing defense. The Gaussian std assumes [0, 255] input.
$(SCENARIOS)/%_randomized_smoothing_gaussian%std.json: $(SCENARIOS)/%.json
> cat $< | $(JQ) '.defense.module = "oscar.defences.preprocessor.gaussian_augmentation"' \
         | $(JQ) '.defense.name = "GaussianAugmentation"' \
         | $(JQ) '.defense.type = "Preprocessor"' \
         | $(JQ) '.defense.kwargs = {}' \
         | $(JQ) '.defense.kwargs.augmentation = false' \
         | $(JQ) '.defense.kwargs.clip_values = [0,1]' \
         | $(JQ) '.defense.kwargs.sigma = $*/255.' > $@

# Our ablation defense.
$(SCENARIOS)/%_bgablation.json: $(SCENARIOS)/%.json
> cat $< | $(JQ) '.defense.module = "oscar.defences.preprocessor.ablator"' \
         | $(JQ) '.defense.name = "Ablator"' \
         | $(JQ) '.defense.type = "Preprocessor"' \
         | $(JQ) '.defense.kwargs.mask_color = [114.7748/255, 107.7354/255, 99.4750/255]' \
         | $(JQ) '.defense.kwargs.detectron2.module = "oscar.defences.preprocessor.detectron2"' \
         | $(JQ) '.defense.kwargs.detectron2.name = "Detectron2Preprocessor"' \
         | $(JQ) '.defense.kwargs.detectron2.kwargs.config_path = "detectron2://COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"' \
         | $(JQ) '.defense.kwargs.detectron2.kwargs.weights_path = "detectron2://COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x/139653917/model_final_2d9806.pkl"' > $@

# Our ablation defense with Gaussian-DT2. Lower the threshold to decrease false negative rate.
$(SCENARIOS)/%_bgablation_gaussian%std.json: $(SCENARIOS)/%_bgablation.json $(MODEL_ZOO)/detectron2/mask_rcnn_X_101_32x8d_FPN_3x_gaussian64truncated_8gpus/model_final.pth
> cat $< | $(JQ) '.defense.kwargs.detectron2.name = "GaussianDetectron2Preprocessor"' \
         | $(JQ) '.defense.kwargs.detectron2.kwargs.gaussian_sigma = $*/255.' \
         | $(JQ) '.defense.kwargs.detectron2.kwargs.gaussian_clip_values = [0, 1]' \
         | $(JQ) '.defense.kwargs.detectron2.kwargs.score_thresh = 0.1' \
         | $(JQ) '.defense.kwargs.detectron2.kwargs.weights_path = "oscar://detectron2/mask_rcnn_X_101_32x8d_FPN_3x_gaussian64truncated_8gpus/model_final.pth"' > $@

$(SCENARIOS)/%_fgablation.json: $(SCENARIOS)/%_bgablation.json
> cat $< | $(JQ) '.defense.kwargs.invert_mask = true' > $@

# Our paletted semantic segmentation defense
$(SCENARIOS)/%_palettedsemanticseg.json: $(SCENARIOS)/%.json
> cat $< | $(JQ) '.defense.module = "oscar.defences.preprocessor.paletted_semantic_segmentor"' \
         | $(JQ) '.defense.name = "PalettedSemanticSegmentor"' \
         | $(JQ) '.defense.type = "Preprocessor"' \
         | $(JQ) '.defense.kwargs.mask_color = [115/255, 108/255, 99/255]' \
         | $(JQ) '.defense.kwargs.detectron2.module = "oscar.defences.preprocessor.detectron2"' \
         | $(JQ) '.defense.kwargs.detectron2.name = "Detectron2Preprocessor"' \
         | $(JQ) '.defense.kwargs.detectron2.kwargs.config_path = "detectron2://COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"' \
         | $(JQ) '.defense.kwargs.detectron2.kwargs.weights_path = "detectron2://COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x/139653917/model_final_2d9806.pkl"' > $@

# Our paletted semantic segmentation defense with Gaussian-DT2. Lower the threshold to decrease false negative rate.
$(SCENARIOS)/%_palettedsemanticseg_gaussian%std.json: $(SCENARIOS)/%_palettedsemanticseg.json $(MODEL_ZOO)/detectron2/mask_rcnn_X_101_32x8d_FPN_3x_gaussian64truncated_8gpus/model_final.pth
> cat $< | $(JQ) '.defense.kwargs.detectron2.name = "GaussianDetectron2Preprocessor"' \
         | $(JQ) '.defense.kwargs.detectron2.kwargs.gaussian_sigma = $*/255.' \
         | $(JQ) '.defense.kwargs.detectron2.kwargs.gaussian_clip_values = [0, 1]' \
         | $(JQ) '.defense.kwargs.detectron2.kwargs.score_thresh = 0.1' \
         | $(JQ) '.defense.kwargs.detectron2.kwargs.weights_path = "oscar://detectron2/mask_rcnn_X_101_32x8d_FPN_3x_gaussian64truncated_8gpus/model_final.pth"' > $@

# Our multichannel semantic segmentation defense
$(SCENARIOS)/%_mcsemanticseg.json: $(SCENARIOS)/%.json
> cat $< | $(JQ) '.defense.module = "oscar.defences.preprocessor.multichannel_semantic_segmentor"' \
         | $(JQ) '.defense.name = "MultichannelSemanticSegmentor"' \
         | $(JQ) '.defense.type = "Preprocessor"' \
         | $(JQ) '.defense.kwargs.detectron2.module = "oscar.defences.preprocessor.detectron2"' \
         | $(JQ) '.defense.kwargs.detectron2.name = "Detectron2Preprocessor"' \
         | $(JQ) '.defense.kwargs.detectron2.kwargs.config_path = "detectron2://COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"' \
         | $(JQ) '.defense.kwargs.detectron2.kwargs.weights_path = "detectron2://COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x/139653917/model_final_2d9806.pkl"' \
         | $(JQ) '.model.module = "oscar.classifiers.ucf101_mars"' \
         | $(JQ) '.model.model_kwargs.input_channels = 80' \
         | $(JQ) '.model.model_kwargs.preprocessing_mean = [0]' \
         | $(JQ) '.model.model_kwargs.preprocessing_std = [255]' > $@

$(SCENARIOS)/%_scsemanticseg.json: $(SCENARIOS)/%.json
> cat $< | $(JQ) '.defense.module = "oscar.defences.preprocessor.multichannel_semantic_segmentor"' \
         | $(JQ) '.defense.name = "MultichannelSemanticSegmentor"' \
         | $(JQ) '.defense.type = "Preprocessor"' \
         | $(JQ) '.defense.kwargs.nb_channels = 1' \
         | $(JQ) '.defense.kwargs.detectron2.module = "oscar.defences.preprocessor.detectron2"' \
         | $(JQ) '.defense.kwargs.detectron2.name = "Detectron2Preprocessor"' \
         | $(JQ) '.defense.kwargs.detectron2.kwargs.config_path = "detectron2://COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"' \
         | $(JQ) '.defense.kwargs.detectron2.kwargs.weights_path = "detectron2://COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x/139653917/model_final_2d9806.pkl"' \
         | $(JQ) '.model.module = "oscar.classifiers.ucf101_mars"' \
         | $(JQ) '.model.model_kwargs.input_channels = 1' \
         | $(JQ) '.model.model_kwargs.preprocessing_mean = [127.5]' \
         | $(JQ) '.model.model_kwargs.preprocessing_std = [1]' > $@

# Frame saliency using one_shot method
$(SCENARIOS)/%_frame_saliency_oneshot_eps_0p004.json: $(SCENARIOS)/%.json
> cat $< | $(JQ) 'del(.attack)' \
         | $(JQ) '.attack.knowledge = "white"' \
         | $(JQ) '.attack.module = "armory.art_experimental.attacks.frame"' \
         | $(JQ) '.attack.name = "get_frame_saliency"' \
         | $(JQ) '.attack.use_label = true' \
         | $(JQ) '.attack.kwargs.verbose = false' \
         | $(JQ) '.attack.kwargs.batch_size = 1' \
         | $(JQ) '.attack.kwargs.frame_index = 1' \
         | $(JQ) '.attack.kwargs.method = "one_shot"' \
         | $(JQ) '.attack.kwargs.inner_config.module = "art.attacks.evasion"' \
         | $(JQ) '.attack.kwargs.inner_config.name = "ProjectedGradientDescent"' \
         | $(JQ) '.attack.kwargs.inner_config.kwargs.batch_size = 1' \
         | $(JQ) '.attack.kwargs.inner_config.kwargs.eps = 0.004' \
         | $(JQ) '.attack.kwargs.inner_config.kwargs.eps_step = 0.001' \
         | $(JQ) '.attack.kwargs.inner_config.kwargs.max_iter = 10' \
         | $(JQ) '.attack.kwargs.inner_config.kwargs.targeted = false' \
         | $(JQ) '.attack.kwargs.inner_config.kwargs.verbose = false' > $@

# Masked PGD attack
$(SCENARIOS)/%_maskedpgd_patchratio_0p1.json: $(SCENARIOS)/%.json
> cat $< | $(JQ) 'del(.attack)' \
         | $(JQ) '.attack.knowledge = "white"' \
         | $(JQ) '.attack.module = "armory.art_experimental.attacks.pgd_patch"' \
         | $(JQ) '.attack.name = "PGDPatch"' \
         | $(JQ) '.attack.use_label = true' \
         | $(JQ) '.attack.kwargs.batch_size = 1' \
         | $(JQ) '.attack.kwargs.eps = 255/255' \
         | $(JQ) '.attack.kwargs.eps_step = 0.2' \
         | $(JQ) '.attack.kwargs.max_iter = 10' \
         | $(JQ) '.attack.kwargs.num_random_init = 0' \
         | $(JQ) '.attack.kwargs.random_eps = false' \
         | $(JQ) '.attack.kwargs.targeted = false' \
         | $(JQ) '.attack.kwargs.verbose = false' \
         | $(JQ) '.attack.generate_kwargs.patch_ratio = 0.1' \
         | $(JQ) '.attack.generate_kwargs.xmin = 0' \
         | $(JQ) '.attack.generate_kwargs.ymin = 0' \
         | $(JQ) '.attack.generate_kwargs.video_input = true'  > $@

# Our multichannel semantic segmentation defense with Gaussian-DT2. Lower the threshold to decrease false negative rate.
$(SCENARIOS)/%_mcsemanticseg_gaussian%std.json: $(SCENARIOS)/%_mcsemanticseg.json $(MODEL_ZOO)/detectron2/mask_rcnn_X_101_32x8d_FPN_3x_gaussian64truncated_8gpus/model_final.pth
> cat $< | $(JQ) '.defense.kwargs.detectron2.name = "GaussianDetectron2Preprocessor"' \
         | $(JQ) '.defense.kwargs.detectron2.kwargs.gaussian_sigma = $*/255.' \
         | $(JQ) '.defense.kwargs.detectron2.kwargs.gaussian_clip_values = [0, 1]' \
         | $(JQ) '.defense.kwargs.detectron2.kwargs.score_thresh = 0.1' \
         | $(JQ) '.defense.kwargs.detectron2.kwargs.weights_path = "oscar://detectron2/mask_rcnn_X_101_32x8d_FPN_3x_gaussian64truncated_8gpus/model_final.pth"' > $@

# XXX: Scenario configs that use our dataset module are only supported by our scenarios! This is due
#      to .dataset.kwargs not being properly passed into the dataset module using the official scenario
$(SCENARIOS)/ucf101_%_oscar.json: $(SCENARIOS)/ucf101_%.json
> cat $< | $(JQ) '.dataset.module = "oscar.data.datasets"' \
         | $(JQ) '.dataset.batch_size = 1' \
         | $(JQ) '.dataset.kwargs.frame_dir = "$(UCF101_FRAMES)"' \
         | $(JQ) '.dataset.kwargs.annotation_path = "$(UCF101_ANNOT)"' \
         | $(JQ) '.dataset.kwargs.sample_size = [240, 320]' \
         | $(JQ) '.dataset.kwargs.only_RGB = true' \
         | $(JQ) '.dataset.kwargs.n_workers = 4' \
         | $(JQ) '.dataset.kwargs.test_size = 505' \
         | $(JQ) '.dataset.kwargs.mid_clip_only = false' \
         | $(JQ) '.scenario.module = "oscar.scenarios.video_ucf101_scenario"' > $@

$(SCENARIOS)/ucf101_%_oscar_skiing.json: $(SCENARIOS)/ucf101_%_oscar.json
> cat $< | $(JQ) '.dataset.kwargs.split = 2' \
         | $(JQ) '.dataset.kwargs.test_index = [3003]' \
         | $(JQ) '.scenario.export_samples = 1' > $@

$(SCENARIOS)/ucf101_%_oscar_midcliponly.json: $(SCENARIOS)/ucf101_%_oscar.json
> cat $< | $(JQ) '.dataset.kwargs.mid_clip_only = true' > $@

# This will pass masks in the alpha channel, which the ablation preprocessor will notice and use, rather than computing maks on the fly
$(SCENARIOS)/%_precomputed_bgablation.json: $(SCENARIOS)/%_bgablation.json
> cat $< | $(JQ) '.dataset.kwargs.frame_mask_dir = "$(UCF101_FRAMES_MASKS)"' \
         | $(JQ) '.dataset.kwargs.mask_in_alpha = true' > $@

$(SCENARIOS)/%_precomputed_fgablation.json: $(SCENARIOS)/%_precomputed_bgablation.json
> cat $< | $(JQ) '.defense.kwargs.invert_mask = true' > $@

# This will pass ablated images into the ablation preprocessor, which will "recursively" ablate the already-ablated videos.
$(SCENARIOS)/%_preablated_bgablation.json: $(SCENARIOS)/%_bgablation.json
> cat $< | $(JQ) '.dataset.kwargs.frame_mask_dir = "$(UCF101_FRAMES_MASKS)"' > $@

$(SCENARIOS)/%_preablated_fgablation.json: $(SCENARIOS)/%_preablated_bgablation.json
> cat $< | $(JQ) '.defense.kwargs.invert_mask = true' > $@

# Take smoothed input.
$(SCENARIOS)/%_smoothed.json: $(SCENARIOS)/%.json
> cat $< | $(JQ) '.defense.module = "oscar.defences.preprocessor.smoothing"' \
         | $(JQ) '.defense.name = "Smoothing"' \
         | $(JQ) '.defense.type = "Preprocessor"' \
         | $(JQ) '.defense.kwargs.is_input_standardized = false' > $@

# Evaluate a model trained with smoothed input.
$(SCENARIOS)/ucf101_fulltune_r50_smoothed.json: $(SCENARIOS)/ucf101_baseline_pretrained_smoothed.json
> cat $< | $(JQ) '.model.weights_file = "$(MODEL_ZOO)/MARS/RGB/fulltune_kinetics400_ucf101_smoothed_frames/UCF101/PreKin_UCF101_1_RGB_train_batch32_sample112_clip16_nestFalse_damp0.9_weight_decay1e-05_manualseed1_modelresnext101_ftbeginidx0_varLR50.pth"' > $@

# Integrate smoothing defense into the model.
$(SCENARIOS)/ucf101_fulltune_r50_smoothed_integrated_defense.json: $(SCENARIOS)/ucf101_fulltune_r50_smoothed.json
> cat $< | $(JQ) '.model.module = "oscar.classifiers.ucf101_mars_chaining_defenses"' \
         | $(JQ) '.model.model_kwargs.defenses = [.defense]' \
         | $(JQ) '.defense = null'  > $@

# Nonsmoothed input for smoothed-trained model.
$(SCENARIOS)/ucf101_fulltune_r50_smoothed_nonsmoothed_input.json: $(SCENARIOS)/ucf101_baseline_pretrained.json
> cat $< | $(JQ) '.model.weights_file = "$(MODEL_ZOO)/MARS/RGB/fulltune_kinetics400_ucf101_smoothed_frames/UCF101/PreKin_UCF101_1_RGB_train_batch32_sample112_clip16_nestFalse_damp0.9_weight_decay1e-05_manualseed1_modelresnext101_ftbeginidx0_varLR50.pth"' > $@

# Evaluate a model trained with residual frames.
#     with residual frames.
$(SCENARIOS)/ucf101_fulltune_r50_residual_with_residual.json: $(SCENARIOS)/ucf101_baseline_pretrained.json
> cat $< | $(JQ) '.defense.module = "oscar.defences.preprocessor.residual_frames"' \
         | $(JQ) '.defense.name = "ResidualFrames"' \
         | $(JQ) '.defense.type = "Preprocessor"' \
         | $(JQ) '.defense.kwargs.residual_absolute_value = true' \
         | $(JQ) '.defense.kwargs.residual_scaling = true' \
         | $(JQ) '.defense.kwargs.residual_after_keyframe = false' \
         | $(JQ) '.defense.kwargs.means = [114.7748, 107.7354, 99.475]' \
         | $(JQ) '.model.weights_file = "$(MODEL_ZOO)/MARS/RGB/fulltune_kinetics400_ucf101_residual_frames/UCF101/PreKin_UCF101_1_RGB_train_batch32_sample112_clip16_nestFalse_damp0.9_weight_decay1e-05_manualseed1_modelresnext101_ftbeginidx0_varLR50.pth"' \
         | $(JQ) '.model.module = "oscar.classifiers.ucf101_mars_chaining_defenses"' \
         | $(JQ) '.model.model_kwargs.defenses = [.defense]' \
         | $(JQ) '.defense = null' > $@

# Evaluate a model trained with jitter residual frames.
#     with residual frames.
$(SCENARIOS)/ucf101_fulltune_r50_residual_jitter_with_residual.json: $(SCENARIOS)/ucf101_fulltune_r50_residual_with_residual.json
> cat $< | $(JQ) '.model.weights_file = "$(MODEL_ZOO)/MARS/RGB/fulltune_kinetics400_ucf101_residual_frames_jitter/UCF101/PreKin_UCF101_1_RGB_train_batch32_sample112_clip16_nestFalse_damp0.9_weight_decay1e-05_manualseed1_modelresnext101_ftbeginidx0_varLR50.pth"' > $@

# Evaluate a model trained with smoothed residual frames.
#     with residual frames.
$(SCENARIOS)/ucf101_fulltune_r50_smoothed_residual_with_residual.json: $(SCENARIOS)/ucf101_fulltune_r50_residual_with_residual.json
> cat $< | $(JQ) '.model.weights_file = "$(MODEL_ZOO)/MARS/RGB/fulltune_kinetics400_ucf101_residual_smoothed_frames/UCF101/PreKin_UCF101_1_RGB_train_batch32_sample112_clip16_nestFalse_damp0.9_weight_decay1e-05_manualseed1_modelresnext101_ftbeginidx0_varLR50.pth"' > $@

# Evaluate a model trained with smoothed residual frames
#     with smoothed residual frames
#     by integrating two differentiable defenses and the MARS preprocessing into the model.
# Note: .defense is required by Armory.
$(SCENARIOS)/ucf101_fulltune_r50_smoothed_residual_with_smoothed_residual.json: $(SCENARIOS)/ucf101_fulltune_r50_smoothed_residual_with_residual.json
> cat $< | $(JQ) '.defense2.module = "oscar.defences.preprocessor.smoothing"' \
         | $(JQ) '.defense2.name = "Smoothing"' \
         | $(JQ) '.defense2.type = "Preprocessor"' \
         | $(JQ) '.defense2.kwargs.is_input_standardized = false' \
         | $(JQ) '.model.weights_file = "$(MODEL_ZOO)/MARS/RGB/fulltune_kinetics400_ucf101_residual_smoothed_frames/UCF101/PreKin_UCF101_1_RGB_train_batch32_sample112_clip16_nestFalse_damp0.9_weight_decay1e-05_manualseed1_modelresnext101_ftbeginidx0_varLR50.pth"' \
         | $(JQ) '.model.module = "oscar.classifiers.ucf101_mars_chaining_defenses"' \
         | $(JQ) '.model.model_kwargs.defenses = [.defense2] + .model.model_kwargs.defenses' \
         | $(JQ) 'del(.defense2)' > $@

# Generic scenario to evaluate using our science scenario
$(SCENARIOS)/ucf101%_science.json: $(SCENARIOS)/ucf101_%.json
> cat $< | $(JQ) '.scenario.module = "oscar.scenarios.video_ucf101_scenario_science"' \
         | $(JQ) 'del(.attack)' \
         | $(JQ) '.attack.knowledge = "white"' \
         | $(JQ) '.attack.module = "art.attacks.evasion"' \
         | $(JQ) '.attack.name = "ProjectedGradientDescent"' \
         | $(JQ) '.attack.kwargs.verbose = false' \
         | $(JQ) '.attack.kwargs.batch_size = 1' \
         | $(JQ) '.attack.kwargs.num_random_init = 1' \
         | $(JQ) '.attack.kwargs.targeted = false' \
         | $(JQ) '.attack.kwargs.eps = [1/255, 2/255, 4/255, 8/255, 16/255, 32/255, 64/255, 128/255]' \
         | $(JQ) '.attack.kwargs.eps_step = [1/255, 2/255, 4/255, 8/255, 16/255, 32/255, 64/255, 128/255]' \
         | $(JQ) '.attack.kwargs.max_iter = [9, 9, 9, 7, 7, 3, 3, 1]' > $@

# Shuffle ucf101 clean dataset at oscar data pipeline at the inference stage. The input is expected to have shape (batch, frames, height, width, channels). If shuffle_axes is [0], shuffling will be done at batch dimension; if shuffle_axes is [1], shuffling will be done at frames dimension; if shuffle_axes is [0, 1], shuffling will be done at both batch and frames dimension.
# This config file shuffles ucf101_shuffle dataset in oscar.data.datasets.
$(SCENARIOS)/ucf101%_shuffle_video.json: $(SCENARIOS)/ucf101%.json
> cat $< | $(JQ) '.dataset.module = "oscar.data.datasets"' \
         | $(JQ) '.dataset.name = "ucf101clean_shuffle"' \
         | $(JQ) '.dataset.kwargs.shuffle_axes = [0, 1]'  > $@

# This config file shuffles ucf101_adversarial_112x112 of oscar.data.adversarial_datasets.
$(SCENARIOS)/ucf101%_science_shuffle_frames.json: $(SCENARIOS)/ucf101%_science.json
> cat $< | $(JQ) '.dataset.kwargs.shuffle_axes = [1]'  > $@

# Generic scenario to shuffle various axes
$(SCENARIOS)/ucf101%_shuffle.json: $(SCENARIOS)/ucf101%.json
> cat $< | $(JQ) '.dataset.module = "oscar.data.datasets"' \
         | $(JQ) '.dataset.name = "ucf101clean_shuffle"' \
         | $(JQ) '.dataset.shuffle = true' \
         | $(JQ) '.dataset.shuffle_axis = 1'  > $@

$(SCENARIOS)/ucf101%_science_shuffle.json: $(SCENARIOS)/ucf101%_science.json
> cat $< | $(JQ) '.dataset.kwargs.shuffle = true' \
         | $(JQ) '.dataset.kwargs.shuffle_axis = 1' > $@

# ST-GCN classifier with DT2 keypoint detector classifier
$(SCENARIOS)/ucf101clean_stgcn.json: $(SCENARIOS)/ucf101clean_baseline_pretrained.json $(MODEL_ZOO)/ST_GCN/stgcn_lightning_v15.pth
> cat $< | $(JQ) '.defense.module = "oscar.defences.preprocessor.keypoints"' \
         | $(JQ) '.defense.name = "VideoToKeypointsPreprocessor"' \
         | $(JQ) '.defense.type = "Preprocessor"' \
         | $(JQ) '.defense.kwargs.config_path = "detectron2://COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"' \
         | $(JQ) '.defense.kwargs.weights_path = "detectron2://COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x/137849621/model_final_a6e10b.pkl"' \
         | $(JQ) '.model.module = "oscar.classifiers.ucf101_stgcn"' \
         | $(JQ) '.model.weights_file = "ST_GCN/stgcn_lightning_v15.pth"' \
         | $(JQ) '.model.model_kwargs = {}' \
         | $(JQ) '.model.fit_kwargs = {}' > $@

$(SCENARIOS)/ucf101clean_stgcn_fgsm04.json: $(SCENARIOS)/ucf101clean_stgcn.json
> cat $< | $(JQ) '.attack.kwargs.eps = 0.0157' > $@

$(SCENARIOS)/ucf101clean_stgcn_fgsm08.json: $(SCENARIOS)/ucf101clean_stgcn.json
> cat $< | $(JQ) '.attack.kwargs.eps = 0.0314' > $@

$(SCENARIOS)/ucf101clean_stgcn_fgsm16.json: $(SCENARIOS)/ucf101clean_stgcn.json
> cat $< | $(JQ) '.attack.kwargs.eps = 0.0627' > $@


###################################### Armory Scenarios ######################################
######################################        End       ######################################


###################################### MARS RGB Training ######################################
######################################        Begin      ######################################

# UCF101 Training
$(MODEL_ZOO)/MiDaS/model.pt: | .venv
> $(POETRY) run gdown "https://drive.google.com/uc?id=1nqW_Hwj86kslfsXR7EnXpEWdO2csz1cC&export=download" -O $@

$(DATASETS)/UCF101.rar: | $(DATASETS)
> @echo "$(YELLOW)Couldn't find $@. Run the commands below, if you really want to generate this data.$(RESET)"
> @echo "WARNING: $(RED)Do not run these commands on spr-gpu*, unless you are absolutely sure what you are doing!$(RESET)"
> @echo wget -O $@ https://www.crcv.ucf.edu/data/UCF101/UCF101.rar

$(UCF101): | $(DATASETS)/UCF101.rar
> @echo "$(YELLOW)Couldn't find $@. Run the commands below, if you really want to generate this data.$(RESET)"
> @echo "WARNING: $(RED)Do not run these commands on spr-gpu*, unless you are absolutely sure what you are doing!$(RESET)"
> @echo unrar x $< $(DATASETS)

.PHONY: extract_depth
extract_depth: | $(MODEL_ZOO)/MiDaS/model.pt .venv $(UCF101) ## Extract depth from UCF-101 using MiDaS
> @echo "$(YELLOW)Couldn't find $@. Run the commands below, if you really want to generate this data.$(RESET)"
> @echo "WARNING: $(RED)Do not run these commands on spr-gpu*, unless you are absolutely sure what you are doing!$(RESET)"
> @echo $(POETRY) run python ./bin/process_videos.py MonocularDepth "$(UCF101)/**/*.avi" --postfix=_depth

$(UCF101_ANNOT): | $(UCF101_VARIANTS)/UCF101TrainTestSplits-RecognitionTask.zip
> @echo "$(YELLOW)Couldn't find $@. Run the commands below, if you really want to generate this data.$(RESET)"
> @echo "WARNING: $(RED)Do not run these commands on spr-gpu*, unless you are absolutely sure what you are doing!$(RESET)"
> @echo unzip -D -d $(UCF101_VARIANTS) $<

$(UCF101_VARIANTS): | $(DATASETS)
> @echo "$(YELLOW)Couldn't find $@. Run the commands below, if you really want to generate this data.$(RESET)"
> @echo "WARNING: $(RED)Do not run these commands on spr-gpu*, unless you are absolutely sure what you are doing!$(RESET)"
> @echo mkdir -p $@

$(UCF101_VARIANTS)/UCF101TrainTestSplits-RecognitionTask.zip: | $(UCF101_VARIANTS)
> @echo "$(YELLOW)Couldn't find $@. Run the commands below, if you really want to generate this data.$(RESET)"
> @echo "WARNING: $(RED)Do not run these commands on spr-gpu*, unless you are absolutely sure what you are doing!$(RESET)"
> @echo wget -O $@ https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip

# The program arguments are: annotation_path, frame_dir, frames_preds_folder, start_idx, end_idx, is_train
$(UCF101_FRAMES_MASKS): | .venv $(UCF101_ANNOT) $(UCF101_FRAMES)
> @echo "$(YELLOW)Couldn't find $@. Run the commands below, if you really want to generate this data.$(RESET)"
> @echo "WARNING: $(RED)Do not run these commands on spr-gpu*, unless you are absolutely sure what you are doing!$(RESET)"
> @echo $(POETRY) run python ./bin/ucf101_mask_bg.py $(UCF101_ANNOT) $(UCF101_FRAMES) $@ 0 3783 0
> @echo $(POETRY) run python ./bin/ucf101_mask_bg.py $(UCF101_ANNOT) $(UCF101_FRAMES) $@ 0 9537 1

$(MODEL_ZOO)/MARS_UCF101_16f.pth: | .venv $(MODEL_ZOO)
> $(POETRY) run gdown "https://drive.google.com/uc?id=1TRTKuJF2lSSWHy3dX91KV4aiD0ni0hhI&export=download" -O $@

$(MODEL_ZOO)/MARS_Kinetics_16f.pth: | .venv $(MODEL_ZOO)
> $(POETRY) run gdown "https://drive.google.com/uc?id=14jZVPIa-Ye2y45icD2MdO_Zg4mlSiTyx&export=download" -O $@

$(MODEL_ZOO)/RGB_UCF101_16f.pth: | .venv $(MODEL_ZOO)
> $(POETRY) run gdown "https://drive.google.com/uc?id=1WOR1AcZ3K3LC0JOXtYrVq9EnZVjYZF96&export=download" -O $@

# Train single-stream RGB models. These are legacy and don't make sense since they only train on RGB features and not like MARS.
# Each folder represents:
#   "Training Features"/"Training Type"_"Network Initialization"
# Where the options are:
#   {RGB,RGBMasked}/{fulltune}_{random,kinetics400,ucf101}
TRAIN_RUN = $(POETRY) run python -m oscar.classifiers.ucf101_mars_lightning --gpus 1
TRAIN_OPT_DEFAULTS = --dataset UCF101 --only_RGB --batch_size 32 --frame_dir $(UCF101_FRAMES) --annotation_path $(UCF101_ANNOT) --frame_mask_dir $(UCF101_FRAMES_MASKS) --result_path $@ --checkpoint 1 --n_classes 101 --n_finetune_classes 101 --ft_begin_index 0 --lr_patience 50 --min_epochs 400 --n_workers 16


.PHONY: $(MODEL_ZOO)/MARS/RGB/fulltune_random
$(MODEL_ZOO)/MARS/RGB/fulltune_random: | $(UCF101_FRAMES) $(UCF101_ANNOT)
> $(TRAIN_RUN) $(TRAIN_OPT_DEFAULTS) --modality RGB $(ARGS)

.PHONY: $(MODEL_ZOO)/MARS/RGBMasked/fulltune_random
$(MODEL_ZOO)/MARS/RGBMasked/fulltune_random: | $(UCF101_FRAMES) $(UCF101_ANNOT) $(UCF101_FRAMES_MASKS)
> $(TRAIN_RUN) $(TRAIN_OPT_DEFAULTS) --modality RGBMasked $(ARGS)

# Gaussian augmentation
# FIXME: It is not possible to mark this target as .PHONY because .PHONY targets will skip implicit/patterned rules
#$(MODEL_ZOO)/MARS/RGBGaussian%/fulltune_ucf101: | $(UCF101_FRAMES) $(UCF101_ANNOT) $(MODEL_ZOO)/MARS_UCF101_16f.pth
#> $(TRAIN_RUN) $(TRAIN_OPT_DEFAULTS) $(TRAIN_OPT_UCF101_RGB) --gaussian_augmentation_std $* $(ARGS)
#
#$(MODEL_ZOO)/MARS/RGBGaussian%truncated/fulltune_ucf101: | $(UCF101_FRAMES) $(UCF101_ANNOT) $(MODEL_ZOO)/MARS_UCF101_16f.pth
#> $(TRAIN_RUN) $(TRAIN_OPT_DEFAULTS) $(TRAIN_OPT_UCF101_RGB) --gaussian_augmentation_std $* --gaussian_augmentation_truncated $(ARGS)

.PHONY: $(MODEL_ZOO)/MARS/RGBSeg/fulltune_random
$(MODEL_ZOO)/MARS/RGBSeg/fulltune_random: | $(UCF101_FRAMES) $(UCF101_ANNOT) $(UCF101_FRAMES_MASKS)
> $(TRAIN_RUN) $(TRAIN_OPT_DEFAULTS) --modality RGBSeg $(ARGS)

.PHONY: $(MODEL_ZOO)/MARS/RGBSegMC/fulltune_random
$(MODEL_ZOO)/MARS/RGBSegMC/fulltune_random: | $(UCF101_FRAMES) $(UCF101_ANNOT) $(UCF101_FRAMES_MASKS)
> $(TRAIN_RUN) $(TRAIN_OPT_DEFAULTS) --modality RGBSegMC $(ARGS)

###################################### MARS RGB Training ######################################
######################################         End       ######################################

#################################### MARS RGB-Flow Training ###################################
##################################            Begin          ##################################

# Utilities necessary to extract optical flow
lib/opencv-4.3.0.zip:
> curl -L https://github.com/opencv/opencv/archive/4.3.0.zip > $@

lib/opencv-4.3.0: | lib/opencv-4.3.0.zip
> unzip -d lib $|

lib/opencv_contrib-4.3.0.zip:
> curl -L https://github.com/opencv/opencv_contrib/archive/4.3.0.zip > $@

lib/opencv_contrib-4.3.0: | lib/opencv_contrib-4.3.0.zip
> unzip -d lib $|

lib/opencv-4.3.0/build:
> mkdir -p lib/opencv-4.3.0/build

lib/opencv-4.3.0/build/Makefile: | lib/opencv-4.3.0 lib/opencv_contrib-4.3.0 lib/opencv-4.3.0/build
> cd lib/opencv-4.3.0/build && cmake -D CMAKE_BUILD_TYPE=Release \
                                        -D CMAKE_INSTALL_PREFIX=../install \
                                        -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.3.0/modules \
                                        -D WITH_CUDA=ON \
                                        -D BUILD_NEW_PYTHON_SUPPORT=ON \
                                        -D BUILD_opencv_python3=ON \
                                        -D HAVE_opencv_python3=ON \
                                        -D PYTHON_DEFAULT_EXECUTABLE=`which python3` \
                                        -D BUILD_opencv_apps=OFF \
                                        -DBUILD_LIST="core,imgproc,objdetect,features2d,highgui,imgcodecs,cudaoptflow,cudaarithm,cudev" \
                                        ..

lib/opencv-4.3.0/install: | lib/opencv-4.3.0/build/Makefile
> cd lib/opencv-4.3.0/build && make -j`nproc` install

lib/MARS/MARS/utils1/tvl1_videoframes: | lib/opencv-4.3.0/install
> g++ -std=c++11 $@.cpp -o $@ \
      -I$|/include/opencv4 \
      -L$|/lib \
      -lopencv_objdetect -lopencv_features2d -lopencv_imgproc \
      -lopencv_highgui -lopencv_core -lopencv_imgcodecs \
      -lopencv_cudaoptflow -lopencv_cudaarithm

$(UCF101_FRAMES): #| lib/MARS/MARS/utils1/tvl1_videoframes
> @echo "$(YELLOW)Couldn't find $@. Run the commands below, if you really want to generate this data.$(RESET)"
> @echo "WARNING: $(RED)Do not run these commands on spr-gpu*, unless you are absolutely sure what you are doing!$(RESET)"
> @echo LD_LIBRARY_PATH=lib/opencv-4.3.0/install/lib PATH=$(PATH):lib/MARS/MARS/utils1 $(POETRY) run python lib/MARS/MARS/utils1/extract_frames_flows.py $(UCF101)/ $@ 0 101

.PHONY: extract_optical_flow
extract_optical_flow: | $(UCF101_FRAMES) ## Extract frames and optical flow using MARS

# Pre-trained optical flow models
$(MODEL_ZOO)/Flow_UCF101_16f.pth: | .venv $(MODEL_ZOO)
> $(POETRY) run gdown "https://drive.google.com/uc?id=1gUf3tKllGi2LMTXVLY2lO8-V0XUcTuck&export=download" -O $@

$(MODEL_ZOO)/Flow_Kinetics_16f.pth: | .venv $(MODEL_ZOO)
> $(POETRY) run gdown "https://drive.google.com/uc?id=1v6cIw4kPoAwPXQFThV5tXJ9WSk2UWXPJ&export=download" -O $@

# Train dual-stream MARS models. To train these models, we need optical extracted using the utilities above.
# Each folder represents:
#   "Training Features"/"Training Type"_"RGB Network Initialization"_"Flow Network Initialization"
# Where the options are:
#   {RGB_Flow,RGBMasked_FlowMasked}/{fulltune}_{random,kinetics400,ucf101}_{kinetics400,ucf101}
MARS_TRAIN_OPT_DEFAULTS = --dataset UCF101 --batch_size 16 --output_layers avgpool --frame_dir $(UCF101_FRAMES) --annotation_path $(UCF101_ANNOT) --frame_mask_dir $(UCF101_FRAMES_MASKS) --resume_path1 $(MODEL_ZOO)/Flow_Kinetics_16f.pth --result_path $@ --checkpoint 1 --n_classes 101 --n_finetune_classes 101 --ft_begin_index 0 --lr_patience 50 --min_epochs 400 --n_workers 16

.PHONY: $(MODEL_ZOO)/MARS/RGB_Flow/fulltune_random_kinetics400
$(MODEL_ZOO)/MARS/RGB_Flow/fulltune_random_kinetics400: | $(MODEL_ZOO)/Flow_Kinetics_16f.pth $(UCF101_FRAMES) $(UCF101_ANNOT)
> $(TRAIN_RUN) $(MARS_TRAIN_OPT_DEFAULTS) --modality RGB_Flow $(ARGS)

$(MODEL_ZOO)/MARS/RGBGaussian%_Flow/fulltune_random_kinetics400: | $(MODEL_ZOO)/Flow_Kinetics_16f.pth $(UCF101_FRAMES) $(UCF101_ANNOT)
> $(TRAIN_RUN) $(MARS_TRAIN_OPT_DEFAULTS) --modality RGBGaussian_Flow --gaussian_augmentation_std $* $(ARGS)

$(MODEL_ZOO)/MARS/RGBGaussian%truncated_Flow/fulltune_random_kinetics400: | $(MODEL_ZOO)/Flow_Kinetics_16f.pth $(UCF101_FRAMES) $(UCF101_ANNOT)
> $(TRAIN_RUN) $(MARS_TRAIN_OPT_DEFAULTS) --modality RGBGaussianTruncated_Flow --gaussian_augmentation_std $* --gaussian_augmentation_truncated $(ARGS)

.PHONY: $(MODEL_ZOO)/MARS/RGBMasked_FlowMasked/fulltune_random_kinetics400
$(MODEL_ZOO)/MARS/RGBMasked_FlowMasked/fulltune_random_kinetics400: | $(MODEL_ZOO)/Flow_Kinetics_16f.pth $(UCF101_FRAMES) $(UCF101_ANNOT)
> $(TRAIN_RUN) $(MARS_TRAIN_OPT_DEFAULTS) --modality RGBMasked_FlowMasked $(ARGS)

.PHONY: $(MODEL_ZOO)/MARS/RGBMasked_Flow/fulltune_random_kinetics400
$(MODEL_ZOO)/MARS/RGBMasked_Flow/fulltune_random_kinetics400: | $(MODEL_ZOO)/Flow_Kinetics_16f.pth $(UCF101_FRAMES) $(UCF101_ANNOT)
> $(TRAIN_RUN) $(MARS_TRAIN_OPT_DEFAULTS) --modality RGBMasked_Flow $(ARGS)

$(MODEL_ZOO)/MARS/RGBMaskedGaussian%_FlowMasked/fulltune_random_kinetics400: | $(MODEL_ZOO)/Flow_Kinetics_16f.pth $(UCF101_FRAMES) $(UCF101_ANNOT)
> $(TRAIN_RUN) $(MARS_TRAIN_OPT_DEFAULTS) --modality RGBMaskedGaussian_FlowMasked --gaussian_augmentation_std $* $(ARGS)

$(MODEL_ZOO)/MARS/RGBMaskedGaussian%truncated_FlowMasked/fulltune_random_kinetics400: | $(MODEL_ZOO)/Flow_Kinetics_16f.pth $(UCF101_FRAMES) $(UCF101_ANNOT)
> $(TRAIN_RUN) $(MARS_TRAIN_OPT_DEFAULTS) --modality RGBMaskedGaussianTruncated_FlowMasked --gaussian_augmentation_std $* --gaussian_augmentation_truncated $(ARGS)

.PHONY: $(MODEL_ZOO)/MARS/RGBSeg_Flow/fulltune_random_kinetics400
$(MODEL_ZOO)/MARS/RGBSeg_Flow/fulltune_random_kinetics400: | $(MODEL_ZOO)/Flow_Kinetics_16f.pth $(UCF101_FRAMES) $(UCF101_ANNOT)
> $(TRAIN_RUN) $(MARS_TRAIN_OPT_DEFAULTS) --modality RGBSeg_Flow $(ARGS)

.PHONY: $(MODEL_ZOO)/MARS/RGBSeg_Flow/fulltune_random_kinetics400_shuffle_frames
$(MODEL_ZOO)/MARS/RGBSeg_Flow/fulltune_random_kinetics400_shuffle_frames: | $(MODEL_ZOO)/Flow_Kinetics_16f.pth $(UCF101_FRAMES) $(UCF101_ANNOT)
> $(TRAIN_RUN) $(MARS_TRAIN_OPT_DEFAULTS) --modality RGBSeg_Flow --shuffle_axes 1 $(ARGS)

.PHONY: $(MODEL_ZOO)/MARS/RGBSegMC_Flow/fulltune_random_kinetics400
$(MODEL_ZOO)/MARS/RGBSegMC_Flow/fulltune_random_kinetics400: | $(MODEL_ZOO)/Flow_Kinetics_16f.pth $(UCF101_FRAMES) $(UCF101_ANNOT)
> $(TRAIN_RUN) $(MARS_TRAIN_OPT_DEFAULTS) --modality RGBSegMC_Flow $(ARGS)

.PHONY: $(MODEL_ZOO)/MARS/RGBSegSC_Flow/fulltune_random_kinetics400
$(MODEL_ZOO)/MARS/RGBSegSC_Flow/fulltune_random_kinetics400: | $(MODEL_ZOO)/Flow_Kinetics_16f.pth $(UCF101_FRAMES) $(UCF101_ANNOT)
> $(TRAIN_RUN) $(MARS_TRAIN_OPT_DEFAULTS) --modality RGBSegSC_Flow $(ARGS)



# + Gaussian augmentation # --gaussian_augmentation_std $*
# FIXME: It is not possible to mark this target as .PHONY because .PHONY targets will skip implicit/patterned rules
#$(MODEL_ZOO)/MARS/RGBGaussian%_Flow/fulltune_ucf101: | $(UCF101_FRAMES) $(UCF101_ANNOT) $(MODEL_ZOO)/MARS_UCF101_16f.pth
#> $(TRAIN_RUN) $(MARS_TRAIN_OPT_DEFAULTS) $(TRAIN_OPT_UCF101_RGB) $(MARS_TRAIN_OPT_UCF101_FLOW) --gaussian_augmentation_std $* $(ARGS)


#################################### MARS RGB-Flow Training ###################################
####################################          End           ###################################


#################################### ST-GCN Training and Models ###################################
####################################           Begin            ###################################

STGCN_TRAIN_RUN = $(POETRY) run python -m oscar.classifiers.ucf101_stgcn --gpus 1

STGCN_TRAIN_DATA = $(PRECOMPUTED_DATA_DIR)/preprocessed.ucf101clean_stgcn.train.h5
STGCN_TEST_DATA = $(PRECOMPUTED_DATA_DIR)/preprocessed.ucf101clean_stgcn.test.h5

STGCN_DATA_ARGS = --precomputed_train_dataset $(STGCN_TRAIN_DATA) --precomputed_test_dataset $(STGCN_TEST_DATA)
STGCN_TRAIN_OPT_DEFAULTS = $(STGCN_DATA_ARGS) --fulltune_model

STGCN_KINETICS_PRETRAINED_MODEL = $(MODEL_ZOO)/ST_GCN/st_gcn.kinetics-6fa43f73.pth

$(STGCN_KINETICS_PRETRAINED_MODEL):
> COMMIT_HASH="40073b653c38c05336a3f28dcef93412dce7fcd9"; \
  FILENAME="st_gcn.kinetics-6fa43f73.pth"; \
  wget "https://github.com/open-mmlab/mmskeleton/raw/$$COMMIT_HASH/checkpoints/$$FILENAME" -O $@

.PHONY: $(MODEL_ZOO)/ST_GCN
$(MODEL_ZOO)/ST_GCN: $(STGCN_TRAIN_DATA) $(STGCN_TEST_DATA) $(STGCN_KINETICS_PRETRAINED_MODEL)
> $(STGCN_TRAIN_RUN) $(STGCN_TRAIN_OPT_DEFAULTS) $(ARGS)

BEST_STGCN_MODEL = $(MODEL_ZOO)/ST_GCN/stgcn_lightning_v15.pth

$(MODEL_ZOO)/ST_GCN/stgcn_lightning_v15.pth:
> wget "https://www.dropbox.com/s/0ib255mi30ol37y/stgcn_lightning_v15.pth?dl=1" -O $@

#################################### ST-GCN Training and Models ###################################
####################################            End             ###################################


####################################### Eval Submission ######################################
#######################################      Begin      ######################################

# XXX: Hack to support "=" in filenames below. Why, PyTorch-Lightning?!, Why?!
equal := =

submission/INTL_ucf101_ablation.json: $(SCENARIOS)/ucf101clean_baseline_pretrained_bgablation.json
> cat $< | $(JQ) '.sysconfig.external_github_repo = ["IntelLabs/GARD@gard-phase1-eval2", "google-research/big_transfer", "yusong-tan/MARS"]' \
         | $(JQ) '.sysconfig.use_gpu = true' \
         | $(JQ) '.model.weights_file = "INTL_fulltune_random_kinetics400_masked.pth"' > $@

submission/INTL_fulltune_kinetics400_ucf101_masked.pth: $(MODEL_ZOO)/UCF101/fulltune_kinetics400_ucf101_masked.pth
> cp $< $@

submission/INTL_fulltune_random_kinetics400_masked.pth: $(MODEL_ZOO)/MARS/RGBMasked_FlowMasked/fulltune_random_kinetics400/version_0/model_epoch$(equal)00498.pth
> cp $< $@

submission/INTL_ucf101_paletted_segmentation.json: $(SCENARIOS)/ucf101clean_baseline_pretrained_palettedsemanticseg.json
> cat $< | $(JQ) '.sysconfig.external_github_repo = ["IntelLabs/GARD@gard-phase1-eval2", "yusong-tan/MARS"]' \
         | $(JQ) '.sysconfig.use_gpu = true' \
         | $(JQ) '.model.weights_file = "INTL_fulltune_random_kinetics400_paletted_segmentation.pth"' > $@

submission/INTL_fulltune_random_kinetics400_paletted_segmentation.pth: $(MODEL_ZOO)/MARS/RGBSeg_Flow/fulltune_random_kinetics400/version_0/model_epoch$(equal)00436.pth
> cp $< $@

submission/INTL_ucf101_paletted_segmentation_gaussian64std.json: $(SCENARIOS)/ucf101clean_baseline_pretrained_palettedsemanticseg_gaussian64std.json submission/INTL_detectron2_segm_mask_rcnn_X_101_32x8d_FPN_3x_gaussian64.pth
> cat $< | $(JQ) '.sysconfig.external_github_repo = ["IntelLabs/GARD@gard-phase1-eval2", "yusong-tan/MARS"]' \
         | $(JQ) '.sysconfig.use_gpu = true' \
         | $(JQ) '.model.weights_file = "INTL_fulltune_random_kinetics400_paletted_segmentation.pth"' \
         | $(JQ) '.defense.kwargs.detectron2.kwargs.weights_path = "armory://INTL_detectron2_segm_mask_rcnn_X_101_32x8d_FPN_3x_gaussian64.pth"' > $@

submission/INTL_ucf101_multichannel_segmentation.json: $(SCENARIOS)/ucf101clean_baseline_pretrained_mcsemanticseg.json
> cat $< | $(JQ) '.sysconfig.external_github_repo = ["IntelLabs/GARD@gard-phase1-eval2", "yusong-tan/MARS"]' \
         | $(JQ) '.sysconfig.use_gpu = true' \
         | $(JQ) '.model.weights_file = "INTL_fulltune_random_kinetics400_multichannel_segmentation.pth"' > $@

submission/INTL_fulltune_random_kinetics400_multichannel_segmentation.pth: $(MODEL_ZOO)/MARS/RGBSegMC_Flow/fulltune_random_kinetics400/version_2/model_epoch$(equal)00303.pth
> cp $< $@

submission/INTL_ucf101_multichannel_segmentation_gaussian64std.json: $(SCENARIOS)/ucf101clean_baseline_pretrained_mcsemanticseg_gaussian64std.json submission/INTL_detectron2_segm_mask_rcnn_X_101_32x8d_FPN_3x_gaussian64.pth
> cat $< | $(JQ) '.sysconfig.external_github_repo = ["IntelLabs/GARD@gard-phase1-eval2", "yusong-tan/MARS"]' \
         | $(JQ) '.sysconfig.use_gpu = true' \
         | $(JQ) '.model.weights_file = "INTL_fulltune_random_kinetics400_multichannel_segmentation.pth"' \
         | $(JQ) '.defense.kwargs.detectron2.kwargs.weights_path = "armory://INTL_detectron2_segm_mask_rcnn_X_101_32x8d_FPN_3x_gaussian64.pth"' > $@


submission/INTL_ucf101_randomized_smoothing64_fulltune_ucf101.json: $(SCENARIOS)/ucf101clean_baseline_randomized_smoothing_gaussian64std.json submission/INTL_ucf101_randomized_smoothing64_fulltune_ucf101.pth
> cat $< | $(JQ) '.sysconfig.external_github_repo = ["IntelLabs/GARD@gard-phase1-eval2", "yusong-tan/MARS"]' \
         | $(JQ) '.sysconfig.use_gpu = true' \
         | $(JQ) '.model.weights_file = "INTL_ucf101_randomized_smoothing64_fulltune_ucf101.pth"' > $@

submission/INTL_ucf101_randomized_smoothing64_fulltune_ucf101.pth: $(MODEL_ZOO)/MARS/RGBGaussian64truncated/fulltune_ucf101/version_0/model_epoch$(equal)00049.ckpt
> $(POETRY) run python ./bin/convert_lightning_ckpt_to_mars.py $< $@


submission/INTL_detectron2_segm_mask_rcnn_X_101_32x8d_FPN_3x_gaussian64.pth: $(MODEL_ZOO)/detectron2/mask_rcnn_X_101_32x8d_FPN_3x_gaussian64truncated_8gpus/model_final.pth
> cp $< $@

submission/INTL_ucf101_stgcn.json: $(SCENARIOS)/ucf101clean_stgcn.json
> cat $< | $(JQ) '.sysconfig.external_github_repo = ["IntelLabs/GARD@gard-phase1-eval2"]' \
         | $(JQ) '.sysconfig.use_gpu = true' \
         | $(JQ) '.defense.kwargs.batch_frames_per_video = 8' \
         | $(JQ) '.model.weights_file = "INTL_ucf101_stgcn.pth"' > $@

submission/INTL_ucf101_stgcn.pth: $(BEST_STGCN_MODEL)
> cp $< $@


.PHONY: ucf101_submission
ucf101_submission: submission/INTL_ucf101_paletted_segmentation.json \
                   submission/INTL_fulltune_random_kinetics400_paletted_segmentation.pth \
                   submission/INTL_ucf101_paletted_segmentation_gaussian64std.json \
                   submission/INTL_ucf101_multichannel_segmentation.json \
                   submission/INTL_ucf101_multichannel_segmentation_gaussian64std.json \
                   submission/INTL_fulltune_random_kinetics400_multichannel_segmentation.pth \
                   submission/INTL_ucf101_randomized_smoothing64_fulltune_ucf101.json \
                   submission/INTL_ucf101_randomized_smoothing64_fulltune_ucf101.pth \
                   submission/INTL_detectron2_segm_mask_rcnn_X_101_32x8d_FPN_3x_gaussian64.pth \
                   submission/INTL_ucf101_stgcn.json \
                   submission/INTL_ucf101_stgcn.pth
> @echo "Created UCF101 submission!"

.PHONY: run_ucf101_submission
run_ucf101_submission: ucf101_submission | .venv
> $(info If fails, you will probably need to run 'make docker_image')
> cp submission/INTL_fulltune_random_kinetics400_paletted_segmentation.pth $(ARMORY_SAVED_MODEL_DIR)
> cp submission/INTL_fulltune_random_kinetics400_multichannel_segmentation.pth $(ARMORY_SAVED_MODEL_DIR)
> cp submission/INTL_ucf101_randomized_smoothing64_fulltune_ucf101.pth $(ARMORY_SAVED_MODEL_DIR)
> cp submission/INTL_detectron2_segm_mask_rcnn_X_101_32x8d_FPN_3x_gaussian64.pth $(ARMORY_SAVED_MODEL_DIR)
> cp submission/INTL_ucf101_stgcn.pth $(ARMORY_SAVED_MODEL_DIR)
> $(POETRY) run armory run submission/INTL_ucf101_ablation.json
> $(POETRY) run armory run submission/INTL_ucf101_randomized_smoothing64_fulltune_ucf101.json
> $(POETRY) run armory run submission/INTL_ucf101_stgcn.json

####################################### Eval Submission ######################################
#######################################       End       ######################################


####################################### Submission Testing ######################################
#######################################        Begin       ######################################


.PHONY: gt_eval2_ucf101_submission
gt_eval2_ucf101_submission: submission/INTL_ucf101_stgcn.json \
                            submission/INTL_ucf101_stgcn.pth
> @echo "Created GT Eval2 UCF101 submission!"

.PHONY: armory_prepare_gt_eval2_ucf101_submission
armory_prepare_gt_eval2_ucf101_submission: docker_image gt_eval2_ucf101_submission | .venv
> cp submission/INTL_ucf101_stgcn.pth $(ARMORY_SAVED_MODEL_DIR)

.PHONY: check_gt_eval2_ucf101_submission
check_gt_eval2_ucf101_submission: armory_prepare_gt_eval2_ucf101_submission
> $(POETRY) run armory run --check submission/INTL_ucf101_stgcn.json $(ARGS)

.PHONY: run_gt_eval2_ucf101_submission
run_gt_eval2_ucf101_submission: armory_prepare_gt_eval2_ucf101_submission
> $(POETRY) run armory run submission/INTL_ucf101_stgcn.json $(ARGS)


####################################### Submission Testing ######################################
#######################################         End        ######################################


.PHONY: test_ucf101
test_ucf101:
> $(MAKE) $(RESULTS)/MARS/RGBMasked_FlowMasked/fulltune_random_ucf101/UCF101/ucf101_baseline_pretrained.json.armory_check
> $(MAKE) $(RESULTS)/MARS/RGBMasked_FlowMasked/fulltune_random_ucf101/UCF101/ucf101_baseline_pretrained_bgablation.json.armory_check
> $(MAKE) $(RESULTS)/MARS/RGBMasked_FlowMasked/fulltune_random_ucf101/UCF101/ucf101_baseline_pretrained_fgablation.json.armory_check
> $(MAKE) $(RESULTS)/MARS/RGBMasked_FlowMasked/fulltune_random_ucf101/UCF101/ucf101_baseline_pretrained_oscar.json.armory_check
> $(MAKE) $(RESULTS)/MARS/RGBMasked_FlowMasked/fulltune_random_ucf101/UCF101/ucf101_baseline_pretrained_bgablation_oscar.json.armory_check
> $(MAKE) $(RESULTS)/MARS/RGBMasked_FlowMasked/fulltune_random_ucf101/UCF101/ucf101_baseline_pretrained_fgablation_oscar.json.armory_check
> $(MAKE) $(RESULTS)/MARS/RGBMasked_FlowMasked/fulltune_random_ucf101/UCF101/ucf101_baseline_pretrained_precomputed_bgablation_oscar.json.armory_check
> $(MAKE) $(RESULTS)/MARS/RGBMasked_FlowMasked/fulltune_random_ucf101/UCF101/ucf101_baseline_pretrained_precomputed_fgablation_oscar.json.armory_check
> $(MAKE) $(RESULTS)/MARS/RGBMasked_FlowMasked/fulltune_random_ucf101/UCF101/ucf101_baseline_pretrained_preablated_bgablation_oscar.json.armory_check
> $(MAKE) $(RESULTS)/MARS/RGBMasked_FlowMasked/fulltune_random_ucf101/UCF101/ucf101_baseline_pretrained_preablated_fgablation_oscar.json.armory_check
> $(MAKE) $(RESULTS)/MARS/RGBMasked_FlowMasked/fulltune_random_ucf101/UCF101/ucf101_baseline_pretrained_oscar_science.json.armory_check
> $(MAKE) $(RESULTS)/MARS/RGBMasked_FlowMasked/fulltune_random_ucf101/UCF101/ucf101_baseline_pretrained_bgablation_oscar_science.json.armory_check
> $(MAKE) $(RESULTS)/MARS/RGBMasked_FlowMasked/fulltune_random_ucf101/UCF101/ucf101_baseline_pretrained_fgablation_oscar_science.json.armory_check
> $(MAKE) $(RESULTS)/MARS/RGBMasked_FlowMasked/fulltune_random_ucf101/UCF101/ucf101_baseline_pretrained_oscar_science.json.armory_check
> $(MAKE) $(RESULTS)/MARS/RGBMasked_FlowMasked/fulltune_random_ucf101/UCF101/ucf101_baseline_pretrained_bgablation_oscar_science.json.armory_check
> $(MAKE) $(RESULTS)/MARS/RGBMasked_FlowMasked/fulltune_random_ucf101/UCF101/ucf101_baseline_pretrained_fgablation_oscar_science.json.armory_check
> $(MAKE) $(RESULTS)/MARS/RGBMasked_FlowMasked/fulltune_random_ucf101/UCF101/ucf101_baseline_pretrained_precomputed_bgablation_oscar_science.json.armory_check
> $(MAKE) $(RESULTS)/MARS/RGBMasked_FlowMasked/fulltune_random_ucf101/UCF101/ucf101_baseline_pretrained_precomputed_fgablation_oscar_science.json.armory_check
> $(MAKE) $(RESULTS)/MARS/RGBMasked_FlowMasked/fulltune_random_ucf101/UCF101/ucf101_baseline_pretrained_preablated_bgablation_oscar_science.json.armory_check
> $(MAKE) $(RESULTS)/MARS/RGBMasked_FlowMasked/fulltune_random_ucf101/UCF101/ucf101_baseline_pretrained_preablated_fgablation_oscar_science.json.armory_check
