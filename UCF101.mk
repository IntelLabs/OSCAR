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

scenario_configs/oscar/ucf101_%.json: scenario_configs/ucf101_%.json scenario_configs/oscar/
> cat $< | $(JQ) '.sysconfig.external_github_repo = ""' > $@

# Convert PyTorch Lightning checkpoint .ckpt to .pth to be loaded by MARS.
# Handle normal conversion of ckpt to pth
$(MODEL_ZOO)/MARS/%.pth: $(MODEL_ZOO)/MARS/%.ckpt
> $(POETRY) run python ./bin/convert_lightning_ckpt_to_mars.py $< $@

# Handle special case when pth is '*'
$(MODEL_ZOO)/MARS/%/*.pth: $(MODEL_ZOO)/MARS/%/model_epoch*.ckpt
> $(if $(word 2, $^),$(error There are $(words $^) ckpt files in $(MODEL_ZOO)/$%. There should only be one to convert: $^),)
> $(POETRY) run python ./bin/convert_lightning_ckpt_to_mars.py $< $(patsubst %.ckpt, %.pth, $<)

# Baseline JPEG defense.
# This does not work due a bug: https://github.com/twosixlabs/armory/issues/838
scenario_configs/oscar/ucf101_baseline_pretrained_jpeg.json: scenario_configs/oscar/ucf101_baseline_pretrained.json
> cat $< | $(JQ) '.defense.module = "armory.art_experimental.defences.jpeg_compression_normalized"' \
         | $(JQ) '.defense.name = "JpegCompressionNormalized"' \
         | $(JQ) '.defense.type = "Preprocessor"' \
         | $(JQ) '.defense.kwargs.apply_fit = false' \
         | $(JQ) '.defense.kwargs.apply_predict = true' \
         | $(JQ) '.defense.kwargs.channel_index = 3' \
         | $(JQ) '.defense.kwargs.clip_values = [0, 1] ' \
         | $(JQ) '.defense.kwargs.quality = 50' > $@

# Baseline H264 defense.
scenario_configs/oscar/ucf101_baseline_pretrained_h264.json: scenario_configs/oscar/ucf101_baseline_pretrained.json
> cat $< | $(JQ) '.defense.module = "art.defences.preprocessor.video_compression"' \
         | $(JQ) '.defense.name = "VideoCompression"' \
         | $(JQ) '.defense.type = "Preprocessor"' \
         | $(JQ) '.defense.kwargs.channels_first = false' \
         | $(JQ) '.defense.kwargs.video_format = "mp4"' > $@

# Our depth defense.
scenario_configs/oscar/ucf101_baseline_pretrained_depth.json: scenario_configs/oscar/ucf101_baseline_pretrained.json
> cat $< | $(JQ) '.defense.module = "oscar.defences.preprocessor.monocular_depth"' \
         | $(JQ) '.defense.name = "MonocularDepth"' \
         | $(JQ) '.defense.type = "Preprocessor"' \
         | $(JQ) '.defense.kwargs = {}' \
         | $(JQ) '.dataset.batch_size = 1' > $@

# Randomized smoothing defense. The Gaussian std assumes [0, 255] input.
scenario_configs/oscar/ucf101_baseline_randomized_smoothing_gaussian%std.json: scenario_configs/oscar/ucf101_baseline_pretrained.json
> cat $< | $(JQ) '.defense.module = "oscar.defences.preprocessor.gaussian_augmentation_pytorch"' \
         | $(JQ) '.defense.name = "GaussianAugmentationPyTorch"' \
         | $(JQ) '.defense.type = "Preprocessor"' \
         | $(JQ) '.defense.kwargs = {}' \
         | $(JQ) '.defense.kwargs.augmentation = false' \
         | $(JQ) '.defense.kwargs.clip_values = [0,1]' \
         | $(JQ) '.defense.kwargs.sigma = $*/255.' > $@

# Our ablation defense.
scenario_configs/oscar/ucf101_baseline_pretrained_bgablation.json: scenario_configs/oscar/ucf101_baseline_pretrained.json
> cat $< | $(JQ) '.defense.module = "oscar.defences.preprocessor.ablation"' \
         | $(JQ) '.defense.name = "BackgroundAblator"' \
         | $(JQ) '.defense.type = "Preprocessor"' \
         | $(JQ) '.defense.kwargs.mask_color = [114.7748/255, 107.7354/255, 99.4750/255]' \
         | $(JQ) '.defense.kwargs.detectron2_config_path = "detectron2://COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"' \
         | $(JQ) '.defense.kwargs.detectron2_weights_path = "detectron2://COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x/139653917/model_final_2d9806.pkl"' > $@

# Our ablation defense with Gaussian-DT2. Lower the threshold to decrease false negative rate.
scenario_configs/oscar/ucf101_baseline_pretrained_bgablation_gaussian%std.json: scenario_configs/oscar/ucf101_baseline_pretrained_bgablation.json $(MODEL_ZOO)/detectron2/mask_rcnn_X_101_32x8d_FPN_3x_gaussian64truncated_8gpus/model_final.pth
> cat $< | $(JQ) '.defense.kwargs.gaussian_sigma = $*/255.' \
         | $(JQ) '.defense.kwargs.gaussian_clip_values = [0, 1]' \
         | $(JQ) '.defense.kwargs.detectron2_score_thresh = 0.1' \
         | $(JQ) '.defense.kwargs.detectron2_weights_path = "oscar://detectron2/mask_rcnn_X_101_32x8d_FPN_3x_gaussian64truncated_8gpus/model_final.pth"' > $@

scenario_configs/oscar/ucf101_baseline_pretrained_fgablation.json: scenario_configs/oscar/ucf101_baseline_pretrained_bgablation.json
> cat $< | $(JQ) '.defense.kwargs.invert_mask = true' > $@

# Our paletted semantic segmentation defense
scenario_configs/oscar/ucf101_baseline_pretrained_palettedsemanticseg.json: scenario_configs/oscar/ucf101_baseline_pretrained.json
> cat $< | $(JQ) '.defense.module = "oscar.defences.preprocessor.paletted_semantic_segmentator"' \
         | $(JQ) '.defense.name = "PalettedSemanticSegmentor"' \
         | $(JQ) '.defense.type = "Preprocessor"' \
         | $(JQ) '.defense.kwargs.mask_color = [115/255, 108/255, 99/255]' \
         | $(JQ) '.defense.kwargs.detectron2_config_path = "detectron2://COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"' \
         | $(JQ) '.defense.kwargs.detectron2_weights_path = "detectron2://COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x/139653917/model_final_2d9806.pkl"' > $@

# Our paletted semantic segmentation defense with Gaussian-DT2. Lower the threshold to decrease false negative rate.
scenario_configs/oscar/ucf101_baseline_pretrained_palettedsemanticseg_gaussian%std.json: scenario_configs/oscar/ucf101_baseline_pretrained_palettedsemanticseg.json $(MODEL_ZOO)/detectron2/mask_rcnn_X_101_32x8d_FPN_3x_gaussian64truncated_8gpus/model_final.pth
> cat $< | $(JQ) '.defense.kwargs.gaussian_sigma = $*/255.' \
         | $(JQ) '.defense.kwargs.gaussian_clip_values = [0, 1]' \
         | $(JQ) '.defense.kwargs.detectron2_score_thresh = 0.1' \
         | $(JQ) '.defense.kwargs.detectron2_weights_path = "oscar://detectron2/mask_rcnn_X_101_32x8d_FPN_3x_gaussian64truncated_8gpus/model_final.pth"' > $@

# Our multichannel semantic segmentation defense
scenario_configs/oscar/ucf101_baseline_pretrained_mcsemanticseg.json: scenario_configs/oscar/ucf101_baseline_pretrained.json
> cat $< | $(JQ) '.defense.module = "oscar.defences.preprocessor.multichannel_semantic_segmentator"' \
         | $(JQ) '.defense.name = "MultichannelSemanticSegmentor"' \
         | $(JQ) '.defense.type = "Preprocessor"' \
         | $(JQ) '.defense.kwargs.detectron2_config_path = "detectron2://COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"' \
         | $(JQ) '.defense.kwargs.detectron2_weights_path = "detectron2://COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x/139653917/model_final_2d9806.pkl"' \
         | $(JQ) '.model.module = "oscar.classifiers.ucf101_mars"' \
         | $(JQ) '.model.model_kwargs.input_channels = 80' \
         | $(JQ) '.model.model_kwargs.preprocessing_mean = [0]' \
         | $(JQ) '.model.model_kwargs.preprocessing_std = [255]' > $@

# Our multichannel semantic segmentation defense with Gaussian-DT2. Lower the threshold to decrease false negative rate.
scenario_configs/oscar/ucf101_baseline_pretrained_mcsemanticseg_gaussian%std.json: scenario_configs/oscar/ucf101_baseline_pretrained_mcsemanticseg.json $(MODEL_ZOO)/detectron2/mask_rcnn_X_101_32x8d_FPN_3x_gaussian64truncated_8gpus/model_final.pth
> cat $< | $(JQ) '.defense.kwargs.gaussian_sigma = $*/255.' \
         | $(JQ) '.defense.kwargs.gaussian_clip_values = [0, 1]' \
         | $(JQ) '.defense.kwargs.detectron2_score_thresh = 0.1' \
         | $(JQ) '.defense.kwargs.detectron2_weights_path = "oscar://detectron2/mask_rcnn_X_101_32x8d_FPN_3x_gaussian64truncated_8gpus/model_final.pth"' > $@

# XXX: Scenario configs that use our dataset module are only supported by our scenarios! This is due
#      to .dataset.kwargs not being properly passed into the dataset module using the official scenario
scenario_configs/oscar/ucf101_%_oscar.json: scenario_configs/oscar/ucf101_%.json
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

scenario_configs/oscar/ucf101_%_oscar_midcliponly.json: scenario_configs/oscar/ucf101_%_oscar.json
> cat $< | $(JQ) '.dataset.kwargs.mid_clip_only = true' > $@

# This will pass masks in the alpha channel, which the ablation preprocessor will notice and use, rather than computing maks on the fly
scenario_configs/oscar/ucf101_baseline_pretrained_precomputed_bgablation.json: scenario_configs/oscar/ucf101_baseline_pretrained_bgablation.json
> cat $< | $(JQ) '.dataset.kwargs.frame_mask_dir = "$(UCF101_FRAMES_MASKS)"' \
         | $(JQ) '.dataset.kwargs.mask_in_alpha = true' > $@

scenario_configs/oscar/ucf101_baseline_pretrained_precomputed_fgablation.json: scenario_configs/oscar/ucf101_baseline_pretrained_precomputed_bgablation.json
> cat $< | $(JQ) '.defense.kwargs.invert_mask = true' > $@

# This will pass ablated images into the ablation preprocessor, which will "recursively" ablate the already-ablated videos.
scenario_configs/oscar/ucf101_baseline_pretrained_preablated_bgablation.json: scenario_configs/oscar/ucf101_baseline_pretrained_bgablation.json
> cat $< | $(JQ) '.dataset.kwargs.frame_mask_dir = "$(UCF101_FRAMES_MASKS)"' > $@

scenario_configs/oscar/ucf101_baseline_pretrained_preablated_fgablation.json: scenario_configs/oscar/ucf101_baseline_pretrained_preablated_bgablation.json
> cat $< | $(JQ) '.defense.kwargs.invert_mask = true' > $@

# Take smoothed input.
scenario_configs/oscar/ucf101_baseline_pretrained_smoothed.json: scenario_configs/oscar/ucf101_baseline_pretrained.json
> cat $< | $(JQ) '.defense.module = "oscar.defences.preprocessor.smoothing"' \
         | $(JQ) '.defense.name = "Smoothing"' \
         | $(JQ) '.defense.type = "Preprocessor"' \
         | $(JQ) '.defense.kwargs.is_input_standardized = false' > $@

# Evaluate a model trained with smoothed input.
scenario_configs/oscar/ucf101_fulltune_r50_smoothed.json: scenario_configs/oscar/ucf101_baseline_pretrained_smoothed.json
> cat $< | $(JQ) '.model.weights_file = "$(MODEL_ZOO)/MARS/RGB/fulltune_kinetics400_ucf101_smoothed_frames/UCF101/PreKin_UCF101_1_RGB_train_batch32_sample112_clip16_nestFalse_damp0.9_weight_decay1e-05_manualseed1_modelresnext101_ftbeginidx0_varLR50.pth"' > $@

# Integrate smoothing defense into the model.
scenario_configs/oscar/ucf101_fulltune_r50_smoothed_integrated_defense.json: scenario_configs/oscar/ucf101_fulltune_r50_smoothed.json
> cat $< | $(JQ) '.model.module = "oscar.classifiers.ucf101_mars_chaining_defenses"' \
         | $(JQ) '.model.model_kwargs.defenses = [.defense]' \
         | $(JQ) '.defense = null'  > $@

# Nonsmoothed input for smoothed-trained model.
scenario_configs/oscar/ucf101_fulltune_r50_smoothed_nonsmoothed_input.json: scenario_configs/oscar/ucf101_baseline_pretrained.json
> cat $< | $(JQ) '.model.weights_file = "$(MODEL_ZOO)/MARS/RGB/fulltune_kinetics400_ucf101_smoothed_frames/UCF101/PreKin_UCF101_1_RGB_train_batch32_sample112_clip16_nestFalse_damp0.9_weight_decay1e-05_manualseed1_modelresnext101_ftbeginidx0_varLR50.pth"' > $@

# Evaluate a model trained with residual frames.
#     with residual frames.
scenario_configs/oscar/ucf101_fulltune_r50_residual_with_residual.json: scenario_configs/oscar/ucf101_baseline_pretrained.json
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
scenario_configs/oscar/ucf101_fulltune_r50_residual_jitter_with_residual.json: scenario_configs/oscar/ucf101_fulltune_r50_residual_with_residual.json
> cat $< | $(JQ) '.model.weights_file = "$(MODEL_ZOO)/MARS/RGB/fulltune_kinetics400_ucf101_residual_frames_jitter/UCF101/PreKin_UCF101_1_RGB_train_batch32_sample112_clip16_nestFalse_damp0.9_weight_decay1e-05_manualseed1_modelresnext101_ftbeginidx0_varLR50.pth"' > $@

# Evaluate a model trained with smoothed residual frames.
#     with residual frames.
scenario_configs/oscar/ucf101_fulltune_r50_smoothed_residual_with_residual.json: scenario_configs/oscar/ucf101_fulltune_r50_residual_with_residual.json
> cat $< | $(JQ) '.model.weights_file = "$(MODEL_ZOO)/MARS/RGB/fulltune_kinetics400_ucf101_residual_smoothed_frames/UCF101/PreKin_UCF101_1_RGB_train_batch32_sample112_clip16_nestFalse_damp0.9_weight_decay1e-05_manualseed1_modelresnext101_ftbeginidx0_varLR50.pth"' > $@

# Evaluate a model trained with smoothed residual frames
#     with smoothed residual frames
#     by integrating two differentiable defenses and the MARS preprocessing into the model.
# Note: .defense is required by Armory.
scenario_configs/oscar/ucf101_fulltune_r50_smoothed_residual_with_smoothed_residual.json: scenario_configs/oscar/ucf101_fulltune_r50_smoothed_residual_with_residual.json
> cat $< | $(JQ) '.defense2.module = "oscar.defences.preprocessor.smoothing"' \
         | $(JQ) '.defense2.name = "Smoothing"' \
         | $(JQ) '.defense2.type = "Preprocessor"' \
         | $(JQ) '.defense2.kwargs.is_input_standardized = false' \
         | $(JQ) '.model.weights_file = "$(MODEL_ZOO)/MARS/RGB/fulltune_kinetics400_ucf101_residual_smoothed_frames/UCF101/PreKin_UCF101_1_RGB_train_batch32_sample112_clip16_nestFalse_damp0.9_weight_decay1e-05_manualseed1_modelresnext101_ftbeginidx0_varLR50.pth"' \
         | $(JQ) '.model.module = "oscar.classifiers.ucf101_mars_chaining_defenses"' \
         | $(JQ) '.model.model_kwargs.defenses = [.defense2] + .model.model_kwargs.defenses' \
         | $(JQ) 'del(.defense2)' > $@

# Generic scenario to evaluate using our science scenario
scenario_configs/oscar/ucf101_%_science.json: scenario_configs/oscar/ucf101_%.json
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

# Shuffle ucf101 dataset at oscar data pipeline at the inference stage. If shuffle_axes is [0], shuffling will be done at clip dimension; if shuffle_axes is [2], shuffling will be done at frame dimension; if shuffle_axes is [0, 2], shuffling will be done at both clip and frame dimension.
# This config file shuffles ucf101_shuffle dataset in oscar.data.datasets.
scenario_configs/oscar/ucf101_%_shuffle_video.json: scenario_configs/oscar/ucf101_%.json
> cat $< | $(JQ) '.dataset.module = "oscar.data.datasets"' \
         | $(JQ) '.dataset.name = "ucf101_shuffle"' \
         | $(JQ) '.dataset.kwargs.shuffle_axes = [0, 2]'  > $@

# This config file shuffles ucf101_adversarial_112x112 of oscar.data.adversarial_datasets.
scenario_configs/oscar/ucf101_%_science_shuffle_frames.json: scenario_configs/oscar/ucf101_%_science.json
> cat $< | $(JQ) '.dataset.kwargs.shuffle_axes = [2]'  > $@

# Generic scenario to shuffle various axes
scenario_configs/oscar/ucf101_%_shuffle.json: scenario_configs/oscar/ucf101_%.json
> cat $< | $(JQ) '.dataset.module = "oscar.data.datasets"' \
         | $(JQ) '.dataset.name = "ucf101_shuffle"' \
         | $(JQ) '.dataset.shuffle = true' \
         | $(JQ) '.dataset.shuffle_axis = 2'  > $@

scenario_configs/oscar/ucf101_%_science_shuffle.json: scenario_configs/oscar/ucf101_%_science.json
> cat $< | $(JQ) '.dataset.kwargs.shuffle = true' \
         | $(JQ) '.dataset.kwargs.shuffle_axis = 2' > $@

# ST-GCN classifier with DT2 keypoint detector classifier
scenario_configs/oscar/ucf101_stgcn.json: scenario_configs/oscar/ucf101_baseline_pretrained.json $(MODEL_ZOO)/ST_GCN/stgcn_lightning_v15.pth
> cat $< | $(JQ) '.dataset.name = "ucf101_clean"' \
         | $(JQ) '.defense.module = "oscar.defences.preprocessor.keypoints"' \
         | $(JQ) '.defense.name = "VideoToKeypointsPreprocessor"' \
         | $(JQ) '.defense.type = "Preprocessor"' \
         | $(JQ) '.defense.kwargs.config_path = "detectron2://COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"' \
         | $(JQ) '.defense.kwargs.weights_path = "detectron2://COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x/137849621/model_final_a6e10b.pkl"' \
         | $(JQ) '.model.module = "oscar.classifiers.ucf101_stgcn"' \
         | $(JQ) '.model.weights_file = "ST_GCN/stgcn_lightning_v15.pth"' \
         | $(JQ) '.model.model_kwargs = {}' \
         | $(JQ) '.model.fit_kwargs = {}' > $@

scenario_configs/oscar/ucf101_stgcn_fgsm04.json: scenario_configs/oscar/ucf101_stgcn.json
> cat $< | $(JQ) '.attack.kwargs.eps = 0.0157' > $@

scenario_configs/oscar/ucf101_stgcn_fgsm08.json: scenario_configs/oscar/ucf101_stgcn.json
> cat $< | $(JQ) '.attack.kwargs.eps = 0.0314' > $@

scenario_configs/oscar/ucf101_stgcn_fgsm16.json: scenario_configs/oscar/ucf101_stgcn.json
> cat $< | $(JQ) '.attack.kwargs.eps = 0.0627' > $@


###################################### Armory Scenarios ######################################
######################################        End       ######################################


###################################### MARS RGB Training ######################################
######################################        Begin      ######################################

# UCF101 Training
oscar/model_zoo/MiDaS/model.pt: | .venv
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
extract_depth: | oscar/model_zoo/MiDaS/model.pt .venv $(UCF101) ## Extract depth from UCF-101 using MiDaS
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
#   {RGB,RGBMasked,RGBHalfMasked}/{fulltune,finetune}_{random,kinetics400,ucf101}
TRAIN_RUN = $(POETRY) run python ./bin/train_ucf101_mars.py --gpus 1 --distributed_backend ddp
TRAIN_OPT_DEFAULTS = --dataset UCF101 --modality RGB --split 1 --only_RGB --batch_size 32 --log 1 --sample_duration 16 --model resnext --model_depth 101 --frame_dir $(UCF101_FRAMES) --annotation_path $(UCF101_ANNOT) --result_path $@ --checkpoint 1 --n_finetune_classes 101 --lr_patience 50 --min_epochs 400
TRAIN_OPT_MASKS = --frame_mask_dir $(UCF101_FRAMES_MASKS) --frame_mask_color 115 108 99
TRAIN_OPT_HALFMASKS = $(TRAIN_OPT_MASKS) --frame_mask_prob 0.5
TRAIN_OPT_KINETICS400_RGB = --n_classes 400 --pretrain_path $(MODEL_ZOO)/MARS_Kinetics_16f.pth
TRAIN_OPT_UCF101_RGB = --n_classes 101 --pretrain_path $(MODEL_ZOO)/MARS_UCF101_16f.pth


$(MODEL_ZOO)/MARS/RGB/finetune_kinetics400: | $(UCF101_FRAMES) $(UCF101_ANNOT) $(MODEL_ZOO)/MARS_Kinetics_16f.pth
> $(TRAIN_RUN) $(TRAIN_OPT_DEFAULTS) $(TRAIN_OPT_KINETICS400_RGB) $(ARGS)

$(MODEL_ZOO)/MARS/RGBMasked/finetune_kinetics400: | $(UCF101_FRAMES) $(UCF101_ANNOT) $(MODEL_ZOO)/MARS_Kinetics_16f.pth $(UCF101_FRAMES_MASKS)
> $(TRAIN_RUN) $(TRAIN_OPT_DEFAULTS) $(TRAIN_OPT_KINETICS400_RGB) $(TRAIN_OPT_MASKS) $(ARGS)

$(MODEL_ZOO)/MARS/RGBHalfMasked/finetune_kinetics400: | $(UCF101_FRAMES) $(UCF101_ANNOT) $(MODEL_ZOO)/MARS_Kinetics_16f.pth $(UCF101_FRAMES_MASKS)
> $(TRAIN_RUN) $(TRAIN_OPT_DEFAULTS) $(TRAIN_OPT_KINETICS400_RGB) $(TRAIN_OPT_HALFMASKS) $(ARGS)


$(MODEL_ZOO)/MARS/RGB/finetune_ucf101: | $(UCF101_FRAMES) $(UCF101_ANNOT) $(MODEL_ZOO)/MARS_UCF101_16f.pth
> $(TRAIN_RUN) $(TRAIN_OPT_DEFAULTS) $(TRAIN_OPT_UCF101_RGB) $(ARGS)

$(MODEL_ZOO)/MARS/RGBMasked/finetune_ucf101: | $(UCF101_FRAMES) $(UCF101_ANNOT) $(MODEL_ZOO)/MARS_UCF101_16f.pth $(UCF101_FRAMES_MASKS)
> $(TRAIN_RUN) $(TRAIN_OPT_DEFAULTS) $(TRAIN_OPT_UCF101_RGB) $(TRAIN_OPT_MASKS) $(ARGS)

$(MODEL_ZOO)/MARS/RGBHalfMasked/finetune_ucf101: | $(UCF101_FRAMES) $(UCF101_ANNOT) $(MODEL_ZOO)/MARS_UCF101_16f.pth $(UCF101_FRAMES_MASKS)
> $(TRAIN_RUN) $(TRAIN_OPT_DEFAULTS) $(TRAIN_OPT_UCF101_RGB) $(TRAIN_OPT_HALFMASKS) $(ARGS)



.PHONY: $(MODEL_ZOO)/MARS/RGB/fulltune_random
$(MODEL_ZOO)/MARS/RGB/fulltune_random: | $(UCF101_FRAMES) $(UCF101_ANNOT)
> $(TRAIN_RUN) $(TRAIN_OPT_DEFAULTS) --n_classes 101 --ft_begin_index 0 $(ARGS)

.PHONY: $(MODEL_ZOO)/MARS/RGBMasked/fulltune_random
$(MODEL_ZOO)/MARS/RGBMasked/fulltune_random: | $(UCF101_FRAMES) $(UCF101_ANNOT) $(UCF101_FRAMES_MASKS)
> $(TRAIN_RUN) $(TRAIN_OPT_DEFAULTS) --n_classes 101 $(TRAIN_OPT_MASKS) --ft_begin_index 0 $(ARGS)

$(MODEL_ZOO)/MARS/RGBMasked/fulltune_random2: | $(UCF101_FRAMES) $(UCF101_ANNOT) $(UCF101_FRAMES_MASKS)
> $(TRAIN_RUN) $(TRAIN_OPT_DEFAULTS) --n_classes 101 $(TRAIN_OPT_MASKS) --ft_begin_index 0 --batch_size 16 $(ARGS)

$(MODEL_ZOO)/MARS/RGBMasked/fulltune_random3: | $(UCF101_FRAMES) $(UCF101_ANNOT) $(UCF101_FRAMES_MASKS)
> $(TRAIN_RUN) $(TRAIN_OPT_DEFAULTS) --n_classes 101 $(TRAIN_OPT_MASKS) --ft_begin_index 0 --batch_size 16 --frame_dir $(UCF101_TRUNCATED_FRAMES) $(ARGS)

$(MODEL_ZOO)/MARS/RGBMasked/fulltune_random4: | $(UCF101_FRAMES) $(UCF101_ANNOT) $(UCF101_FRAMES_MASKS)
> $(TRAIN_RUN) $(TRAIN_OPT_DEFAULTS) --n_classes 101 $(TRAIN_OPT_MASKS) --ft_begin_index 0 --frame_dir $(UCF101_TRUNCATED_FRAMES) $(ARGS)

$(MODEL_ZOO)/MARS/RGBMasked/fulltune_random5: | $(UCF101_FRAMES) $(UCF101_ANNOT) $(UCF101_FRAMES_MASKS)
> $(TRAIN_RUN) $(TRAIN_OPT_DEFAULTS) --n_classes 101 $(TRAIN_OPT_MASKS) --ft_begin_index 0 --frame_dir $(UCF101_TRUNCATED_FRAMES) --cooldown 40 $(ARGS)

$(MODEL_ZOO)/MARS/RGBMasked/fulltune_random6: | $(UCF101_FRAMES) $(UCF101_ANNOT) $(UCF101_FRAMES_MASKS)
> $(TRAIN_RUN) $(TRAIN_OPT_DEFAULTS) --n_classes 101 $(TRAIN_OPT_MASKS) --ft_begin_index 0 --batch_size 16 --frame_dir $(UCF101_TRUNCATED_FRAMES) --cooldown 40 $(ARGS)

$(MODEL_ZOO)/MARS/RGBMasked/fulltune_random7: | $(UCF101_FRAMES) $(UCF101_ANNOT) $(UCF101_FRAMES_MASKS)
> $(TRAIN_RUN) $(TRAIN_OPT_DEFAULTS) --n_classes 101 $(TRAIN_OPT_MASKS) --ft_begin_index 0 --batch_size 16 --cooldown 40 $(ARGS)

$(MODEL_ZOO)/MARS/RGBMasked/fulltune_random8: | $(UCF101_FRAMES) $(UCF101_ANNOT) $(UCF101_FRAMES_MASKS)
> $(TRAIN_RUN) $(TRAIN_OPT_DEFAULTS) --n_classes 101 $(TRAIN_OPT_MASKS) --ft_begin_index 0 --cooldown 40 $(ARGS)


$(MODEL_ZOO)/MARS/RGBHalfMasked/fulltune_random: | $(UCF101_FRAMES) $(UCF101_ANNOT) $(UCF101_FRAMES_MASKS)
> $(TRAIN_RUN) $(TRAIN_OPT_DEFAULTS) --n_classes 101 $(TRAIN_OPT_HALFMASKS) --ft_begin_index 0 $(ARGS)


$(MODEL_ZOO)/MARS/RGB/fulltune_kinetics400: | $(UCF101_FRAMES) $(UCF101_ANNOT) $(MODEL_ZOO)/MARS_Kinetics_16f.pth
> $(TRAIN_RUN) $(TRAIN_OPT_DEFAULTS) $(TRAIN_OPT_KINETICS400_RGB) --ft_begin_index 0 $(ARGS)

$(MODEL_ZOO)/MARS/RGBGaussian%/fulltune_kinetics400: | $(UCF101_FRAMES) $(UCF101_ANNOT) $(MODEL_ZOO)/MARS_Kinetics_16f.pth
> $(TRAIN_RUN) $(TRAIN_OPT_DEFAULTS) $(TRAIN_OPT_KINETICS400_RGB) --ft_begin_index 0 --gaussian_augmentation_std $* $(ARGS)

$(MODEL_ZOO)/MARS/RGBGaussian%truncated/fulltune_kinetics400: | $(UCF101_FRAMES) $(UCF101_ANNOT) $(MODEL_ZOO)/MARS_Kinetics_16f.pth
> $(TRAIN_RUN) $(TRAIN_OPT_DEFAULTS) $(TRAIN_OPT_KINETICS400_RGB) --ft_begin_index 0 --gaussian_augmentation_std $* --gaussian_augmentation_truncated $(ARGS)

$(MODEL_ZOO)/MARS/RGBMasked/fulltune_kinetics400: | $(UCF101_FRAMES) $(UCF101_ANNOT) $(MODEL_ZOO)/MARS_Kinetics_16f.pth $(UCF101_FRAMES_MASKS)
> $(TRAIN_RUN) $(TRAIN_OPT_DEFAULTS) $(TRAIN_OPT_KINETICS400_RGB) $(TRAIN_OPT_MASKS) --ft_begin_index 0 $(ARGS)

$(MODEL_ZOO)/MARS/RGBMaskedGaussian%/fulltune_kinetics400: | $(UCF101_FRAMES) $(UCF101_ANNOT) $(MODEL_ZOO)/MARS_Kinetics_16f.pth $(UCF101_FRAMES_MASKS)
> $(TRAIN_RUN) $(TRAIN_OPT_DEFAULTS) $(TRAIN_OPT_KINETICS400_RGB) $(TRAIN_OPT_MASKS) --ft_begin_index 0 --gaussian_augmentation_std $* $(ARGS)

$(MODEL_ZOO)/MARS/RGBHalfMasked/fulltune_kinetics400: | $(UCF101_FRAMES) $(UCF101_ANNOT) $(MODEL_ZOO)/MARS_Kinetics_16f.pth $(UCF101_FRAMES_MASKS)
> $(TRAIN_RUN) $(TRAIN_OPT_DEFAULTS) $(TRAIN_OPT_KINETICS400_RGB) $(TRAIN_OPT_HALFMASKS) --ft_begin_index 0 $(ARGS)

$(MODEL_ZOO)/MARS/RGB/fulltune_ucf101: | $(UCF101_FRAMES) $(UCF101_ANNOT) $(MODEL_ZOO)/MARS_UCF101_16f.pth
> $(TRAIN_RUN) $(TRAIN_OPT_DEFAULTS) $(TRAIN_OPT_UCF101_RGB) --ft_begin_index 0 $(ARGS)

# Gaussian augmentation
# FIXME: It is not possible to mark this target as .PHONY because .PHONY targets will skip implicit/patterned rules
$(MODEL_ZOO)/MARS/RGBGaussian%/fulltune_ucf101: | $(UCF101_FRAMES) $(UCF101_ANNOT) $(MODEL_ZOO)/MARS_UCF101_16f.pth
> $(TRAIN_RUN) $(TRAIN_OPT_DEFAULTS) $(TRAIN_OPT_UCF101_RGB) --ft_begin_index 0 --gaussian_augmentation_std $* $(ARGS)

$(MODEL_ZOO)/MARS/RGBGaussian%truncated/fulltune_ucf101: | $(UCF101_FRAMES) $(UCF101_ANNOT) $(MODEL_ZOO)/MARS_UCF101_16f.pth
> $(TRAIN_RUN) $(TRAIN_OPT_DEFAULTS) $(TRAIN_OPT_UCF101_RGB) --ft_begin_index 0 --gaussian_augmentation_std $* --gaussian_augmentation_truncated $(ARGS)

.PHONY: $(MODEL_ZOO)/MARS/RGBSeg/fulltune_random
$(MODEL_ZOO)/MARS/RGBSeg/fulltune_random: | $(UCF101_FRAMES) $(UCF101_ANNOT) $(UCF101_FRAMES_MASKS)
> $(TRAIN_RUN) $(TRAIN_OPT_DEFAULTS) --n_classes 101 $(TRAIN_OPT_MASKS) --ft_begin_index 0 --frame_segment=RGB $(ARGS)

.PHONY: $(MODEL_ZOO)/MARS/RGBSegMC/fulltune_random
$(MODEL_ZOO)/MARS/RGBSegMC/fulltune_random: | $(UCF101_FRAMES) $(UCF101_ANNOT) $(UCF101_FRAMES_MASKS)
> $(TRAIN_RUN) $(TRAIN_OPT_DEFAULTS) --n_classes 101 $(TRAIN_OPT_MASKS) --ft_begin_index 0 --frame_segment=MC $(ARGS)

$(MODEL_ZOO)/MARS/RGBMasked/fulltune_ucf101: | $(UCF101_FRAMES) $(UCF101_ANNOT) $(MODEL_ZOO)/MARS_UCF101_16f.pth $(UCF101_FRAMES_MASKS)
> $(TRAIN_RUN) $(TRAIN_OPT_DEFAULTS) $(TRAIN_OPT_UCF101_RGB) $(TRAIN_OPT_MASKS) --ft_begin_index 0 $(ARGS)

$(MODEL_ZOO)/MARS/RGBHalfMasked/fulltune_ucf101: | $(UCF101_FRAMES) $(UCF101_ANNOT) $(MODEL_ZOO)/MARS_UCF101_16f.pth $(UCF101_FRAMES_MASKS)
> $(TRAIN_RUN) $(TRAIN_OPT_DEFAULTS) $(TRAIN_OPT_UCF101_RGB) $(TRAIN_OPT_HALFMASKS) --ft_begin_index 0 $(ARGS)

$(MODEL_ZOO)/MARS/RGBHalfMaskedGaussian%/fulltune_ucf101: | $(UCF101_FRAMES) $(UCF101_ANNOT) $(MODEL_ZOO)/MARS_UCF101_16f.pth $(UCF101_FRAMES_MASKS)
> $(TRAIN_RUN) $(TRAIN_OPT_UCF101_RGB) $(TRAIN_OPT_HALFMASKS) --ft_begin_index 0 --gaussian_augmentation_std $* $(ARGS)

$(MODEL_ZOO)/Flow/UCF101: $(UCF101_FRAMES) $(UCF101_ANNOT)
> $(TRAIN_RUN) $(TRAIN_OPT_DEFAULTS) --modality Flow --n_classes 101 --frame_dir $(UCF101_FRAMES) $(ARGS)

$(MODEL_ZOO)/FlowMasked/UCF101: $(UCF101_FRAMES) $(UCF101_ANNOT)
> $(TRAIN_RUN) $(TRAIN_OPT_DEFAULTS) --modality Flow --n_classes 101 --frame_dir $(UCF101_FRAMES) $(TRAIN_OPT_MASKS) $(ARGS)

# Test models using PyTorch Lightning.
$(MODEL_ZOO)/MARS/%/test_resnext101_UCF101_1_RGB_16_lightning.txt:  $(MODEL_ZOO)/MARS/%/*.ckpt
> $(TRAIN_RUN) --n_classes 101 \
               --test_output_filename $@ \
               --test_only $< \
               --limit_test_batches 1. \
               --result_path $(@D)

# Test models using PyTorch Lightning, with Gaussian augmentation std=50.
$(MODEL_ZOO)/MARS/%/test_resnext101_UCF101_1_RGB_16_lightning_gaussian50.txt:  $(MODEL_ZOO)/MARS/%/*.ckpt
> $(TRAIN_RUN) $(TRAIN_OPT_DEFAULTS) --n_classes 101 \
               --gaussian_augmentation_std 50 \
               --test_output_filename $@ \
               --test_only $< \
               --limit_test_batches 1. \
               --result_path $(@D)

###################################### MARS RGB Training ######################################
######################################         End       ######################################

################################## MARS RGB-Residual Training #################################
##################################            Begin          ##################################

# Residual frames as model input.
TRAIN_OPT_RESIDUAL_FRAMES = --residual_frames
TRAIN_OPT_RESIDUAL_FRAMES_SIGNED = --residual_frames --residual_frames_signed
TRAIN_OPT_RESIDUAL_FRAMES_SIGNED_AFTER_KEYFRAME = --residual_frames --residual_frames_signed --residual_frames_after_keyframe
TRAIN_OPTS_FULLTUNE = --n_epochs 50 --ft_begin_index 0 --learning_rate 0.001
TRAIN_OPTS_DEBUG = --n_workers 0
TRANS_OPT_SMOOTHED_FRAMES = --frame_dir $(UCF101_FRAMES_SMOOTHED)

# Training
$(MODEL_ZOO)/MARS/RGB/fulltune_kinetics400_ucf101_residual_frames: | $(MODEL_ZOO)/MARS_Kinetics_16f.pth $(UCF101_FRAMES) $(UCF101_ANNOT)
> $(TRAIN_RUN) $(TRAIN_OPT_DEFAULTS) $(TRAIN_OPT_KINETICS400_RGB) $(TRAIN_OPTS_FULLTUNE) $(TRAIN_OPT_RESIDUAL_FRAMES) $(ARGS)

$(MODEL_ZOO)/MARS/RGB/fulltune_kinetics400_ucf101_residual_frames_jitter: | $(MODEL_ZOO)/MARS_Kinetics_16f.pth $(UCF101_FRAMES) $(UCF101_ANNOT)
> $(TRAIN_RUN) $(TRAIN_OPT_DEFAULTS) $(TRAIN_OPT_KINETICS400_RGB) $(TRAIN_OPTS_FULLTUNE) $(TRAIN_OPT_RESIDUAL_FRAMES) --jitter $(ARGS)

$(MODEL_ZOO)/MARS/RGB/fulltune_kinetics400_ucf101_residual_frames_signed_after_keyframe: | $(MODEL_ZOO)/MARS_Kinetics_16f.pth $(UCF101_FRAMES) $(UCF101_ANNOT)
> $(TRAIN_RUN) $(TRAIN_OPT_DEFAULTS) $(TRAIN_OPT_KINETICS400_RGB) $(TRAIN_OPTS_FULLTUNE) $(TRAIN_OPT_RESIDUAL_FRAMES_SIGNED_AFTER_KEYFRAME) $(ARGS)

$(MODEL_ZOO)/MARS/RGB/fulltune_kinetics400_ucf101_residual_frames_signed_after_keyframe_jitter: | $(MODEL_ZOO)/MARS_Kinetics_16f.pth $(UCF101_FRAMES) $(UCF101_ANNOT)
> $(TRAIN_RUN) $(TRAIN_OPT_DEFAULTS) $(TRAIN_OPT_KINETICS400_RGB) $(TRAIN_OPTS_FULLTUNE) $(TRAIN_OPT_RESIDUAL_FRAMES_SIGNED_AFTER_KEYFRAME) --jitter $(ARGS)

$(MODEL_ZOO)/MARS/RGB/fulltune_kinetics400_ucf101_smoothed_frames: | $(MODEL_ZOO)/MARS_Kinetics_16f.pth $(UCF101_FRAMES) $(UCF101_ANNOT)
> $(TRAIN_RUN) $(TRAIN_OPT_DEFAULTS) $(TRAIN_OPT_KINETICS400_RGB) $(TRAIN_OPTS_FULLTUNE) $(TRANS_OPT_SMOOTHED_FRAMES) $(ARGS)

$(MODEL_ZOO)/MARS/RGB/fulltune_kinetics400_ucf101_residual_smoothed_frames: | $(MODEL_ZOO)/MARS_Kinetics_16f.pth $(UCF101_FRAMES) $(UCF101_ANNOT)
> $(TRAIN_RUN) $(TRAIN_OPT_DEFAULTS) $(TRAIN_OPT_KINETICS400_RGB) $(TRAIN_OPTS_FULLTUNE) $(TRANS_OPT_SMOOTHED_FRAMES) $(TRAIN_OPT_RESIDUAL_FRAMES) $(ARGS)

################################## MARS RGB-Residual Training #################################
##################################             End            #################################


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
#   {RGB_Flow,RGBMasked_FlowMasked,RGBHalfMasked_FlowHalfMasked}/{fulltune,finetune}_{random,kinetics400,ucf101}_{kinetics400,ucf101}
MARS_TRAIN_OPT_DEFAULTS = --dataset UCF101 --modality RGB_Flow --batch_size 16 --log 1 --sample_duration 16 --model resnext --model_depth 101 --output_layers avgpool --MARS_alpha 50 --frame_dir $(UCF101_FRAMES) --annotation_path $(UCF101_ANNOT) --result_path $@ --checkpoint 1 --n_finetune_classes 101 --lr_patience 50 --min_epochs 400
MARS_TRAIN_OPT_UCF101_FLOW = --resume_path1 $(MODEL_ZOO)/Flow_UCF101_16f.pth
MARS_TRAIN_OPT_KINETICS400_FLOW = --resume_path1 $(MODEL_ZOO)/Flow_Kinetics_16f.pth

$(MODEL_ZOO)/MARS/RGB_Flow/finetune_ucf101_ucf101: | $(MODEL_ZOO)/MARS_UCF101_16f.pth $(MODEL_ZOO)/Flow_UCF101_16f.pth $(UCF101_FRAMES) $(UCF101_ANNOT)
> $(TRAIN_RUN) $(MARS_TRAIN_OPT_DEFAULTS) $(TRAIN_OPT_UCF101_RGB) $(MARS_TRAIN_OPT_UCF101_FLOW) $(ARGS)

$(MODEL_ZOO)/MARS/RGBMasked_FlowMasked/finetune_ucf101_ucf101: | $(MODEL_ZOO)/MARS_UCF101_16f.pth $(MODEL_ZOO)/Flow_UCF101_16f.pth $(UCF101_FRAMES) $(UCF101_ANNOT)
> $(TRAIN_RUN) $(MARS_TRAIN_OPT_DEFAULTS) $(TRAIN_OPT_UCF101_RGB) $(MARS_TRAIN_OPT_UCF101_FLOW) $(TRAIN_OPT_MASKS) $(ARGS)

$(MODEL_ZOO)/MARS/RGBHalfMasked_FlowHalfMasked/finetune_ucf101_ucf101: | $(MODEL_ZOO)/MARS_UCF101_16f.pth $(MODEL_ZOO)/Flow_UCF101_16f.pth $(UCF101_FRAMES) $(UCF101_ANNOT)
> $(TRAIN_RUN) $(MARS_TRAIN_OPT_DEFAULTS) $(TRAIN_OPT_UCF101_RGB) $(MARS_TRAIN_OPT_UCF101_FLOW) $(TRAIN_OPT_HALFMASKS) $(ARGS)


$(MODEL_ZOO)/MARS/RGB_Flow/finetune_ucf101_kinetics400: | $(MODEL_ZOO)/MARS_UCF101_16f.pth $(MODEL_ZOO)/Flow_Kinetics_16f.pth $(UCF101_FRAMES) $(UCF101_ANNOT)
> $(TRAIN_RUN) $(MARS_TRAIN_OPT_DEFAULTS) $(TRAIN_OPT_UCF101_RGB) $(MARS_TRAIN_OPT_KINETICS400_FLOW) $(ARGS)

$(MODEL_ZOO)/MARS/RGBMasked_FlowMasked/finetune_ucf101_kinetics400: | $(MODEL_ZOO)/MARS_UCF101_16f.pth $(MODEL_ZOO)/Flow_Kinetics_16f.pth $(UCF101_FRAMES) $(UCF101_ANNOT)
> $(TRAIN_RUN) $(MARS_TRAIN_OPT_DEFAULTS) $(TRAIN_OPT_UCF101_RGB) $(MARS_TRAIN_OPT_KINETICS400_FLOW) $(TRAIN_OPT_MASKS) $(ARGS)

$(MODEL_ZOO)/MARS/RGBHalfMasked_FlowHalfMasked/finetune_ucf101_kinetics400: | $(MODEL_ZOO)/MARS_UCF101_16f.pth $(MODEL_ZOO)/Flow_Kinetics_16f.pth $(UCF101_FRAMES) $(UCF101_ANNOT)
> $(TRAIN_RUN) $(MARS_TRAIN_OPT_DEFAULTS) $(TRAIN_OPT_UCF101_RGB) $(MARS_TRAIN_OPT_KINETICS400_FLOW) $(TRAIN_OPT_HALFMASKS) $(ARGS)


$(MODEL_ZOO)/MARS/RGB_Flow/finetune_kinetics400_ucf101: | $(MODEL_ZOO)/MARS_Kinetics_16f.pth $(MODEL_ZOO)/Flow_UCF101_16f.pth $(UCF101_FRAMES) $(UCF101_ANNOT)
> $(TRAIN_RUN) $(MARS_TRAIN_OPT_DEFAULTS) $(TRAIN_OPT_KINETICS400_RGB) $(MARS_TRAIN_OPT_UCF101_FLOW) $(ARGS)

$(MODEL_ZOO)/MARS/RGBMasked_FlowMasked/finetune_kinetics400_ucf101: | $(MODEL_ZOO)/MARS_Kinetics_16f.pth $(UCF101_FRAMES) $(UCF101_ANNOT)
> $(TRAIN_RUN) $(MARS_TRAIN_OPT_DEFAULTS) $(TRAIN_OPT_KINETICS400_RGB) $(MARS_TRAIN_OPT_UCF101_FLOW) $(TRAIN_OPT_MASKS) $(ARGS)

$(MODEL_ZOO)/MARS/RGBHalfMasked_FlowHalfMasked/finetune_kinetics400_ucf101: | $(MODEL_ZOO)/MARS_Kinetics_16f.pth $(UCF101_FRAMES) $(UCF101_ANNOT)
> $(TRAIN_RUN) $(MARS_TRAIN_OPT_DEFAULTS) $(TRAIN_OPT_KINETICS400_RGB) $(MARS_TRAIN_OPT_UCF101_FLOW) $(TRAIN_OPT_HALFMASKS) $(ARGS)


$(MODEL_ZOO)/MARS/RGB_Flow/finetune_kinetics400_kinetics400: | $(MODEL_ZOO)/MARS_Kinetics_16f.pth $(MODEL_ZOO)/Flow_Kinetics_16f.pth $(UCF101_FRAMES) $(UCF101_ANNOT)
> $(TRAIN_RUN) $(MARS_TRAIN_OPT_DEFAULTS) $(TRAIN_OPT_KINETICS400_RGB) $(MARS_TRAIN_OPT_KINETICS400_FLOW) $(ARGS)

$(MODEL_ZOO)/MARS/RGBMasked_FlowMasked/finetune_kinetics400_kinetics400: | $(MODEL_ZOO)/MARS_Kinetics_16f.pth $(MODEL_ZOO)/Flow_Kinetics_16f.pth $(UCF101_FRAMES) $(UCF101_ANNOT)
> $(TRAIN_RUN) $(MARS_TRAIN_OPT_DEFAULTS) $(TRAIN_OPT_KINETICS400_RGB) $(MARS_TRAIN_OPT_KINETICS400_FLOW) $(TRAIN_OPT_MASKS) $(ARGS)

$(MODEL_ZOO)/MARS/RGBHalfMasked_FlowHalfMasked/finetune_kinetics400_kinetics400: | $(MODEL_ZOO)/MARS_Kinetics_16f.pth $(MODEL_ZOO)/Flow_Kinetics_16f.pth $(UCF101_FRAMES) $(UCF101_ANNOT)
> $(TRAIN_RUN) $(MARS_TRAIN_OPT_DEFAULTS) $(TRAIN_OPT_KINETICS400_RGB) $(MARS_TRAIN_OPT_KINETICS400_FLOW) $(TRAIN_OPT_HALFMASKS) $(ARGS)



.PHONY: $(MODEL_ZOO)/MARS/RGB_Flow/fulltune_random_kinetics400
$(MODEL_ZOO)/MARS/RGB_Flow/fulltune_random_kinetics400: | $(MODEL_ZOO)/Flow_Kinetics_16f.pth $(UCF101_FRAMES) $(UCF101_ANNOT)
> $(TRAIN_RUN) $(MARS_TRAIN_OPT_DEFAULTS) --n_classes 101 $(MARS_TRAIN_OPT_KINETICS400_FLOW) --ft_begin_index 0 $(ARGS)

$(MODEL_ZOO)/MARS/RGBGaussian%_Flow/fulltune_random_kinetics400: | $(MODEL_ZOO)/Flow_Kinetics_16f.pth $(UCF101_FRAMES) $(UCF101_ANNOT)
> $(TRAIN_RUN) $(MARS_TRAIN_OPT_DEFAULTS) --n_classes 101 $(MARS_TRAIN_OPT_KINETICS400_FLOW) --ft_begin_index 0 --gaussian_augmentation_std $* $(ARGS)

$(MODEL_ZOO)/MARS/RGBGaussian%truncated_Flow/fulltune_random_kinetics400: | $(MODEL_ZOO)/Flow_Kinetics_16f.pth $(UCF101_FRAMES) $(UCF101_ANNOT)
> $(TRAIN_RUN) $(MARS_TRAIN_OPT_DEFAULTS) --n_classes 101 $(MARS_TRAIN_OPT_KINETICS400_FLOW) --ft_begin_index 0 --gaussian_augmentation_std $* --gaussian_augmentation_truncated $(ARGS)

.PHONY: $(MODEL_ZOO)/MARS/RGBMasked_FlowMasked/fulltune_random_kinetics400
$(MODEL_ZOO)/MARS/RGBMasked_FlowMasked/fulltune_random_kinetics400: | $(MODEL_ZOO)/Flow_Kinetics_16f.pth $(UCF101_FRAMES) $(UCF101_ANNOT)
> $(TRAIN_RUN) $(MARS_TRAIN_OPT_DEFAULTS) --n_classes 101 $(MARS_TRAIN_OPT_KINETICS400_FLOW) $(TRAIN_OPT_MASKS) --ft_begin_index 0 $(ARGS)

$(MODEL_ZOO)/MARS/RGBMaskedGaussian%_FlowMasked/fulltune_random_kinetics400: | $(MODEL_ZOO)/Flow_Kinetics_16f.pth $(UCF101_FRAMES) $(UCF101_ANNOT)
> $(TRAIN_RUN) $(MARS_TRAIN_OPT_DEFAULTS) --n_classes 101 $(MARS_TRAIN_OPT_KINETICS400_FLOW) $(TRAIN_OPT_MASKS) --ft_begin_index 0 --gaussian_augmentation_std $* $(ARGS)

$(MODEL_ZOO)/MARS/RGBMaskedGaussian%truncated_FlowMasked/fulltune_random_kinetics400: | $(MODEL_ZOO)/Flow_Kinetics_16f.pth $(UCF101_FRAMES) $(UCF101_ANNOT)
> $(TRAIN_RUN) $(MARS_TRAIN_OPT_DEFAULTS) --n_classes 101 $(MARS_TRAIN_OPT_KINETICS400_FLOW) $(TRAIN_OPT_MASKS) --ft_begin_index 0 --gaussian_augmentation_std $* --gaussian_augmentation_truncated $(ARGS)

$(MODEL_ZOO)/MARS/RGBMasked_FlowMasked/fulltune_random_kinetics400_cooldown: | $(MODEL_ZOO)/Flow_Kinetics_16f.pth $(UCF101_FRAMES) $(UCF101_ANNOT)
> $(TRAIN_RUN) $(MARS_TRAIN_OPT_DEFAULTS) --n_classes 101 $(MARS_TRAIN_OPT_KINETICS400_FLOW) $(TRAIN_OPT_MASKS) --ft_begin_index 0 --cooldown 40 $(ARGS)

$(MODEL_ZOO)/MARS/RGBMasked_FlowMasked/fulltune_random_kinetics400_cooldown_alpha0: | $(MODEL_ZOO)/Flow_Kinetics_16f.pth $(UCF101_FRAMES) $(UCF101_ANNOT)
> $(TRAIN_RUN) $(MARS_TRAIN_OPT_DEFAULTS) --n_classes 101 $(MARS_TRAIN_OPT_KINETICS400_FLOW) $(TRAIN_OPT_MASKS) --ft_begin_index 0 --cooldown 40 --MARS_alpha 0 $(ARGS)

$(MODEL_ZOO)/MARS/RGBMasked_FlowMasked/fulltune_random_kinetics400_cooldown_alpha1: | $(MODEL_ZOO)/Flow_Kinetics_16f.pth $(UCF101_FRAMES) $(UCF101_ANNOT)
> $(TRAIN_RUN) $(MARS_TRAIN_OPT_DEFAULTS) --n_classes 101 $(MARS_TRAIN_OPT_KINETICS400_FLOW) $(TRAIN_OPT_MASKS) --ft_begin_index 0 --cooldown 40 --MARS_alpha 1 $(ARGS)

$(MODEL_ZOO)/MARS/RGBMasked_FlowMasked/fulltune_random_kinetics400_cooldown_alpha10: | $(MODEL_ZOO)/Flow_Kinetics_16f.pth $(UCF101_FRAMES) $(UCF101_ANNOT)
> $(TRAIN_RUN) $(MARS_TRAIN_OPT_DEFAULTS) --n_classes 101 $(MARS_TRAIN_OPT_KINETICS400_FLOW) $(TRAIN_OPT_MASKS) --ft_begin_index 0 --cooldown 40 --MARS_alpha 10 $(ARGS)

$(MODEL_ZOO)/MARS/RGBMasked_FlowMasked/fulltune_random_kinetics400_cooldown_alpha25: | $(MODEL_ZOO)/Flow_Kinetics_16f.pth $(UCF101_FRAMES) $(UCF101_ANNOT)
> $(TRAIN_RUN) $(MARS_TRAIN_OPT_DEFAULTS) --n_classes 101 $(MARS_TRAIN_OPT_KINETICS400_FLOW) $(TRAIN_OPT_MASKS) --ft_begin_index 0 --cooldown 40 --MARS_alpha 25 $(ARGS)

$(MODEL_ZOO)/MARS/RGBMasked_FlowMasked/fulltune_random_kinetics400_cooldown_alpha100: | $(MODEL_ZOO)/Flow_Kinetics_16f.pth $(UCF101_FRAMES) $(UCF101_ANNOT)
> $(TRAIN_RUN) $(MARS_TRAIN_OPT_DEFAULTS) --n_classes 101 $(MARS_TRAIN_OPT_KINETICS400_FLOW) $(TRAIN_OPT_MASKS) --ft_begin_index 0 --cooldown 40 --MARS_alpha 100 $(ARGS)

$(MODEL_ZOO)/MARS/RGBMasked_FlowMasked/fulltune_random_kinetics400_cooldown_alpha200: | $(MODEL_ZOO)/Flow_Kinetics_16f.pth $(UCF101_FRAMES) $(UCF101_ANNOT)
> $(TRAIN_RUN) $(MARS_TRAIN_OPT_DEFAULTS) --n_classes 101 $(MARS_TRAIN_OPT_KINETICS400_FLOW) $(TRAIN_OPT_MASKS) --ft_begin_index 0 --cooldown 40 --MARS_alpha 200 $(ARGS)

$(MODEL_ZOO)/MARS/RGBMasked_FlowMasked/fulltune_random_kinetics400_cooldown_alpha1000: | $(MODEL_ZOO)/Flow_Kinetics_16f.pth $(UCF101_FRAMES) $(UCF101_ANNOT)
> $(TRAIN_RUN) $(MARS_TRAIN_OPT_DEFAULTS) --n_classes 101 $(MARS_TRAIN_OPT_KINETICS400_FLOW) $(TRAIN_OPT_MASKS) --ft_begin_index 0 --cooldown 40 --MARS_alpha 1000 $(ARGS)

$(MODEL_ZOO)/MARS/RGBMasked_FlowMasked/fulltune_random_kinetics400_alpha0: | $(MODEL_ZOO)/Flow_Kinetics_16f.pth $(UCF101_FRAMES) $(UCF101_ANNOT)
> $(TRAIN_RUN) $(MARS_TRAIN_OPT_DEFAULTS) --n_classes 101 $(MARS_TRAIN_OPT_KINETICS400_FLOW) $(TRAIN_OPT_MASKS) --ft_begin_index 0 --MARS_alpha 0 $(ARGS)

$(MODEL_ZOO)/MARS/RGBMasked_FlowMasked/fulltune_random_kinetics400_alpha1: | $(MODEL_ZOO)/Flow_Kinetics_16f.pth $(UCF101_FRAMES) $(UCF101_ANNOT)
> $(TRAIN_RUN) $(MARS_TRAIN_OPT_DEFAULTS) --n_classes 101 $(MARS_TRAIN_OPT_KINETICS400_FLOW) $(TRAIN_OPT_MASKS) --ft_begin_index 0 --MARS_alpha 1 $(ARGS)

$(MODEL_ZOO)/MARS/RGBMasked_FlowMasked/fulltune_random_kinetics400_alpha10: | $(MODEL_ZOO)/Flow_Kinetics_16f.pth $(UCF101_FRAMES) $(UCF101_ANNOT)
> $(TRAIN_RUN) $(MARS_TRAIN_OPT_DEFAULTS) --n_classes 101 $(MARS_TRAIN_OPT_KINETICS400_FLOW) $(TRAIN_OPT_MASKS) --ft_begin_index 0 --MARS_alpha 10 $(ARGS)

$(MODEL_ZOO)/MARS/RGBMasked_FlowMasked/fulltune_random_kinetics400_alpha25: | $(MODEL_ZOO)/Flow_Kinetics_16f.pth $(UCF101_FRAMES) $(UCF101_ANNOT)
> $(TRAIN_RUN) $(MARS_TRAIN_OPT_DEFAULTS) --n_classes 101 $(MARS_TRAIN_OPT_KINETICS400_FLOW) $(TRAIN_OPT_MASKS) --ft_begin_index 0 --MARS_alpha 25 $(ARGS)

$(MODEL_ZOO)/MARS/RGBMasked_FlowMasked/fulltune_random_kinetics400_alpha100: | $(MODEL_ZOO)/Flow_Kinetics_16f.pth $(UCF101_FRAMES) $(UCF101_ANNOT)
> $(TRAIN_RUN) $(MARS_TRAIN_OPT_DEFAULTS) --n_classes 101 $(MARS_TRAIN_OPT_KINETICS400_FLOW) $(TRAIN_OPT_MASKS) --ft_begin_index 0 --MARS_alpha 100 $(ARGS)

$(MODEL_ZOO)/MARS/RGBMasked_FlowMasked/fulltune_random_kinetics400_alpha200: | $(MODEL_ZOO)/Flow_Kinetics_16f.pth $(UCF101_FRAMES) $(UCF101_ANNOT)
> $(TRAIN_RUN) $(MARS_TRAIN_OPT_DEFAULTS) --n_classes 101 $(MARS_TRAIN_OPT_KINETICS400_FLOW) $(TRAIN_OPT_MASKS) --ft_begin_index 0 --MARS_alpha 200 $(ARGS)

$(MODEL_ZOO)/MARS/RGBMasked_FlowMasked/fulltune_random_kinetics400_alpha1000: | $(MODEL_ZOO)/Flow_Kinetics_16f.pth $(UCF101_FRAMES) $(UCF101_ANNOT)
> $(TRAIN_RUN) $(MARS_TRAIN_OPT_DEFAULTS) --n_classes 101 $(MARS_TRAIN_OPT_KINETICS400_FLOW) $(TRAIN_OPT_MASKS) --ft_begin_index 0 --MARS_alpha 1000 $(ARGS)

$(MODEL_ZOO)/MARS/RGBHalfMasked_FlowHalfMasked/fulltune_random_kinetics400: | $(MODEL_ZOO)/Flow_Kinetics_16f.pth $(UCF101_FRAMES) $(UCF101_ANNOT)
> $(TRAIN_RUN) $(MARS_TRAIN_OPT_DEFAULTS) --n_classes 101 $(MARS_TRAIN_OPT_KINETICS400_FLOW) $(TRAIN_OPT_HALFMASKS) --ft_begin_index 0 $(ARGS)

.PHONY: $(MODEL_ZOO)/MARS/RGBSeg_Flow/fulltune_random_kinetics400
$(MODEL_ZOO)/MARS/RGBSeg_Flow/fulltune_random_kinetics400: | $(MODEL_ZOO)/Flow_Kinetics_16f.pth $(UCF101_FRAMES) $(UCF101_ANNOT)
> $(TRAIN_RUN) $(MARS_TRAIN_OPT_DEFAULTS) --n_classes 101 $(MARS_TRAIN_OPT_KINETICS400_FLOW) $(TRAIN_OPT_MASKS) --ft_begin_index 0 --frame_segment=RGB $(ARGS)

.PHONY: $(MODEL_ZOO)/MARS/RGBSegMC_Flow/fulltune_random_kinetics400
$(MODEL_ZOO)/MARS/RGBSegMC_Flow/fulltune_random_kinetics400: | $(MODEL_ZOO)/Flow_Kinetics_16f.pth $(UCF101_FRAMES) $(UCF101_ANNOT)
> $(TRAIN_RUN) $(MARS_TRAIN_OPT_DEFAULTS) --n_classes 101 $(MARS_TRAIN_OPT_KINETICS400_FLOW) $(TRAIN_OPT_MASKS) --ft_begin_index 0 --frame_segment=MC $(ARGS)


$(MODEL_ZOO)/MARS/RGB_Flow/fulltune_ucf101_kinetics400: | $(MODEL_ZOO)/MARS_UCF101_16f.pth $(MODEL_ZOO)/Flow_Kinetics_16f.pth $(UCF101_FRAMES) $(UCF101_ANNOT)
> $(TRAIN_RUN) $(MARS_TRAIN_OPT_DEFAULTS) $(TRAIN_OPT_UCF101_RGB) $(MARS_TRAIN_OPT_KINETICS400_FLOW) --ft_begin_index 0 $(ARGS)

$(MODEL_ZOO)/MARS/RGBMasked_FlowMasked/fulltune_ucf101_kinetics400: | $(MODEL_ZOO)/MARS_UCF101_16f.pth $(MODEL_ZOO)/Flow_Kinetics_16f.pth $(UCF101_FRAMES) $(UCF101_ANNOT)
> $(TRAIN_RUN) $(MARS_TRAIN_OPT_DEFAULTS) $(TRAIN_OPT_UCF101_RGB) $(MARS_TRAIN_OPT_KINETICS400_FLOW) $(TRAIN_OPT_MASKS) --ft_begin_index 0 $(ARGS)

$(MODEL_ZOO)/MARS/RGBHalfMasked_FlowHalfMasked/fulltune_ucf101_kinetics400: | $(MODEL_ZOO)/MARS_UCF101_16f.pth $(MODEL_ZOO)/Flow_Kinetics_16f.pth $(UCF101_FRAMES) $(UCF101_ANNOT)
> $(TRAIN_RUN) $(MARS_TRAIN_OPT_DEFAULTS) $(TRAIN_OPT_UCF101_RGB) $(MARS_TRAIN_OPT_KINETICS400_FLOW) $(TRAIN_OPT_HALFMASKS) --ft_begin_index 0 $(ARGS)

$(MODEL_ZOO)/MARS/RGB_Flow/fulltune_kinetics400_kinetics400: | $(MODEL_ZOO)/MARS_Kinetics_16f.pth $(MODEL_ZOO)/Flow_Kinetics_16f.pth $(UCF101_FRAMES) $(UCF101_ANNOT)
> $(TRAIN_RUN) $(MARS_TRAIN_OPT_DEFAULTS) $(TRAIN_OPT_KINETICS400_RGB) $(MARS_TRAIN_OPT_KINETICS400_FLOW) --ft_begin_index 0 $(ARGS)

$(MODEL_ZOO)/MARS/RGBGaussian%_Flow/fulltune_kinetics400_kinetics400: | $(MODEL_ZOO)/MARS_Kinetics_16f.pth $(MODEL_ZOO)/Flow_Kinetics_16f.pth $(UCF101_FRAMES) $(UCF101_ANNOT)
> $(TRAIN_RUN) $(MARS_TRAIN_OPT_DEFAULTS) $(TRAIN_OPT_KINETICS400_RGB) $(MARS_TRAIN_OPT_KINETICS400_FLOW) --ft_begin_index 0 --gaussian_augmentation_std $* $(ARGS)

$(MODEL_ZOO)/MARS/RGBGaussian%truncated_Flow/fulltune_kinetics400_kinetics400: | $(MODEL_ZOO)/MARS_Kinetics_16f.pth $(MODEL_ZOO)/Flow_Kinetics_16f.pth $(UCF101_FRAMES) $(UCF101_ANNOT)
> $(TRAIN_RUN) $(MARS_TRAIN_OPT_DEFAULTS) $(TRAIN_OPT_KINETICS400_RGB) $(MARS_TRAIN_OPT_KINETICS400_FLOW) --ft_begin_index 0 --gaussian_augmentation_std $* --gaussian_augmentation_truncated $(ARGS)

$(MODEL_ZOO)/MARS/RGBMasked_FlowMasked/fulltune_kinetics400_kinetics400: | $(MODEL_ZOO)/MARS_Kinetics_16f.pth $(MODEL_ZOO)/Flow_Kinetics_16f.pth $(UCF101_FRAMES) $(UCF101_ANNOT)
> $(TRAIN_RUN) $(MARS_TRAIN_OPT_DEFAULTS) $(TRAIN_OPT_KINETICS400_RGB) $(MARS_TRAIN_OPT_KINETICS400_FLOW) $(TRAIN_OPT_MASKS) --ft_begin_index 0 $(ARGS)

$(MODEL_ZOO)/MARS/RGBHalfMasked_FlowHalfMasked/fulltune_kinetics400_kinetics400: | $(MODEL_ZOO)/MARS_Kinetics_16f.pth $(MODEL_ZOO)/Flow_Kinetics_16f.pth $(UCF101_FRAMES) $(UCF101_ANNOT)
> $(TRAIN_RUN) $(MARS_TRAIN_OPT_DEFAULTS) $(TRAIN_OPT_KINETICS400_RGB) $(MARS_TRAIN_OPT_KINETICS400_FLOW) $(TRAIN_OPT_HALFMASKS) --ft_begin_index 0 $(ARGS)




$(MODEL_ZOO)/MARS/RGB_Flow/fulltune_random_ucf101: | $(MODEL_ZOO)/Flow_UCF101_16f.pth $(UCF101_FRAMES) $(UCF101_ANNOT)
> $(TRAIN_RUN) $(MARS_TRAIN_OPT_DEFAULTS) --n_classes 101 $(MARS_TRAIN_OPT_UCF101_FLOW) --ft_begin_index 0 $(ARGS)

$(MODEL_ZOO)/MARS/RGBGaussian%_Flow/fulltune_random_ucf101: | $(MODEL_ZOO)/Flow_UCF101_16f.pth $(UCF101_FRAMES) $(UCF101_ANNOT)
> $(TRAIN_RUN) $(MARS_TRAIN_OPT_DEFAULTS) --n_classes 101 $(MARS_TRAIN_OPT_UCF101_FLOW) --ft_begin_index 0 --gaussian_augmentation_std $* $(ARGS)

$(MODEL_ZOO)/MARS/RGBGaussian%truncated_Flow/fulltune_random_ucf101: | $(MODEL_ZOO)/Flow_UCF101_16f.pth $(UCF101_FRAMES) $(UCF101_ANNOT)
> $(TRAIN_RUN) $(MARS_TRAIN_OPT_DEFAULTS) --n_classes 101 $(MARS_TRAIN_OPT_UCF101_FLOW) --ft_begin_index 0 --gaussian_augmentation_std $* --gaussian_augmentation_truncated $(ARGS)

$(MODEL_ZOO)/MARS/RGB_Flow/fulltune_random_ucf101_shuffle_frames: | $(MODEL_ZOO)/Flow_UCF101_16f.pth $(UCF101_FRAMES) $(UCF101_ANNOT)
> $(TRAIN_RUN) $(MARS_TRAIN_OPT_DEFAULTS) --n_classes 101 $(MARS_TRAIN_OPT_UCF101_FLOW) --ft_begin_index 0 --shuffle_axes 1 $(ARGS)

$(MODEL_ZOO)/MARS/RGBMasked_FlowMasked/fulltune_random_ucf101: | $(MODEL_ZOO)/Flow_UCF101_16f.pth $(UCF101_FRAMES) $(UCF101_ANNOT)
> $(TRAIN_RUN) $(MARS_TRAIN_OPT_DEFAULTS) --n_classes 101 $(MARS_TRAIN_OPT_UCF101_FLOW) $(TRAIN_OPT_MASKS) --ft_begin_index 0 $(ARGS)

$(MODEL_ZOO)/MARS/RGBMasked_FlowMasked/fulltune_random_ucf101_shuffle_frames: | $(MODEL_ZOO)/Flow_UCF101_16f.pth $(UCF101_FRAMES) $(UCF101_ANNOT)
> $(TRAIN_RUN) $(MARS_TRAIN_OPT_DEFAULTS) --n_classes 101 $(MARS_TRAIN_OPT_UCF101_FLOW) $(TRAIN_OPT_MASKS) --ft_begin_index 0 --shuffle_axes 1 $(ARGS)

$(MODEL_ZOO)/MARS/RGBHalfMasked_FlowHalfMasked/fulltune_random_ucf101: | $(MODEL_ZOO)/Flow_UCF101_16f.pth $(UCF101_FRAMES) $(UCF101_ANNOT)
> $(TRAIN_RUN) $(MARS_TRAIN_OPT_DEFAULTS) --n_classes 101 $(MARS_TRAIN_OPT_UCF101_FLOW) $(TRAIN_OPT_HALFMASKS) --ft_begin_index 0 $(ARGS)

$(MODEL_ZOO)/MARS/RGB_Flow/fulltune_ucf101_ucf101: | $(MODEL_ZOO)/MARS_UCF101_16f.pth $(MODEL_ZOO)/Flow_UCF101_16f.pth $(UCF101_FRAMES) $(UCF101_ANNOT)
> $(TRAIN_RUN) $(MARS_TRAIN_OPT_DEFAULTS) $(TRAIN_OPT_UCF101_RGB) $(MARS_TRAIN_OPT_UCF101_FLOW) --ft_begin_index 0 $(ARGS)

# + Gaussian augmentation # --gaussian_augmentation_std $*
# FIXME: It is not possible to mark this target as .PHONY because .PHONY targets will skip implicit/patterned rules
$(MODEL_ZOO)/MARS/RGBGaussian%_Flow/fulltune_ucf101: | $(UCF101_FRAMES) $(UCF101_ANNOT) $(MODEL_ZOO)/MARS_UCF101_16f.pth
> $(TRAIN_RUN) $(MARS_TRAIN_OPT_DEFAULTS) $(TRAIN_OPT_UCF101_RGB) $(MARS_TRAIN_OPT_UCF101_FLOW) --ft_begin_index 0 --gaussian_augmentation_std $* $(ARGS)

$(MODEL_ZOO)/MARS/RGBMasked_FlowMasked/fulltune_ucf101_ucf101: | $(MODEL_ZOO)/MARS_UCF101_16f.pth $(MODEL_ZOO)/Flow_UCF101_16f.pth $(UCF101_FRAMES) $(UCF101_ANNOT)
> $(TRAIN_RUN) $(MARS_TRAIN_OPT_DEFAULTS) $(TRAIN_OPT_UCF101_RGB) $(MARS_TRAIN_OPT_UCF101_FLOW) $(TRAIN_OPT_MASKS) --ft_begin_index 0 $(ARGS)

$(MODEL_ZOO)/MARS/RGBHalfMasked_FlowHalfMasked/fulltune_ucf101_ucf101: | $(MODEL_ZOO)/MARS_UCF101_16f.pth $(MODEL_ZOO)/Flow_UCF101_16f.pth $(UCF101_FRAMES) $(UCF101_ANNOT)
> $(TRAIN_RUN) $(MARS_TRAIN_OPT_DEFAULTS) $(TRAIN_OPT_UCF101_RGB) $(MARS_TRAIN_OPT_UCF101_FLOW) $(TRAIN_OPT_HALFMASKS) --ft_begin_index 0 $(ARGS)


$(MODEL_ZOO)/MARS/RGB_Flow/fulltune_kinetics400_ucf101: | $(MODEL_ZOO)/MARS_Kinetics_16f.pth $(MODEL_ZOO)/Flow_UCF101_16f.pth $(UCF101_FRAMES) $(UCF101_ANNOT)
> $(TRAIN_RUN) $(MARS_TRAIN_OPT_DEFAULTS) $(TRAIN_OPT_KINETICS400_RGB) $(MARS_TRAIN_OPT_UCF101_FLOW) --ft_begin_index 0 $(ARGS)

$(MODEL_ZOO)/MARS/RGBMasked_FlowMasked/fulltune_kinetics400_ucf101: | $(MODEL_ZOO)/MARS_Kinetics_16f.pth $(MODEL_ZOO)/Flow_UCF101_16f.pth $(UCF101_FRAMES) $(UCF101_ANNOT)
> $(TRAIN_RUN) $(MARS_TRAIN_OPT_DEFAULTS) $(TRAIN_OPT_KINETICS400_RGB) $(MARS_TRAIN_OPT_UCF101_FLOW) $(TRAIN_OPT_MASKS) --ft_begin_index 0 $(ARGS)

$(MODEL_ZOO)/MARS/RGBHalfMasked_FlowHalfMasked/fulltune_kinetics400_ucf101: | $(MODEL_ZOO)/MARS_Kinetics_16f.pth $(MODEL_ZOO)/Flow_UCF101_16f.pth $(UCF101_FRAMES) $(UCF101_ANNOT)
> $(TRAIN_RUN) $(MARS_TRAIN_OPT_DEFAULTS) $(TRAIN_OPT_KINETICS400_RGB) $(MARS_TRAIN_OPT_UCF101_FLOW) $(TRAIN_OPT_HALFMASKS) --ft_begin_index 0 $(ARGS)


$(MODEL_ZOO)/MARS/RGB_Flow/f/MARS/RGBMasked_FlowMasked/fulltune_random_ucf101ulltune_random_ucf101masked: | $(MODEL_ZOO)/Flow/UCF101 $(UCF101_FRAMES) $(UCF101_ANNOT)
> $(TRAIN_RUN) $(MARS_TRAIN_OPT_DEFAULTS) --n_classes 101 --resume_path1 $(MODEL_ZOO)/FlowMasked/UCF101/UCF101/*.pth --ft_begin_index 0 $(ARGS)

$(MODEL_ZOO)/MARS/RGB_Flow/fulltune_random_ucf101normal: | $(MODEL_ZOO)/Flow/UCF101 $(UCF101_FRAMES) $(UCF101_ANNOT)
> $(TRAIN_RUN) $(MARS_TRAIN_OPT_DEFAULTS) --n_classes 101 --resume_path1 $(MODEL_ZOO)/Flow/UCF101/UCF101/*.pth --ft_begin_index 0 $(ARGS)

$(MODEL_ZOO)/MARS/RGBMasked_FlowMasked/fulltune_random_ucf101masked: | $(MODEL_ZOO)/FlowMasked/UCF101 $(UCF101_FRAMES) $(UCF101_ANNOT)
> $(TRAIN_RUN) $(MARS_TRAIN_OPT_DEFAULTS) --n_classes 101 --resume_path1 $(MODEL_ZOO)/FlowMasked/UCF101/UCF101/*.pth $(TRAIN_OPT_MASKS) --ft_begin_index 0 $(ARGS)

$(MODEL_ZOO)/MARS/RGBMasked_FlowMasked/fulltune_random_ucf101normal: | $(MODEL_ZOO)/Flow/UCF101 $(UCF101_FRAMES) $(UCF101_ANNOT)
> $(TRAIN_RUN) $(MARS_TRAIN_OPT_DEFAULTS) --n_classes 101 --resume_path1 $(MODEL_ZOO)/Flow/UCF101/UCF101/*.pth $(TRAIN_OPT_MASKS) --ft_begin_index 0 $(ARGS)

#################################### MARS RGB-Flow Training ###################################
####################################          End           ###################################


#################################### ST-GCN Training and Models ###################################
####################################           Begin            ###################################

BEST_STGCN_MODEL = $(MODEL_ZOO)/ST_GCN/stgcn_lightning_v15.pth

$(MODEL_ZOO)/ST_GCN/stgcn_lightning_v15.pth:
> wget "https://www.dropbox.com/s/0ib255mi30ol37y/stgcn_lightning_v15.pth?dl=1" -O $@

#################################### ST-GCN Training and Models ###################################
####################################            End             ###################################


####################################### Eval Submission ######################################
#######################################      Begin      ######################################

submission/INTL_ucf101_ablation.json: scenario_configs/oscar/ucf101_baseline_pretrained_bgablation.json
> cat $< | $(JQ) '.dataset.name = "ucf101_clean"' \
         | $(JQ) '.sysconfig.external_github_repo = ["IntelLabs/GARD@gard-phase1-eval1", "google-research/big_transfer", "yusong-tan/MARS"]' \
         | $(JQ) '.sysconfig.use_gpu = true' \
         | $(JQ) '.model.weights_file = "INTL_fulltune_kinetics400_ucf101_masked.pth"' > $@

submission/INTL_fulltune_kinetics400_ucf101_masked.pth: oscar/model_zoo/UCF101/fulltune_kinetics400_ucf101_masked.pth
> cp $< $@

# XXX: Hack to support "=" in filenames below. Why, PyTorch-Lightning?!, Why?!
equal := =

submission/INTL_ucf101_paletted_segmentation.json: scenario_configs/oscar/ucf101_baseline_pretrained_palettedsemanticseg.json
> cat $< | $(JQ) '.dataset.name = "ucf101_clean"' \
         | $(JQ) '.sysconfig.external_github_repo = ["IntelLabs/GARD@gard-phase1-eval2", "yusong-tan/MARS"]' \
         | $(JQ) '.sysconfig.use_gpu = true' \
         | $(JQ) '.model.weights_file = "INTL_fulltune_random_kinetics400_paletted_segmentation.pth"' > $@

submission/INTL_fulltune_random_kinetics400_paletted_segmentation.pth: oscar/model_zoo/MARS/RGBSeg_Flow/fulltune_random_kinetics400/version_0/model_epoch$(equal)00436.pth
> cp $< $@

submission/INTL_ucf101_paletted_segmentation_gaussian64std.json: scenario_configs/oscar/ucf101_baseline_pretrained_palettedsemanticseg_gaussian64std.json submission/INTL_detectron2_segm_mask_rcnn_X_101_32x8d_FPN_3x_gaussian64.pth
> cat $< | $(JQ) '.dataset.name = "ucf101_clean"' \
         | $(JQ) '.sysconfig.external_github_repo = ["IntelLabs/GARD@gard-phase1-eval2", "yusong-tan/MARS"]' \
         | $(JQ) '.sysconfig.use_gpu = true' \
         | $(JQ) '.model.weights_file = "INTL_fulltune_random_kinetics400_paletted_segmentation.pth"' \
         | $(JQ) '.defense.kwargs.detectron2_weights_path = "armory://INTL_detectron2_segm_mask_rcnn_X_101_32x8d_FPN_3x_gaussian64.pth"' > $@

submission/INTL_ucf101_multichannel_segmentation.json: scenario_configs/oscar/ucf101_baseline_pretrained_mcsemanticseg.json
> cat $< | $(JQ) '.dataset.name = "ucf101_clean"' \
         | $(JQ) '.sysconfig.external_github_repo = ["IntelLabs/GARD@gard-phase1-eval2", "yusong-tan/MARS"]' \
         | $(JQ) '.sysconfig.use_gpu = true' \
         | $(JQ) '.model.weights_file = "INTL_fulltune_random_kinetics400_multichannel_segmentation.pth"' > $@

submission/INTL_fulltune_random_kinetics400_multichannel_segmentation.pth: oscar/model_zoo/MARS/RGBSegMC_Flow/fulltune_random_kinetics400/version_2/model_epoch$(equal)00303.pth
> cp $< $@

submission/INTL_ucf101_multichannel_segmentation_gaussian64std.json: scenario_configs/oscar/ucf101_baseline_pretrained_mcsemanticseg_gaussian64std.json submission/INTL_detectron2_segm_mask_rcnn_X_101_32x8d_FPN_3x_gaussian64.pth
> cat $< | $(JQ) '.dataset.name = "ucf101_clean"' \
         | $(JQ) '.sysconfig.external_github_repo = ["IntelLabs/GARD@gard-phase1-eval2", "yusong-tan/MARS"]' \
         | $(JQ) '.sysconfig.use_gpu = true' \
         | $(JQ) '.model.weights_file = "INTL_fulltune_random_kinetics400_multichannel_segmentation.pth"' \
         | $(JQ) '.defense.kwargs.detectron2_weights_path = "armory://INTL_detectron2_segm_mask_rcnn_X_101_32x8d_FPN_3x_gaussian64.pth"' > $@


submission/INTL_ucf101_randomized_smoothing64_fulltune_ucf101.json: scenario_configs/oscar/ucf101_baseline_randomized_smoothing_gaussian64std.json submission/INTL_ucf101_randomized_smoothing64_fulltune_ucf101.pth
> cat $< | $(JQ) '.dataset.name = "ucf101_clean"' \
         | $(JQ) '.sysconfig.external_github_repo = ["IntelLabs/GARD@gard-phase1-eval2", "yusong-tan/MARS"]' \
         | $(JQ) '.sysconfig.use_gpu = true' \
         | $(JQ) '.model.weights_file = "INTL_ucf101_randomized_smoothing64_fulltune_ucf101.pth"' > $@

submission/INTL_ucf101_randomized_smoothing64_fulltune_ucf101.pth: oscar/model_zoo/MARS/RGBGaussian64truncated/fulltune_ucf101/version_0/model_epoch$(equal)00049.pth
> cp $< $@


submission/INTL_ucf101_ablation_detectron2_randomized_smoothing64.json: submission/INTL_ucf101_ablation.json
> cat $< | $(JQ) '.dataset.name = "ucf101_clean"' \
         | $(JQ) '.defense.name = "GaussianBackgroundAblator"' \
         | $(JQ) '.defense.kwargs.gaussian_sigma = 64/255.' \
         | $(JQ) '.defense.kwargs.gaussian_clip_values = [0, 1]' \
         | $(JQ) '.defense.kwargs.detectron2_weights_path = "armory://INTL_detectron2_segm_mask_rcnn_X_101_32x8d_FPN_3x_gaussian64.pth"' > $@

submission/INTL_detectron2_segm_mask_rcnn_X_101_32x8d_FPN_3x_gaussian64.pth: oscar/model_zoo/detectron2/mask_rcnn_X_101_32x8d_FPN_3x_gaussian64truncated_8gpus/model_final.pth
> cp $< $@

submission/INTL_ucf101_stgcn.json: scenario_configs/oscar/ucf101_stgcn.json
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
> $(POETRY) run armory run submission/INTL_ucf101_ablation_detectron2_randomized_smoothing64.json
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


####################################### Result Analysis  ######################################
######################################       Begin       ######################################

.PHONY: ucf101_unmasked_results.csv
ucf101_unmasked_results.csv:
> echo "Filename,Attack Name,Attack Epsilon,Top-1 Accuracy,Top-5 Accuracy" > $@
# Baseline model outputs looks like: ucf101_baseline_adversarial_science.json_TIMESTAMP.json
> find results/MARS -name 'ucf101_baseline_adversarial_science.json_*.json' \
                    -exec jq --argjson filename \"{}\" -r \
                            '.results as $$results | if .results.attacks == null then [] else .results.attacks + ["perturbation", "patch", "benign"] end | map([$$filename, ., if . == "benign" then 0, $$results[. + "_mean_categorical_accuracy"], $$results[. + "_mean_top_5_categorical_accuracy"] else $$results[. + "_perturbation_mean_linf", . + "_adversarial_mean_categorical_accuracy", . + "_adversarial_mean_top_5_categorical_accuracy"] end]) | .[] | @csv' {} \; \
                    >> $@
# Model outputs looks like: ucf101_baseline_pretrained_eval_science.json_TIMESTAMP.json
> find results/MARS -name 'ucf101_baseline_pretrained_eval_science.json_*.json' \
                    -exec jq --argjson filename \"{}\" -r \
                            '.results as $$results | if .results.attacks == null then [] else .results.attacks + ["perturbation", "patch", "benign"] end | map([$$filename, ., if . == "benign" then 0, $$results[. + "_mean_categorical_accuracy"], $$results[. + "_mean_top_5_categorical_accuracy"] else $$results[. + "_perturbation_mean_linf", . + "_adversarial_mean_categorical_accuracy", . + "_adversarial_mean_top_5_categorical_accuracy"] end]) | .[] | @csv' {} \; \
                    >> $@
# FGSM results
> find results/MARS -name 'ucf101_baseline_pretrained_bgablation_eval1_science.json_*.json' \
                    -exec jq --argjson filename \"{}\" -r \
                             '[$$filename, "benign", 0, .results.benign_mean_categorical_accuracy, .results.benign_mean_top_5_categorical_accuracy] | @csv' {} \; \
                    >> $@
> find results/MARS -name 'ucf101_baseline_pretrained_bgablation_eval1_science.json_*.json' \
                    -exec jq --argjson filename \"{}\" -r \
                             '[$$filename, "patch", .results.patch_perturbation_mean_linf, .results.patch_adversarial_mean_categorical_accuracy, .results.patch_adversarial_mean_top_5_categorical_accuracy] | @csv' {} \; \
                    >> $@
> find results/MARS -name 'ucf101_baseline_pretrained_bgablation_eval1_science.json_*.json' \
                    -exec jq --argjson filename \"{}\" -r \
                             '[$$filename, "perturbation", .results.perturbation_perturbation_mean_linf, .results.perturbation_adversarial_mean_categorical_accuracy, .results.perturbation_adversarial_mean_top_5_categorical_accuracy] | @csv' {} \; \
                    >> $@
> find results/MARS -name 'ucf101_baseline_pretrained_eval?*_science.json_*.json' \
                    -exec jq --argjson filename \"{}\" -r \
                             '[$$filename, .config.attack.name, .results.attack_perturbation_mean_linf, .results.attack_adversarial_mean_categorical_accuracy, .results.attack_adversarial_mean_top_5_categorical_accuracy] | @csv' {} \; \
                    >> $@

.PHONY: ucf101_masked_results.csv
ucf101_masked_results.csv:
> echo "Filename,Attack Name,Attack Epsilon,Top-1 Accuracy,Top-5 Accuracy" > $@
# Baseline model outputs looks like: ucf101_baseline_adversarial_science.json_TIMESTAMP.json
> find results/MARS -name 'ucf101_baseline_adversarial_science.json_*.json' \
                    -exec jq --argjson filename \"{}\" -r \
                            '.results as $$results | if .results.attacks == null then [] else .results.attacks + ["perturbation", "patch", "benign"] end | map([$$filename, ., if . == "benign" then 0, $$results[. + "_mean_categorical_accuracy"], $$results[. + "_mean_top_5_categorical_accuracy"] else $$results[. + "_perturbation_mean_linf", . + "_adversarial_mean_categorical_accuracy", . + "_adversarial_mean_top_5_categorical_accuracy"] end]) | .[] | @csv' {} \; \
                    >> $@
# BG-Ablated model outputs looks like: ucf101_baseline_pretrained_bgablation_eval_science.json_TIMESTAMP.json
> find results/MARS -name 'ucf101_baseline_pretrained_bgablation_eval_science.json_*.json' \
                    -exec jq --argjson filename \"{}\" -r \
                            '.results as $$results | if .results.attacks == null then [] else .results.attacks + ["perturbation", "patch", "benign"] end | map([$$filename, ., if . == "benign" then 0, $$results[. + "_mean_categorical_accuracy"], $$results[. + "_mean_top_5_categorical_accuracy"] else $$results[. + "_perturbation_mean_linf", . + "_adversarial_mean_categorical_accuracy", . + "_adversarial_mean_top_5_categorical_accuracy"] end]) | .[] | @csv' {} \; \
                    >> $@
# FGSM Results
> find results/MARS -name 'ucf101_baseline_pretrained_eval1_science.json_*.json' \
                    -exec jq --argjson filename \"{}\" -r \
                             '[$$filename, "benign", 0, .results.benign_mean_categorical_accuracy, .results.benign_mean_top_5_categorical_accuracy] | @csv' {} \; \
                    >> $@
> find results/MARS -name 'ucf101_baseline_pretrained_eval1_science.json_*.json' \
                    -exec jq --argjson filename \"{}\" -r \
                             '[$$filename, "patch", .results.patch_perturbation_mean_linf, .results.patch_adversarial_mean_categorical_accuracy, .results.patch_adversarial_mean_top_5_categorical_accuracy] | @csv' {} \; \
                    >> $@
> find results/MARS -name 'ucf101_baseline_pretrained_eval1_science.json_*.json' \
                    -exec jq --argjson filename \"{}\" -r \
                             '[$$filename, "perturbation", .results.perturbation_perturbation_mean_linf, .results.perturbation_adversarial_mean_categorical_accuracy, .results.perturbation_adversarial_mean_top_5_categorical_accuracy] | @csv' {} \; \
                    >> $@
> find results/MARS -name 'ucf101_baseline_pretrained_bgablation_eval?*_science.json_*.json' \
                    -exec jq --argjson filename \"{}\" -r \
                             '[$$filename, .config.attack.name, .results.attack_perturbation_mean_linf, .results.attack_adversarial_mean_categorical_accuracy, .results.attack_adversarial_mean_top_5_categorical_accuracy] | @csv' {} \; \
                    >> $@

.PHONY: ucf101_results.csv
ucf101_results.csv: ucf101_masked_results.csv ucf101_unmasked_results.csv
> echo "Filename,Attack Name,Attack Epsilon,Top-1 Accuracy,Top-5 Accuracy" > $@
> tail -q -n+2 $^ | sort | uniq >> $@

####################################### Result Analysis  ######################################
######################################        End        ######################################


.PHONY: test_ucf101
test_ucf101:
> $(MAKE) results/MARS/RGBMasked_FlowMasked/fulltune_random_ucf101/UCF101/ucf101_baseline_pretrained.json.armory_check
> $(MAKE) results/MARS/RGBMasked_FlowMasked/fulltune_random_ucf101/UCF101/ucf101_baseline_pretrained_bgablation.json.armory_check
> $(MAKE) results/MARS/RGBMasked_FlowMasked/fulltune_random_ucf101/UCF101/ucf101_baseline_pretrained_fgablation.json.armory_check
> $(MAKE) results/MARS/RGBMasked_FlowMasked/fulltune_random_ucf101/UCF101/ucf101_baseline_pretrained_oscar.json.armory_check
> $(MAKE) results/MARS/RGBMasked_FlowMasked/fulltune_random_ucf101/UCF101/ucf101_baseline_pretrained_bgablation_oscar.json.armory_check
> $(MAKE) results/MARS/RGBMasked_FlowMasked/fulltune_random_ucf101/UCF101/ucf101_baseline_pretrained_fgablation_oscar.json.armory_check
> $(MAKE) results/MARS/RGBMasked_FlowMasked/fulltune_random_ucf101/UCF101/ucf101_baseline_pretrained_precomputed_bgablation_oscar.json.armory_check
> $(MAKE) results/MARS/RGBMasked_FlowMasked/fulltune_random_ucf101/UCF101/ucf101_baseline_pretrained_precomputed_fgablation_oscar.json.armory_check
> $(MAKE) results/MARS/RGBMasked_FlowMasked/fulltune_random_ucf101/UCF101/ucf101_baseline_pretrained_preablated_bgablation_oscar.json.armory_check
> $(MAKE) results/MARS/RGBMasked_FlowMasked/fulltune_random_ucf101/UCF101/ucf101_baseline_pretrained_preablated_fgablation_oscar.json.armory_check
> $(MAKE) results/MARS/RGBMasked_FlowMasked/fulltune_random_ucf101/UCF101/ucf101_baseline_pretrained_oscar_science.json.armory_check
> $(MAKE) results/MARS/RGBMasked_FlowMasked/fulltune_random_ucf101/UCF101/ucf101_baseline_pretrained_bgablation_oscar_science.json.armory_check
> $(MAKE) results/MARS/RGBMasked_FlowMasked/fulltune_random_ucf101/UCF101/ucf101_baseline_pretrained_fgablation_oscar_science.json.armory_check
> $(MAKE) results/MARS/RGBMasked_FlowMasked/fulltune_random_ucf101/UCF101/ucf101_baseline_pretrained_oscar_science.json.armory_check
> $(MAKE) results/MARS/RGBMasked_FlowMasked/fulltune_random_ucf101/UCF101/ucf101_baseline_pretrained_bgablation_oscar_science.json.armory_check
> $(MAKE) results/MARS/RGBMasked_FlowMasked/fulltune_random_ucf101/UCF101/ucf101_baseline_pretrained_fgablation_oscar_science.json.armory_check
> $(MAKE) results/MARS/RGBMasked_FlowMasked/fulltune_random_ucf101/UCF101/ucf101_baseline_pretrained_precomputed_bgablation_oscar_science.json.armory_check
> $(MAKE) results/MARS/RGBMasked_FlowMasked/fulltune_random_ucf101/UCF101/ucf101_baseline_pretrained_precomputed_fgablation_oscar_science.json.armory_check
> $(MAKE) results/MARS/RGBMasked_FlowMasked/fulltune_random_ucf101/UCF101/ucf101_baseline_pretrained_preablated_bgablation_oscar_science.json.armory_check
> $(MAKE) results/MARS/RGBMasked_FlowMasked/fulltune_random_ucf101/UCF101/ucf101_baseline_pretrained_preablated_fgablation_oscar_science.json.armory_check
