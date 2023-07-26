$(MODEL_ZOO)/CARLA_TRACKING:
> mkdir -p $@

$(MODEL_ZOO)/CARLA_TRACKING/pytorch_goturn.pth: $(MODEL_ZOO)/CARLA_TRACKING
> $(POETRY) run gdown https://drive.google.com/uc?id=1szpx3J-hfSrBEi_bze3d0PjSfQwNij7X -O $@

#########################################################################
# Eval 4 carla object tracking
#########################################################################

# test the following config with:
# make clean_scenarios && make results/CARLA_TRACKING/carla-goturn-background_subtraction.json.armory_check
$(SCENARIOS)/carla-goturn-background_subtraction.json: $(ARMORY_SCENARIOS)/carla_video_tracking_goturn_advtextures_undefended.json | $(SCENARIOS)/
> cat $< | $(JQ) '.attack.kwargs.max_iter = 1000' \
         | $(JQ) '.defense = {}' \
         | $(JQ) '.defense.kwargs = {}' \
         | $(JQ) '.defense.module = "oscar.defences.preprocessor.background_subtraction"' \
         | $(JQ) '.defense.name = "BackgroundSubtraction"' \
         | $(JQ) '.defense.type = "Preprocessor"' \
         | $(JQ) '.sysconfig.use_gpu = true' \
         | $(JQ) '.scenario.export_samples = 20' > $@

#########################################################################
# Eval 4 carla object tracking - Generate preprocessed frames
#########################################################################

NUM_SAVED_SAMPLES ?= 3
SEQ_SAVED_SAMPLES := $(shell seq 0 1 $$(($(NUM_SAVED_SAMPLES) - 1)))

results/CARLA_TRACKING/%/saved_samples/.done_benign_bgmask_videos:
> for SAMPLE_ID in $(SEQ_SAVED_SAMPLES); do \
        $(POETRY) run python -m oscar.defences.preprocessor.background_subtraction \
            "$(abspath $(@D))/$${SAMPLE_ID}/frame_*_benign.png"; \
        cd $(@D)/$${SAMPLE_ID}; \
        ffmpeg -r 10 -f image2 -s 800x600 -i frame_%04d_benign_bgmask.png \
            -vcodec libx264 -pix_fmt yuv420p video_benign_bgmask.mp4; \
        cd $(abspath .); \
    done
> touch $@

results/CARLA_TRACKING/%/saved_samples/.done_adversarial_bgmask_videos:
> for SAMPLE_ID in $(SEQ_SAVED_SAMPLES); do \
        $(POETRY) run python -m oscar.defences.preprocessor.background_subtraction \
            "$(abspath $(@D))/$${SAMPLE_ID}/frame_*_adversarial.png"; \
        cd $(@D)/$${SAMPLE_ID}; \
        ffmpeg -r 10 -f image2 -s 800x600 -i frame_%04d_adversarial_bgmask.png \
            -vcodec libx264 -pix_fmt yuv420p video_adversarial_bgmask.mp4; \
        cd $(abspath .); \
    done
> touch $@

results/CARLA_TRACKING/%/saved_samples/.done_vis_tracking: \
    results/CARLA_TRACKING/%/saved_samples/.done_adversarial_bgmask_videos
> $(POETRY) run python bin/visualize_tracking.py $(abspath $(@D)) --glob "frame_*_adversarial_bgmask.png"
> touch $@

results/CARLA_TRACKING/%/saved_samples/.done_adversarial_bgmask_tracking_videos: \
    results/CARLA_TRACKING/%/saved_samples/.done_vis_tracking
> for SAMPLE_ID in $(SEQ_SAVED_SAMPLES); do \
        cd $(@D)/$${SAMPLE_ID}; \
        ffmpeg -r 10 -f image2 -s 800x600 -i frame_%04d_adversarial_bgmask_tracking.png \
            -vcodec libx264 -pix_fmt yuv420p video_adversarial_bgmask_tracking.mp4; \
        cd $(abspath .); \
    done
> touch $@

# One target to make all videos
# e.g., make results/CARLA_TRACKING/<EVAL_ID>/saved_samples/.done_videos NUM_SAVED_SAMPLES=42
results/CARLA_TRACKING/%/saved_samples/.done_videos: \
    results/CARLA_TRACKING/%/saved_samples/.done_benign_bgmask_videos \
    results/CARLA_TRACKING/%/saved_samples/.done_adversarial_bgmask_videos \
    results/CARLA_TRACKING/%/saved_samples/.done_adversarial_bgmask_tracking_videos
> touch $@

#########################################################################
# Eval 4 carla object tracking - Submission
#########################################################################

submission/INTL_carla_video_tracking_goturn_advtextures_bgmask.json: \
    $(SCENARIOS)/carla-goturn-background_subtraction.json | submission/
> cat $< | $(JQ) 'del(.scenario.export_samples)' \
         | $(JQ) '.sysconfig.external_github_repo = ["IntelLabs/OSCAR@gard-eval4", "amoudgl/pygoturn"]' \
         | $(JQ) '.sysconfig.use_gpu = false' > $@

.PHONY: carla_tracking_submission
carla_tracking_submission: submission/INTL_carla_video_tracking_goturn_advtextures_bgmask.json
> @echo "Created CARLA Tracking submission!"

.PHONY: run_carla_tracking_submission
run_carla_tracking_submission: carla_tracking_submission | .venv
> $(POETRY) run armory run submission/INTL_carla_video_tracking_goturn_advtextures_bgmask.json $(ARGS)

#########################################################################
# Eval 5 carla object tracking
#########################################################################

# undefended
$(SCENARIOS)/carla-goturn-eval5-undefended.json: $(ARMORY_SCENARIOS)/carla_video_tracking.json | $(SCENARIOS)/
> cat $< | $(JQ) '.attack.kwargs.max_iter = 100' \
         | $(JQ) '.sysconfig.use_gpu = true' \
         | $(JQ) '.sysconfig.num_eval_batches = 20' \
         | $(JQ) '.scenario.export_batches = 20' > $@

# test the dynamic background subtraction defense (eval5 submission) with:
# make clean_scenarios && make results/CARLA_TRACKING/carla-goturn-dynamic_background_subtraction.json.armory_run
$(SCENARIOS)/carla-goturn-dynamic_background_subtraction.json: $(SCENARIOS)/carla-goturn-eval5-undefended.json
> cat $< | $(JQ) '.defense = {}' \
         | $(JQ) '.defense.module = "oscar.defences.preprocessor.dynamic_background_subtraction"' \
         | $(JQ) '.defense.name = "DynamicBackgroundSubtraction"' \
         | $(JQ) '.defense.type = "Preprocessor"' \
         | $(JQ) '.defense.kwargs = {}' \
         | $(JQ) '.defense.kwargs.orb_good_match_percent = 1' \
         | $(JQ) '.defense.kwargs.orb_levels = 5' \
         | $(JQ) '.defense.kwargs.orb_scale_factor = 1.05' \
         | $(JQ) '.defense.kwargs.orb_gaussian_ksize = [5,5]' \
         | $(JQ) '.defense.kwargs.subtraction_gaussian_ksize = [5,5]' \
         | $(JQ) '.defense.kwargs.median_ksize = [9,9]' \
         | $(JQ) '.defense.kwargs.bg_sub_thre = 25/255' > $@

#########################################################################
# Eval 5 carla object tracking - Generate preprocessed frames
#########################################################################

# make results/CARLA_TRACKING/vid012-fp2-thre20/saved_samples/.done_eval5_adversarial_bgmask_videos
results/CARLA_TRACKING/%/saved_samples/.done_eval5_adversarial_bgmask_videos:
> for SAMPLE_ID in $(SEQ_SAVED_SAMPLES); do \
        $(POETRY) run python -m oscar.defences.preprocessor.dynamic_background_subtraction \
            "$(abspath $(@D))/$${SAMPLE_ID}/frame_*_adversarial.png"; \
        cd $(@D)/$${SAMPLE_ID}; \
        ffmpeg -r 5 -f image2 -s 1280x960 -i frame_%04d_adversarial_bgmask.png \
            -vcodec libx264 -pix_fmt yuv420p video_adversarial_bgmask.mp4; \
        cd $(abspath .); \
    done
> touch $@

# make results/CARLA_TRACKING/vid012-fp2-thre20/saved_samples/.done_eval5_vis_tracking
results/CARLA_TRACKING/%/saved_samples/.done_eval5_vis_tracking: \
    results/CARLA_TRACKING/%/saved_samples/.done_eval5_adversarial_bgmask_videos
> $(POETRY) run python bin/visualize_tracking.py $(abspath $(@D)) --glob "frame_*_adversarial_bgmask.png"
> touch $@

# make results/CARLA_TRACKING/vid012-nbm-thre20/saved_samples/.done_eval5_adversarial_bgmask_tracking_videos
results/CARLA_TRACKING/%/saved_samples/.done_eval5_adversarial_bgmask_tracking_videos: \
    results/CARLA_TRACKING/%/saved_samples/.done_eval5_vis_tracking
> for SAMPLE_ID in $(SEQ_SAVED_SAMPLES); do \
        cd $(@D)/$${SAMPLE_ID}; \
        ffmpeg -r 5 -f image2 -s 1280x960 -i frame_%04d_adversarial_bgmask_tracking.png \
            -vcodec libx264 -pix_fmt yuv420p video_adversarial_bgmask_tracking.mp4; \
        cd $(abspath .); \
    done
> touch $@

#########################################################################
# Eval 5 carla object tracking - Submission
#########################################################################

# make clean_scenarios && CUDA_VISIBLE_DEVICES=1 make results/CARLA_TRACKING/carla-goturn-dynamic_background_subtraction.json.armory_run
submission/INTL_carla_video_tracking_goturn_advtextures_dynamic_bg_sub.json: \
    $(ARMORY_SCENARIOS)/eval5/carla_video_tracking/carla_video_tracking_goturn_advtextures_undefended.json | submission/
> cat $< | $(JQ) '.sysconfig.external_github_repo = ["IntelLabs/OSCAR@gard-eval5", "amoudgl/pygoturn"]' \
         | $(JQ) '.sysconfig.use_gpu = true' \
         | $(JQ) '.sysconfig.docker_image = "$(DOCKER_IMAGE_TAG_OSCAR)"' \
         | $(JQ) '.defense = {}' \
         | $(JQ) '.defense.module = "oscar.defences.preprocessor.dynamic_background_subtraction"' \
         | $(JQ) '.defense.name = "DynamicBackgroundSubtraction"' \
         | $(JQ) '.defense.type = "Preprocessor"' \
         | $(JQ) '.defense.kwargs = {}' \
         | $(JQ) '.defense.kwargs.orb_good_match_percent = 1' \
         | $(JQ) '.defense.kwargs.orb_levels = 5' \
         | $(JQ) '.defense.kwargs.orb_scale_factor = 1.05' \
         | $(JQ) '.defense.kwargs.orb_gaussian_ksize = [5,5]' \
         | $(JQ) '.defense.kwargs.subtraction_gaussian_ksize = [5,5]' \
         | $(JQ) '.defense.kwargs.median_ksize = [9,9]' \
         | $(JQ) '.defense.kwargs.bg_sub_thre = 25/255' > $@

#########################################################################
# Eval 6 carla object tracking
#########################################################################
# Undfended
$(SCENARIOS)/carla_mot_adversarialpatch_undefended.json: $(ARMORY_SCENARIOS)/eval6/carla_mot/carla_mot_adversarialpatch_undefended.json | $(SCENARIOS)/
> cat $< | $(JQ) '.attack.kwargs.max_iter = 5' \
         | $(JQ) '.attack.kwargs.batch_frame_size = 3' \
         | $(JQ) '.model.model_kwargs.min_size = 960 ' \
         | $(JQ) '.model.model_kwargs.max_size = 1280 ' \
         | $(JQ) '.sysconfig.use_gpu = true' \
	     | $(JQ) '.sysconfig.num_eval_batches = 20' \
         | $(JQ) '.sysconfig.docker_image = "$(DOCKER_IMAGE_TAG_OSCAR)"' > $@

# Defended - Bgsub+SS or Bgsub alone
$(SCENARIOS)/carla_mot_adversarialpatch_ss_bgsub.json: $(SCENARIOS)/carla_mot_adversarialpatch_undefended.json | $(SCENARIOS)/
> cat $< | $(JQ) '.defense = {}' \
         | $(JQ) '.defense.type = "Preprocessor"' \
     	 | $(JQ) '.defense.kwargs = {}' \
         | $(JQ) '.defense.module = "oscar.defences.preprocessor.dynamic_background_subtraction"' \
         | $(JQ) '.defense.name = "DynamicBackgroundSubtraction"' \
         | $(JQ) '.defense.kwargs.orb_good_match_percent = 0.15' \
         | $(JQ) '.defense.kwargs.orb_levels = 8' \
         | $(JQ) '.defense.kwargs.orb_scale_factor = 1.2' \
         | $(JQ) '.defense.kwargs.bg_sub_thre = 25/255' \
	     | $(JQ) '.defense.kwargs.postfilter_ss = "false"' \
	     | $(JQ) '.defense.kwargs.prefilter_ss = "false"' \
         | $(JQ) '.defense.kwargs.edge_filter_size = 5' \
	     | $(JQ) '.defense.kwargs.gaussian_filter_size = 11' \
         | $(JQ) '.defense.kwargs.edge_sub_thres = 15/255' > $@

# Defended - SS only
$(SCENARIOS)/carla_mot_adversarialpatch_ss.json: $(SCENARIOS)/carla_mot_adversarialpatch_undefended.json | $(SCENARIOS)/
> cat $< | $(JQ) '.defense = {}' \
         | $(JQ) '.defense.type = "Preprocessor"' \
     	 | $(JQ) '.defense.kwargs = {}' \
         | $(JQ) '.defense.module = "oscar.defences.preprocessor.small_scale_filter"' \
         | $(JQ) '.defense.name = "SmallScaleFilter"' \
         | $(JQ) '.defense.kwargs.edge_filter_size = 5' \
	     | $(JQ) '.defense.kwargs.gaussian_filter_size = 11' \
         | $(JQ) '.defense.kwargs.edge_sub_thres = 15/255' > $@


# Defended - single video
$(SCENARIOS)/carla_mot_adversarialpatch_ssfilter_vid_%.json: $(SCENARIOS)/carla_mot_adversarialpatch_ssfilter.json | $(SCENARIOS)/
> cat $< | $(JQ) '.sysconfig.num_eval_batches = 1' \
         | $(JQ) '.dataset.index = [$*]' > $@

################
#Eval7
################
#Undefended
$(SCENARIOS)/carla_mot_adversarialpatch_undefended.json: $(ARMORY_SCENARIOS)/eval7/carla_mot/carla_mot_adversarialpatch_undefended.json | $(SCENARIOS)/
> cat $< | $(JQ) '.attack.kwargs.max_iter = 5' \
         | $(JQ) '.attack.kwargs.batch_frame_size = 3' \
         | $(JQ) '.model.model_kwargs.min_size = 960 ' \
         | $(JQ) '.model.model_kwargs.max_size = 1280 ' \
         | $(JQ) '.sysconfig.use_gpu = true' \
	     | $(JQ) '.sysconfig.num_eval_batches = 20' \
         | $(JQ) '.sysconfig.docker_image = "$(DOCKER_IMAGE_TAG_OSCAR)"' > $@

# Defended - Bgsub alone
$(SCENARIOS)/INTL_mot_bgsub.json: $(SCENARIOS)/carla_mot_adversarialpatch_undefended.json
> cat $< | $(JQ) '.defense = {}' \
         | $(JQ) '.attack.kwargs.batch_frame_size = 20' \
	     | $(JQ) '.attack.kwargs.max_iter = 5' \
         | $(JQ) '.defense.type = "Preprocessor"' \
     	 | $(JQ) '.defense.kwargs = {}' \
         | $(JQ) '.defense.module = "oscar.defences.preprocessor.dynamic_background_subtraction"' \
         | $(JQ) '.defense.name = "DynamicBackgroundSubtraction"' \
         | $(JQ) '.defense.kwargs.orb_good_match_percent = 0.1' \
         | $(JQ) '.defense.kwargs.orb_levels = 9' \
         | $(JQ) '.defense.kwargs.orb_scale_factor = 1.25' \
         | $(JQ) '.defense.kwargs.bg_sub_thre = 7/255' \
	     | $(JQ) '.defense.kwargs.postfilter_ss = "false"' \
	     | $(JQ) '.defense.kwargs.prefilter_ss = "false"' \
         | $(JQ) '.sysconfig.docker_image = "intellabs/oscar:0.18.0"' \
         | $(JQ) '.sysconfig.external_github_repo = ["IntelLabs/OSCAR@gard-eval7", "JonathonLuiten/TrackEval"]' > $@
