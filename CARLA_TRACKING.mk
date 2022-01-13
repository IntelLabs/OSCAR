$(MODEL_ZOO)/CARLA_TRACKING:
> mkdir -p $@

$(MODEL_ZOO)/CARLA_TRACKING/pytorch_goturn.pth: $(MODEL_ZOO)/CARLA_TRACKING
> $(POETRY) run gdown https://drive.google.com/uc?id=1szpx3J-hfSrBEi_bze3d0PjSfQwNij7X -O $@

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

##################################################

NUM_SAVED_SAMPLES ?= 1
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

##################################################
# Submission
##################################################

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
