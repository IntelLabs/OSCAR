# Non robust
DT2_CONFIG = detectron2://COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml
DT2_WEIGHTS = detectron2://COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x/139653917/model_final_2d9806.pkl

# Robust-ish
DT2_ROBUST_R_50_DC5_F2_CONFIG = DT2/COCO-InstanceSegmentation/mask_rcnn_adv_R_50_DC5_3x/config.yaml
DT2_ROBUST_R_50_DC5_F2_WEIGHTS = DT2/COCO-InstanceSegmentation/mask_rcnn_adv_R_50_DC5_3x/model_final.pth

DT2_ROBUST_R_50_FPN_F2_CONFIG = DT2/COCO-InstanceSegmentation/mask_rcnn_adv_R_50_FPN_3x/config.yaml
DT2_ROBUST_R_50_FPN_F2_WEIGHTS = DT2/COCO-InstanceSegmentation/mask_rcnn_adv_R_50_FPN_3x/model_final.pth

# Robust
DT2_ROBUST_R_50_DC5_F5_CONFIG = DT2/COCO-InstanceSegmentation/mask_rcnn_adv_F5_R_50_DC5_3x/config.yaml
DT2_ROBUST_R_50_DC5_F5_WEIGHTS = DT2/COCO-InstanceSegmentation/mask_rcnn_adv_F5_R_50_DC5_3x/model_final.pth

DT2_ROBUST_R_50_FPN_F5_CONFIG = DT2/COCO-InstanceSegmentation/mask_rcnn_adv_F5_R_50_FPN_3x/config.yaml
DT2_ROBUST_R_50_FPN_F5_WEIGHTS = DT2/COCO-InstanceSegmentation/mask_rcnn_adv_F5_R_50_FPN_3x/model_final.pth


$(RESULTS)/DAPRICOT/dapricot_%.json: | $(SCENARIOS)/dapricot_%.json
> mkdir -p $(@D)
> cat $| | $(JQ) ".sysconfig.output_filename = \"$(@F)\"" > $@

$(SCENARIOS)/dapricot_%.json: $(ARMORY_SCENARIOS)/dapricot_%.json $(SCENARIOS)/
> cat $< | $(JQ) '.sysconfig.external_github_repo = ""' > $@

# Non robust
$(SCENARIOS)/dapricot_dt2_masked_pgd.json: $(SCENARIOS)/dapricot_frcnn_masked_pgd.json
> cat $< | $(JQ) '.model.module = "oscar.classifiers.detectron2estimator"' \
         | $(JQ) '.model.model_kwargs.config_file = "armory://$(DT2_CONFIG)"' \
         | $(JQ) '.model.weights_file = "$(DT2_WEIGHTS)"' > $@

# Robust-ish
$(SCENARIOS)/dapricot_dt2_robust_r_50_dc5_f2_masked_pgd.json: $(SCENARIOS)/dapricot_dt2_masked_pgd.json
> cat $< | $(JQ) '.model.model_kwargs.config_file = "armory://$(DT2_ROBUST_R_50_DC5_F2_CONFIG)"' \
         | $(JQ) '.model.weights_file = "$(DT2_ROBUST_R_50_DC5_F2_WEIGHTS)"' > $@

$(SCENARIOS)/dapricot_dt2_robust_r_50_fpn_f2_masked_pgd.json: $(SCENARIOS)/dapricot_dt2_masked_pgd.json
> cat $< | $(JQ) '.model.model_kwargs.config_file = "armory://$(DT2_ROBUST_R_50_FPN_F2_CONFIG)"' \
         | $(JQ) '.model.weights_file = "$(DT2_ROBUST_R_50_FPN_F2_WEIGHTS)"' > $@

# Robust
$(SCENARIOS)/dapricot_dt2_robust_r_50_dc5_f5_masked_pgd.json: $(SCENARIOS)/dapricot_dt2_masked_pgd.json
> cat $< | $(JQ) '.model.model_kwargs.config_file = "armory://$(DT2_ROBUST_R_50_DC5_F5_CONFIG)"' \
         | $(JQ) '.model.weights_file = "$(DT2_ROBUST_R_50_DC5_F5_WEIGHTS)"' > $@

$(SCENARIOS)/dapricot_dt2_robust_r_50_fpn_f5_masked_pgd.json: $(SCENARIOS)/dapricot_dt2_masked_pgd.json
> cat $< | $(JQ) '.model.model_kwargs.config_file = "armory://$(DT2_ROBUST_R_50_FPN_F5_CONFIG)"' \
         | $(JQ) '.model.weights_file = "$(DT2_ROBUST_R_50_FPN_F5_WEIGHTS)"' > $@


# Submission
submission/INTL_dapricot_dt2_config.yaml: $(MODEL_ZOO)/$(DT2_ROBUST_R_50_DC5_F5_CONFIG)
> cat $< > $@

submission/INTL_dapricot_dt2_weights.pth: $(MODEL_ZOO)/$(DT2_ROBUST_R_50_DC5_F5_WEIGHTS)
> cat $< > $@

submission/INTL_dapricot_dt2_masked_pgd.json: $(SCENARIOS)/dapricot_dt2_masked_pgd.json submission/INTL_dapricot_dt2_config.yaml submission/INTL_dapricot_dt2_weights.pth
> cat $< | $(JQ) '.model.model_kwargs.config_file = "armory://INTL_dapricot_dt2_config.yaml"' \
         | $(JQ) '.model.weights_file = "INTL_dapricot_dt2_weights.pth"' \
         | $(JQ) '.sysconfig.external_github_repo = ["IntelLabs/OSCAR@gard-eval3"]' \
         | $(JQ) '.sysconfig.use_gpu = false' > $@

.PHONY: dapricot_submission
dapricot_submission: submission/INTL_dapricot_dt2_masked_pgd.json
> @echo "Created DAPRICOT submission!"

.PHONY: run_dapricot_submission
run_dapricot_submission: dapricot_submission | .venv
> cp submission/INTL_dapricot_dt2_config.yaml $(ARMORY_SAVED_MODEL_DIR)
> cp submission/INTL_dapricot_dt2_weights.pth $(ARMORY_SAVED_MODEL_DIR)
> $(POETRY) run armory run submission/INTL_dapricot_dt2_masked_pgd.json $(ARGS)
