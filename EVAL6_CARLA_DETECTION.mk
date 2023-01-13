# Download and extract dataset of carla_over_obj_det
CARLA_OVERHEAD_DATASET = $(DATASETS)/carla_over_obj_det/train/kwcoco_annotations_all.json $(DATASETS)/carla_over_obj_det/val/kwcoco_annotations_all.json

$(CARLA_OVERHEAD_DATASET):
> $(error Missing dataset)

%/eval6_dev_benign/test_prediction.json: %/.hydra/config.yaml %/.hydra/hydra.yaml %/checkpoints/last.ckpt
> $(MART) task_name=$(shell $(YQ) .task_name < $*/.hydra/config.yaml) \
          experiment=$(shell $(YQ) .hydra.runtime.choices.experiment < $*/.hydra/hydra.yaml) \
          resume=$*/checkpoints/last.ckpt \
          fit=false test=true \
          hydra.run.dir=$(@D) \
          datamodule.test_dataset.root=$(DATASETS)/carla_over_obj_det/dev \
          datamodule.test_dataset.annFile=$(DATASETS)/carla_over_obj_det/dev/kwcoco_annotations.json \
          datamodule.ims_per_batch=1

%/eval6_test_benign/test_prediction.json: %/.hydra/config.yaml %/.hydra/hydra.yaml %/checkpoints/last.ckpt
> $(MART) task_name=$(shell $(YQ) .task_name < $*/.hydra/config.yaml) \
          experiment=$(shell $(YQ) .hydra.runtime.choices.experiment < $*/.hydra/hydra.yaml) \
          resume=$*/checkpoints/last.ckpt \
          fit=false test=true \
          hydra.run.dir=$(@D) \
          datamodule.test_dataset.root=$(DATASETS)/carla_over_obj_det/test \
          datamodule.test_dataset.annFile=$(DATASETS)/carla_over_obj_det/test/kwcoco_annotations.json \
          datamodule.ims_per_batch=1

.PHONY: carla_over_train
carla_over_train: ArmoryCarlaOverObjDet_TorchvisionFasterRCNN ## Train Faster R-CNN with the CarlaOverObjDet dataset from Armory.

.PHONY: ArmoryCarlaOverObjDet_TorchvisionFasterRCNN
ArmoryCarlaOverObjDet_TorchvisionFasterRCNN: .venv $(CARLA_OVERHEAD_DATASET) ## Train Faster R-CNN on truncated annotations but validation and test on all annotations
> $(MART) experiment=ArmoryCarlaOverObjDetAll_TorchvisionFasterRCNN \
          task_name=$@ \
          "datamodule.train_dataset.annFile=$$\{paths.data_dir\}/carla_over_obj_det/train/kwcoco_annotations.json"

.PHONY: carla_over_train_all
carla_over_train_all: ArmoryCarlaOverObjDetAll_TorchvisionFasterRCNN ## Train Faster R-CNN with the CarlaOverObjDet dataset from Armory.

.PHONY: ArmoryCarlaOverObjDetAll_TorchvisionFasterRCNN
ArmoryCarlaOverObjDetAll_TorchvisionFasterRCNN: .venv $(CARLA_OVERHEAD_DATASET) ## Train Faster R-CNN on all annotations and validate and test on all annotations
> $(MART) experiment=ArmoryCarlaOverObjDetAll_TorchvisionFasterRCNN \
          task_name=$@ \
          "+model.modules.losses_and_detections.model.model.trainable_backbone_layers=5" \
          "+model.modules.losses_and_detections.model.model.min_size=960" \
          "+model.modules.losses_and_detections.model.model.max_size=1280"

.PHONY: ArmoryCarlaOverObjDetAll_TorchvisionFasterRCNN_betteranchors
ArmoryCarlaOverObjDetAll_TorchvisionFasterRCNN_betteranchors: .venv $(CARLA_OVERHEAD_DATASET) ## Train Faster R-CNN with better anchors on all annotations and validate and test on all annotations
> $(MART) experiment=ArmoryCarlaOverObjDetAll_TorchvisionFasterRCNN \
          task_name=$@ \
          model=torchvision_faster_rcnn_better_anchors \
          "+model.modules.losses_and_detections.model.model.trainable_backbone_layers=5" \
          "+model.modules.losses_and_detections.model.model.min_size=960" \
          "+model.modules.losses_and_detections.model.model.max_size=1280"

# modular faster rcnn implementation should be similar to torchvision
.PHONY: ArmoryCarlaOverObjDetAll_FasterRCNN
ArmoryCarlaOverObjDetAll_FasterRCNN: .venv $(CARLA_OVERHEAD_DATASET)
> $(MART) experiment=ArmoryCarlaOverObjDetAll_FasterRCNN \
          task_name=$@

# train no layers
.PHONY: ArmoryCarlaOverObjDetAll_FasterRCNN_trainable0
ArmoryCarlaOverObjDetAll_FasterRCNN_trainable0: .venv $(CARLA_OVERHEAD_DATASET)
> $(MART) experiment=ArmoryCarlaOverObjDetAll_FasterRCNN \
          task_name=$@ \
          model.modules.backbone.trainable_layers=0

# train 1 layer
.PHONY: ArmoryCarlaOverObjDetAll_FasterRCNN_trainable1
ArmoryCarlaOverObjDetAll_FasterRCNN_trainable1: .venv $(CARLA_OVERHEAD_DATASET)
> $(MART) experiment=ArmoryCarlaOverObjDetAll_FasterRCNN \
          task_name=$@ \
          model.modules.backbone.trainable_layers=1

# train 2 layers
ArmoryCarlaOverObjDetAll_FasterRCNN_trainable2: .venv $(CARLA_OVERHEAD_DATASET)
> $(MART) experiment=ArmoryCarlaOverObjDetAll_FasterRCNN \
          task_name=$@ \
          model.modules.backbone.trainable_layers=2

# train 3 layers
.PHONY: ArmoryCarlaOverObjDetAll_FasterRCNN_trainable3
ArmoryCarlaOverObjDetAll_FasterRCNN_trainable3: .venv $(CARLA_OVERHEAD_DATASET)
> $(MART) experiment=ArmoryCarlaOverObjDetAll_FasterRCNN \
          task_name=$@ \
          model.modules.backbone.trainable_layers=3

# train 4 layers
.PHONY: ArmoryCarlaOverObjDetAll_FasterRCNN_trainable4
ArmoryCarlaOverObjDetAll_FasterRCNN_trainable4: .venv $(CARLA_OVERHEAD_DATASET)
> $(MART) experiment=ArmoryCarlaOverObjDetAll_FasterRCNN \
          task_name=$@ \
          model.modules.backbone.trainable_layers=4

# better anchors
.PHONY: ArmoryCarlaOverObjDetAll_FasterRCNN_betteranchors
ArmoryCarlaOverObjDetAll_FasterRCNN_betteranchors: .venv $(CARLA_OVERHEAD_DATASET)
> $(MART) experiment=ArmoryCarlaOverObjDetAll_FasterRCNN \
          task_name=$@ \
          model.modules.rpn.head.num_anchors=9 \
          +model.modules.rpn.anchor_generator._convert_=partial \
          "model.modules.rpn.anchor_generator.sizes.0=[0, 4, 16]" \
          "model.modules.rpn.anchor_generator.sizes.1=[0, 8, 32]" \
          "model.modules.rpn.anchor_generator.sizes.2=[4, 16, 64]" \
          "model.modules.rpn.anchor_generator.sizes.3=[8, 32, 128]" \
          "model.modules.rpn.anchor_generator.sizes.4=[128, 256, 512]"

.PHONY: ArmoryCarlaOverObjDetAll_ModularFasterRCNN
ArmoryCarlaOverObjDetAll_ModularFasterRCNN: .venv $(CARLA_OVERHEAD_DATASET)
> $(MART) experiment=ArmoryCarlaOverObjDetAll_ModularFasterRCNN \
          task_name=$@

# smaller anchors
.PHONY: ArmoryCarlaOverObjDetAll_FasterRCNN_betteranchors2
ArmoryCarlaOverObjDetAll_FasterRCNN_betteranchors2: .venv $(CARLA_OVERHEAD_DATASET)
> $(MART) experiment=ArmoryCarlaOverObjDetAll_FasterRCNN \
          task_name=$@ \
          "model.modules.rpn.anchor_generator.sizes=[16, 32, 64, 128, 256]"

.PHONY: ArmoryCarlaOverObjDetAll_FasterRCNN_betteranchors2_longer
ArmoryCarlaOverObjDetAll_FasterRCNN_betteranchors2_longer: .venv $(CARLA_OVERHEAD_DATASET)
> $(MART) experiment=ArmoryCarlaOverObjDetAll_FasterRCNN \
          task_name=$@ \
          "model.modules.rpn.anchor_generator.sizes=[16, 32, 64, 128, 256]"
          trainer.max_steps=$(shell python -c "import math; print(math.ceil(3600/2 * 12))")

# smaller anchors
.PHONY: ArmoryCarlaOverObjDetAll_FasterRCNN_betteranchors3
ArmoryCarlaOverObjDetAll_FasterRCNN_betteranchors3: .venv $(CARLA_OVERHEAD_DATASET)
> $(MART) experiment=ArmoryCarlaOverObjDetAll_FasterRCNN \
          task_name=$@ \
          "model.modules.rpn.anchor_generator.sizes=[8, 16, 32, 64, 128]"

# even smaller anchors
.PHONY: ArmoryCarlaOverObjDetAll_FasterRCNN_betteranchors4
ArmoryCarlaOverObjDetAll_FasterRCNN_betteranchors4: .venv $(CARLA_OVERHEAD_DATASET)
> $(MART) experiment=ArmoryCarlaOverObjDetAll_FasterRCNN \
          task_name=$@ \
          "model.modules.rpn.anchor_generator.sizes=[4, 8, 16, 32, 64]"

# even smaller anchors with 1/2 learning rate
.PHONY: ArmoryCarlaOverObjDetAll_FasterRCNN_betteranchors4_lr0.00625
ArmoryCarlaOverObjDetAll_FasterRCNN_betteranchors4_lr0.00625: .venv $(CARLA_OVERHEAD_DATASET)
> $(MART) experiment=ArmoryCarlaOverObjDetAll_FasterRCNN \
          task_name=$@ \
          "model.modules.rpn.anchor_generator.sizes=[4, 8, 16, 32, 64]"
          model.optimizer.lr=0.00625

# even smaller anchors with 1/4 learning rate
.PHONY: ArmoryCarlaOverObjDetAll_FasterRCNN_betteranchors4_lr0.003125
ArmoryCarlaOverObjDetAll_FasterRCNN_betteranchors4_lr0.003125: .venv $(CARLA_OVERHEAD_DATASET)
> $(MART) experiment=ArmoryCarlaOverObjDetAll_FasterRCNN \
          task_name=$@ \
          "model.modules.rpn.anchor_generator.sizes=[4, 8, 16, 32, 64]"
          model.optimizer.lr=0.003125

# smaller anchors with no last-level max pool
.PHONY: ArmoryCarlaOverObjDetAll_FasterRCNN_betteranchors5
ArmoryCarlaOverObjDetAll_FasterRCNN_betteranchors5: .venv $(CARLA_OVERHEAD_DATASET)
> $(MART) experiment=ArmoryCarlaOverObjDetAll_FasterRCNN \
          task_name=$@ \
          "model.modules.rpn.anchor_generator.sizes=[16, 32, 64, 128]" \
          model.modules.backbone.extra_blocks=null

# even smaller anchors with no last-level max pool
.PHONY: ArmoryCarlaOverObjDetAll_FasterRCNN_betteranchors6
ArmoryCarlaOverObjDetAll_FasterRCNN_betteranchors6: .venv $(CARLA_OVERHEAD_DATASET)
> $(MART) experiment=ArmoryCarlaOverObjDetAll_FasterRCNN \
          task_name=$@ \
          "model.modules.rpn.anchor_generator.sizes=[8, 16, 32, 64]" \
          model.modules.backbone.extra_blocks=null

# smaller anchors with removed layer4 and last-level max pool
.PHONY: ArmoryCarlaOverObjDetAll_FasterRCNN_betteranchors7
ArmoryCarlaOverObjDetAll_FasterRCNN_betteranchors7: .venv $(CARLA_OVERHEAD_DATASET)
> $(MART) experiment=ArmoryCarlaOverObjDetAll_FasterRCNN \
          task_name=$@ \
          "model.modules.rpn.anchor_generator.sizes=[16, 32, 64]" \
          "~model.modules.backbone.return_layers.layer4" \
          "model.modules.backbone.in_channels_list=[256, 512, 1024]" \
          model.modules.backbone.extra_blocks=null

# smaller anchors with smaller representation
.PHONY: ArmoryCarlaOverObjDetAll_FasterRCNN_betteranchors8
ArmoryCarlaOverObjDetAll_FasterRCNN_betteranchors8: .venv $(CARLA_OVERHEAD_DATASET)
> $(MART) experiment=ArmoryCarlaOverObjDetAll_FasterRCNN \
          task_name=$@ \
          "model.modules.rpn.anchor_generator.sizes=[16, 32, 64, 128, 256]"
          model.modules.backbone.out_channels=128 \
          model.modules.rpn.head.in_channels=128 \
          model.modules.box_head.box_head.in_channels=$(shell python -c "print(128*7**2)")

# smaller anchors with roialignv2
.PHONY: ArmoryCarlaOverObjDetAll_FasterRCNN_betteranchors9
ArmoryCarlaOverObjDetAll_FasterRCNN_betteranchors9: .venv $(CARLA_OVERHEAD_DATASET)
> $(MART) experiment=ArmoryCarlaOverObjDetAll_FasterRCNN \
          task_name=$@ \
          "model.modules.rpn.anchor_generator.sizes=[16, 32, 64, 128, 256]"
          model.modules.box_head.box_roi_pool.aligned=true

# Remove layer4 but keep last-level max pool on layer3
.PHONY: ArmoryCarlaOverObjDetAll_FasterRCNN_betteranchors10
ArmoryCarlaOverObjDetAll_FasterRCNN_betteranchors10: .venv $(CARLA_OVERHEAD_DATASET)
> $(MART) experiment=ArmoryCarlaOverObjDetAll_FasterRCNN \
          task_name=$@ \
          "model.modules.rpn.anchor_generator.sizes=[16, 32, 64, 128]" \
          "~model.modules.backbone.return_layers.layer4" \
          "model.modules.backbone.in_channels_list=[256, 512, 1024]"

.PHONY: ArmoryCarlaOverObjDetAll_FasterRCNN_betteranchors10_lr0.00625
ArmoryCarlaOverObjDetAll_FasterRCNN_betteranchors10_lr0.00625: .venv $(CARLA_OVERHEAD_DATASET)
> $(MART) experiment=ArmoryCarlaOverObjDetAll_FasterRCNN \
          task_name=$@ \
          "model.modules.rpn.anchor_generator.sizes=[16, 32, 64, 128]" \
          "~model.modules.backbone.return_layers.layer4" \
          "model.modules.backbone.in_channels_list=[256, 512, 1024]" \
          model.optimizer.lr=0.00625

.PHONY: ArmoryCarlaOverObjDetAll_FasterRCNN_betteranchors10_lr0.01
ArmoryCarlaOverObjDetAll_FasterRCNN_betteranchors10_lr0.01: .venv $(CARLA_OVERHEAD_DATASET)
> $(MART) experiment=ArmoryCarlaOverObjDetAll_FasterRCNN \
          task_name=$@ \
          "model.modules.rpn.anchor_generator.sizes=[16, 32, 64, 128]" \
          "~model.modules.backbone.return_layers.layer4" \
          "model.modules.backbone.in_channels_list=[256, 512, 1024]" \
          model.optimizer.lr=0.01

# smaller anchors with removed layer4 and last-level max pool and roialignv2
.PHONY: ArmoryCarlaOverObjDetAll_FasterRCNN_betteranchors11
ArmoryCarlaOverObjDetAll_FasterRCNN_betteranchors11: .venv $(CARLA_OVERHEAD_DATASET)
> $(MART) experiment=ArmoryCarlaOverObjDetAll_FasterRCNN \
          task_name=$@ \
          "model.modules.rpn.anchor_generator.sizes=[16, 32, 64]" \
          "~model.modules.backbone.return_layers.layer4" \
          "model.modules.backbone.in_channels_list=[256, 512, 1024]" \
          model.modules.backbone.extra_blocks=null \
          model.modules.box_head.box_roi_pool.aligned=true

# smaller anchors with no last-level max pool and roialignv2
.PHONY: ArmoryCarlaOverObjDetAll_FasterRCNN_betteranchors12_lr0.00625
ArmoryCarlaOverObjDetAll_FasterRCNN_betteranchors12_lr0.00625: .venv $(CARLA_OVERHEAD_DATASET)
> $(MART) experiment=ArmoryCarlaOverObjDetAll_FasterRCNN \
          task_name=$@ \
          "model.modules.rpn.anchor_generator.sizes=[16, 32, 64, 128]" \
          model.modules.backbone.extra_blocks=null \
          model.modules.box_head.box_roi_pool.aligned=true \
          model.optimizer.lr=0.00625

.PHONY: ArmoryCarlaOverObjDetAll_FasterRCNN_betteranchors12
ArmoryCarlaOverObjDetAll_FasterRCNN_betteranchors12: .venv $(CARLA_OVERHEAD_DATASET)
> $(MART) experiment=ArmoryCarlaOverObjDetAll_FasterRCNN \
          task_name=$@ \
          "model.modules.rpn.anchor_generator.sizes=[16, 32, 64, 128]" \
          model.modules.backbone.extra_blocks=null \
          model.modules.box_head.box_roi_pool.aligned=true

# Remove layer4 but keep last-level max pool on layer3 and roialignv2
.PHONY: ArmoryCarlaOverObjDetAll_FasterRCNN_betteranchors13
ArmoryCarlaOverObjDetAll_FasterRCNN_betteranchors13: .venv $(CARLA_OVERHEAD_DATASET)
> $(MART) experiment=ArmoryCarlaOverObjDetAll_FasterRCNN \
          task_name=$@ \
          "model.modules.rpn.anchor_generator.sizes=[16, 32, 64, 128]" \
          "~model.modules.backbone.return_layers.layer4" \
          "model.modules.backbone.in_channels_list=[256, 512, 1024]" \
          model.modules.box_head.box_roi_pool.aligned=true

# even smaller anchors with no last-level max pool and roialignv2
.PHONY: ArmoryCarlaOverObjDetAll_FasterRCNN_betteranchors14
ArmoryCarlaOverObjDetAll_FasterRCNN_betteranchors14: .venv $(CARLA_OVERHEAD_DATASET)
> $(MART) experiment=ArmoryCarlaOverObjDetAll_FasterRCNN \
          task_name=$@ \
          "model.modules.rpn.anchor_generator.sizes=[8, 16, 32, 64]" \
          model.modules.backbone.extra_blocks=null \
          model.modules.box_head.box_roi_pool.aligned=true

.PHONY: ArmoryCarlaOverObjDetAll_FasterRCNN_betteranchors12_scratch
ArmoryCarlaOverObjDetAll_FasterRCNN_betteranchors12_scratch: .venv
> $(MART) experiment=ArmoryCarlaOverObjDetAll_FasterRCNN \
          task_name=$@ \
          "model.modules.rpn.anchor_generator.sizes=[16, 32, 64, 128]" \
          model.modules.backbone.extra_blocks=null \
          model.modules.box_head.box_roi_pool.aligned=true \
          model.modules.backbone.backbone.weights=false \
          model.modules.backbone.backbone.norm_layer.path=mart.nn.nn.GroupNorm32 \
          model.optimizer.lr=0.01 \
          trainer.max_steps=$(shell python -c "import math; print(math.ceil(3600/32 * 6))") \
          datamodule.ims_per_batch=32 \
          datamodule.world_size=4

.PHONY: ArmoryCarlaOverObjDetAll_FasterRCNN_betteranchors12_scratch_longer
ArmoryCarlaOverObjDetAll_FasterRCNN_betteranchors12_scratch_longer: .venv
> $(MART) experiment=ArmoryCarlaOverObjDetAll_FasterRCNN \
          task_name=$@ \
          "model.modules.rpn.anchor_generator.sizes=[16, 32, 64, 128]" \
          model.modules.backbone.extra_blocks=null \
          model.modules.box_head.box_roi_pool.aligned=true \
          model.modules.backbone.backbone.weights=false \
          model.modules.backbone.backbone.norm_layer.path=mart.nn.nn.GroupNorm32 \
          model.optimizer.lr=0.075 \
          model.optimizer.weight_decay=0 \
          trainer.max_steps=$(shell python -c "import math; print(math.ceil(3600/32 * 120))") \
          datamodule.ims_per_batch=32 \
          datamodule.world_size=4

.PHONY: ArmoryCarlaOverObjDetAll_FasterRCNN_betteranchors14_scratch
ArmoryCarlaOverObjDetAll_FasterRCNN_betteranchors14_scratch: .venv
> $(MART) experiment=ArmoryCarlaOverObjDetAll_FasterRCNN \
          task_name=$@ \
          "model.modules.rpn.anchor_generator.sizes=[8, 16, 32, 64]" \
          model.modules.backbone.extra_blocks=null \
          model.modules.box_head.box_roi_pool.aligned=true \
          model.modules.backbone.backbone.weights=false \
          model.modules.backbone.backbone.norm_layer.path=mart.nn.nn.GroupNorm32 \
          model.optimizer.lr=0.05 \
          trainer.max_steps=$(shell python -c "import math; print(math.ceil(3600/32 * 6))") \
          datamodule.ims_per_batch=32 \
          datamodule.world_size=4

.PHONY: ArmoryCarlaOverObjDetAll_FasterRCNN_betteranchors14_scratch_longer
ArmoryCarlaOverObjDetAll_FasterRCNN_betteranchors14_scratch_longer: .venv
> $(MART) experiment=ArmoryCarlaOverObjDetAll_FasterRCNN \
          task_name=$@ \
          "model.modules.rpn.anchor_generator.sizes=[8, 16, 32, 64]" \
          model.modules.backbone.extra_blocks=null \
          model.modules.box_head.box_roi_pool.aligned=true \
          model.modules.backbone.backbone.weights=false \
          model.modules.backbone.backbone.norm_layer.path=mart.nn.nn.GroupNorm32 \
          model.optimizer.lr=0.075 \
          model.optimizer.weight_decay=0 \
          trainer.max_steps=$(shell python -c "import math; print(math.ceil(3600/32 * 120))") \
          datamodule.ims_per_batch=32 \
          datamodule.world_size=4

# smaller anchors w/ roialignv2
.PHONY: ArmoryCarlaOverObjDetAll_FasterRCNN_betteranchors15
ArmoryCarlaOverObjDetAll_FasterRCNN_betteranchors15: .venv $(CARLA_OVERHEAD_DATASET)
> $(MART) experiment=ArmoryCarlaOverObjDetAll_FasterRCNN \
          task_name=$@ \
          "model.modules.rpn.anchor_generator.sizes=[8, 16, 32, 64, 128]" \
          model.modules.box_head.box_roi_pool.aligned=true

# even smaller anchors w/ roialignv2
.PHONY: ArmoryCarlaOverObjDetAll_FasterRCNN_betteranchors16
ArmoryCarlaOverObjDetAll_FasterRCNN_betteranchors16: .venv $(CARLA_OVERHEAD_DATASET)
> $(MART) experiment=ArmoryCarlaOverObjDetAll_FasterRCNN \
          task_name=$@ \
          "model.modules.rpn.anchor_generator.sizes=[4, 8, 16, 32, 64]" \
          model.modules.box_head.box_roi_pool.aligned=true

.PHONY: ArmoryCarlaOverObjDetAllDepth_FasterRCNN
ArmoryCarlaOverObjDetAllDepth_FasterRCNN: .venv $(CARLA_OVERHEAD_DATASET)
> $(MART) experiment=ArmoryCarlaOverObjDetAll_FasterRCNN \
          task_name=$@ \
          "datamodule.train_dataset.modalities=[depth]" \
          "datamodule.val_dataset.modalities=[depth]" \
          "datamodule.test_dataset.modalities=[depth]" \
          "model.modules.preprocessor.image_mean=[127.86, 107.73, 7.6331]" \
          "model.modules.preprocessor.image_std=[74.119, 2.4791, 68.022]"

.PHONY: ArmoryCarlaOverObjDetAllDepth_FasterRCNN_2
ArmoryCarlaOverObjDetAllDepth_FasterRCNN_2: .venv $(CARLA_OVERHEAD_DATASET)
> $(MART) experiment=ArmoryCarlaOverObjDetAll_FasterRCNN \
          task_name=$@ \
          "datamodule.train_dataset.modalities=[depth]" \
          "datamodule.val_dataset.modalities=[depth]" \
          "datamodule.test_dataset.modalities=[depth]" \
          "model.modules.preprocessor.image_mean=[127.86, 107.73, 7.6331]" \
          "model.modules.preprocessor.image_std=[74.119, 2.4791, 68.022]" \
          "model.modules.rpn.anchor_generator.sizes=[16, 32, 64, 128, 256]"

.PHONY: ArmoryCarlaOverObjDetAllDepth_FasterRCNN_3
ArmoryCarlaOverObjDetAllDepth_FasterRCNN_3: .venv $(CARLA_OVERHEAD_DATASET)
> $(MART) experiment=ArmoryCarlaOverObjDetAll_FasterRCNN \
          task_name=$@ \
          "datamodule.train_dataset.modalities=[depth]" \
          "datamodule.val_dataset.modalities=[depth]" \
          "datamodule.test_dataset.modalities=[depth]" \
          "model.modules.preprocessor.image_mean=[127.86, 107.73, 7.6331]" \
          "model.modules.preprocessor.image_std=[74.119, 2.4791, 68.022]" \
          "model.modules.rpn.anchor_generator.sizes=[16, 32, 64, 128]" \
          model.modules.backbone.extra_blocks=null

.PHONY: ArmoryCarlaOverObjDetAllDepth_FasterRCNN_4
ArmoryCarlaOverObjDetAllDepth_FasterRCNN_4: .venv $(CARLA_OVERHEAD_DATASET)
> $(MART) experiment=ArmoryCarlaOverObjDetAll_FasterRCNN \
          task_name=$@ \
          "datamodule.train_dataset.modalities=[depth]" \
          "datamodule.val_dataset.modalities=[depth]" \
          "datamodule.test_dataset.modalities=[depth]" \
          "model.modules.preprocessor.image_mean=[127.86, 107.73, 7.6331]" \
          "model.modules.preprocessor.image_std=[74.119, 2.4791, 68.022]" \
          "model.modules.rpn.anchor_generator.sizes=[16, 32, 64, 128]" \
          model.modules.backbone.extra_blocks=null \
          model.modules.box_head.box_roi_pool.aligned=true

.PHONY: ArmoryCarlaOverObjDetAll1Depth_FasterRCNN
ArmoryCarlaOverObjDetAll1Depth_FasterRCNN: .venv $(CARLA_OVERHEAD_DATASET)
> $(MART) experiment=ArmoryCarlaOverObjDetAll_FasterRCNN \
          task_name=$@ \
          "datamodule=armory_carla_over_objdet_depth" \
          "model.modules.backbone.backbone.in_channels=1" \
          "model.modules.preprocessor.image_mean=[31.468]" \
          "model.modules.preprocessor.image_std=[9.5084]"

.PHONY: ArmoryCarlaOverObjDetAll1Depth_FasterRCNN_2
ArmoryCarlaOverObjDetAll1Depth_FasterRCNN_2: .venv $(CARLA_OVERHEAD_DATASET)
> $(MART) experiment=ArmoryCarlaOverObjDetAll_FasterRCNN \
          task_name=$@ \
          "datamodule=armory_carla_over_objdet_depth" \
          "model.modules.backbone.backbone.in_channels=1" \
          "model.modules.preprocessor.image_mean=[31.468]" \
          "model.modules.preprocessor.image_std=[9.5084]" \
          "model.modules.rpn.anchor_generator.sizes=[16, 32, 64, 128, 256]"

.PHONY: ArmoryCarlaOverObjDetAll1Depth_FasterRCNN_3
ArmoryCarlaOverObjDetAll1Depth_FasterRCNN_3: .venv $(CARLA_OVERHEAD_DATASET)
> $(MART) experiment=ArmoryCarlaOverObjDetAll_FasterRCNN \
          task_name=$@ \
          "datamodule=armory_carla_over_objdet_depth" \
          "model.modules.backbone.backbone.in_channels=1" \
          "model.modules.preprocessor.image_mean=[31.468]" \
          "model.modules.preprocessor.image_std=[9.5084]" \
          "model.modules.rpn.anchor_generator.sizes=[16, 32, 64, 128]" \
          model.modules.backbone.extra_blocks=null

.PHONY: ArmoryCarlaOverObjDetAll1Depth_FasterRCNN_4
ArmoryCarlaOverObjDetAll1Depth_FasterRCNN_4: .venv $(CARLA_OVERHEAD_DATASET)
> $(MART) experiment=ArmoryCarlaOverObjDetAll_FasterRCNN \
          task_name=$@ \
          "datamodule=armory_carla_over_objdet_depth" \
          "model.modules.backbone.backbone.in_channels=1" \
          "model.modules.preprocessor.image_mean=[31.468]" \
          "model.modules.preprocessor.image_std=[9.5084]" \
          "model.modules.rpn.anchor_generator.sizes=[16, 32, 64, 128]" \
          model.modules.backbone.extra_blocks=null \
          model.modules.box_head.box_roi_pool.aligned=true

.PHONY: ArmoryCarlaOverObjDetAll1Depth_FasterRCNN_5
ArmoryCarlaOverObjDetAll1Depth_FasterRCNN_5: .venv $(CARLA_OVERHEAD_DATASET)
> $(MART) experiment=ArmoryCarlaOverObjDetAll_FasterRCNN \
          task_name=$@ \
          "datamodule=armory_carla_over_objdet_depth" \
          "datamodule.train_dataset.modalities=[rgb, depth]" \
          "datamodule.val_dataset.modalities=[rgb, depth]" \
          "datamodule.test_dataset.modalities=[rgb, depth]" \
          "+model.modules.backbone.channel_slice=[3, 4]" \
          "model.modules.backbone.backbone.in_channels=1" \
          "model.modules.preprocessor.image_mean=[123.675, 116.28, 103.53, 31.468]" \
          "model.modules.preprocessor.image_std=[58.395, 57.12, 57.375, 9.5084]" \
          "model.modules.rpn.anchor_generator.sizes=[16, 32, 64, 128]" \
          model.modules.backbone.extra_blocks=null \
          model.modules.box_head.box_roi_pool.aligned=true

.PHONY: ArmoryCarlaOverObjDetAllMultiModal_FasterRCNN
ArmoryCarlaOverObjDetAllMultiModal_FasterRCNN: .venv $(CARLA_OVERHEAD_DATASET)
> $(MART) experiment=ArmoryCarlaOverObjDetAll_FasterRCNN \
          task_name=$@ \
          "datamodule.train_dataset.modalities=[rgb, depth]" \
          "datamodule.val_dataset.modalities=[rgb, depth]" \
          "datamodule.test_dataset.modalities=[rgb, depth]" \
          "model.modules.backbone.backbone.in_channels=6" \
          "model.modules.preprocessor.image_mean=[123.675, 116.28, 103.53, 127.86, 107.73, 7.6331]" \
          "model.modules.preprocessor.image_std=[58.395, 57.12, 57.375, 74.119, 2.4791, 68.022]"

.PHONY: ArmoryCarlaOverObjDetAllMultiModal_FasterRCNN_2
ArmoryCarlaOverObjDetAllMultiModal_FasterRCNN_2: .venv $(CARLA_OVERHEAD_DATASET)
> $(MART) experiment=ArmoryCarlaOverObjDetAll_FasterRCNN \
          task_name=$@ \
          "datamodule.train_dataset.modalities=[rgb, depth]" \
          "datamodule.val_dataset.modalities=[rgb, depth]" \
          "datamodule.test_dataset.modalities=[rgb, depth]" \
          "model.modules.backbone.backbone.in_channels=6" \
          "model.modules.preprocessor.image_mean=[123.675, 116.28, 103.53, 127.86, 107.73, 7.6331]" \
          "model.modules.preprocessor.image_std=[58.395, 57.12, 57.375, 74.119, 2.4791, 68.022]" \
          "model.modules.rpn.anchor_generator.sizes=[16, 32, 64, 128, 256]"

.PHONY: ArmoryCarlaOverObjDetAllMultiModal_FasterRCNN_3
ArmoryCarlaOverObjDetAllMultiModal_FasterRCNN_3: .venv $(CARLA_OVERHEAD_DATASET)
> $(MART) experiment=ArmoryCarlaOverObjDetAll_FasterRCNN \
          task_name=$@ \
          "datamodule.train_dataset.modalities=[rgb, depth]" \
          "datamodule.val_dataset.modalities=[rgb, depth]" \
          "datamodule.test_dataset.modalities=[rgb, depth]" \
          "model.modules.backbone.backbone.in_channels=6" \
          "model.modules.preprocessor.image_mean=[123.675, 116.28, 103.53, 127.86, 107.73, 7.6331]" \
          "model.modules.preprocessor.image_std=[58.395, 57.12, 57.375, 74.119, 2.4791, 68.022]" \
          "model.modules.rpn.anchor_generator.sizes=[16, 32, 64, 128]" \
          model.modules.backbone.extra_blocks=null

.PHONY: ArmoryCarlaOverObjDetAllMultiModal_FasterRCNN_4
ArmoryCarlaOverObjDetAllMultiModal_FasterRCNN_4: .venv $(CARLA_OVERHEAD_DATASET)
> $(MART) experiment=ArmoryCarlaOverObjDetAll_FasterRCNN \
          task_name=$@ \
          "datamodule.train_dataset.modalities=[rgb, depth]" \
          "datamodule.val_dataset.modalities=[rgb, depth]" \
          "datamodule.test_dataset.modalities=[rgb, depth]" \
          "model.modules.backbone.backbone.in_channels=6" \
          "model.modules.preprocessor.image_mean=[123.675, 116.28, 103.53, 127.86, 107.73, 7.6331]" \
          "model.modules.preprocessor.image_std=[58.395, 57.12, 57.375, 74.119, 2.4791, 68.022]" \
          "model.modules.rpn.anchor_generator.sizes=[16, 32, 64, 128]" \
          model.modules.backbone.extra_blocks=null \
          model.modules.box_head.box_roi_pool.aligned=true

.PHONY: ArmoryCarlaOverObjDetAllMultiModal_FasterRCNN_1depth
ArmoryCarlaOverObjDetAllMultiModal_FasterRCNN_1depth: .venv $(CARLA_OVERHEAD_DATASET)
> $(MART) experiment=ArmoryCarlaOverObjDetAll_FasterRCNN \
          task_name=$@ \
          "datamodule=armory_carla_over_objdet_depth" \
          "datamodule.train_dataset.modalities=[rgb, depth]" \
          "datamodule.val_dataset.modalities=[rgb, depth]" \
          "datamodule.test_dataset.modalities=[rgb, depth]" \
          "model.modules.backbone.backbone.in_channels=4" \
          "model.modules.preprocessor.image_mean=[123.675, 116.28, 103.53, 31.468]" \
          "model.modules.preprocessor.image_std=[58.395, 57.12, 57.375, 9.5084]"

.PHONY: ArmoryCarlaOverObjDetAllMultiModal_FasterRCNN_1depth_2
ArmoryCarlaOverObjDetAllMultiModal_FasterRCNN_1depth_2: .venv $(CARLA_OVERHEAD_DATASET)
> $(MART) experiment=ArmoryCarlaOverObjDetAll_FasterRCNN \
          task_name=$@ \
          "datamodule=armory_carla_over_objdet_depth" \
          "datamodule.train_dataset.modalities=[rgb, depth]" \
          "datamodule.val_dataset.modalities=[rgb, depth]" \
          "datamodule.test_dataset.modalities=[rgb, depth]" \
          "model.modules.backbone.backbone.in_channels=4" \
          "model.modules.preprocessor.image_mean=[123.675, 116.28, 103.53, 31.468]" \
          "model.modules.preprocessor.image_std=[58.395, 57.12, 57.375, 9.5084]" \
          "model.modules.rpn.anchor_generator.sizes=[16, 32, 64, 128, 256]"

.PHONY: ArmoryCarlaOverObjDetAllMultiModal_FasterRCNN_1depth_3
ArmoryCarlaOverObjDetAllMultiModal_FasterRCNN_1depth_3: .venv $(CARLA_OVERHEAD_DATASET)
> $(MART) experiment=ArmoryCarlaOverObjDetAll_FasterRCNN \
          task_name=$@ \
          "datamodule=armory_carla_over_objdet_depth" \
          "datamodule.train_dataset.modalities=[rgb, depth]" \
          "datamodule.val_dataset.modalities=[rgb, depth]" \
          "datamodule.test_dataset.modalities=[rgb, depth]" \
          "model.modules.backbone.backbone.in_channels=4" \
          "model.modules.preprocessor.image_mean=[123.675, 116.28, 103.53, 31.468]" \
          "model.modules.preprocessor.image_std=[58.395, 57.12, 57.375, 9.5084]" \
          "model.modules.rpn.anchor_generator.sizes=[16, 32, 64, 128]" \
          model.modules.backbone.extra_blocks=null

.PHONY: ArmoryCarlaOverObjDetAllMultiModal_FasterRCNN_1depth_4
ArmoryCarlaOverObjDetAllMultiModal_FasterRCNN_1depth_4: .venv $(CARLA_OVERHEAD_DATASET)
> $(MART) experiment=ArmoryCarlaOverObjDetAll_FasterRCNN \
          task_name=$@ \
          "datamodule=armory_carla_over_objdet_depth" \
          "datamodule.train_dataset.modalities=[rgb, depth]" \
          "datamodule.val_dataset.modalities=[rgb, depth]" \
          "datamodule.test_dataset.modalities=[rgb, depth]" \
          "model.modules.backbone.backbone.in_channels=4" \
          "model.modules.preprocessor.image_mean=[123.675, 116.28, 103.53, 31.468]" \
          "model.modules.preprocessor.image_std=[58.395, 57.12, 57.375, 9.5084]" \
          "model.modules.rpn.anchor_generator.sizes=[16, 32, 64, 128]" \
          model.modules.backbone.extra_blocks=null \
          model.modules.box_head.box_roi_pool.aligned=true

.PHONY: ArmoryCarlaOverObjDetAllMultiModal_ModularFasterRCNN
ArmoryCarlaOverObjDetAllMultiModal_ModularFasterRCNN: .venv $(CARLA_OVERHEAD_DATASET)
> $(MART) experiment=ArmoryCarlaOverObjDetAll_ModularFasterRCNN \
          task_name=$@ \
          "datamodule=armory_carla_over_objdet_depth" \
          "datamodule.train_dataset.modalities=[rgb, depth]" \
          "datamodule.train_dataset.transforms.transforms.3.lambd.scale=255" \
          "+datamodule.train_dataset.transforms.transforms.3.lambd.far=1" \
          "datamodule.val_dataset.modalities=[rgb, depth]" \
          "datamodule.val_dataset.transforms.transforms.2.lambd.scale=255" \
          "+datamodule.val_dataset.transforms.transforms.2.lambd.far=1" \
          "datamodule.test_dataset.modalities=[rgb, depth]" \
          "datamodule.test_dataset.transforms.transforms.2.lambd.scale=255" \
          "+datamodule.test_dataset.transforms.transforms.2.lambd.far=1" \
          "model.modules.losses_and_detections.backbone.backbone.in_channels=4" \
          "model.modules.losses_and_detections.preprocessor.image_mean=[0.485, 0.456, 0.406, 0.031468]" \
          "model.modules.losses_and_detections.preprocessor.image_std=[0.229, 0.224, 0.225, 0.0095084]"

.PHONY: ArmoryCarlaOverObjDetAllMultiModal_FasterRCNN_Semantic
ArmoryCarlaOverObjDetAllMultiModal_FasterRCNN_Semantic: .venv $(CARLA_OVERHEAD_DATASET)
> $(MART) experiment=ArmoryCarlaOverObjDetAll_FasterRCNN_Semantic \
          task_name=$@ \
          "datamodule=armory_carla_over_objdet_depth" \
          "datamodule.train_dataset.modalities=[rgb, depth]" \
          "datamodule.val_dataset.modalities=[rgb, depth]" \
          "datamodule.test_dataset.modalities=[rgb, depth]" \
          "model.modules.preprocessor.image_mean=[123.675, 116.28, 103.53, 31.468]" \
          "model.modules.preprocessor.image_std=[58.395, 57.12, 57.375, 9.5084]" \
          "model.modules.rpn.anchor_generator.sizes=[16, 32, 64, 128]" \
          "model.modules.rpn_aux.anchor_generator.sizes=[16, 32, 64, 128]" \
          model.modules.backbone.rpn.extra_blocks=null \
          model.modules.backbone.box.extra_blocks=null \
          model.modules.box_head.box_roi_pool.aligned=true \
          model.modules.box_head_aux.box_roi_pool.aligned=true

.PHONY: ArmoryCarlaOverObjDetAllMultiModal_FasterRCNN_Semantic_2
ArmoryCarlaOverObjDetAllMultiModal_FasterRCNN_Semantic_2: .venv $(CARLA_OVERHEAD_DATASET)
> $(MART) experiment=ArmoryCarlaOverObjDetAll_FasterRCNN_Semantic \
          task_name=$@ \
          "datamodule=armory_carla_over_objdet_depth" \
          "datamodule.train_dataset.modalities=[rgb, depth]" \
          "datamodule.val_dataset.modalities=[rgb, depth]" \
          "datamodule.test_dataset.modalities=[rgb, depth]" \
          "model.modules.preprocessor.image_mean=[123.675, 116.28, 103.53, 31.468]" \
          "model.modules.preprocessor.image_std=[58.395, 57.12, 57.375, 9.5084]" \
          "model.modules.rpn.anchor_generator.sizes=[16, 32, 64, 128]" \
          "model.modules.rpn_aux.anchor_generator.sizes=[16, 32, 64, 128]" \
          model.modules.backbone.rpn.extra_blocks=null \
          model.modules.backbone.box.extra_blocks=null

.PHONY: ArmoryCarlaOverObjDetAllMultiModal_FasterRCNN_Semantic_3
ArmoryCarlaOverObjDetAllMultiModal_FasterRCNN_Semantic_3: .venv $(CARLA_OVERHEAD_DATASET)
> $(MART) experiment=ArmoryCarlaOverObjDetAll_FasterRCNN_Semantic \
          task_name=$@ \
          "datamodule=armory_carla_over_objdet_depth" \
          "datamodule.train_dataset.modalities=[rgb, depth]" \
          "datamodule.val_dataset.modalities=[rgb, depth]" \
          "datamodule.test_dataset.modalities=[rgb, depth]" \
          "model.modules.preprocessor.image_mean=[123.675, 116.28, 103.53, 31.468]" \
          "model.modules.preprocessor.image_std=[58.395, 57.12, 57.375, 9.5084]" \
          "model.modules.rpn.anchor_generator.sizes=[16, 32, 64, 128, 256]" \
          "model.modules.rpn_aux.anchor_generator.sizes=[16, 32, 64, 128, 256]"

.PHONY: ArmoryCarlaOverObjDetAllMultiModal_FasterRCNN_Semantic_4
ArmoryCarlaOverObjDetAllMultiModal_FasterRCNN_Semantic_4: .venv $(CARLA_OVERHEAD_DATASET)
> $(MART) experiment=ArmoryCarlaOverObjDetAll_FasterRCNN_Semantic \
          task_name=$@ \
          "datamodule=armory_carla_over_objdet_depth" \
          "datamodule.train_dataset.modalities=[rgb, depth]" \
          "datamodule.val_dataset.modalities=[rgb, depth]" \
          "datamodule.test_dataset.modalities=[rgb, depth]" \
          "model.modules.preprocessor.image_mean=[123.675, 116.28, 103.53, 31.468]" \
          "model.modules.preprocessor.image_std=[58.395, 57.12, 57.375, 9.5084]"

.PHONY: ArmoryCarlaOverObjDetAllMultiModal_FasterRCNN_Semantic3Depth
ArmoryCarlaOverObjDetAllMultiModal_FasterRCNN_Semantic3Depth: .venv $(CARLA_OVERHEAD_DATASET)
> $(MART) experiment=ArmoryCarlaOverObjDetAll_FasterRCNN_Semantic \
          task_name=$@ \
          "datamodule.train_dataset.modalities=[rgb, depth]" \
          "datamodule.val_dataset.modalities=[rgb, depth]" \
          "datamodule.test_dataset.modalities=[rgb, depth]" \
          model.modules.backbone.rpn.backbone.in_channels=3 \
          "model.modules.backbone.rpn.channel_slice=[3, 6]" \
          "model.modules.preprocessor.image_mean=[123.675, 116.28, 103.53, 127.86, 107.73, 7.6331]" \
          "model.modules.preprocessor.image_std=[58.395, 57.12, 57.375, 74.119, 2.4791, 68.022]" \
          "model.modules.rpn.anchor_generator.sizes=[16, 32, 64, 128]" \
          "model.modules.rpn_aux.anchor_generator.sizes=[16, 32, 64, 128]" \
          model.modules.backbone.rpn.extra_blocks=null \
          model.modules.backbone.box.extra_blocks=null \
          model.modules.box_head.box_roi_pool.aligned=true \
          model.modules.box_head_aux.box_roi_pool.aligned=true

.PHONY: ArmoryCarlaOverObjDetAllMultiModal_FasterRCNN_Semantic3Depth_2
ArmoryCarlaOverObjDetAllMultiModal_FasterRCNN_Semantic3Depth_2: .venv $(CARLA_OVERHEAD_DATASET)
> $(MART) experiment=ArmoryCarlaOverObjDetAll_FasterRCNN_Semantic \
          task_name=$@ \
          "datamodule.train_dataset.modalities=[rgb, depth]" \
          "datamodule.val_dataset.modalities=[rgb, depth]" \
          "datamodule.test_dataset.modalities=[rgb, depth]" \
          model.modules.backbone.rpn.backbone.in_channels=3 \
          "model.modules.backbone.rpn.channel_slice=[3, 6]" \
          "model.modules.preprocessor.image_mean=[123.675, 116.28, 103.53, 127.86, 107.73, 7.6331]" \
          "model.modules.preprocessor.image_std=[58.395, 57.12, 57.375, 74.119, 2.4791, 68.022]" \
          "model.modules.rpn.anchor_generator.sizes=[16, 32, 64, 128]" \
          "model.modules.rpn_aux.anchor_generator.sizes=[16, 32, 64, 128]" \
          model.modules.backbone.rpn.extra_blocks=null \
          model.modules.backbone.box.extra_blocks=null

.PHONY: ArmoryCarlaOverObjDetAllMultiModal_FasterRCNN_Semantic3Depth_3
ArmoryCarlaOverObjDetAllMultiModal_FasterRCNN_Semantic3Depth_3: .venv $(CARLA_OVERHEAD_DATASET)
> $(MART) experiment=ArmoryCarlaOverObjDetAll_FasterRCNN_Semantic \
          task_name=$@ \
          "datamodule.train_dataset.modalities=[rgb, depth]" \
          "datamodule.val_dataset.modalities=[rgb, depth]" \
          "datamodule.test_dataset.modalities=[rgb, depth]" \
          model.modules.backbone.rpn.backbone.in_channels=3 \
          "model.modules.backbone.rpn.channel_slice=[3, 6]" \
          "model.modules.preprocessor.image_mean=[123.675, 116.28, 103.53, 127.86, 107.73, 7.6331]" \
          "model.modules.preprocessor.image_std=[58.395, 57.12, 57.375, 74.119, 2.4791, 68.022]" \
          "model.modules.rpn.anchor_generator.sizes=[16, 32, 64, 128, 256]" \
          "model.modules.rpn_aux.anchor_generator.sizes=[16, 32, 64, 128, 256]"

.PHONY: ArmoryCarlaOverObjDetAllMultiModal_FasterRCNN_Semantic3Depth_4
ArmoryCarlaOverObjDetAllMultiModal_FasterRCNN_Semantic3Depth_4: .venv $(CARLA_OVERHEAD_DATASET)
> $(MART) experiment=ArmoryCarlaOverObjDetAll_FasterRCNN_Semantic \
          task_name=$@ \
          "datamodule.train_dataset.modalities=[rgb, depth]" \
          "datamodule.val_dataset.modalities=[rgb, depth]" \
          "datamodule.test_dataset.modalities=[rgb, depth]" \
          model.modules.backbone.rpn.backbone.in_channels=3 \
          "model.modules.backbone.rpn.channel_slice=[3, 6]" \
          "model.modules.preprocessor.image_mean=[123.675, 116.28, 103.53, 127.86, 107.73, 7.6331]" \
          "model.modules.preprocessor.image_std=[58.395, 57.12, 57.375, 74.119, 2.4791, 68.022]"

.PHONY: ArmoryCarlaOverObjDetAllMultiModal_FasterRCNN_InverseSemantic
ArmoryCarlaOverObjDetAllMultiModal_FasterRCNN_InverseSemantic: .venv $(CARLA_OVERHEAD_DATASET)
> $(MART) experiment=ArmoryCarlaOverObjDetAll_FasterRCNN_Semantic \
          task_name=$@ \
          "datamodule=armory_carla_over_objdet_depth" \
          "datamodule.train_dataset.modalities=[rgb, depth]" \
          "datamodule.val_dataset.modalities=[rgb, depth]" \
          "datamodule.test_dataset.modalities=[rgb, depth]" \
          "model.modules.preprocessor.image_mean=[123.675, 116.28, 103.53, 31.468]" \
          "model.modules.preprocessor.image_std=[58.395, 57.12, 57.375, 9.5084]" \
          "model.modules.rpn.anchor_generator.sizes=[16, 32, 64, 128]" \
          "model.modules.rpn_aux.anchor_generator.sizes=[16, 32, 64, 128]" \
          "model.modules.backbone.rpn.channel_slice=[0, 3]" \
          model.modules.backbone.rpn.backbone.in_channels=3 \
          model.modules.backbone.rpn.extra_blocks=null \
          "model.modules.backbone.box.channel_slice=[3, 4]" \
          model.modules.backbone.box.backbone.in_channels=1 \
          model.modules.backbone.box.extra_blocks=null \
          model.modules.box_head.box_roi_pool.aligned=true \
          model.modules.box_head_aux.box_roi_pool.aligned=true

.PHONY: ArmoryCarlaOverObjDetAllTrainValMultiModal_FasterRCNN_Semantic
ArmoryCarlaOverObjDetAllTrainValMultiModal_FasterRCNN_Semantic: .venv $(CARLA_OVERHEAD_DATASET)
> $(MART) experiment=ArmoryCarlaOverObjDetAll_FasterRCNN_Semantic \
          task_name=$@ \
          "datamodule=armory_carla_over_objdet_depth" \
          trainer.max_steps=$(shell python -c "import math; print(math.ceil(4800/2 * 96))") \
          "datamodule.train_dataset.root=$$\{paths.data_dir\}/carla_over_obj_det/train_val" \
          "datamodule.train_dataset.annFile=$$\{paths.data_dir\}/carla_over_obj_det/train_val/kwcoco_annotations_all_iscrowd0.json" \
          "datamodule.train_dataset.modalities=[rgb, depth]" \
          "datamodule.val_dataset.modalities=[rgb, depth]" \
          "datamodule.test_dataset.modalities=[rgb, depth]" \
          "model.modules.preprocessor.image_mean=[123.675, 116.28, 103.53, 31.468]" \
          "model.modules.preprocessor.image_std=[58.395, 57.12, 57.375, 9.5084]" \
          "model.modules.rpn.anchor_generator.sizes=[16, 32, 64, 128]" \
          "model.modules.rpn_aux.anchor_generator.sizes=[16, 32, 64, 128]" \
          model.modules.backbone.rpn.extra_blocks=null \
          model.modules.backbone.box.extra_blocks=null \
          model.modules.box_head.box_roi_pool.aligned=true \
          model.modules.box_head_aux.box_roi_pool.aligned=true

.PHONY: ArmoryCarlaOverObjDetAllTrainValDevMultiModal_FasterRCNN_Semantic
ArmoryCarlaOverObjDetAllTrainValDevMultiModal_FasterRCNN_Semantic: .venv $(CARLA_OVERHEAD_DATASET)
> $(MART) experiment=ArmoryCarlaOverObjDetAll_FasterRCNN_Semantic \
          task_name=$@ \
          "datamodule=armory_carla_over_objdet_depth" \
          trainer.max_steps=$(shell python -c "import math; print(math.ceil(4820/2 * 96))") \
          "datamodule.train_dataset.root=$$\{paths.data_dir\}/carla_over_obj_det/train_val_dev" \
          "datamodule.train_dataset.annFile=$$\{paths.data_dir\}/carla_over_obj_det/train_val_dev/kwcoco_annotations_all_iscrowd0.json" \
          "datamodule.train_dataset.modalities=[rgb, depth]" \
          "datamodule.val_dataset.modalities=[rgb, depth]" \
          "datamodule.test_dataset.modalities=[rgb, depth]" \
          "model.modules.preprocessor.image_mean=[123.675, 116.28, 103.53, 31.468]" \
          "model.modules.preprocessor.image_std=[58.395, 57.12, 57.375, 9.5084]" \
          "model.modules.rpn.anchor_generator.sizes=[16, 32, 64, 128]" \
          "model.modules.rpn_aux.anchor_generator.sizes=[16, 32, 64, 128]" \
          model.modules.backbone.rpn.extra_blocks=null \
          model.modules.backbone.box.extra_blocks=null \
          model.modules.box_head.box_roi_pool.aligned=true \
          model.modules.box_head_aux.box_roi_pool.aligned=true

.PHONY: ArmoryCarlaOverObjDetAllTrainVal1Depth_FasterRCNN_5
ArmoryCarlaOverObjDetAllTrainVal1Depth_FasterRCNN_5: .venv $(CARLA_OVERHEAD_DATASET)
> $(MART) experiment=ArmoryCarlaOverObjDetAll_FasterRCNN \
          task_name=$@ \
          "datamodule=armory_carla_over_objdet_depth" \
          trainer.max_steps=$(shell python -c "import math; print(math.ceil(4800/2 * 96))") \
          "datamodule.train_dataset.root=$$\{paths.data_dir\}/carla_over_obj_det/train_val" \
          "datamodule.train_dataset.annFile=$$\{paths.data_dir\}/carla_over_obj_det/train_val/kwcoco_annotations_all_iscrowd0.json" \
          "datamodule.train_dataset.modalities=[rgb, depth]" \
          "datamodule.val_dataset.modalities=[rgb, depth]" \
          "datamodule.test_dataset.modalities=[rgb, depth]" \
          "+model.modules.backbone.channel_slice=[3, 4]" \
          "model.modules.backbone.backbone.in_channels=1" \
          "model.modules.preprocessor.image_mean=[123.675, 116.28, 103.53, 31.468]" \
          "model.modules.preprocessor.image_std=[58.395, 57.12, 57.375, 9.5084]" \
          "model.modules.rpn.anchor_generator.sizes=[16, 32, 64, 128]" \
          model.modules.backbone.extra_blocks=null \
          model.modules.box_head.box_roi_pool.aligned=true

.PHONY: ArmoryCarlaOverObjDetAllTrainValDev1Depth_FasterRCNN_5
ArmoryCarlaOverObjDetAllTrainValDev1Depth_FasterRCNN_5: .venv $(CARLA_OVERHEAD_DATASET)
> $(MART) experiment=ArmoryCarlaOverObjDetAll_FasterRCNN \
          task_name=$@ \
          "datamodule=armory_carla_over_objdet_depth" \
          trainer.max_steps=$(shell python -c "import math; print(math.ceil(4820/2 * 96))") \
          "datamodule.train_dataset.root=$$\{paths.data_dir\}/carla_over_obj_det/train_val_dev" \
          "datamodule.train_dataset.annFile=$$\{paths.data_dir\}/carla_over_obj_det/train_val_dev/kwcoco_annotations_all_iscrowd0.json" \
          "datamodule.train_dataset.modalities=[rgb, depth]" \
          "datamodule.val_dataset.modalities=[rgb, depth]" \
          "datamodule.test_dataset.modalities=[rgb, depth]" \
          "+model.modules.backbone.channel_slice=[3, 4]" \
          "model.modules.backbone.backbone.in_channels=1" \
          "model.modules.preprocessor.image_mean=[123.675, 116.28, 103.53, 31.468]" \
          "model.modules.preprocessor.image_std=[58.395, 57.12, 57.375, 9.5084]" \
          "model.modules.rpn.anchor_generator.sizes=[16, 32, 64, 128]" \
          model.modules.backbone.extra_blocks=null \
          model.modules.box_head.box_roi_pool.aligned=true

# We put yaml and pth into the folder $(MODEL_ZOO) so that Armory finds them.
.PRECIOUS: submission/%.yaml
submission/%.yaml: $(MODEL_ZOO)/%.yaml
> cp $< $@

.PRECIOUS: submission/%.pth
submission/%.pth: $(MODEL_ZOO)/%.pth
> cp $< $@

.PRECIOUS: submission/%.ckpt
submission/%.ckpt: $(MODEL_ZOO)/%.ckpt
> cp $< $@

# We will submit truncated YAMLs to run models in Armory, because
#   1. We only need the model definition at inference.
#   2. Variable interpolation in raw YAMLs may not work outside our MART-OSCAR environment.
#   3. We need to reset input_adv_* to NoAdversary anyway.
.PRECIOUS: $(MODEL_ZOO)/INTL_%.yaml
$(MODEL_ZOO)/INTL_%.yaml: logs/%/.hydra/config.yaml
> cat $< | $(YQ) '.model.modules.input_adv_training={"_target_": "mart.attack.NoAdversary"}' \
         | $(YQ) '.model.modules.input_adv_validation={"_target_": "mart.attack.NoAdversary"}' \
         | $(YQ) '.model.modules.input_adv_test={"_target_": "mart.attack.NoAdversary"}' \
         | $(YQ) '{"_target_": .model._target_, "optimizer": null, "modules": .model.modules, "training_sequence": .model.training_sequence}' > $@

.PRECIOUS: $(MODEL_ZOO)/INTL_%.pth
$(MODEL_ZOO)/INTL_%.pth: logs/%/checkpoints/last.ckpt
> $(POETRY) run $(PYTHON) -c 'import torch; state_dict = torch.load("$<", map_location="cpu")["state_dict"]; torch.save(state_dict, "$@")'


logs/%/.hydra/config.yaml:
> $(error No configuration for model $* found! Create a symlink in logs via: ln -s <experiment>/<datetime> logs/$*")

logs/%/checkpoints/last.ckpt:
> $(error No checkpoint for model $* found! Create a symlink in logs via: ln -s <experiment>/<datetime> logs/$*")

# Always export batches
$(SCENARIOS)/carla_obj_det_%.json: $(ARMORY_SCENARIOS)/eval6/carla_overhead_object_detection/carla_obj_det_%.json
> cat $< | $(JQ) '.sysconfig.docker_image = "$(DOCKER_IMAGE_TAG_OSCAR)"' \
         | $(JQ) '.scenario.export_batches = true' > $@

# Replace dataset with test dataset
$(SCENARIOS)/carla_obj_det_test_%.json: $(SCENARIOS)/carla_obj_det_%.json
> cat $< | ${JQ} '.dataset.name = "carla_over_obj_det_test"' \
         | ${JQ} '.dataset.eval_split = "test"' > $@

$(SCENARIOS)/INTL_carla_over_obj_det_test_adversarialpatch_%.json: $(SCENARIOS)/carla_obj_det_test_adversarialpatch_undefended.json
> cat $< | $(JQ) 'del(.model)' \
         | $(JQ) '.model.fit = false' \
         | $(JQ) '.model.fit_kwargs = {}' \
         | $(JQ) '.model.wrapper_kwargs = {}' \
         | $(JQ) '.model.module = "oscar.models.art_estimator"' \
         | $(JQ) '.model.name = "get_art_model"' \
         | $(JQ) '.model.weights_file.checkpoint = "INTL_$*.pth"' \
         | $(JQ) '.model.weights_file.config_yaml = "INTL_$*.yaml"' > $@

$(SCENARIOS)/INTL_carla_over_obj_det_test_multimodal_adversarialpatch_%.json: $(SCENARIOS)/carla_obj_det_test_multimodal_adversarialpatch_undefended.json
> cat $< | $(JQ) 'del(.model)' \
         | $(JQ) '.model.fit = false' \
         | $(JQ) '.model.fit_kwargs = {}' \
         | $(JQ) '.model.wrapper_kwargs = {}' \
         | $(JQ) '.model.module = "oscar.models.art_estimator"' \
         | $(JQ) '.model.name = "get_art_model"' \
         | $(JQ) '.model.weights_file.checkpoint = "INTL_$*.pth"' \
         | $(JQ) '.model.weights_file.config_yaml = "INTL_$*.yaml"' > $@

submission/INTL_carla_over_obj_det_rgb_%.json: $(SCENARIOS)/INTL_carla_over_obj_det_test_adversarialpatch_%.json \
                                               submission/INTL_%.pth \
                                               submission/INTL_%.yaml
> cat $< | $(JQ) '.model.model_kwargs.modality = "rgb"' > $@

submission/INTL_carla_over_obj_det_depth_%.json: $(SCENARIOS)/INTL_carla_over_obj_det_test_multimodal_adversarialpatch_%.json \
                                                 submission/INTL_%.pth \
                                                 submission/INTL_%.yaml
> cat $< | $(JQ) '.model.model_kwargs.modality = "d"' > $@

submission/INTL_carla_over_obj_det_multimodal_%.json: $(SCENARIOS)/INTL_carla_over_obj_det_test_multimodal_adversarialpatch_%.json \
                                                      submission/INTL_%.pth \
                                                      submission/INTL_%.yaml
> cat $< | $(JQ) '.model.model_kwargs.modality = "rgbd"' > $@

submission/eval6_baseline_%.json: $(SCENARIOS)/%_test.json
> cp $< $@

.PHONY: results/Eval_%

results/Eval_%: submission/%.json
> ${ARMORY} run $< --no-docker --output-dir Eval_$*

results/EvalBenign_%: submission/%.json
> ${ARMORY} run $< --no-docker --skip-attack --output-dir EvalBenign_$*
