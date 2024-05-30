from detrex.config import get_config
from .models.composed_detr_r50 import model
dataloader = get_config("common/data/coco_detr.py").dataloader
# lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_50ep
optimizer = get_config("common/optim.py").AdamW
train = get_config("common/train.py").train

# modify training config
train.init_checkpoint = "detectron2://ImageNetPretrained/torchvision/R-50.pkl"
train.output_dir = "./pre_trained/detr_r50_300ep"

# modify lr_multiplier
# lr_multiplier.scheduler.milestones = [369600, 554400]


# modify optimizer config
optimizer.weight_decay = 1e-4
optimizer.lr = 1e-4 * 3
optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1

# modify dataloader config
dataloader.train.num_workers = 32
dataloader.train.total_batch_size = 12
train.checkpointer.period = 118000 // dataloader.train.total_batch_size
train.eval_period = 118000 // dataloader.train.total_batch_size
train.log_period = 500
lr_multiplier = get_config('common/coco_schedule.py').lr_multiplier_50ep
lr_multiplier.scheduler.milestones = [118000 // dataloader.train.total_batch_size * 50,
                                      118000 // dataloader.train.total_batch_size * 75]
train.max_iter = 118000 // dataloader.train.total_batch_size * 75
train.amp.enabled = False