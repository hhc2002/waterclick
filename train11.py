from isegm.utils.exp_imports.default import *
from isegm.model.modeling.transformer_helper.cross_entropy_loss import CrossEntropyLoss

MODEL_NAME = 'cocolvis_vit_base'

def main(cfg):
    model, model_cfg = init_model(cfg)
    train(model, cfg, model_cfg)

def init_model(cfg):
    model_cfg = edict()
    model_cfg.crop_size = (256, 256)
    model_cfg.num_max_points = 24

    # 创建模型
    model = PlainVitModel(
        use_disks=True,
        norm_radius=5,
        with_prev_mask=True,
        backbone_params={},  # 这里可以省略详细参数，因为模型已经预训练好
        neck_params={},
        head_params={},
        random_split=cfg.random_split,
    )

    # 加载预训练模型权重
    pretrained_weights_path = cfg.IMAGENET_PRETRAINED_MODELS.COCO_BASE
    model.load_state_dict(torch.load(pretrained_weights_path))
    model.to(cfg.device)

    return model, model_cfg

def train(model, cfg, model_cfg):
    cfg.batch_size = 32 if cfg.batch_size < 1 else cfg.batch_size
    cfg.val_batch_size = cfg.batch_size
    crop_size = model_cfg.crop_size

    loss_cfg = edict()
    loss_cfg.instance_loss = NormalizedFocalLossSigmoid(alpha=0.5, gamma=2)
    loss_cfg.instance_loss_weight = 1.0

    train_augmentator = Compose([
        UniformRandomResize(scale_range=(0.75, 1.40)),
        HorizontalFlip(),
        PadIfNeeded(min_height=crop_size[0], min_width=crop_size[1], border_mode=0),
        RandomCrop(*crop_size),
        RandomBrightnessContrast(brightness_limit=(-0.25, 0.25), contrast_limit=(-0.15, 0.4), p=0.75),
        RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.75)
    ], p=1.0)

    val_augmentator = Compose([
        PadIfNeeded(min_height=crop_size[0], min_width=crop_size[1], border_mode=0),
        RandomCrop(*crop_size)
    ], p=1.0)

    points_sampler = MultiPointSampler(model_cfg.num_max_points, prob_gamma=0.80,
                                       merge_objects_prob=0.15,
                                       max_num_merged_objects=2)

    trainset = CocoLvisDataset(
        "/home/ipad_3d/hhc/WaterMask/data/UDW/",
        split='train',
        augmentator=train_augmentator,
        min_object_area=1000,
        keep_background_prob=0.05,
        points_sampler=points_sampler,
        epoch_len=30000,
        stuff_prob=0.30
    )

    valset = CocoLvisDataset(
        "/home/ipad_3d/hhc/WaterMask/data/UDW/",
        split='val',
        augmentator=val_augmentator,
        min_object_area=1000,
        points_sampler=points_sampler,
        epoch_len=2000
    )

    optimizer_params = {
        'lr': 5e-5, 'betas': (0.9, 0.999), 'eps': 1e-8
    }

    lr_scheduler = partial(torch.optim.lr_scheduler.MultiStepLR,
                           milestones=[50, 55], gamma=0.1)
    trainer = ISTrainer(model, cfg, model_cfg, loss_cfg,
                        trainset, valset,
                        optimizer='adam',
                        optimizer_params=optimizer_params,
                        layerwise_decay=cfg.layerwise_decay,
                        lr_scheduler=lr_scheduler,
                        checkpoint_interval=[(0, 20), (50, 1)],
                        image_dump_interval=300,
                        metrics=[AdaptiveIoU()],
                        max_interactive_points=model_cfg.num_max_points,
                        max_num_next_clicks=3)
    trainer.run(num_epochs=55, validation=False)

if __name__ == '__main__':
    cfg = edict()
    cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg.upsample = 'x4'  # 根据你的需求设置
    cfg.random_split = False  # 根据你的需求设置
    cfg.batch_size = 32
    cfg.layerwise_decay = 0.95  # 根据你的需求设置

    cfg.IMAGENET_PRETRAINED_MODELS = {
        'COCO_BASE': './weights/cocolvis_vit_base.pth'
    }

    main(cfg)
