import os
import torch
import data_setup, deeplab_model, engine
from torch.utils.data import DataLoader
from utils import create_writer,save_model
import albumentations as A
from albumentations.pytorch import ToTensorV2
import argparse
import numpy as np
import random


def load_model(args):
    if args.pruned_model:
        model_path = f'results/models/{args.pruned_model}'
        model = torch.load(model_path)
        # Conditionally set requires_grad based on the train_pruned argument
        if args.train_pruned_fully:
            for param in model.backbone.parameters():
                param.requires_grad = True
        else:
            for param in model.backbone.parameters():
                param.requires_grad = False
    else:
        model = deeplab_model.initialize_model(num_classes=8, keep_feature_extract=args.keep_feature_extract)
        # If pretrained weights are to be loaded
        if args.load_pretrained_weights:
            model.load_state_dict(torch.load(f'results/models/{args.load_pretrained_weights}'))

    return model

def set_seed(seed_value):
    """Set seed for reproducibility."""
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # if you are using multi-GPU.
    np.random.seed(seed_value)  # Numpy module.
    random.seed(seed_value)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main(train_dir, val_dir, name, model):
    NUM_EPOCHS = 400
    LEARNING_RATE = 0.001

    NUM_WORKERS = os.cpu_count()
    NUM_CLASSES = 23
    BATCH_SIZE = 16

    #set_seed(42)

    # Setup target device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device}")

    print("Initializing Datasets and Dataloaders...")

    train_transform = A.Compose(
            [
                A.LongestMaxSize(max_size=512, interpolation=1),
                A.RandomCrop(224, 224, p=1),
                A.PadIfNeeded(min_height=224, min_width=224),
                A.VerticalFlip(p=0.3),
                A.HorizontalFlip(p=0.3),
                A.RandomRotate90(p=0.3),
                A.OneOf([
                    A.ElasticTransform(alpha=90, sigma=90 * 0.05, alpha_affine=90 * 0.03, p=0.25),
                    A.GridDistortion(p=0.25),
                    A.CoarseDropout(max_holes=6, max_height=24, max_width=24, min_holes=2, min_height=8, min_width=8,
                                    fill_value=0, p=0.25),
                    A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=20, val_shift_limit=15, p=0.25),
                    A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=20, p=0.25),
                    A.CLAHE(p=0.25),
                ], p=0.7),
                A.OneOf([
                    A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.25),
                    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.25),
                    A.Blur(blur_limit=(3, 5), p=0.25),
                    A.RandomFog(p=0.33),
                    A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=0.33),
                    A.ColorJitter(p=0.33),
                ], p=0.7),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

    val_transform = A.Compose(
        [
            A.LongestMaxSize(max_size=512, interpolation=1),
            A.CenterCrop(224, 224),
            A.PadIfNeeded(min_height=224, min_width=224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    train_data = data_setup.DataLoaderSegmentation(train_dir, transform=train_transform)
    val_data = data_setup.DataLoaderSegmentation(val_dir, transform=val_transform)

    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    print("Initializing Model...")

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=LEARNING_RATE)

    tensorboard_writer = create_writer(experiment_name="deeplabv3_resnet101",
                                   model_name=f"{name}")

    print("Begin training...")

    # Start training with help from engine.py
    engine.train(model=model,
                 train_dataloader=train_dataloader,
                 val_dataloader=val_dataloader,
                 loss_fn=loss_fn,
                 optimizer=optimizer,
                 epochs=NUM_EPOCHS,
                 device=device,
                 writer=tensorboard_writer,
                 name=name,
                 train_pruned=False)

def args_preprocess():
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("train_dir", help='Directory path, should contain train/Images, train/Labels_grayscale')
    parser.add_argument("val_dir", help='Directory path, should contain val/Images and val/Labels_grayscale')
    parser.add_argument("name", type=str, help="Name of the current training variant")
    parser.add_argument("--keep_feature_extract", type=bool, default=True, help="Keep feature extraction layers frozen")
    parser.add_argument("--load_pretrained_weights", type=str)


    args = parser.parse_args()

    model = deeplab_model.initialize_model(num_classes=23, keep_feature_extract=args.keep_feature_extract)
    # If pretrained weights are to be loaded
    if args.load_pretrained_weights:
        model.load_state_dict(torch.load(f'results/models/{args.load_pretrained_weights}'))

    main(args.train_dir, args.val_dir, args.name, model)


if __name__ == '__main__':
    args_preprocess()

