import argparse
import os

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

from deeplab_model import initialize_model


class SemanticSegmentationTarget:
    def __init__(self, category, mask):
        self.category = category
        self.mask = torch.from_numpy(mask)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()

    def __call__(self, model_output):
        return (model_output[self.category, :, :] * self.mask).sum()


class SegmentationModelOutputWrapper(torch.nn.Module):
    def __init__(self, model):
        super(SegmentationModelOutputWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)["out"]


def inspect_layer_activation(state_dict, image, s2ds_class, pruned_model, use_pruned):
    if use_pruned:
        model = torch.load(pruned_model)
    else:
        model = initialize_model(num_classes=8)

    model.load_state_dict(torch.load(state_dict))
    model = SegmentationModelOutputWrapper(model)
    model.eval()

    original_image = Image.open(image)
    resized_image = np.array(original_image.resize((512, 512)))
    rgb_img = np.float32(resized_image) / 255
    input_tensor = preprocess_image(rgb_img,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    if torch.cuda.is_available():
        model = model.cuda()
        input_tensor = input_tensor.cuda()

    output = model(input_tensor)

    normalized_masks = torch.nn.functional.softmax(output, dim=1).cpu()
    sem_classes = [
        '__background__', 'Control Point', 'Vegetation', 'Efflorescence', 'Corrosion', 'Spalling', 'Crack', 'Boundary'
    ]
    sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}

    target_class_idx = sem_class_to_idx[s2ds_class]
    target_class_mask = normalized_masks[0, :, :, :].argmax(axis=0).detach().cpu().numpy()
    target_class_mask_float = np.float32(target_class_mask == target_class_idx)

    for layer_number in range(1, 5):
        target_layers = [getattr(model.model.backbone, f'layer{layer_number}')]
        targets = [SemanticSegmentationTarget(target_class_idx, target_class_mask_float)]
        with GradCAM(model=model, target_layers=target_layers, ) as cam:
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        cam_image_pil = Image.fromarray(cam_image)
        image_name = image.split('/')[-1].split('.')[0]  # Extract the image name without extension
        if use_pruned:
            cam_image_pil.save(f"results/layer_activations/{image_name}_layer{layer_number}_pruned.png")
        else:
            cam_image_pil.save(f"results/layer_activations/{image_name}_layer{layer_number}_old.png")


def args_preprocess():
    parser = argparse.ArgumentParser()
    parser.add_argument("state_dict", help='Path and name of the state dict')
    parser.add_argument("image", help="Path and Name of the image")
    parser.add_argument("s2ds_class", type=str,
                        help="'__background__', 'Control Point', 'Vegetation', 'Efflorescence', 'Corrosion', 'Spalling', 'Crack', 'Boundary'")
    parser.add_argument("--pruned_model", default=None, help='Path to the pruned model file')
    parser.add_argument("--use_pruned", type=bool, default=False, help='Flag to use the pruned model')
    args = parser.parse_args()
    inspect_layer_activation(args.state_dict, args.image, args.s2ds_class, args.pruned_model, args.use_pruned)


if __name__ == "__main__":
    args_preprocess()
