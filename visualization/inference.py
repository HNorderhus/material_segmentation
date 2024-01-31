import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from PIL import Image
import argparse
import deeplab_model
import os
import matplotlib.pyplot as plt
from tqdm import tqdm


def extract_image_name(image_path):
    return os.path.splitext(os.path.basename(image_path))[0]


def run_inference(model, image_path, mode, save_figure):
    """
    Perform image segmentation using the provided DeepLab model and save or display the results based on the chosen mode.

    Parameters:
        state_dict (str): Path to the saved state dictionary of the DeepLab model.
        image_path (str): Path to the input image for segmentation.
        mode (str): Visualization mode. Options: "side_by_side" (default), "overlay", "save_mask".
        save_figure (bool): True to save the resulting figure, False otherwise.

    Returns:
        None
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #model = model.to(device)
    #model.load_state_dict(state_dict)

    model.eval()

    transforms_image = A.Compose(
        [
            A.LongestMaxSize(max_size=768, interpolation=1),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    # Load the image
    image = Image.open(image_path)
    image_np = np.asarray(image)

    transformed = transforms_image(image=image_np)
    image_transformed = transformed["image"]
    image_tensor = image_transformed.unsqueeze(0).to(device)

    outputs = model(image_tensor)["out"]
    #outputs = model(image_tensor)

    # Perform segmentation predictions
    _, preds = torch.max(outputs, 1)
    preds = preds.to("cpu")
    preds_np = preds.squeeze(0).cpu().numpy().astype(np.uint8)

    # Convert segmentation predictions to colored image using the custom colormap
    custom_colormap = np.array([
        [0, 0, 0],        # Background (Black)
        [0, 0, 255],      # Control Point (Blue)
        [0, 255, 0],      # Vegetation (Green)
        [0, 255, 255],    # Efflorescence (Cyan)
        [255, 255, 0],    # Corrosion (Yellow)
        [255, 0, 0],      # Spalling (Red)
        [255,255,255]   # Crack (White)
    ])

    preds_color = custom_colormap[preds_np]
    preds_color = preds_color.astype(np.uint8)
    preds_pil = Image.fromarray(preds_color)

    if mode == "side_by_side":
        # Plot the original image and the generated mask side by side
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Plot the original image
        axes[0].imshow(image_np)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # Plot the generated mask
        axes[1].imshow(preds_pil)
        axes[1].set_title('Generated Mask')
        axes[1].axis('off')

        # Show the comparison plot
        plt.tight_layout()
        if save_figure:
            image_name = extract_image_name(image_path)
            plt.savefig(f"results/{image_name}_90percent.png")
            #print("Saved resulting plot")
        #plt.show()
        plt.close()
    elif mode == "overlay":
        # Overlay the mask on the original image with 30% opacity
        mask_with_opacity = cv2.addWeighted(image_np, 0.7, preds_color, 0.3, 0)
        plt.figure(figsize=(6,6))
        plt.imshow(mask_with_opacity)
        plt.title('Mask Overlay')
        plt.axis('off')
        if save_figure:
            image_name = extract_image_name(image_path)
            plt.savefig(f"results/{image_name}_overlay_ignore_F1.png")
            #print("Saved overlaid figure")
        #plt.show()
        plt.close()
    elif mode == "save_mask":
        # Extract image name from the image path
        image_name = extract_image_name(image_path)
        # Save the generated mask with the image name as the file name
        mask_filename = f"results/{image_name}_mask.png"
        preds_color_rgb = cv2.cvtColor(preds_color, cv2.COLOR_BGR2RGB)
        cv2.imwrite(mask_filename, preds_color_rgb)
        print(f"Saved generated mask.")


def args_preprocess():
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--state_dict", help='Path and name of the state dict')
    parser.add_argument("--image", help="Path and Name of the image")
    parser.add_argument("--mode", choices=["side_by_side", "overlay", "save_mask"],
                        default="side_by_side", help="Visualization mode (default: side_by_side)")
    parser.add_argument("--save_figure", type=bool,
                        default=False, help="Save the resulting figure")
    parser.add_argument("--pruned_model", help='Path to the pruned model file')
    parser.add_argument("--use_pruned",  action="store_true", help='Flag to use the pruned model')

    args = parser.parse_args()

    # Get a list of all image files in the specified directory
    image_files = [os.path.join(args.image, file) for file in os.listdir(args.image) if file.lower().endswith(('.png', '.jpg'))]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.use_pruned:
        model = torch.load(args.pruned_model)
    else:
        model = deeplab_model.initialize_model(num_classes=8, keep_feature_extract=True)

    model.load_state_dict(torch.load(args.state_dict, map_location=device))
    model = model.to(device)

    # Run inference for each image
    for image_file in tqdm(image_files):
        run_inference(model, image_file, args.mode, args.save_figure)


    #run_inference(args.state_dict, args.image, args.mode, args.save_figure)

if __name__ == "__main__":
    args_preprocess()
