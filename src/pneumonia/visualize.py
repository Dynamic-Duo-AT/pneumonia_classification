import os

import matplotlib.pyplot as plt
import torch
from dotenv import load_dotenv

load_dotenv()


def plot_cotrast_image(path: str):
    image = torch.load(path)
    mean = float(os.getenv("MEAN_PIXEL"))
    std = float(os.getenv("STD_PIXEL"))

    # multiply all pixel values by 3.5 to increase contrast
    image_contrast = image * 3.5

    # denormalize images for visualization
    image = torch.clamp(image * std + mean, 0, 1)
    image_contrast = torch.clamp(image_contrast * std + mean, 0, 1)

    # plot the images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image.permute(1, 2, 0))
    plt.title("Original Image")
    plt.set_cmap("gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(image_contrast.permute(1, 2, 0))
    plt.title("Contrast Image")
    plt.set_cmap("gray")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("contrast_image_comparison.png")
    plt.show()


if __name__ == "__main__":
    plot_cotrast_image("data/processed/train/IM-0122-0001.pt")
