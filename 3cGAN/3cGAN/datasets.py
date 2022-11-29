import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, "A") + "/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root, "B") + "/*.*"))
        self.files_C = sorted(glob.glob(os.path.join(root, "C") + "/*.*"))


    def __getitem__(self, index):
        image_A = Image.open(self.files_A[index % len(self.files_A)])

        if self.unaligned:
            image_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)])
        else:
            image_B = Image.open(self.files_B[index % len(self.files_B)])

        # Convert grayscale images to rgb
        #if image_A.mode != "L":
        #    image_A = image_A.convert('L')
        #    image_A = to_rgb(image_A)  # we convert it into rgb with 3 identical channels
        #if image_B.mode != "L":
            #image_B = image_B.convert('L')

        image_C = Image.open(self.files_C[index % len(self.files_C)])


        # Banach modified 20/04/20 - so that we estimate a grayscale depth map
        # if image_B.mode != "RGB":
        #    image_B = to_rgb(image_B)

        #if image_A.mode != "RGB":
        #    image_A = to_rgb(image_A)
        #if image_B.mode != "RGB":
        #    image_B = to_rgb(image_B)

        item_A = self.transform(image_A)
        item_B = self.transform(image_B)
        item_C = self.transform(image_C)
        return {"A": item_A, "B": item_B, "C": item_C}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B), len(self.files_C))
