from imports import *
from datasets import load_dataset, Image, Dataset
from torchvision.transforms import Compose, ToTensor, CenterCrop, Resize, Lambda

def show_images(dataset, num_samples=20, cols = 3):
  plt.figure(figsize=(15,15))
  for i, img in enumerate(dataset):
    if i == num_samples:
      break
    plt.subplot(int(num_samples/cols+1), cols, i+1)
    plt.imshow(img['images'][0])

def show_tensor_image(image):
  reverse_transforms = transforms.Compose([
    transforms.Lambda(lambda t: (t + 1) / 2),
    transforms.Lambda(lambda t: t.permute(1, 2, 0)),
    transforms.Lambda(lambda t: t * 255.),
    transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
    transforms.ToPILImage(),
  ])

  if len(image.shape) == 4:
    image = image[0, :, :, :]
  plt.imshow(reverse_transforms(image))

