from torchvision import transforms


def show_torch_image(image_tensor):
    """Converts torch Tensor to PIL image and shows it"""
    toPILtransform = transforms.ToPILImage()
    sample_image = toPILtransform(image_tensor)
    sample_image.show()
