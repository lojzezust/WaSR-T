import albumentations as A
import torchvision.transforms as T
import numpy as np
import cv2


def get_augmentation_transform():
    color_transform = A.Compose([
        A.ColorJitter(p=0.7, hue=0.05),
        A.RandomGamma(p=1, gamma_limit=(70,120))], p=0.5)

    noise_transform = A.Compose([
        A.GaussNoise(p=0.5),
        A.ISONoise(p=0.5)], p=0.3)

    transform = A.Compose([
        A.HorizontalFlip(),
        A.ShiftScaleRotate(scale_limit=[0,0.3], rotate_limit=15, border_mode=0, p=0.7),
        color_transform,
        noise_transform
    ])

    return AlbumentationsTransform(transform)

class AlbumentationsTransform(object):
    def __init__(self, transform, image_feature='image', max_extra_images=5, mask_features=['segmentation', 'imu_mask', 'objects', 'pa_similarity', 'instance_seg']):
        self.transform = transform
        self.image_feature = image_feature
        self.mask_features = mask_features

        self.transform.add_targets({('image_%i' % i): 'image' for i in range(max_extra_images)})

    def __call__(self, x):
        valid_mask_features = [feat for feat in self.mask_features if feat in x]
        masks = [x[feat] for feat in valid_mask_features]
        extra_targets = {}
        if 'extra_images' in x:
            for i, img in enumerate(x['extra_images']):
                extra_targets['image_%i' % i] = img

        res = self.transform(image=x[self.image_feature], masks=masks, **extra_targets)

        output = {}
        output[self.image_feature] = res['image']
        for feat, mask in zip(valid_mask_features, res['masks']):
            output[feat] = mask

        if 'extra_images' in x:
            extra_images = []
            for i in range(len(x['extra_images'])):
                extra_images.append(res['image_%i' % i])
            output['extra_images'] = extra_images

        for feat in x:
            if feat not in output:
                output[feat] = x[feat]

        return output


def PytorchHubNormalization():
    """Transform that normalizes the image to pytorch hub models (DeepLab, ResNet,...) expected range.
    See: https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/"""

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    transform = T.Compose([
        T.ToTensor(), # CHW order, divide by 255
        T.Normalize(mean, std)
    ])

    return transform
