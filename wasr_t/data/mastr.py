from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as TF
import yaml
import os
from .utils import load_pa_sim
from tqdm.auto import tqdm

def read_mask(path):
    mask = np.array(Image.open(path))

    # Masks stored in RGB channels or as class ids
    if mask.ndim == 3:
        mask = mask.astype(np.float32) / 255.0
    else:
        mask = np.stack([mask==0, mask==1, mask==2], axis=-1).astype(np.float32)

    return mask

def read_image_list(path):
    with open(path, 'r') as file:
        images = [line.strip() for line in file]
    return images

def compat_yaml2map(path):
    with open(path, 'r') as file:
        data = yaml.safe_load(file)

    img_list_path = (Path(path).parent / data['image_list']).resolve()
    images = read_image_list(img_list_path)

    img_dir = data['image_dir']
    mask_dir = (Path(path).parent / data['mask_dir']).resolve()
    imu_dir = (Path(path).parent / data['imu_dir']).resolve()

    ds_map = []
    for image in images:
        img_p = os.path.join(data['image_dir'], f'{image}.jpg')
        mask_p = os.path.join(data['mask_dir'], f'{image}m.png')
        imu_p = os.path.join(data['imu_dir'], f'{image}.png')
        ds_map.append((img_p, mask_p, imu_p))

    return ds_map

def pad_instances(masks, fixed_size=32):
    n_instances = masks.shape[0]
    n_pad = fixed_size - n_instances
    ndim = masks.ndim
    masks_p = np.pad(masks, ((0,n_pad),) + ((0,0),) * (ndim-1))
    return masks_p

class MaSTr1325Dataset(torch.utils.data.Dataset):
    """MaSTr1325 dataset wrapper

    Args:
        dataset_file (str): Path to the MaSTr1325 dataset file, containing pairs
            (or triplets for imu) of relative image and mask (and imu) paths.
        transform (optional): Tranform to apply to image and masks
        normalize_t (optional): Transform that normalizes the input image
        object_masks_limit (optional): Max number of objects in an image (can be larger than actual)
        include_original (optional): Include original (non-normalized) version of the image in the features
        masks_subdir (optional): Overrides the masks subdir specified in dataset file.
        preload (bool, optional): Preload the dataset into memory. Default: False.
    """
    def __init__(self, dataset_file, transform=None, normalize_t=None,
                 object_masks_limit=32, include_original=False, masks_subdir=None, preload=False):
        dataset_file = Path(dataset_file)
        self.dataset_dir = dataset_file.parent
        with dataset_file.open('r') as file:
            data = yaml.safe_load(file)

            # Set data directories
            self.image_dir = (self.dataset_dir / Path(data['image_dir'])).resolve()
            self.image_hist_dir = (self.dataset_dir / Path(data['image_hist_dir'])).resolve() if 'image_hist_dir' in data else None
            self.image_hist_len = data['image_hist_len'] if 'image_hist_len' in data else 0
            self.mask_dir = (self.dataset_dir / Path(data['mask_dir'])).resolve() if 'mask_dir' in data else None
            self.imu_dir = (self.dataset_dir / Path(data['imu_dir'])).resolve() if 'imu_dir' in data else None
            self.object_masks_dir = (self.dataset_dir / Path(data['object_masks_dir'])).resolve() if 'object_masks_dir' in data else None
            self.instance_masks_dir = (self.dataset_dir / Path(data['instance_masks_dir'])).resolve() if 'instance_masks_dir' in data else None
            self.pa_sim_dir = (self.dataset_dir / Path(data['pa_sim_dir'])).resolve() if 'pa_sim_dir' in data else None

            # Mask dir override
            if masks_subdir is not None:
                self.mask_dir = (self.dataset_dir / Path(masks_subdir)).resolve()

            # Entries
            image_list = (self.dataset_dir / data['image_list']).resolve()
            self.images = read_image_list(image_list)

        self.cache = None
        if preload:
            self.preload_into_memory()

        self.transform = transform
        self.normalize_t = normalize_t
        self.object_masks_limit = object_masks_limit
        self.include_original = include_original

    def preload_into_memory(self):
        self.cache = []
        for idx in tqdm(range(len(self)), desc="Preloading dataset into memory"):
            self.cache.append(self._read_sample(idx))

    def _read_hist_images(self, img_name):
        images = []
        for i in range(self.image_hist_len):
            img = np.array(Image.open(self.image_hist_dir / ('%s_%d.jpg' % (img_name, i))))
            images.append(img)

        return images[::-1]

    def _read_sample(self, idx):
        img_name = self.images[idx]
        img_filename = '%s.jpg' % img_name
        img_path = str(self.image_dir / img_filename)
        mask_filename = '%sm.png' % img_name

        img = np.array(Image.open(img_path))

        data = {
            'image': img,
            'img_name': img_name,
            'img_filename': img_filename,
            'mask_filename': mask_filename
        }

        if self.image_hist_dir is not None:
            data['extra_images'] = self._read_hist_images(img_name)

        if self.mask_dir is not None:
            mask_path = str(self.mask_dir / mask_filename)
            mask = read_mask(mask_path)
            data['segmentation'] = mask

        if self.imu_dir is not None:
            imu_path = str(self.imu_dir / ('%s.png' % img_name))
            imu_mask = np.array(Image.open(imu_path))
            data['imu_mask'] = imu_mask

        if self.object_masks_dir is not None:
            obj_masks_path = str(self.object_masks_dir / ('%s.npz' % img_name))
            obj_masks = np.load(obj_masks_path)['arr_0'].transpose(1,2,0)
            data['objects'] = obj_masks

        if self.instance_masks_dir is not None:
            inst_masks_path = str(self.instance_masks_dir / ('%s.npz' % img_name))
            inst_masks = np.load(inst_masks_path)['arr_0'].transpose(1,2,0,3)
            ih,iw,im,ic = inst_masks.shape
            data['instance_seg'] = inst_masks.reshape(ih,iw,im*ic)
            data['instance_seg_shape'] = (ih,iw,im,ic)

        if self.pa_sim_dir is not None:
            pa_sim_path = str(self.pa_sim_dir / ('%s.npz' % img_name))
            pa_sim = load_pa_sim(pa_sim_path).transpose(1,2,0)
            data['pa_similarity'] = pa_sim

        return data

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.cache is not None:
            data = self.cache[idx] # Read from memory
        else:
            data = self._read_sample(idx)

        img = data['image']
        img_original = data['image']
        img_name = data['img_name']
        img_filename = data['img_filename']
        mask_filename = data['mask_filename']
        extra_images = data['extra_images'] if 'extra_images' in data else []

        # Transform images and masks if transform is provided
        if self.transform is not None:
            data = self.transform(data)
            img = data['image']
            extra_images = data['extra_images'] if 'extra_images' in data else []

        if self.normalize_t is not None:
            img = self.normalize_t(img)
            extra_images = [self.normalize_t(img) for img in extra_images]
        else:
            # Default: divide by 255
            img = TF.to_tensor(img)
            extra_images = [TF.to_tensor(img) for img in extra_images]

        features = {
            'image': img
        }

        labels = {}

        if len(extra_images) > 0:
            extra_images = torch.stack(extra_images)
            features['hist_images'] = extra_images

        if self.include_original:
            features['image_original'] = torch.from_numpy(img_original.transpose(2,0,1))

        if 'segmentation' in data:
            labels['segmentation'] = torch.from_numpy(data['segmentation'].transpose(2,0,1))

        if 'imu_mask' in data:
            features['imu_mask'] = torch.from_numpy(data['imu_mask'].astype(np.bool))

        if 'objects' in data:
            objects = data['objects'].transpose(2,0,1)
            n_objects = objects.shape[0]
            objects_p = pad_instances(objects, self.object_masks_limit)
            labels['objects'] = torch.from_numpy(objects_p)
            labels['n_objects'] = n_objects

        if 'instance_seg' in data:
            ih,iw,im,ic = data['instance_seg_shape']
            instances = data['instance_seg'].reshape(ih,iw,im,ic).transpose(2,3,0,1)
            inst_p = pad_instances(instances, self.object_masks_limit)
            labels['instance_seg'] = torch.from_numpy(inst_p)

        if 'pa_similarity' in data:
            labels['pa_similarity'] = torch.from_numpy(data['pa_similarity'].transpose(2,0,1))

        # Add metadata to labels
        metadata = {
            'img_name': img_name,
            'image_path': img_filename,
            'mask_filename': mask_filename
        }
        labels.update(metadata)

        return features, labels
