import os
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import torch
import glob


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()])


class LavalIndoorDataset(Dataset):
    def __init__(self, data_root, isTrain=True):
        super().__init__()
        self.data_root = data_root
        self.hdr_dir = data_root + '/EnvMap_SP_128_128_wraped'
        self.crop_dir = data_root + '/PhotoInput'

        test_img_ids = np.load('./test_photo.npy', allow_pickle=True)

        test_id_map = {}
        for test_id in test_img_ids:
            test_id_map[test_id] = test_id

        pairs = []

        if isTrain:
            hdr_names = os.listdir(self.hdr_dir)
            for hdr_name in hdr_names:
                img_id = hdr_name[0:-4]
                img_id_prefix = img_id.split('_')[0]
                if img_id_prefix not in test_id_map:
                    hdr_path = f'{self.hdr_dir}/{hdr_name}'
                    photo_name = f'{img_id}.png'
                    crop_path = f'{self.crop_dir}/{photo_name}'
                    pairs.append([crop_path, hdr_path, img_id])
        else:
            for img_id_prefix in test_img_ids:
                for i in range(10):
                    img_id = f'{img_id_prefix}_v0_h{i}'

                    photo_name = f'{img_id}.png'
                    crop_path = f'{self.crop_dir}/{photo_name}'

                    hdr_name = f'{img_id}.exr'
                    hdr_path = f'{self.hdr_dir}/{hdr_name}'

                    pairs.append([crop_path, hdr_path, img_id])

        self.pairs = pairs

        self.dataset_size = len(self.pairs)

    def __len__(self) -> int:
        return self.dataset_size

    def __getitem__(self, index: int):

        crop_path, hdr_path, img_id = self.pairs[index]

        photo_img = Image.open(crop_path)
        photo = transform(photo_img)

        hdr = cv2.imread(hdr_path, cv2.IMREAD_UNCHANGED)
        hdr = hdr[:, :, [2, 1, 0]]  # bgr -> rgb
        hdr = np.transpose(hdr, (2, 0, 1))  # (3, h, w)
        hdr_gt = torch.from_numpy(hdr).contiguous().float()
        hdr_gt = torch.log(hdr_gt + 1)

        return hdr_gt, photo, img_id


class TestImgDataset(Dataset):
    def __init__(self, input_dir):
        super(TestImgDataset, self).__init__()

        self.img_list = []

        self.img_list = sorted(glob.glob(f'{input_dir}/*.png'))
        assert len(self.img_list) > 0

        self.length = len(self.img_list)
        self.input_dir = input_dir

    def __getitem__(self, index):
        img_path = self.img_list[index]
        img_name = os.path.basename(img_path)
        img_id = img_name.replace('.png', '')

        img = Image.open(img_path)

        img_tensor = transform(img).to(torch.float32)

        return img_tensor, img_id

    def __len__(self):
        return self.length


if __name__ == '__main__':
    pass
