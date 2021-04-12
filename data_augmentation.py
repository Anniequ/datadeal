# HorizontalFlip 翻转
import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as albu
import os
import shutil
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from PIL import Image
# cv
"""
if image_arr.shape[2] == 1:
    image_arr_rgb = cv2.cvtColor(image_arr, cv2.COLOR_GRAY2RGB) # grey ->RGB

image_arr_rgb_ = cv2.cvtColor(image_arr, cv2.COLOR_BGR2RGB) # imread 读的彩色图按照BGR像素存储，如果转换成RGB则需要用cvtColor函数进行转换
# imread 读的图片按照 H,W,C 格式存储
# H,W,C格式转换到C,H,W格式
image_arr_rgb_chw = np.transpose(image_arr_rgb_, (2,0,1))

"""
x_train_dir = r"D:\project\data_qjp\406\MYVOC224224\JPEGImages"
y_train_dir = r"D:\project\data_qjp\406\MYVOC224224\SegmentationClass"
save_as_dir = r'D:\project\data_qjp\406\MYVOCaug224224'


# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()


class Dataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    CLASSES = ['__background__', 'tape', 'scissors', 'nailpolish', 'lighter']

    def __init__(
            self,
            images_dir,
            masks_dir,
            augmentation=None,
            preprocessing=None,
    ):
        self.img_ids = os.listdir(images_dir)
        self.mask_ids = os.listdir(masks_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.img_ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.mask_ids]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i])  # 灰度图读取
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        # print(image.type())
        # print(img.type())

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.img_ids)


def get_training_augmentation():
    train_transform = [

        albu.HorizontalFlip(p=0.5), #水平翻转

        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0), # 平移缩放旋转

        albu.PadIfNeeded(min_height=224, min_width=224, always_apply=True, border_mode=0),  # 加padding
        albu.RandomCrop(height=224, width=224, always_apply=True), # 随机剪裁

        albu.IAAAdditiveGaussianNoise(p=0.2),  # Add gaussian noise to the input image.
        albu.IAAPerspective(p=0.5),  # Perform a random four point perspective transform of the input

        albu.OneOf(
            [
                albu.CLAHE(p=1), # 对比度受限情况下的自适应直方图均衡化算法
                albu.RandomBrightnessContrast(p=1), # Randomly change brightness and contrast
                albu.RandomGamma(p=1), # Gamma变换
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.IAASharpen(p=1), # Sharpen the input image and overlays the result with the original image.
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.RandomBrightnessContrast(p=1),
                albu.HueSaturationValue(p=1),  # Randomly change hue, saturation and value of the input image.
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(384, 480)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


def aug(SC):
    augmented_dataset = Dataset(
        x_train_dir,
        y_train_dir,
        augmentation=get_training_augmentation(),
    )

    # same image with different random transforms
    for k in range(SC):
        path1 = os.path.join(save_as_dir, 'JPEGImages{}'.format(k))
        if not os.path.exists(path1):
            os.mkdir(path1)
        path2 = os.path.join(save_as_dir, 'SegmentationClass{}'.format(k))
        if not os.path.exists(path2):
            os.mkdir(path2)
        # print(path1,path2)
        for i, [img, mask] in enumerate(augmented_dataset):
            img_path = os.path.join(path1, str(i).zfill(3) + '.jpg')
            cv2.imwrite(img_path, img)
            mask_path = os.path.join(path2, str(i).zfill(3) + '.jpg')
            cv2.imwrite(mask_path, mask)
        print('JPEGImages{} and SegmentationClass{} has finished.'.format(k, k))

def src_rename(path, n, tye):
    # change to ***.png
    fileList = os.listdir(path)
    for i in fileList:
        oldName = path + os.sep + i  # os.sep 添加系统分隔符
        newName = path + os.sep + str(n).zfill(4) + tye
        if oldName == newName:
            n += 1
            continue
        os.rename(oldName, newName)
        print(oldName, '====>', newName)
        n += 1
    print("{} pictures has finished...".format(n))
    return n

def move(path, path_to):
    filepath_list = os.listdir(path)
    for file_path in filepath_list:
        shutil.move(os.path.join(path, file_path), path_to)


def merg(SC):
    n = 0
    for k in range(SC):
        path1 = os.path.join(save_as_dir, 'JPEGImages{}'.format(k))
        path1_to = os.path.join(save_as_dir, 'JPEGImages')
        src_rename(path1, n, '.jpg')
        path2 = os.path.join(save_as_dir, 'SegmentationClass{}'.format(k))
        path2_to = os.path.join(save_as_dir, 'SegmentationClass')
        n = src_rename(path2, n, ".png")
        move(path1,path1_to)
        move(path2, path2_to)
        print("{} pictures has moved".format(n))

if __name__ == "__main__":
    # Lets look at data we have
    """
    dataset = Dataset(x_train_dir, y_train_dir, classes=['tape'])

    image, mask = dataset[4] # get some sample
    # print(mask)
    visualize(
        image=image,
        mask=mask,
    )
    """
    SC = 6
    #### Visualize resulted augmented images and masks
    # aug(SC)
    merg(SC)

        # print(image.shape)
        # visualize(image=image, mask=mask)
