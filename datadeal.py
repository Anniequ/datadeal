# _*_ coding: utf-8 _*_
# @author: anniequ
# @file: datadeal.py
# @time: 2021/1/18 18:01
# @Software: PyCharm

import shutil
import json
import os
import os.path as osp

from labelme import utils
import base64
import cv2 as cv
PATH_x = r"D:\project\data_qjp\406\imgjson"
PATH_TRAIN = r"D:\project\data_sets\data_sci\train"  # path为已重命名后的X射线图片存放的路径
PATH_VAL = r"D:\project\data_sets\datatest\val"  # path为验证图片存放的路径


def src_rename(path):
    # change to ***.png
    fileList = os.listdir(path)
    n = 298
    for i in fileList:
        oldName = path + os.sep + i  # os.sep 添加系统分隔符

        newName = path + os.sep + str(n).zfill(3) + '.json'
        if oldName == newName:
            n += 1
            continue
        os.rename(oldName, newName)
        print(oldName, '====>', newName)
        n += 1
    print("{} pictures has finished...".format(n))


def json_to_mask(json_file):
    json_file = json_file
    # 下面类别就是自己要分的类别数目，我的共有9类，而且下面这个字典的关键字（单引号内的）是自己在用labelme时标注的类别
    # 所以我自己又加了0-9的关键字，你的改成你自己的关键字就可以了，然后冒号后面的我猜代表的是颜色的标号，这样就会出来不同的颜色。
    label_name_to_value = {'_background_': 0,
                           'tape': 1,
                           'scissors': 2,
                           'pen': 3,
                           '3': 4,
                           '4': 5,
                           '5': 6,
                           '6': 7,
                           '7': 8,
                           '8': 9}
    out_dir = osp.join(osp.dirname(json_file), osp.basename(json_file).split('_')[0] + 'mask_set')
    if not osp.exists(out_dir):
        os.mkdir(out_dir)
    count = os.listdir(json_file)
    for i in count:
        path = os.path.join(json_file, i)
        if os.path.isfile(path):
            data = json.load(open(path))

            if data['imageData']:
                imageData = data['imageData']
            else:
                imagePath = os.path.join(os.path.dirname(path), data['imagePath'])
                with open(imagePath, 'rb') as f:
                    imageData = f.read()
                    imageData = base64.b64encode(imageData).decode('utf-8')
            img = utils.img_b64_to_arr(imageData)

            for shape in data['shapes']:
                label_name = shape['label']
                if label_name in label_name_to_value:
                    label_value = label_name_to_value[label_name]
                else:
                    label_value = len(label_name_to_value)
                    label_name_to_value[label_name] = label_value

            # label_values must be dense
            label_values, label_names = [], []
            for ln, lv in sorted(label_name_to_value.items(), key=lambda x: x[1]):
                label_values.append(lv)
                label_names.append(ln)
            assert label_values == list(range(len(label_values)))

            lbl = utils.shapes_to_label(img.shape, data['shapes'], label_name_to_value)

            name = osp.basename(i).replace('.', '_').split('_')
            # PIL.Image.fromarray(lbl).save(osp.join(out_dir, rename+'label.png'))
            utils.lblsave(osp.join(out_dir, name[0] + '_mask.png'), lbl)

    print('Saved to {} mask pictures'.format(len(os.listdir(out_dir))))
    return out_dir


def change_to_binary(file_path):
    res_file = osp.join(osp.dirname(file_path), 'binary')
    if not osp.isdir(res_file):
        os.mkdir(res_file)
    img_set = os.listdir(file_path)
    for img_name in img_set:
        img_path = file_path + os.sep + img_name
        # print(img_path)
        try:
            img = cv.imread(img_path)
            img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            ret, img_binary = cv.threshold(img_gray, 30, 255, cv.THRESH_BINARY)  # green 75 so < 75
            cv.imwrite(res_file + os.sep + img_name, img_binary)
        except:
            print("error: {} fail to change to binary...".format(img_name))

    print('Changing to binary finish...')
    return res_file


def re_size(filepath, xy=(512, 512)):
    img_set = os.listdir(filepath)
    for img_name in img_set:
        img_path = filepath + os.sep + img_name
        try:
            if len(img_name.split('_')) == 2:
                img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
            else:
                img = cv.imread(img_path)
            new_img = cv.resize(img, xy)
            cv.imwrite(img_path, new_img)
        except:
            print("error: {} fail to resize...".format(img_name))

    print("Finish resizing...")


def label_deal(path):
    # 将原 picture set下的json文件移入***_jsonset文件夹中以备处理
    json_dir = osp.join(osp.dirname(path), osp.basename(path) + '_jsonset')
    filepath_list = os.listdir(path)
    if not osp.isdir(json_dir):
        os.mkdir(json_dir)
    for file_path in filepath_list:
        if file_path.split('.')[1] == 'json':
            try:
                shutil.move(osp.join(path, file_path), json_dir)
            except:
                print("error: move {} fail...".format(file_path))
    print("All json has copy to json set...")
    mask_file = json_to_mask(json_dir)  # get mask picture, saved as mask_set
    # 将得到的mask图片二值化并移回train or val
    binary = change_to_binary(mask_file)
    try:
        binary_file = os.listdir(binary)
        for file in binary_file:
            shutil.move(osp.join(binary, file), path)
    except:
        print('error: move mask to train or val fail...')

    # resize 图片到指定大小
    re_size(path, (512, 512))
    print("Mask pretreatment has finished...")


# 原X射线图片重命名
src_rename(PATH_x)
# then, 标记图片
# label_deal(PATH_VAL)
