# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
#
# import _init_paths
#
# import os
# import cv2
# import numpy as np
#
# from opts import opts
# from detectors.detector_factory import detector_factory
# #
# image_ext = ['jpg', 'jpeg', 'png', 'webp']
# video_ext = ['mp4', 'mov', 'avi', 'mkv']
# time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
#
#
# def demo(opt):
#     os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
#     opt.debug = max(opt.debug, 1)
#     Detector = detector_factory[opt.task]
#     detector = Detector(opt)
#     frame_id = 0
#
#     if os.path.isdir(opt.demo):
#         image_names = []
#         ls = os.listdir(opt.demo)
#         for file_name in sorted(ls):
#             ext = file_name[file_name.rfind('.') + 1:].lower()
#             if ext in image_ext:
#                 image_names.append(os.path.join(opt.demo, file_name))
#     else:
#         image_names = [opt.demo]
#
#     for (image_name) in image_names:
#         if len(image_names) == 1:
#             img_id = 'video'
#             ret = detector.run(image_name, Img_Id=img_id)
#             time_str = ''
#             for stat in time_stats:
#                 time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
#             print(time_str)
#         else:
#             # img_id = image_name.split('/')[-1].split('.')[0]
#             img_id = image_name.split('/')[-1].split('.')[0]
#             output = detector.run(image_name, Img_Id=img_id)
#             ret = output['results']
#             for k, v in zip(ret.keys(), ret.values()):
#                 result = np.zeros((v.shape[0], 6))
#                 result[:, 0] = frame_id
#                 result[:, 1] = k
#                 result[:, 2:4] = v[:, 2:4]
#                 result[:, 4:] = v[:, :2]
#                 np.savetxt('result.csv', result, fmt='%f', delimiter=',')
#             frame_id += 1
#
#
# if __name__ == '__main__':
#     opt = opts().init()
#     demo(opt)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import cv2
import numpy as np

from opts import opts
from detectors.detector_factory import detector_factory

def readFile(path):
    # 打开文件（注意路径）
    f = open(path)
    # 逐行进行处理
    img_list = []
    for data in f.readlines():
        ## 去掉每行的换行符，"\n"
        data = data.strip('\n')
        ## 按照 空格进行分割。
        nums = data.split(' ')[0]
        ## 添加到 matrix 中。
        img_list.append(nums)
    return img_list
#
image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']


def demo(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.debug = max(opt.debug, 1)
    Detector = detector_factory[opt.task]
    detector = Detector(opt)
    save_thresh = 0.5
    dataset = opt.test_dataset

    class_list = [1, 2, 3, 4, 6, 8]  # 1:person, 2:bicycle, 3:car, 4:motorcycle, 6:bus, 8:truck

    dataset_list = os.listdir(opt.dataset_path)
    dataset_list.sort()

    for data_dir in dataset_list:
        dataset_path = os.path.join(opt.dataset_path, data_dir)
        frame_id = 0
        n = 1

        if dataset == 'KITTI':
            img_path = os.path.join(dataset_path, 'image_2')
            if os.path.isdir(img_path):
                image_names = []
                ls = os.listdir(img_path)
                for file_name in sorted(ls):
                    ext = file_name[file_name.rfind('.') + 1:].lower()
                    if ext in image_ext:
                        image_names.append(os.path.join(opt.demo, file_name))
        elif dataset == 'TUM':
            img_path = os.path.join(dataset_path, 'rgb')
            file_path = os.path.join(dataset_path, 'associations.txt')
            image_names = []
            ls = readFile(file_path)
            for file_name in ls:
                image_names.append(os.path.join(opt.demo, file_name))

        for (image_name) in image_names:
            image = os.path.join(img_path, image_name)
            image = image + '.png'
            ret = detector.run(image)['results']
            for k, v in zip(ret.keys(), ret.values()):
                valid_row_list = []
                if len(v) > 0 and k in class_list:
                    result = np.zeros((v.shape[0], 6))
                    result[:, 0] = frame_id
                    result[:, 1] = k
                    result[:, 2:4] = v[:, 2:4]
                    result[:, 4:] = v[:, :2]
                    for i in range(v.shape[0]):
                        if v[i, -1] > save_thresh:
                            valid_row_list.append(i)

                    if len(valid_row_list) > 0:
                        valid_row = np.array(valid_row_list)
                        result = result[valid_row]
                        if n == 1:
                            output = result
                        output = np.vstack((output, result))
                        n += 1
                else:
                    pass
            frame_id += 1

        output_path = os.path.join(opt.output_path, data_dir)
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        np.savetxt(output_path + '/result.csv', output, fmt='%f', delimiter=',', newline='\n')

if __name__ == '__main__':
    opt = opts().init()
    demo(opt)


# import sys
# import os
# import numpy as np
# CENTERNET_PATH = '/home/wangshuo/PycharmProjects/detection/CenterNet/src/lib/'
# sys.path.insert(0, CENTERNET_PATH)
#
# from detectors.detector_factory import detector_factory
# from opts import opts
#
# def readFile(path):
#     # 打开文件（注意路径）
#     f = open(path)
#     # 逐行进行处理
#     img_list = []
#     for data in f.readlines():
#         ## 去掉每行的换行符，"\n"
#         data = data.strip('\n')
#         ## 按照 空格进行分割。
#         nums = data.split(' ')[0]
#         ## 添加到 matrix 中。
#         img_list.append(nums)
#     return img_list
#
# frame_id = 0
# n = 1
# save_thresh = 0.5
# dataset = 'TUM' # 'TUM/KITTI'
# image_ext = ['jpg', 'jpeg', 'png', 'webp']
# MODEL_PATH = '/home/wangshuo/PycharmProjects/detection/CenterNet/models/ctdet_coco_dla_2x.pth'
# TASK = 'ctdet' # or 'multi_pose' for human pose estimation
# opt = opts().init('{} --load_model {}'.format(TASK, MODEL_PATH).split(' '))
# detector = detector_factory[opt.task](opt)
#
# img_path = '/home/wangshuo/Datasets/TUM/moving_objects/rgbd_dataset_freiburg3_walking_halfsphere/rgb'
# file_path = '/home/wangshuo/Datasets/TUM/moving_objects/rgbd_dataset_freiburg3_walking_halfsphere/associations.txt'
# class_list = [1, 2, 3, 4, 6, 8] # 1:person, 2:bicycle, 3:car, 4:motorcycle, 6:bus, 8:truck
#
# if dataset == 'KITTI':
#     if os.path.isdir(img_path):
#         image_names = []
#         ls = os.listdir(img_path)
#         for file_name in sorted(ls):
#             ext = file_name[file_name.rfind('.') + 1:].lower()
#             if ext in image_ext:
#                 image_names.append(os.path.join(opt.demo, file_name))
# elif dataset == 'TUM':
#     image_names = []
#     ls = readFile(file_path)
#     for file_name in ls:
#         image_names.append(os.path.join(opt.demo, file_name))
#
# for (image_name) in image_names:
#     image = os.path.join(img_path, image_name)
#     image = image + '.png'
#     ret = detector.run(image)['results']
#     for k, v in zip(ret.keys(), ret.values()):
#         valid_row_list = []
#         if len(v) > 0 and k in class_list :
#             result = np.zeros((v.shape[0], 6))
#             result[:, 0] = frame_id
#             result[:, 1] = k
#             result[:, 2:4] = v[:, 2:4]
#             result[:, 4:] = v[:, :2]
#             for i in range(v.shape[0]):
#                 if v[i, -1] > save_thresh:
#                     valid_row_list.append(i)
#
#             if len(valid_row_list) > 0:
#                 valid_row = np.array(valid_row_list)
#                 result = result[valid_row]
#                 if n == 1:
#                     output = result
#                 output = np.vstack((output, result))
#                 n += 1
#         else:
#             pass
#     frame_id += 1
#
# np.savetxt('result.csv', output, fmt='%f', delimiter=',', newline='\n')
# print('done!')