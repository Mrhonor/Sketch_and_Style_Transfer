from torch.utils.data import Dataset
from PIL import Image
import os
import SDoG
import scipy.io as sio
import numpy as np


class CUB_200(Dataset):
    def __init__(self, root, train=True, transform=None):
        super(CUB_200, self).__init__()
        self.root = root
        self.train = train
        self.transform_ = transform
        self.classes_file = os.path.join(root, 'classes.txt')  # <class_id> <class_name>
        self.image_class_labels_file = os.path.join(root, 'image_class_labels.txt')  # <image_id> <class_id>
        self.images_file = os.path.join(root, 'images.txt')  # <image_id> <image_name>
        self.train_test_split_file = os.path.join(root, 'train_test_split.txt')  # <image_id> <is_training_image>
        self.bounding_boxes_file = os.path.join(root, 'bounding_boxes.txt')  # <image_id> <x> <y> <width> <height>

        self._train_ids = []
        self._test_ids = []
        self._image_id_label = {}
        self._train_path_label = []
        self._test_path_label = []
        self._bounding_boxes = {}

        self._train_test_split()
        self._get_id_to_label()
        self._get_path_label()
        self._get_bounding_box()

    def _train_test_split(self):

        for line in open(self.train_test_split_file):
            image_id, label = line.strip('\n').split()
            if label == '1':
                self._train_ids.append(image_id)
            elif label == '0':
                self._test_ids.append(image_id)
            else:
                raise Exception('label error')

    def _get_id_to_label(self):
        for line in open(self.image_class_labels_file):
            image_id, class_id = line.strip('\n').split()
            self._image_id_label[image_id] = class_id

    def _get_path_label(self):
        for line in open(self.images_file):
            image_id, image_name = line.strip('\n').split()
            label = self._image_id_label[image_id]
            if image_id in self._train_ids:
                self._train_path_label.append((image_name, label, image_id))
            else:
                self._test_path_label.append((image_name, label, image_id))

    def __getitem__(self, index):
        if self.train:
            image_name, label, image_id = self._train_path_label[index]
        else:
            image_name, label, image_id = self._test_path_label[index]
        image_path = os.path.join(self.root, 'images', image_name)
        img = Image.open(image_path)
        if img.mode == 'L':
            img = img.convert('RGB')
        label = int(label) - 1
        box = self._bounding_boxes[image_id]
        # print(box)
        img = img.crop(box)
        if self.transform_ is not None:
            img = self.transform_(img)
        return img, label

    def __len__(self):
        if self.train:
            return len(self._train_ids)
        else:
            return len(self._test_ids)

    def _get_bounding_box(self):
        for line in open(self.bounding_boxes_file):
            image_id, x, y, width, height = line.strip('\n').split()
            self._bounding_boxes[image_id] = (float(x), float(y), float(x) + float(width), float(y) + float(height))

class Car(Dataset):
    def __init__(self, root, train=True, transform=None):
        super(Car, self).__init__()
        self.root = root
        self.train = train
        self.transform_ = transform
        self.bounding_boxes_file = os.path.join(root, 'car_devkit/devkit/cars_train_annos.mat')
        self.mat = sio.loadmat(self.bounding_boxes_file)
        self.boxes = self.mat['annotations'][0]

    def getitem(self, index):
        num = int(index / 10)
        if num == 0:
            image_path = os.path.join(self.root, 'cars_train',  '0000'+str(index)+'.jpg')
        if num == 1:
            image_path = os.path.join(self.root, 'cars_train',  '000'+str(index)+'.jpg')
        if num == 2:
            image_path = os.path.join(self.root, 'cars_train',  '00'+str(index)+'.jpg')
        if num == 3:
            image_path = os.path.join(self.root, 'cars_train',  '0'+str(index)+'.jpg')

        print(image_path)
        img = Image.open(image_path)
        if img.mode == 'L':
            img = img.convert('RGB')

        box = self.boxes[index-1]
        print(box.shape)
        print(box)
        list_box = (int(box[0]), int(box[1]), int(box[2]), int(box[3])) 
        print(list_box)
        img = img.crop(list_box)
        if self.transform_ is not None:
            img = self.transform_(img)
        return img

def Concatenate(root, path1, path2):
    j=0
    for i in range(15000):
        sketch_path = os.path.join(path1, 'sketch'+str(i)+'.jpg')
        origin_path = os.path.join(path2, 'origin'+str(i)+'.jpg')
        if(os.path.exists(sketch_path) and os.path.exists(origin_path)):
            img1 = Image.open(sketch_path)
            rgb = img1.convert('RGB')
            rgb = rgb.resize((64,64), Image.ANTIALIAS)
            rgb_npy = np.array(rgb)
            img2=Image.open(origin_path)
            img2 = img2.resize((64,64), Image.ANTIALIAS)
            img2_npy = np.array(img2)
            img = np.concatenate((rgb_npy,img2_npy), 1)
            img = Image.fromarray(img)
            try:
                os.makedirs(root + 'dataset/bird/')
            except OSError:
                pass
            img.save(root + 'dataset/bird/img' + str(j) + '.jpg')
            j+=1


if __name__ == '__main__':
    cub200_root = '../'
    origin_cub_train = CUB_200(cub200_root)
    sketch_cub_train = CUB_200(cub200_root, transform=SDoG.XDOG)
    origin_cub_test = CUB_200(cub200_root, train=None)
    sketch_cub_test = CUB_200(cub200_root, train=None, transform=SDoG.XDOG)
    sketch_path = cub200_root + 'sketch/'
    origin_path = cub200_root + 'origin/'
    try:
        os.makedirs(sketch_path)
        os.makedirs(origin_path)
    except OSError:
        pass

    i = 0
    for img, label in origin_cub_train:
        print(type(img))
        print(label)
        img.save(origin_path+'origin'+str(i)+'.jpg')
        i = i+1
        if i >= origin_cub_train.__len__():
            break
    print(i)
    stop = i + origin_cub_test.__len__()
    for img, label in origin_cub_test:
        print(type(img))
        print(label)
        img.save(origin_path+'origin'+str(i)+'.jpg')
        i = i+1
        if i >= stop:
            break

    i = 0
    for img, label in sketch_cub_train:
        print(type(img))
        print(label)
        img.save(sketch_path+'sketch'+str(i)+'.jpg')
        i = i+1
        if i >= sketch_cub_train.__len__():
            break

    stop = i + sketch_cub_test.__len__()
    for img, label in sketch_cub_test:
        print(type(img))
        print(label)
        img.save(sketch_path+'sketch'+str(i)+'.jpg')
        i = i+1
        if i >= stop:
            break

    Concatenate(cub200_root, sketch_path, origin_path)

    
