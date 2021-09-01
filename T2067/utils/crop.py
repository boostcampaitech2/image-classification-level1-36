import pandas as pd
import cv2
import os
import numpy as np
from torchvision import transforms
from torchvision.transforms import Resize, ToTensor
from PIL import Image
import albumentations
import albumentations.pytorch
from face_detection import RetinaFace
from tqdm import tqdm


#pip install git+https://github.com/elliottzheng/face-detection.git@master


def crop_faces(img):
    flag = False
    detector = RetinaFace(gpu_id=0)
    cen_crop = albumentations.CenterCrop(400, 335)
    transform = transforms.Compose([
        Resize((300, 300), Image.BILINEAR),
        ToTensor()])

    x_pad = 20
    y_pad = 30
    imag = cv2.imread(img)
    H, W, _ = imag.shape
    detected = detector(imag)
    if detected:
        flag = True
        face = detected[0]
        xmin, ymin, xmax, ymax = map(int, list(face[0]))

        image = albumentations.augmentations.crops.transforms.Crop(max(0, xmin - x_pad),
                                                                   max(0, ymin - y_pad),
                                                                   min(W, xmax + x_pad),
                                                                   min(H, ymax + y_pad),
                                                                   always_apply=True, p=1.0)(image=imag)
    else:
        image = Image.open(img)
        image1 = np.array(image)
        image = cen_crop(image=image1)
    cropped = image['image']
    cropped = transforms.ToTensor()(cropped)
    cropped = transforms.ToPILImage()(cropped)
    cropped = transform(cropped)
    cropped = transforms.ToPILImage()(cropped)
    cropped = np.array(cropped)
    return flag, cropped


def crop_train_images(data_path, output_path):
    #data_path = '/opt/ml/image_classification/data/train'
    #output_path = "cropped"
    dataframe = pd.read_csv(data_path + '/train.csv')
    total_faces = len(dataframe)
    print("No. of Images: {}".format(total_faces))
    found_faces = 0
    count = 0
    for i in tqdm(range(len(dataframe)), desc="Cropping In Progress"):
        image_folder = data_path +'/images/'+ dataframe.iloc[i]['path']
        for im in os.listdir(image_folder):
            if not im.startswith("._"):
                img = os.path.join(image_folder, im)
                #img = image_folder + '/'+im
                flag, cropped = crop_faces(img)
                if flag:
                    found_faces += 1
                cv2.imwrite(os.path.join(output_path, "".join(img.split("/")[-1:])), cropped)
                print(count)
                count += 1
    print("Face Cropping for Train Data Complete!")
    print("Found {} faces out of {} images.".format(found_faces, total_faces))


def crop_test_images(test_path, output_path):
    #test_path = 'data/eval'
    image_dir = os.path.join(test_path, 'images')
    found_faces = 0
    count = 0
    test_images = [os.path.join(image_dir,i) for i in os.listdir(image_dir) if not i.startswith("._")]
    total_faces = len(test_images)
    for i in tqdm(range(len(test_images)), desc="Test Images Cropping"):
        im = test_images[i]
        if not im.startswith("._"):
            img = os.path.join(image_dir, im)
            # img = image_folder + '/'+im
            flag, cropped = crop_faces(img)
            if flag:
                found_faces += 1
            cv2.imwrite(os.path.join(output_path, "".join(img.split("/")[-2:])), cropped)
            print(count)
            count += 1
    print("Face Cropping for Test Data Complete!")
    print("Found {} faces out of {} images.".format(found_faces, total_faces))