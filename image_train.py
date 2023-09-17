import argparse
from tqdm import tqdm
from pathlib import Path
import numpy as np
import cv2

from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from demo import load_state, normalize, pad_width
from models.with_mobilenet import PoseEstimationWithMobileNet, PoseClassificationWithMobileNet
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, transform=None) -> None:
        super().__init__()
        self.transform = transform
        self.data_list, self.labels = self.load_data()

    def load_data(self):
        data_list, labels = [], []
        paths = Path('./data/lying_pictures/').glob('*.jpg')
        for path in paths:
            data_list.append(str(path))
            labels.append(1)
        paths = Path('./data/mpii/mpii_human_pose_v1/images/').glob('*.jpg')
        for path in paths:
            data_list.append(str(path))
            labels.append(0)
        return data_list, labels
    
    def __getitem__(self, index):
        image = cv2.imread(self.data_list[index], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = self.transform(image)
        return image, self.labels[index]
    
    def __len__(self):
        return len(self.data_list)
    
class VideoReader(object):
    def __init__(self, file_name):
        self.file_name = file_name
        try:  # OpenCV needs int to read from webcam
            self.file_name = int(file_name)
        except ValueError:
            pass

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.file_name)
        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.file_name))
        return self

    def __next__(self):
        was_read, img = self.cap.read()
        if not was_read:
            raise StopIteration
        return img

def train(net):
    transform = transforms.Compose([
        transforms.Resize(size=(128, 128), antialias=True),
        transforms.RandomApply([
            # transforms.RandomResizedCrop((128, 128), scale=(0.9, 1.1), ratio=(0.9, 1.1), antialias=True), 
            transforms.RandomRotation(10),
            # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
        ], p=0.3),
        # transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    dataset = CustomDataset(transform)
    trainlen = int(len(dataset) * 0.8)
    testlen = len(dataset) - trainlen
    train_dataset, test_dataset = random_split(dataset, [trainlen, testlen])

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=8)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=8)

    optimizer = torch.optim.AdamW(net.classifier.parameters(), lr=0.001, betas=(0.5, 0.999))
    loss_func = torch.nn.BCEWithLogitsLoss()
    
    for e in range(5):
        net.train()
        loss_avg, acc = 0, 0
        print(f'epoch : {e+1}')
        for idx, batch in enumerate(tqdm(train_dataloader)):
            images, labels = batch
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.float().cuda()
            elif torch.backends.mps.is_available():
                images = images.to('mps')
                labels = labels.float().to('mps')

            pred, _ = net(images)
            optimizer.zero_grad()
            loss = loss_func(pred.squeeze(), labels)
            loss.backward()
            optimizer.step()

            loss_avg += loss.cpu().detach().item()
            acc += ((torch.nn.functional.sigmoid(pred.squeeze()) > 0.5) == labels.squeeze()).float().mean().item()

        print('train loss :', loss_avg / (idx+1))
        print('train acc :', acc*100 / (idx+1) )

        net.eval()
        loss_avg, acc = 0, 0
        for idx, batch in enumerate(tqdm(test_dataloader)):
            images, labels = batch
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.float().cuda()
            elif torch.backends.mps.is_available():
                images = images.to('mps')
                labels = labels.float().to('mps')


            pred, _ = net(images)
            loss = loss_func(pred.squeeze(), labels)

            loss_avg += loss.cpu().detach().item()
            acc += ((torch.nn.functional.sigmoid(pred.squeeze()) > 0.5) == labels.squeeze()).float().mean().item()

        print('test loss :', loss_avg / (idx+1))
        print('test acc :', acc*100 / (idx+1))
    
    return net

def inference(net, img, stride=8, net_input_height_size=256, cpu=False,
              pad_value=(0, 0, 0), img_mean=np.array([128, 128, 128], np.float32), img_scale=np.float32(1/256)):
    height, width, _ = img.shape
    scale = net_input_height_size / height

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    # scaled_img = normalize(scaled_img, img_mean, img_scale)
    scaled_img = (scaled_img.astype(np.float32) - 128)/128
    # print(scaled_img.max(), scaled_img.min())
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

    tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
    if not cpu:
        if torch.cuda.is_available():
            tensor_img = tensor_img.cuda()
        elif torch.backends.mps.is_available():
            tensor_img = tensor_img.to('mps')

    output, keypoints = net(tensor_img)
    return (torch.nn.functional.sigmoid(output.squeeze()) > 0.5).cpu().item(), output.squeeze().cpu().item()

def main(args):
    delay = 1

    pretrained = PoseEstimationWithMobileNet()
    checkpoint = torch.load('./weights/checkpoint_iter_370000.pth', map_location='cpu')
    load_state(pretrained, checkpoint)

    net = PoseClassificationWithMobileNet(pretrained)
    # net = net.eval()
    if args.checkpoint_path:
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
        load_state(net, checkpoint)

    if torch.cuda.is_available():
        net = net.cuda()
    elif torch.backends.mps.is_available():
        net = net.to('mps')

    net = net.eval()

    if not args.checkpoint_path:
        train(net)
        
    img_provider = VideoReader(0)
    for idx, img in enumerate(img_provider):
        output, value = inference(net, img)
        print(idx, output, value)

        cv2.imshow('Lightweight Human Pose Estimation Python Demo', img)
        key = cv2.waitKey(delay)
        if key == 27:  # esc
            return
        elif key == 112:  # 'p'
            if delay == 1:
                delay = 0
            else:
                delay = 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-path', type=str, default='', help='path to the checkpoint')
    args = parser.parse_args()

    main(args)