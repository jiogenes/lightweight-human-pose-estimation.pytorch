import sys
from pathlib import Path
from PIL import Image

import cv2
import numpy as np
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (QApplication, QLabel, QMainWindow, QPushButton, 
                             QComboBox, QFrame, QFileDialog, QFrame, QInputDialog, QMessageBox)

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from models.with_mobilenet import PoseEstimationWithMobileNet, PoseClassificationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.pose import Pose
from modules.load_state import load_state
from val import normalize, pad_width

class App(QMainWindow):

    def __init__(self):
        super().__init__()
        self.is_new_motion = False
        self.height_size = 256
        self.stride = 8
        self.upsample_ratio = 4
        self.num_keypoints = Pose.num_kpts
        self.previous_poses = []

        pretrained = PoseEstimationWithMobileNet()
        checkpoint = torch.load('./weights/checkpoint_iter_370000.pth', map_location='cpu')
        load_state(pretrained, checkpoint)

        self.net = PoseClassificationWithMobileNet(pretrained)

        self.setWindowTitle("Motion Detection Application")
        self.setGeometry(100, 100, 700, 700)

        self.capture = None
        self.timer = QTimer(self)
        self.wait_timer = QTimer(self)

        # ComboBox for Mode Selector
        # self.modeSelector = QComboBox(self)
        # self.modeSelector.addItem('Mode 1')
        # self.modeSelector.addItem('Mode 2')
        # self.modeSelector.currentIndexChanged.connect(self.select_mode)
        # self.modeSelector.move(20, 20)

        # Start Video Feed Button
        self.startButton = QPushButton("Detection", self)
        self.startButton.clicked.connect(self.start_video)
        self.startButton.move(20, 60)

        self.stopButton = QPushButton('Stop', self)
        self.stopButton.clicked.connect(self.stop_video)
        self.stopButton.move(120, 60)

        # Status Label
        self.statusLabel = QLabel("Status: Select the motions you want to detect or set your own motions ", self)
        # self.statusLabel.move(20, 100)
        self.statusLabel.setGeometry(20, 20, 600, 20)

        # Status Color Pane
        # self.colorPane = QFrame(self)
        # self.colorPane.setStyleSheet("QWidget { background-color: %s }" % 'red')
        # self.colorPane.setGeometry(150, 100, 50, 20)

        # Video Label
        self.videoLabel = QLabel(self)
        self.videoLabel.setGeometry(50, 200, 640, 480)

        # Division Line
        self.divisionLine = QFrame(self)
        self.divisionLine.setFrameShape(QFrame.Shape.HLine)
        self.divisionLine.setFrameShadow(QFrame.Shadow.Sunken)
        self.divisionLine.setGeometry(0, 180, 800, 1)

        # Save and Load Buttons
        self.saveButton = QPushButton("New Motion", self)
        self.saveButton.clicked.connect(self.make_new_motion)
        self.saveButton.move(20, 130)

        self.loadButton = QPushButton("Select Motion", self)
        self.loadButton.clicked.connect(self.load_motion)
        self.loadButton.move(120, 130)

        self.show()

    def show_info(self, text):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setText(text)
        msg.setWindowTitle("Info")
        msg.exec()

    def select_mode(self):
        current_mode = self.modeSelector.currentText()
        print(f"{current_mode} selected")

    def start_video(self):
        if not self.capture:
            self.capture = cv2.VideoCapture(0)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.timer.timeout.connect(self.update_frame)
            self.timer.start(30)
            self.statusLabel.setText("Status: Running")

    def stop_video(self):
        if self.capture:
            self.timer.stop()
            self.capture.release()
            self.capture = None
            self.videoLabel.clear()
            self.statusLabel.setText('Status: Stop')

    def update_frame(self):
        ret, img = self.capture.read()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if not self.is_new_motion:
            orig_img = img.copy()
            output, heatmaps, pafs, scale, pad = self.infer_fast(img, self.height_size, self.stride, self.upsample_ratio)

            total_keypoints_num = 0
            all_keypoints_by_type = []
            for kpt_idx in range(self.num_keypoints):  # 19th for bg
                total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

            pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
            for kpt_id in range(all_keypoints.shape[0]):
                all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * self.stride / self.upsample_ratio - pad[1]) / scale
                all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * self.stride / self.upsample_ratio - pad[0]) / scale
            current_poses = []
            for n in range(len(pose_entries)):
                if len(pose_entries[n]) == 0:
                    continue
                pose_keypoints = np.ones((self.num_keypoints, 2), dtype=np.int32) * -1
                for kpt_id in range(self.num_keypoints):
                    if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                        pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                        pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
                pose = Pose(pose_keypoints, pose_entries[n][18])
                current_poses.append(pose)

            for pose in current_poses:
                pose.draw(img)
            img = cv2.addWeighted(orig_img, 0.6, img, 0.4, 0)
            for pose in current_poses:
                if output:
                    color = (255, 0, 0)
                    self.statusLabel.setText("Status: Detected!")
                else:
                    color = (0, 255, 0)
                    self.statusLabel.setText("Status: Not Detected")

                cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
                            (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), color)

            self.display_image(img)

        else:
            self. capture_image(img)

    def display_image(self, img):
        qformat = QImage.Format.Format_RGB888
        img = QImage(img, img.shape[1], img.shape[0], qformat)
        self.videoLabel.setPixmap(QPixmap.fromImage(img))

    def capture_image(self, img):
        qformat = QImage.Format.Format_RGB888
        frame = QImage(img, img.shape[1], img.shape[0], qformat)
        self.videoLabel.setPixmap(QPixmap.fromImage(frame))

        Path(f'./data/{self.new_motion_name}/').mkdir(parents=True, exist_ok=True)

        image_count = len(list(Path(f'./data/{self.new_motion_name}/').glob('*.jpg')))
        if image_count <= 200:
            cv2.imwrite(f'./data/{self.new_motion_name}/frame_{image_count}.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        else:
            self.stop_video()
            self.is_new_motion = False


    def load_motion(self):
        options = QFileDialog.Option(QFileDialog.Option.HideNameFilterDetails)
        file_name, _ = QFileDialog.getOpenFileName(self, "Load File", "~/", "All Files (*);;Text Files (*.txt)", options=options)
        if file_name:
            checkpoint = torch.load(file_name, map_location='cpu')
            load_state(self.net, checkpoint)

            if torch.cuda.is_available():
                self.net = self.net.cuda()
            elif torch.backends.mps.is_available():
                self.net = self.net.to('mps')

    def make_new_motion(self):
        text, ok = QInputDialog.getText(self, 'Motion Name', 'Enter Motion Name:')

        if not ok or not text:
            return
        
        self.is_new_motion = True
        self.new_motion_name = text
        
        self.make_dataset()

        # self.train(self.net)

        # self.save_motion(text)

        # self.is_new_motion = False

    def make_dataset(self):

        self.show_info('Take a picture of your posture. Please take your stance.')
        self.start_video()
        # QTimer.singleShot(2000, self.stop_video)
        # self.stop_video()


        transform = transforms.Compose([
            transforms.Resize(size=(128, 128), antialias=True),
            transforms.RandomApply([
                # transforms.RandomResizedCrop((128, 128), scale=(0.9, 1.1), ratio=(0.9, 1.1), antialias=True), 
                # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
                transforms.RandomRotation(10),
            ], p=0.3),
            # transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])

        # dataset = CustomDataset(transform)


        # return dataset

    def train(net, dataset):
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
            for idx, batch in enumerate(train_dataloader):
                images, labels = batch
                if torch.cuda.is_available():
                    images = images.cuda()
                    labels = labels.float().cuda()
                elif torch.backends.mps.is_available():
                    images = images.to('mps')
                    labels = labels.float().to('mps')
                else:
                    labels = labels.float()

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
            for idx, batch in enumerate(test_dataloader):
                images, labels = batch
                if torch.cuda.is_available():
                    images = images.cuda()
                    labels = labels.float().cuda()
                elif torch.backends.mps.is_available():
                    images = images.to('mps')
                    labels = labels.float().to('mps')
                else:
                    labels = labels.float()


                pred, _ = net(images)
                loss = loss_func(pred.squeeze(), labels)

                loss_avg += loss.cpu().detach().item()
                acc += ((torch.nn.functional.sigmoid(pred.squeeze()) > 0.5) == labels.squeeze()).float().mean().item()

            print('test loss :', loss_avg / (idx+1))
            print('test acc :', acc*100 / (idx+1))
        
        return net

    def save_motion(self, motion_name):
        state_dict = self.net.state_dict()
        for key in list(state_dict.keys()):
            if 'pretrained' in key:
                state_dict.pop(key)

        with open(f'./motions/{motion_name}.pt', 'wb') as f:
            torch.save({'state_dict': state_dict}, f)

    def infer_fast(self, img, net_input_height_size, stride, upsample_ratio,
               pad_value=(0, 0, 0), img_mean=np.array([128, 128, 128], np.float32), img_scale=np.float32(1/256)):
        height, width, _ = img.shape
        scale = net_input_height_size / height

        scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        scaled_img = normalize(scaled_img, img_mean, img_scale)
        min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
        padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

        tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()

        if torch.cuda.is_available():
            tensor_img = tensor_img.cuda()
        elif torch.backends.mps.is_available():
            tensor_img = tensor_img.to('mps')

        output, stages_output = self.net(tensor_img)
        output = (torch.nn.functional.sigmoid(output.squeeze()) > 0.5).cpu().item()

        stage2_heatmaps = stages_output[-2]
        heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
        heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

        stage2_pafs = stages_output[-1]
        pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
        pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

        return output, heatmaps, pafs, scale, pad


class CustomDataset(Dataset):
    def __init__(self, motion_name, transform) -> None:
        super().__init__()
        self.motion_name = motion_name
        self.transform = transform
        self.data_list, self.labels = self.load_data()

    def load_data(self):
        data_list, labels = [], []
        paths = Path(f'./data/{self.motion_name}/').glob('*.jpg')
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

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = App()
    sys.exit(app.exec())
