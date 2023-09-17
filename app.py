import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap, QGuiApplication
from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow, QPushButton, QComboBox, QFrame, QFileDialog, QFrame
import sys

class App(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Video App")
        self.setGeometry(100, 100, 700, 700)

        # ComboBox for Mode Selector
        self.modeSelector = QComboBox(self)
        self.modeSelector.addItem('Mode 1')
        self.modeSelector.addItem('Mode 2')
        self.modeSelector.currentIndexChanged.connect(self.select_mode)
        self.modeSelector.move(20, 20)

        # Start Video Feed Button
        self.startButton = QPushButton("Detection", self)
        self.startButton.clicked.connect(self.start_video)
        self.startButton.move(20, 60)

        # Status Label
        self.statusLabel = QLabel("Status: Select the motions you want to detect or set your own motions ", self)
        # self.statusLabel.move(20, 100)
        self.statusLabel.setGeometry(20, 100, 600, 50)

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
        self.saveButton = QPushButton("Save", self)
        self.saveButton.clicked.connect(self.save_function)
        self.saveButton.move(20, 130)

        self.loadButton = QPushButton("Load", self)
        self.loadButton.clicked.connect(self.load_function)
        self.loadButton.move(100, 130)

        self.show()

    def select_mode(self):
        current_mode = self.modeSelector.currentText()
        print(f"{current_mode} selected")

    def start_video(self):
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(20)

        self.statusLabel.setText("Status: Running")
        # self.colorPane.setStyleSheet("QWidget { background-color: %s }" % 'green')

    def update_frame(self):
        ret, image = self.capture.read()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.displayImage(image)

    def displayImage(self, img):
        qformat = QImage.Format.Format_RGB888
        img = QImage(img, img.shape[1], img.shape[0], qformat)
        # img = QImage(img)
        # img = img.rgbSwapped()
        self.videoLabel.setPixmap(QPixmap.fromImage(img))

    def save_function(self):
        options = QFileDialog.Option(QFileDialog.Option.HideNameFilterDetails)
        fileName, _ = QFileDialog.getSaveFileName(self, "Save File", "~/", "All Files (*);;Text Files (*.txt)", options=options)
        if fileName:
            with open(fileName, 'w') as f:
                f.write("Some save data")

    def load_function(self):
        options = QFileDialog.Option(QFileDialog.Option.HideNameFilterDetails)
        fileName, _ = QFileDialog.getOpenFileName(self, "Load File", "~/", "All Files (*);;Text Files (*.txt)", options=options)
        if fileName:
            with open(fileName, 'r') as f:
                print("Loaded data:", f.read())

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = App()
    sys.exit(app.exec())
