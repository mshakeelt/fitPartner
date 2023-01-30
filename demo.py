import core.inference as inference
from core.inference import TRT_Infer_Body, TRT_Infer_Feet, TRT_Infer_Hands
from core.comparison import Compare
from core.signaling import Signals
from core.graphics import Graphics
from threading import Thread
from queue import Queue
from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import cv2
import os

class ConfirmClosure(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.label_width = None
        self.lable_height = None

    def closeEvent(self, event):
        sys.exit(0)

    def resizeEvent(self, a0: QtGui.QResizeEvent) -> None:
        oldsize = a0.oldSize()
        newsize = QtCore.QSize(self.geometry().width(), self.geometry().height())

class Ui_fitPartner(object):
    def setupUi(self, fitPartner):
        fitPartner.setObjectName("fitPartner")
        fitPartner.resize(1550, 809)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("test_media/window_icon.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        fitPartner.setWindowIcon(icon)
        self.centralwidget = QtWidgets.QWidget(fitPartner)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.logo_windows_groupbox = QtWidgets.QGroupBox(self.centralwidget)
        self.logo_windows_groupbox.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.logo_windows_groupbox.setStyleSheet("QGroupBox#logo_windows_groupbox {border:0;}")
        self.logo_windows_groupbox.setTitle("")
        self.logo_windows_groupbox.setFlat(False)
        self.logo_windows_groupbox.setObjectName("logo_windows_groupbox")
        self.logo_windows_layout = QtWidgets.QGridLayout(self.logo_windows_groupbox)
        self.logo_windows_layout.setObjectName("logo_windows_layout")
        self.windows_labels_groupbox = QtWidgets.QGroupBox(self.logo_windows_groupbox)
        self.windows_labels_groupbox.setStyleSheet("QGroupBox#windows_labels_groupbox {border:0;}")
        self.windows_labels_groupbox.setTitle("")
        self.windows_labels_groupbox.setFlat(False)
        self.windows_labels_groupbox.setCheckable(False)
        self.windows_labels_groupbox.setObjectName("windows_labels_groupbox")
        self.windows_labels_layout = QtWidgets.QGridLayout(self.windows_labels_groupbox)
        self.windows_labels_layout.setObjectName("windows_labels_layout")
        self.trainer_window = QtWidgets.QLabel(self.windows_labels_groupbox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.trainer_window.sizePolicy().hasHeightForWidth())
        self.trainer_window.setSizePolicy(sizePolicy)
        self.trainer_window.setFrameShape(QtWidgets.QFrame.Box)
        self.trainer_window.setLineWidth(0)
        self.trainer_window.setText("")
        self.trainer_window.setAlignment(QtCore.Qt.AlignCenter)
        self.trainer_window.setObjectName("trainer_window")
        self.windows_labels_layout.addWidget(self.trainer_window, 0, 0, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.windows_labels_layout.addItem(spacerItem, 0, 3, 1, 1)
        self.trainee_window = QtWidgets.QLabel(self.windows_labels_groupbox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.trainee_window.sizePolicy().hasHeightForWidth())
        self.trainee_window.setSizePolicy(sizePolicy)
        self.trainee_window.setFrameShape(QtWidgets.QFrame.Box)
        self.trainee_window.setLineWidth(0)
        self.trainee_window.setText("")
        self.trainee_window.setAlignment(QtCore.Qt.AlignCenter)
        self.trainee_window.setObjectName("trainee_window")
        self.windows_labels_layout.addWidget(self.trainee_window, 0, 4, 1, 1)
        self.trainer_label = QtWidgets.QLabel(self.windows_labels_groupbox)
        font = QtGui.QFont()
        font.setFamily("Calibri Light")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.trainer_label.setFont(font)
        self.trainer_label.setAlignment(QtCore.Qt.AlignCenter)
        self.trainer_label.setObjectName("trainer_label")
        self.windows_labels_layout.addWidget(self.trainer_label, 1, 0, 1, 1)
        self.trainee_label = QtWidgets.QLabel(self.windows_labels_groupbox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.trainee_label.sizePolicy().hasHeightForWidth())
        self.trainee_label.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Calibri Light")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.trainee_label.setFont(font)
        self.trainee_label.setAlignment(QtCore.Qt.AlignCenter)
        self.trainee_label.setObjectName("trainee_label")
        self.windows_labels_layout.addWidget(self.trainee_label, 1, 4, 1, 1)
        self.logo_windows_layout.addWidget(self.windows_labels_groupbox, 2, 0, 1, 3)
        self.company_logo = QtWidgets.QLabel(self.logo_windows_groupbox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.company_logo.sizePolicy().hasHeightForWidth())
        self.company_logo.setSizePolicy(sizePolicy)
        self.company_logo.setMaximumSize(QtCore.QSize(150, 50))
        self.company_logo.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.company_logo.setText("")
        self.company_logo.setPixmap(QtGui.QPixmap("test_media/Logo.png"))
        self.company_logo.setScaledContents(True)
        self.company_logo.setAlignment(QtCore.Qt.AlignCenter)
        self.company_logo.setObjectName("company_logo")
        self.logo_windows_layout.addWidget(self.company_logo, 0, 1, 1, 1)
        self.gridLayout.addWidget(self.logo_windows_groupbox, 0, 0, 1, 1)
        fitPartner.setCentralWidget(self.centralwidget)

        self.retranslateUi(fitPartner)
        QtCore.QMetaObject.connectSlotsByName(fitPartner)

        self.draw_graphics = Graphics(colors=colors, comparison_level=body_part.lower())

        file_name = "comparison_results.txt"
        completeName = os.path.join("output", file_name)
        self.output_file = open(completeName, "w+")
        self.output_file.write("Comparison started!")
        self.output_file.write("\n")

        self.draw_comparison = Compare(comparison_level=body_part, threshold=threshold)
        self.generate_signals = Signals(file_to_write=self.output_file)

        self.trt_infer_body = TRT_Infer_Body()
        if body_part.lower() == 'full':
            self.trt_infer_hands = TRT_Infer_Hands()
            self.trt_infer_feet = TRT_Infer_Feet()
        elif body_part.lower() == 'upper':
            self.trt_infer_hands = TRT_Infer_Hands()
        elif body_part.lower() == 'lower':
            self.trt_infer_feet = TRT_Infer_Feet()
        self.show = True
        self.queue_out = Queue()
        self.process_videos(master_video_path)
        self.output_file.close()
        sys.exit(0)

    def retranslateUi(self, fitPartner):
        _translate = QtCore.QCoreApplication.translate
        fitPartner.setWindowTitle(_translate("fitPartner", " "))
        self.trainer_label.setText(_translate("fitPartner", "Trainer"))
        self.trainee_label.setText(_translate("fitPartner", "Trainee"))

    def process_videos(self, master_video_path):
        master_cap = cv2.VideoCapture(master_video_path)
        follower_cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        comparison_frames_available = True
        print("Comparison started.  \n")
        frame_count = 0
        while(comparison_frames_available):
            master_ret, master_frame = master_cap.read()
            follower_ret, follower_frame = follower_cap.read()
            comparison_frames_available = master_ret and follower_ret
            if comparison_frames_available:
                if (frame_count%(frames_to_skip+1)) == 0:
                    master_frame, follower_frame = self.draw_inference(master_frame, follower_frame)
                master_frame = cv2.cvtColor(master_frame, cv2.COLOR_BGR2RGB)
                follower_frame = cv2.cvtColor(follower_frame, cv2.COLOR_BGR2RGB)
                self.Display_Image(master_frame, window='master')
                self.Display_Image(follower_frame, window='follower')
            elif not master_ret and follower_ret:
                print("Master Video is Finished. Exitting...")
            elif not follower_ret and master_ret:
                print("Follower Video is Finished. Exitting...")
            else:
                print("Both videos are finished!")
            frame_count+=1

        master_cap.release()
        follower_cap.release()

    def draw_inference(self, master_image, follower_image):
        trt_inference_results = {}
        t1 = Thread(target=self.trt_infer_body.execute_with_queue, args=(self.queue_out, 'master', master_image))
        t2 = Thread(target=self.trt_infer_body.execute_with_queue, args=(self.queue_out, 'follower', follower_image))
        t1.start()
        t2.start()
        if body_part.lower() == 'full':
            t3 = Thread(target=self.trt_infer_hands.execute_with_queue, args=(self.queue_out, 'master', master_image))
            t4 = Thread(target=self.trt_infer_hands.execute_with_queue, args=(self.queue_out, 'follower', follower_image))
            t5 = Thread(target=self.trt_infer_feet.execute_with_queue, args=(self.queue_out, 'master', master_image))
            t6 = Thread(target=self.trt_infer_feet.execute_with_queue, args=(self.queue_out, 'follower', follower_image))
            t3.start()
            t4.start()
            t5.start()
            t6.start()
        elif body_part.lower() == 'upper':
            trt_inference_results['master_feet'] = None
            trt_inference_results['follower_feet'] = None
            t3 = Thread(target=self.trt_infer_hands.execute_with_queue, args=(self.queue_out, 'master', master_image))
            t4 = Thread(target=self.trt_infer_hands.execute_with_queue, args=(self.queue_out, 'follower', follower_image))
            t3.start()
            t4.start()
        elif body_part.lower() == 'lower':
            trt_inference_results['master_hands'] = None
            trt_inference_results['follower_hands'] = None
            t3 = Thread(target=self.trt_infer_feet.execute_with_queue, args=(self.queue_out, 'master', master_image))
            t4 = Thread(target=self.trt_infer_feet.execute_with_queue, args=(self.queue_out, 'follower', follower_image))
            t3.start()
            t4.start()
        t1.join()
        t2.join()
        if body_part.lower() == 'full':
            t3.join()
            t4.join()
            t5.join()
            t6.join()
        elif body_part.lower() == 'upper' or body_part.lower() == 'lower':
            t3.join()
            t4.join()

        while not self.queue_out.empty():
            trt_inference_results.update(self.queue_out.get())

        t7 = Thread(target=inference.join_trt_keypoints_with_queue, args=(self.queue_out, 'master', trt_inference_results['master_body'], trt_inference_results['master_hands'], trt_inference_results['master_feet']))
        t8 = Thread(target=inference.join_trt_keypoints_with_queue, args=(self.queue_out, 'follower', trt_inference_results['follower_body'], trt_inference_results['follower_hands'], trt_inference_results['follower_feet']))
        t7.start()
        t8.start()
        t7.join()
        t8.join()

        trt_joining_results = {}
        while not self.queue_out.empty():
            trt_joining_results.update(self.queue_out.get())

        trt_master_keypoints = trt_joining_results['master']
        trt_follower_keypoints = trt_joining_results['follower']
        trt_keypoint_comparison_results, trt_joints_comparison_results, trt_angle_errors = self.draw_comparison.compare(trt_master_keypoints, trt_follower_keypoints)
        if self.show:
            fitPartner.show()
            self.show = False
        t9 = Thread(target=self.generate_signals, args=(trt_joints_comparison_results, trt_angle_errors))
        t10 = Thread(target=self.draw_graphics, args=(master_image, trt_master_keypoints, 'master'))
        t11 = Thread(target=self.draw_graphics, args=(follower_image, trt_follower_keypoints, 'follower', trt_keypoint_comparison_results, trt_joints_comparison_results))
        t9.start()
        t10.start()
        t11.start()
        t9.join()
        t10.join()
        t11.join()
        return master_image, follower_image

    def Display_Image(self, img, window):
        if (img.shape[2]) == 4:
            qformat = QtGui.QImage.Format_RGBA8888
        else:
            qformat = QtGui.QImage.Format_RGB888
        h, w, ch = img.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(img.data, w, h, bytes_per_line, qformat)
        disply_width = self.trainer_window.width() #int(640)
        display_height = self.trainer_window.height() #int(480)
        p = convert_to_Qt_format.scaled(disply_width, display_height)#, QtCore.Qt.KeepAspectRatio)
        if window=='master':
            self.trainer_window.setPixmap(QtGui.QPixmap.fromImage(p))
            QtWidgets.QApplication.processEvents()
            self.trainer_window.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        elif window=='follower':
            self.trainee_window.setPixmap(QtGui.QPixmap.fromImage(p))
            QtWidgets.QApplication.processEvents()
            self.trainee_window.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

def set_colors():
    colors = {}
    colors['master_skeleton_color'] = (255, 255, 255)
    colors['master_keypoint_color'] = (128, 128, 128)
    colors['follower_skeleton_match_color'] = (74, 225, 180)
    colors['follower_keypoint_match_color'] = (74, 225, 180)
    colors['follower_skeleton_mismatch_color'] = (80, 97, 230)
    colors['follower_keypoint_mismatch_color'] = (80, 97, 230)
    return colors

if __name__ == '__main__':
    master_video_path = os.path.join("test_media", "exercise.mp4")
    body_part = 'full' # full, upper, or lower
    threshold = 15 #angle difference in degrees for comparison
    frames_to_skip = 0 # Number of frames to skip comparison for fast processing
    colors = set_colors()
    app = QtWidgets.QApplication(sys.argv)
    fitPartner = ConfirmClosure()
    ui = Ui_fitPartner()
    ui.setupUi(fitPartner)
    sys.exit(app.exec_())