# -*- coding: utf-8 -*-

# TrackingBot - A software for video-based animal behavioral tracking and analysis
# Developer: Yutao Bai <yutaobai@hotmail.com>
# Version: 1.02
# https://www.neurotoxlab.com

# Copyright (C) 2022 Yutao Bai
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QStyle, QSplashScreen, QWhatsThis, QProgressBar, \
    QDialog,QVBoxLayout,QLabel
from PyQt5.QtGui import QImage, QPixmap, QPixmapCache
from PyQt5.QtCore import pyqtSignal, Qt, QThread, QObject, QMutex, QMutexLocker, QRect
from qtwidgets import Toggle

import os
import subprocess
import cv2
import time
import numpy as np
from collections import namedtuple
from datetime import datetime, timedelta
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.cm as cm
import serial
import serial.tools.list_ports
import gui
from video_player import VideoThread
from threshold import ThreshVidThread, ThreshCamThread
from tracking import TrackingThread, TrackingCamThread
import graphic_interactive as graphic
from datalog import TrackingTimeStamp, DataLogThread, DataExportThread,CamDataExportThread,\
    TraceExportThread,GraphExportThread, VideoExportThread
from hardware_wizard import Ui_HardwireWizardWindow



class MainWindow(QtWidgets.QMainWindow, gui.Ui_MainWindow):
    STATUS_INIT = 0
    STATUS_PLAYING = 1
    STATUS_PAUSE = 2

    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.setWindowIcon(QtGui.QIcon('icon/icon.png'))

        # # video init
        self.video_file = None
        self.background_frame = None # the first frame of video
        self.last_frame = None # the last frame of video for trace drawing
        self.video_prop = None
        self.camera_prop = None
        self.scale_factor = None
        self.status = self.STATUS_INIT  # 0: init 1:playing 2: pause

        self.playCapture = cv2.VideoCapture()
        self.verticalLayoutWidget.lower()

        #################################################################################
        # timer for video player thread
        #################################################################################
        self.videoThread = VideoThread()
        self.videoThread.timeSignal.signal[str].connect(self.display_video)

        # timer for threshold video player on load tab
        self.threshThread = ThreshVidThread()
        self.threshThread.timeSignal.thresh_signal.connect(self.display_threshold_video)
        self.threshThread.timeSignal.updateSliderPos.connect(self.update_thre_slider)
        self.threshThread.timeSignal.thresh_reset.connect(self.reset_video)
        self.threshThread.timeSignal.detect_cnt.connect(self.update_detect_cnt)

        # timer for tracking video
        self.trackingThread = TrackingThread()
        self.trackingThread.timeSignal.tracking_signal.connect(self.display_tracking_video)
        self.trackingThread.timeSignal.updateSliderPos.connect(self.update_track_slider)

        self.trackingThread.timeSignal.track_results.connect(self.update_track_results)
        self.trackingThread.timeSignal.track_reset.connect(self.reset_video)
        self.trackingThread.timeSignal.track_reset_alarm.connect(self.complete_tracking)
        self.trackingThread.timeSignal.exceed_index_alarm.connect(self.exceed_index_alarm)

        self.start_tic = 0
        self.stop_toc = 0

        self.threshCamThread = ThreshCamThread()
        self.threshCamThread.timeSignal.cam_thresh_signal.connect(self.display_threshold_cam)
        self.threshCamThread.timeSignal.update_clock.connect(self.update_clock)
        self.threshCamThread.timeSignal.update_elapse.connect(self.update_elapse)
        self.threshCamThread.timeSignal.cam_reload.connect(self.reload_camera)
        self.threshCamThread.timeSignal.cam_detect_cnt.connect(self.update_cam_detect_cnt)

        self.trackingCamThread = TrackingCamThread()
        self.trackingCamThread.timeSignal.cam_tracking_signal.connect(self.display_tracking_cam)
        self.trackingCamThread.timeSignal.update_clock.connect(self.update_clock)
        self.trackingCamThread.timeSignal.update_elapse.connect(self.update_elapse)
        self.trackingCamThread.timeSignal.cam_track_results.connect(self.activate_cam_tracking_log)
        self.trackingCamThread.timeSignal.cam_track_results.connect(self.activate_controller_log)
        self.trackingCamThread.timeSignal.cam_reload.connect(self.reload_camera)
        self.trackingCamThread.timeSignal.exceed_index_alarm.connect(self.cam_exceed_index_alarm)

        self.videoExportThread = VideoExportThread()

        self.hardwareWizard = HardwareWizard()
        self.controllerThread = ControllerThread()

        self.dataLogThread = DataLogThread()
        self.dataExportThread = DataExportThread()
        self.trackingTimeStamp = TrackingTimeStamp()
        self.dataProcessDialog = DataProcessDialog()
        self.dataProcessDialog.timesignal.data_export_finish.connect(self.export_data_success)
        self.camDataProcessDialog = CamDataProcessDialog()
        self.camDataProcessDialog.timesignal.cam_data_export_finish.connect(self.export_cam_data_success)
        # signal from datalog thread received by self.dataProgressDialog, go to DataProgressDialog()
        # for detail

        self.traceProcessDialog = TraceProcessDialog()
        self.traceProcessDialog.timesignal.trace_map.connect(self.display_trace)
        self.traceProcessDialog.timesignal.raw_trace_map.connect(self.raw_trace)

        self.graphProcessDialog = GraphProcessDialog()
        self.graphProcessDialog.timesignal.heat_map.connect(self.display_heatmap)

        self.dataframe = None

        self.reset_video()

        self.tabWidget.setTabEnabled(1, False)  # Load video
        self.tabWidget.setTabEnabled(2, False)  # Calibration and ROI
        self.tabWidget.setTabEnabled(3, False)  # Thresholding
        self.tabWidget.setTabEnabled(4, False)  # Tracking
        self.tabWidget.setTabEnabled(5, False)  # Real-time tracking

        # sudo code for debug
        # self.exportDataButton.setEnabled(True)

        ##############################################################
        # signals and widgets for the tab 0
        # need one button for back to main menu
        self.localModeButton.clicked.connect(self.enable_local_mode)
        self.liveModeButton.clicked.connect(self.enable_live_mode)
        self.backToMenuButton.clicked.connect(self.select_main_menu)

        ###############################################################################
        # signals and widgets block for the load video tab
        ###############################################################################
        self.loadVidButton.clicked.connect(self.select_video_file)
        self.loadNewVidButton.clicked.connect(self.select_new_file)

        self.playButton.clicked.connect(self.video_play_control)
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))

        self.stopButton.clicked.connect(self.stop_video)
        self.stopButton.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))

        self.caliTabLinkButton.clicked.connect(self.enable_calibration)
        self.backToLoadButton.clicked.connect(self.select_vid_tab)

        # slider for video player on load tab
        self.vidProgressBar.sliderPressed.connect(self.pause_from_slider)
        self.vidProgressBar.valueChanged.connect(self.update_position)
        self.vidProgressBar.sliderReleased.connect(self.resume_from_slider)

        ##########################################################################
        # signals and widgets for calibration section
        ###########################################################################

        # add a canvas for drawing
        self.scaleCanvas = graphic.Calibration(self.caliTab)
        self.scaleCanvas.scene.inputDialog.scale.get_scale[str].connect(self.convert_scale)
        self.scaleCanvas.scene.inputDialog.scale.reset_scale[str].connect(self.reset_scale)
        # init pixel unit convert ratio
        self.pixel_per_metric = 1
        self.drawScaleButton.clicked.connect(self.draw_scale)
        self.resetScaleButton.clicked.connect(self.reset_scale)
        self.applyScaleButton.clicked.connect(self.apply_scale)
        self.threTabLinkButton.clicked.connect(self.roi_validation)

        ##########################################################################
        # signals and widgets for define ROI section
        ###########################################################################
        self.apply_roi_flag = False
        self.apply_mask_flag = False

        # add a canvas for drawing
        self.roiCanvas = graphic.DefineROI(self.caliTab)
        self.maskCanvas = graphic.DefineMask(self.caliTab)

        # define roi(include)
        self.editROIButton.clicked.connect(self.edit_roi)
        self.rectROIButton.clicked.connect(self.set_rect_roi)
        self.circROIButton.clicked.connect(self.set_circ_roi)
        self.applyROIButton.clicked.connect(self.apply_roi)
        self.resetROIButton.clicked.connect(self.reset_roi)

        # define mask(exclude)
        self.editMaskButton.clicked.connect(self.edit_mask)
        self.rectMaskButton.clicked.connect(self.set_rect_mask)
        self.circMaskButton.clicked.connect(self.set_circ_mask)
        self.applyMaskButton.clicked.connect(self.apply_mask)
        self.resetMaskButton.clicked.connect(self.reset_mask)

        # enable WhatsThis mode
        self.calibrationHelpLabel.enterEvent = self.enable_calibration_help
        self.calibrationHelpLabel.leaveEvent = self.disable_calibration_help
        self.roiHelpLabel.enterEvent = self.enable_roi_help
        self.roiHelpLabel.leaveEvent = self.disable_roi_help
        self.maskHelpLabel.enterEvent = self.enable_mask_help
        self.maskHelpLabel.leaveEvent = self.disable_mask_help

        ############################################################################
        # signals and widgets for threshold section
        ############################################################################

        self.object_num = 1
        self.block_size = ''
        self.offset = ''
        self.min_contour = ''
        self.max_contour = ''
        self.invert_contrast_state = False

        # add scene to display roi and mask as a visual indicator
        self.displayCanvas = graphic.DisplayROI(self.threTab)
        self.backToCaliButton.clicked.connect(self.select_cali_tab)

        self.threPlayButton.clicked.connect(self.thresh_vid_control)
        self.threPlayButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.threStopButton.clicked.connect(self.stop_thresh_vid)
        self.threStopButton.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))

        self.applyObjNumButton.clicked.connect(self.apply_object_num)

        self.blockSizeSlider.sliderPressed.connect(self.pause_thresh_vid)
        self.blockSizeSlider.valueChanged.connect(self.set_blocksize_slider)
        self.blockSizeSlider.sliderReleased.connect(self.resume_thresh_vid)
        self.blockSizeSpin.valueChanged.connect(self.set_blocksize_spin)

        self.offsetSlider.sliderPressed.connect(self.pause_thresh_vid)
        self.offsetSlider.valueChanged.connect(self.set_offset_slider)
        self.offsetSlider.sliderReleased.connect(self.resume_thresh_vid)
        self.offsetSpin.valueChanged.connect(self.set_offset_spin)
        #
        self.cntMinSlider.sliderPressed.connect(self.pause_thresh_vid)
        self.cntMinSlider.valueChanged.connect(self.set_min_cnt_slider)
        self.cntMinSlider.sliderReleased.connect(self.resume_thresh_vid)
        self.cntMinSpin.valueChanged.connect(self.set_min_cnt_spin)
        #
        self.cntMaxSlider.sliderPressed.connect(self.pause_thresh_vid)
        self.cntMaxSlider.valueChanged.connect(self.set_max_cnt_slider)
        self.cntMaxSlider.sliderReleased.connect(self.resume_thresh_vid)
        self.cntMaxSpin.valueChanged.connect(self.set_max_cnt_spin)

        # slider for video player on load tab
        self.threProgressBar.sliderPressed.connect(self.pause_thresh_vid)
        self.threProgressBar.valueChanged.connect(self.update_thre_position)
        self.threProgressBar.sliderReleased.connect(self.resume_thresh_slider)

        self.previewBoxLabel.lower()
        self.previewToggle = Toggle(self.threTab)
        self.previewToggle.setEnabled(False)
        self.previewToggle.setGeometry(QtCore.QRect(1150, 390, 60, 35))
        self.previewToggle.stateChanged.connect(self.enable_thre_preview)

        self.invertContrastToggle = Toggle(self.threTab)
        self.invertContrastToggle.setEnabled(False)
        self.invertContrastToggle.stateChanged.connect(self.invert_contrast)
        self.invertContrastToggle.setGeometry(QtCore.QRect(1050, 210, 220, 35))

        self.applyThreButton.clicked.connect(self.apply_thre_setting)
        self.resetThreButton.clicked.connect(self.reset_thre_setting)

        # enable WhatsThis mode
        self.blocksizeHelpLabel.enterEvent = self.enable_blocksize_help
        self.blocksizeHelpLabel.leaveEvent = self.disable_blocksize_help
        self.offsetHelpLabel.enterEvent = self.enable_offset_help
        self.offsetHelpLabel.leaveEvent = self.disable_offset_help
        self.objectsizeHelpLabel.enterEvent = self.enable_size_help
        self.objectsizeHelpLabel.leaveEvent = self.disable_size_help

        ############################################################################
        # signals and widgets for tracking section
        ############################################################################
        self.track_fin = False
        self.export_data_fin = False
        self.export_graph_fin = False

        self.trackTabLinkButton.clicked.connect(self.enable_tracking)
        self.leaveTrackButton.clicked.connect(self.leave_track_tab)

        self.trackStartButton.clicked.connect(self.tracking_vid_control)
        self.trackStartButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))

        self.trackProgressBar.valueChanged.connect(self.update_track_vid_position)

        ############################################################################
        # signals and widgets for data export section
        ############################################################################

        self.exportDataButton.clicked.connect(self.export_data)
        self.exportTraceButton.clicked.connect(self.export_trace)
        self.exportHeatmapButton.clicked.connect(self.export_heatmap)

        self.traceToggle = Toggle(self.trackingTab)
        self.traceToggle.setEnabled(True)
        self.traceToggle.setGeometry(QtCore.QRect(1210, 95, 85, 35))
        self.traceToggle.stateChanged.connect(self.generate_trace)
        # enable WhatsThis mode
        self.traceHelpLabel.enterEvent = self.enable_trace_help
        self.traceHelpLabel.leaveEvent = self.disable_trace_help

        self.trace_map = None

        self.heatmapToggle = Toggle(self.trackingTab)
        self.heatmapToggle.setEnabled(False)
        self.heatmapToggle.setGeometry(QtCore.QRect(1210, 195, 85, 35))
        self.heatmapToggle.stateChanged.connect(self.generate_heatmap)
        # enable WhatsThis mode
        self.heatmapHelpLabel.enterEvent = self.enable_heatmap_help
        self.heatmapHelpLabel.leaveEvent = self.disable_heatmap_help

        self.figure = Figure()
        # set background color
        self.figure.set_facecolor("black")
        # the Canvas Widget that displays the `figure`
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setFixedSize(1024, 576)
        self.heatmapLayout.addWidget(self.canvas)

        self.heat_map = None

        #####################################################################################
        # signals and widgets for live tracking section
        #####################################################################################

        self.openCamButton.clicked.connect(self.read_camera)
        self.closeCamButton.clicked.connect(self.close_camera)
        self.leaveCamTracking.clicked.connect(self.select_main_menu)



        self.cam_object_num = 1
        self.cam_block_size = ''
        self.cam_offset = ''
        self.cam_min_contour = ''
        self.cam_max_contour = ''
        self.cam_invert_contrast = False

        self.applyLiveObjNum.clicked.connect(self.apply_cam_object_num)

        self.camPreviewBoxLabel.lower()
        self.camPreviewToggle = Toggle(self.liveTrackingTab)
        self.camPreviewToggle.setGeometry(QRect(1150, 390, 60, 35))
        self.camPreviewToggle.setEnabled(False)
        self.camPreviewToggle.stateChanged.connect(self.enable_cam_thre_preview)

        self.camInvertContrastToggle = Toggle(self.liveTrackingTab)
        self.camInvertContrastToggle.setGeometry(QRect(1050, 210, 220, 35))
        self.camInvertContrastToggle.setEnabled(False)
        self.camInvertContrastToggle.stateChanged.connect(self.invert_cam_contrast)

        self.camBlockSizeSlider.valueChanged.connect(self.set_cam_blocksize_slider)
        self.camBlockSizeSpin.valueChanged.connect(self.set_cam_blocksize_spin)
        #
        self.camOffsetSlider.valueChanged.connect(self.set_cam_offset_slider)
        self.camOffsetSpin.valueChanged.connect(self.set_cam_offset_spin)
        #
        self.camCntMinSlider.valueChanged.connect(self.set_cam_min_cnt_slider)
        self.camCntMinSpin.valueChanged.connect(self.set_cam_min_cnt_spin)
        #
        self.camCntMaxSlider.valueChanged.connect(self.set_cam_max_cnt_slider)
        self.camCntMaxSpin.valueChanged.connect(self.set_cam_max_cnt_spin)

        self.applyCamThreButton.clicked.connect(self.apply_thre_cam_setting)
        self.resetCamThreButton.clicked.connect(self.reset_thre_cam_setting)

        self.camTrackingStart.clicked.connect(self.cam_tracking_control)
        self.camTrackingStart.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))

        self.exportCamData.clicked.connect(self.export_cam_data)

        self.camblocksizeHelpLabel.enterEvent = self.enable_cam_blocksize_help
        self.camblocksizeHelpLabel.leaveEvent = self.disable_cam_blocksize_help
        self.camoffsetHelpLabel.enterEvent = self.enable_cam_offset_help
        self.camoffsetHelpLabel.leaveEvent = self.disable_cam_offset_help
        self.camobjectsizeHelpLabel.enterEvent = self.enable_cam_size_help
        self.camobjectsizeHelpLabel.leaveEvent = self.disable_cam_size_help

        ################################################################################
        # signals and widgets for the hardware control
        #################################################################################

        self.camROICanvas = graphic.DefineROI(self.liveTrackingTab)

        self.hardwareWizardToggle = Toggle(self.liveTrackingTab)
        self.hardwareWizardToggle.setEnabled(True)
        self.hardwareWizardToggle.setGeometry(QtCore.QRect(165, 655, 85, 35))
        self.hardwareWizardToggle.stateChanged.connect(self.open_hardware_wizard)

        self.hardwareWizard.editCamROIButton.clicked.connect(self.edit_cam_roi)
        self.hardwareWizard.lineCamROIButton.clicked.connect(self.set_cam_line_roi)
        self.hardwareWizard.rectCamROIButton.clicked.connect(self.set_cam_rect_roi)
        self.hardwareWizard.circCamROIButton.clicked.connect(self.set_cam_circ_roi)
        self.hardwareWizard.applyCamROIButton.clicked.connect(self.apply_cam_roi)
        self.hardwareWizard.resetCamROIButton.clicked.connect(self.reset_cam_roi)


        ###################################################################################
        self.actionAbout.triggered.connect(self.about_info)

    def about_info(self):

        self.about_msg = QMessageBox()
        self.about_msg.setWindowTitle('About')
        self.about_msg.setText('<b>TrackingBot</b>')
        self.about_msg.setInformativeText('An animal behavioural tracking software.' )
        self.about_msg.exec()

    def select_main_menu(self):
        '''
        activate main tab
        '''
        self.tabWidget.setTabEnabled(0, True)
        self.tabWidget.setTabEnabled(1, False)
        self.tabWidget.setTabEnabled(5, False)
        self.tabWidget.setCurrentIndex(0)
        self.reset_video()
        self.close_camera()
        self.reset_cam_track_results()
        self.caliBoxLabel.setEnabled(False)
        self.scaleCanvas.setEnabled(False)
        self.resetScaleButton.setEnabled(False)
        self.applyScaleButton.setEnabled(False)
        self.scaleCanvas.lower()
        self.editROIButton.setEnabled(False)
        self.editMaskButton.setEnabled(False)

    def enable_local_mode(self):
        '''
        activate select video file tab
        '''
        self.tabWidget.setTabEnabled(0, False)
        self.tabWidget.setTabEnabled(1, True)
        self.tabWidget.setCurrentIndex(1)
        self.backToMenuButton.setEnabled(True)

    def enable_live_mode(self):
        '''
        activate load camera source tab then read camera properties
        '''
        self.tabWidget.setTabEnabled(0, False)
        self.tabWidget.setTabEnabled(5, True)
        self.tabWidget.setCurrentIndex(5)
        self.leaveCamTracking.setEnabled(True)

        try:
            self.camera_prop = self.read_cam_prop(cv2.VideoCapture(0, cv2.CAP_DSHOW))
            # print(self.camera_prop)
            cv2.VideoCapture(0, cv2.CAP_DSHOW).release()
        except Exception as e:
            error = str(e)
            self.error_msg = QMessageBox()
            self.error_msg.setWindowTitle('Error')
            self.error_msg.setText('Failed to detect camera.')
            self.error_msg.setInformativeText('Please make sure camera is connected with computer.\n')
            self.error_msg.setIcon(QMessageBox.Warning)
            self.error_msg.setDetailedText(error)
            self.error_msg.exec()

    ################################################################################################
    # Functions for load and preview video file section
    #################################################################################################

    def select_video_file(self):
        '''
        select file and store file path as class instance
        :return:
        '''
        try:
            # set default directory for load files and set file type that only shown
            selected_file = QFileDialog.getOpenFileName(directory='C:/Users/Public/Desktop',
                                                        filter='Videos(*.mp4 *.avi *.mov *.wmv *.mkv *.flv)')
            # if no file selected
            if selected_file[0] == '':
                # self.caliTabLinkButton.setEnabled(False)
                return
            else:
                # video file is global
                self.video_file = selected_file
                # print(self.video_file[0])
                # enable video control buttons
                self.playButton.setEnabled(True)
                self.stopButton.setEnabled(True)
                self.loadNewVidButton.setEnabled(True)
                self.caliTabLinkButton.setEnabled(True)
                # display image on top of other widgets
                # can use either way
                # self.caliBoxCanvasLabel.raise_()
                self.loadVidButton.hide()

                # auto read and display file property
                self.read_video_file(self.video_file[0])

        except Exception as e:
            error = str(e)
            self.error_msg = QMessageBox()
            self.error_msg.setWindowTitle('Error')
            self.error_msg.setText('An error happened when trying to load video file.')
            self.error_msg.setInformativeText('Please ensure the video file is not corrupted.')
            self.error_msg.setIcon(QMessageBox.Warning)
            self.error_msg.setDetailedText('selectVideoFile()Failed. \n'+ error)
            self.error_msg.exec()

    def read_video_file(self, file_path):
        '''
        read property of selected video file and display
        '''

        try:
            video_cap = cv2.VideoCapture(file_path)
            self.read_video_prop(video_cap)
            self.video_name = os.path.split(file_path)

            self.set_vid_progressbar(self.video_prop)
            self.display_video_prop()
            self.set_video_fps()

            # display 1st frame of video in window as preview
            set_background_frame = 1
            video_cap.set(cv2.CAP_PROP_POS_FRAMES, set_background_frame)
            ret, background_frame = video_cap.read()
            # class variable
            self.background_frame = background_frame
            # scale raw frame to fit display window
            scaled_frame = self.scale_frame(background_frame)
            # convert to QPixmap
            display_frame = self.convert_frame(scaled_frame)
            self.set_background_frame(display_frame)

            # get last frame
            set_trace_frame = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)-1
            video_cap.set(cv2.CAP_PROP_POS_FRAMES, set_trace_frame)
            ret, last_frame = video_cap.read()
            self.last_frame = last_frame

            video_cap.release()

        except Exception as e:
            error = str(e)
            self.error_msg = QMessageBox()
            self.error_msg.setWindowTitle('Error')
            self.error_msg.setText('Failed to read video file.')
            self.error_msg.setIcon(QMessageBox.Warning)
            self.error_msg.setDetailedText(error)
            self.error_msg.exec()

    def read_video_prop(self, file_path):
        """
        read parameters of loaded video file and return all values
        Parameters
        ----------
        video_file

        Returns
        -------
        get_video_prop: a named tuple of video parameter
        """
        # calculate total number of seconds of file
        total_sec = file_path.get(cv2.CAP_PROP_FRAME_COUNT) / file_path.get(cv2.CAP_PROP_FPS)
        # convert total seconds to hh:mm:ss format
        video_duration = str(timedelta(seconds=total_sec))
        video_prop = namedtuple('video_prop', ['width', 'height', 'fps', 'length', 'elapse', 'duration'])
        get_video_prop = video_prop(file_path.get(cv2.CAP_PROP_FRAME_WIDTH),
                                    file_path.get(cv2.CAP_PROP_FRAME_HEIGHT),
                                    file_path.get(cv2.CAP_PROP_FPS),
                                    file_path.get(cv2.CAP_PROP_FRAME_COUNT),
                                    file_path.get(cv2.CAP_PROP_POS_MSEC),
                                    video_duration)
        self.video_prop = get_video_prop
        self.threshThread.video_prop = get_video_prop
        self.trackingThread.video_prop = get_video_prop

    def display_video_prop(self):

        self.vidNameText.setText(f'{str(self.video_name[1])}')
        self.vidDurText.setText(f'{str(self.video_prop.duration).split(".")[0]}')
        self.vidFpsText.setText(str(round(self.video_prop.fps, 2)))
        self.vidResText.setText(f'{str(int(self.video_prop.width))} X {str(int(self.video_prop.height))}')

    def set_video_fps(self):
        # set video fps for each thread
        self.video_fps = int(self.video_prop.fps)
        self.videoThread.set_fps(self.video_prop.fps)
        self.threshThread.set_fps(self.video_prop.fps)
        self.trackingThread.set_fps(self.video_prop.fps)

    def scale_frame(self, frame):
        '''
        scale video frame to display window size
        :param frame:
        :return:
        '''
        global scale_frame
        if self.video_prop.width > 1024:  # shrink the raw frame to fit
            self._interpolation_flag = cv2.INTER_AREA
        elif self.video_prop.width < 1024:  # enlarge the raw frame to fit
            self._interpolation_flag = cv2.INTER_LINEAR

        # 4:3
        if self.video_prop.height/self.video_prop.width == 0.75:
            scale_frame = cv2.resize(frame, (768, 576), interpolation=self._interpolation_flag)
            self.scale_factor = int(self.video_prop.width) / 768
        # 16:9
        else:
            scale_frame = cv2.resize(frame, (1024, 576), interpolation=self._interpolation_flag)
            self.scale_factor = int(self.video_prop.width) / 1024
        return scale_frame

    def convert_frame(self, frame):
        '''
        convert image to QImage then to QPixmap
        :param frame:
        :return:
        '''

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_cvt = QImage(frame_rgb, frame_rgb.shape[1], frame_rgb.shape[0], frame_rgb.strides[0],
                           QImage.Format_RGB888)
        frame_display = QPixmap.fromImage(frame_cvt)
        return frame_display

    def set_background_frame(self, frame):
        '''
        set a initial background frame for each tab
        :param frame: Qpixmap
        :return:
        '''

        self.VBoxLabel.setPixmap(frame)
        self.caliBoxLabel.setPixmap(frame)
        self.threBoxLabel.setPixmap(frame)
        self.trackingBoxLabel.setPixmap(frame)

    def select_new_file(self):
        '''
        release current file and select new file
        '''
        # all settings will be reset when call below function
        self.select_video_file()

    def video_play_control(self):
        '''
        Play/pause/resume video
        '''
        if self.video_file[0] == '' or self.video_file[0] is None:
            self.error_msg = QMessageBox()
            self.error_msg.setWindowTitle('TrackingBot')
            self.error_msg.setText('No video file is selected.')
            self.error_msg.setInformativeText('Please select a video file to start')
            self.error_msg.setIcon(QMessageBox.Warning)
            self.error_msg.exec()
            return
        # play
        if self.status is MainWindow.STATUS_INIT:
            try:
                self.playCapture.open(self.video_file[0])
                self.videoThread.start()
                self.status = MainWindow.STATUS_PLAYING
                self.set_pause_icon()
            except Exception as e:
                error = str(e)
                self.error_msg = QMessageBox()
                self.error_msg.setWindowTitle('Error')
                self.error_msg.setText('An error happened when trying to play video file.')
                self.error_msg.setInformativeText('video_play_control() does not execute correctly.')
                self.error_msg.setIcon(QMessageBox.Warning)
                self.error_msg.setDetailedText(error)
                self.error_msg.exec()
        # pause
        elif self.status is MainWindow.STATUS_PLAYING:
            try:
                self.videoThread.stop()
                self.status = MainWindow.STATUS_PAUSE
                self.set_play_icon()
            except Exception as e:
                error = str(e)
                self.error_msg = QMessageBox()
                self.error_msg.setWindowTitle('Error')
                self.error_msg.setText('An error happened when trying to pause video file.')
                self.error_msg.setInformativeText('video_play_control() does not execute correctly.')
                self.error_msg.setIcon(QMessageBox.Warning)
                self.error_msg.setDetailedText(error)
                self.error_msg.exec()
        # resume
        elif self.status is MainWindow.STATUS_PAUSE:
            try:
                self.videoThread.start()
                self.status = MainWindow.STATUS_PLAYING
                self.set_pause_icon()
            except Exception as e:
                error = str(e)
                self.error_msg = QMessageBox()
                self.error_msg.setWindowTitle('Error')
                self.error_msg.setText('An error happened when trying to resume playing.')
                self.error_msg.setInformativeText('video_play_control() does not execute correctly.')
                self.error_msg.setIcon(QMessageBox.Warning)
                self.error_msg.setDetailedText(error)
                self.error_msg.exec()

        '''
        alternative
        '''
        # self.status = (MainWindow.STATUS_PLAYING,
        #                MainWindow.STATUS_PAUSE,
        #                MainWindow.STATUS_PLAYING)[self.status]

    def display_video(self):
        '''
        managed by the video thread
        '''

        self.vidProgressBar.setEnabled(True)

        # click play button will execute playVideo() and open the file
        if self.playCapture.isOpened():

            ret, frame = self.playCapture.read()
            if ret:
                # convert total seconds to timedelta format, not total frames to timedelta
                play_elapse = self.playCapture.get(cv2.CAP_PROP_POS_FRAMES) / self.playCapture.get(cv2.CAP_PROP_FPS)
                # update slider position and label
                self.vidProgressBar.setSliderPosition(int(play_elapse))
                self.vidPosLabel.setText(f"{str(timedelta(seconds=play_elapse)).split('.')[0]}")

                # # if frame.ndim == 3
                frame_scaled = self.scale_frame(frame)
                frame_display = self.convert_frame(frame_scaled)
                self.VBoxLabel.setPixmap(frame_display)

            elif not ret:
                # video finished
                self.reset_video()
                self.playButton.setEnabled(False)
                return
            else:
                self.error_msg = QMessageBox()
                self.error_msg.setWindowTitle('TrackingBot')
                self.error_msg.setText('Failed to read next video frame.')
                self.error_msg.setInformativeText('displayVideo() failed to execute \n'
                                                  'cv2.VideoCapture.read() does not return any frame.')
                self.error_msg.setIcon(QMessageBox.Warning)
                self.error_msg.exec()
        else:
            self.error_msg = QMessageBox()
            self.error_msg.setWindowTitle('Error')
            self.error_msg.setText('Failed to read video file.')
            self.error_msg.setInformativeText('cv2.VideoCapture.isOpen() return false.\n'
                                              'Click "OK" to reset video.')
            self.error_msg.setIcon(QMessageBox.Warning)
            self.error_msg.exec()
            self.reset_video()

    def stop_video(self):

        is_stopped = self.videoThread.is_stopped()
        self.playButton.setEnabled(True)
        # reset when video is paused
        if is_stopped:
            self.playCapture.release()
            self.read_video_file(self.video_file[0])
            self.status = MainWindow.STATUS_INIT
        # reset when video still playing
        elif not is_stopped:
            self.videoThread.stop()
            self.playCapture.release()
            self.read_video_file(self.video_file[0])
            self.status = MainWindow.STATUS_INIT
            self.set_play_icon()

    def reset_video(self):

        self.videoThread.stop()
        self.threshThread.stop()
        self.trackingThread.stop()
        self.dataLogThread.stop()
        self.playCapture.release()
        self.threshThread.playCapture.release()
        self.trackingThread.playCapture.release()
        self.status = MainWindow.STATUS_INIT
        self.set_play_icon()
        self.trackStartButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.max_size.setText('-')
        self.min_size.setText('-')

    def set_vid_progressbar(self, vid_prop):

        self.vidPosLabel.setText('0:00:00')
        self.vidLenLabel.setText(f'{str(vid_prop.duration).split(".")[0]}')
        # total SECONDS ( total frames/fps) use numeric, not timedelta format for range
        self.vidProgressBar.setRange(0, int(vid_prop.length / vid_prop.fps))
        self.vidProgressBar.setValue(0)

        self.vidProgressBar.setSingleStep(int(vid_prop.fps) * 5)  # 5 sec
        self.vidProgressBar.setPageStep(int(vid_prop.fps) * 60)  # 60 sec

        self.threPosLabel.setText('0:00:00')
        self.threLenLabel.setText(f'{str(vid_prop.duration).split(".")[0]}')
        # total SECONDS, use numeric, not timedelta format for range
        self.threProgressBar.setRange(0, int(vid_prop.length / vid_prop.fps))
        self.threProgressBar.setValue(0)

        self.threProgressBar.setSingleStep(int(vid_prop.fps) * 5)  # 5 sec
        self.threProgressBar.setPageStep(int(vid_prop.fps) * 60)  # 60 sec

        self.trackPosLabel.setText('0:00:00')
        self.trackLenLabel.setText(f'{str(vid_prop.duration).split(".")[0]}')
        # # use numeric, not timedelta format for range
        self.trackProgressBar.setRange(0, int(vid_prop.length / vid_prop.fps))
        self.trackProgressBar.setValue(0)
        #
        self.trackProgressBar.setSingleStep(int(vid_prop.fps) * 5)  # 5 sec
        self.trackProgressBar.setPageStep(int(vid_prop.fps) * 60)  # 60 sec

    def update_position(self):
        '''
        when drag slider to new position, update it
        '''

        play_elapse = self.vidProgressBar.value()
        self.vidPosLabel.setText(f"{str(timedelta(seconds=play_elapse)).split('.')[0]}")

    def pause_from_slider(self):

        self.videoThread.stop()
        self.status = MainWindow.STATUS_PAUSE
        self.set_play_icon()

    def resume_from_slider(self):
        # convert current seconds back to frame number
        current_frame = self.playCapture.get(cv2.CAP_PROP_FPS) * int(self.vidProgressBar.value())
        self.playCapture.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

        self.videoThread.start()
        self.status = MainWindow.STATUS_PLAYING
        self.set_pause_icon()

    #####################################Functions for load camera#############

    def read_camera(self):

        self.camBoxLabel.show()

        try:
            # self.cameraThread.start()
            self.videoExportThread.cam_prop = self.camera_prop
            self.trackingCamThread.cam_prop = self.camera_prop
            self.threshCamThread.cam_prop = self.camera_prop
            self.threshCamThread.start()
            self.update_cam_prop()
            self.openCamButton.hide()
            self.closeCamButton.setEnabled(True)

            self.camObjNumBox.setEnabled(True)
            self.applyLiveObjNum.setEnabled(True)
            self.camPreviewToggle.setEnabled(True)
            self.camInvertContrastToggle.setEnabled(True)
            self.camBlockSizeSlider.setEnabled(True)
            self.camBlockSizeSpin.setEnabled(True)
            self.camOffsetSlider.setEnabled(True)
            self.camOffsetSpin.setEnabled(True)
            self.camCntMinSlider.setEnabled(True)
            self.camCntMinSpin.setEnabled(True)
            self.camCntMaxSlider.setEnabled(True)
            self.camCntMaxSpin.setEnabled(True)
            self.applyCamThreButton.setEnabled(True)
            self.resetCamThreButton.setEnabled(True)

        except Exception as e:
            error = str(e)
            self.error_msg = QMessageBox()
            self.error_msg.setWindowTitle('Error')
            self.error_msg.setText('Failed to open camera.')
            self.error_msg.setInformativeText('Please make sure camera is connected with computer.\n')
            self.error_msg.setIcon(QMessageBox.Warning)
            self.error_msg.setDetailedText(error)
            self.error_msg.exec()

    def read_cam_prop(self, cam):

        video_prop = namedtuple('video_prop', ['width', 'height'])
        get_camera_prop = video_prop(cam.get(cv2.CAP_PROP_FRAME_WIDTH),
                                     cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

        return get_camera_prop

    def update_cam_prop(self):

        self.camResText.setText(f'{str(int(self.camera_prop.width))} X {str(int(self.camera_prop.height))}')

    def update_clock(self, clock_time):

        if clock_time:
            self.camClockText.setText(clock_time)
        else:
            self.camClockText.setText('-')

    def update_elapse(self, elapse_time):

        if elapse_time:
            self.camElapseText.setText(elapse_time)
        else:
            self.camElapseText.setText('-')

    def reload_camera(self):
        '''
        auto execute when received no ret alarm signal from cam thread
        '''

        self.threshCamThread.stop()
        self.trackingCamThread.stop()
        self.dataLogThread.stop()

        QPixmapCache.clear()
        self.camBoxLabel.hide()
        self.openCamButton.show()
        self.status = MainWindow.STATUS_INIT

    def close_camera(self):

        self.threshCamThread.stop()
        self.trackingCamThread.stop()
        self.dataLogThread.stop()
        self.trackingCamThread.frame_count = -1
        self.trackingCamThread.trackingTimeStamp.result_index = -1
        self.trackingCamThread.video_elapse = 0
        self.trackingCamThread.is_timeStamp = False

        QPixmapCache.clear()
        self.camBoxLabel.hide()

        self.camResText.setText('-')
        self.camClockText.setText('-')
        self.camElapseText.setText('-')
        self.cam_max_size.setText('-')
        self.cam_min_size.setText('-')
        self.openCamButton.show()
        self.closeCamButton.setEnabled(False)

        self.reset_cam_roi()
        self.camROICanvas.setEnabled(False)
        self.camROICanvas.lower()
        self.camPreviewBoxLabel.hide()
        self.camObjNumBox.setEnabled(False)
        self.applyLiveObjNum.setEnabled(False)
        self.camPreviewToggle.setChecked(False)
        self.camPreviewToggle.setEnabled(False)
        self.camInvertContrastToggle.setEnabled(False)
        self.camBlockSizeSlider.setEnabled(False)
        self.camBlockSizeSpin.setEnabled(False)
        self.camOffsetSlider.setEnabled(False)
        self.camOffsetSpin.setEnabled(False)
        self.camCntMinSlider.setEnabled(False)
        self.camCntMinSpin.setEnabled(False)
        self.camCntMaxSlider.setEnabled(False)
        self.camCntMaxSpin.setEnabled(False)
        self.applyCamThreButton.setEnabled(False)
        self.resetCamThreButton.setEnabled(False)
        self.camTrackingStart.setEnabled(False)
        self.exportCamData.setEnabled(False)

    #############################################################################################
    # Functions for pixel calibration
    #############################################################################################

    def enable_calibration(self):
        '''
        reset status from video player in load video tab and disable tab
        activate calibration tab
        '''
        self.tabWidget.setTabEnabled(1, False)
        self.reset_video()
        self.tabWidget.setTabEnabled(2, True)
        self.tabWidget.setCurrentIndex(2)
        self.backToLoadButton.setEnabled(True)

    def select_vid_tab(self):
        '''
        back to the load video tab, restore all widgets
        '''
        self.tabWidget.setTabEnabled(1, True)
        self.reset_video()
        self.tabWidget.setTabEnabled(2, False)
        self.tabWidget.setCurrentIndex(1)
        self.reset_roi()
        self.reset_mask()
        self.reset_scale()
        # reset thershold settings for the new video
        self.reset_thre_setting()
        self.caliBoxLabel.setEnabled(False)
        self.scaleCanvas.setEnabled(False)
        self.resetScaleButton.setEnabled(False)
        self.applyScaleButton.setEnabled(False)
        self.scaleCanvas.lower()
        self.editROIButton.setEnabled(False)
        self.editMaskButton.setEnabled(False)

    def draw_scale(self):
        '''
        enable canvas label for mouse and paint event
        '''
        self.caliBoxLabel.setEnabled(True)
        self.scaleCanvas.setEnabled(True)
        self.resetScaleButton.setEnabled(True)
        self.applyScaleButton.setEnabled(True)
        self.scaleCanvas.raise_()

    def reset_scale(self):

        self.scaleCanvas.scene.erase()
        self.pixel_per_metric = 1
        self.caliResult.clear()
        self.drawScaleButton.setEnabled(True)
        self.scaleCanvas.setEnabled(True)
        self.resetScaleButton.setEnabled(False)
        self.applyScaleButton.setEnabled(False)
        self.threTabLinkButton.setEnabled(False)

    def convert_scale(self):

        self.drawScaleButton.setEnabled(False)
        self.scaleCanvas.setEnabled(False)
        self.resetScaleButton.setEnabled(True)
        self.applyScaleButton.setEnabled(True)
        try:
            # metric = int(self.metricNumInput.text())
            metric = self.scaleCanvas.scene.inputDialog.scale_value

            if 1 <= metric <= 1000:
                display_pixel_length = self.scaleCanvas.scene.lines[0].line().length()
                # true pixel = display pixel * scale factor
                true_pixel_length = display_pixel_length * self.scale_factor
                # 1mm = x pixel (pix/mm)
                # self.pixel_per_metric = round(round(true_pixel_length, 2) / metric, 3)
                # 1 pix = x mm (mm/pix)
                self.pixel_per_metric = round(metric / round(true_pixel_length, 2), 3)
                self.caliResult.setText(str(self.pixel_per_metric))

            else:
                self.error_msg = QMessageBox()
                self.error_msg.setWindowTitle('TrackingBot')
                self.error_msg.setText('Input value out of range.')
                self.error_msg.setInformativeText('Input can only be numbers between 1 to 1000.')
                self.error_msg.setIcon(QMessageBox.Warning)
                self.error_msg.exec()

        except Exception as e:
            error = str(e)
            self.error_msg = QMessageBox()
            self.error_msg.setWindowTitle('Error')
            self.error_msg.setText('Must draw a scale and define its value.')
            self.error_msg.setIcon(QMessageBox.Warning)
            self.error_msg.setDetailedText(error)
            self.error_msg.exec()
            self.reset_scale()

    def apply_scale(self):

        if not self.scaleCanvas.scene.lines:
            self.error_msg = QMessageBox()
            self.error_msg.setWindowTitle('TrackingBot')
            self.error_msg.setText('Invalid scale.')
            self.error_msg.setInformativeText('Press and drag mouse on video image to draw a scale line.')
            self.error_msg.setIcon(QMessageBox.Warning)
            self.error_msg.exec()
        else:
            self.drawScaleButton.setEnabled(False)
            self.applyScaleButton.setEnabled(False)
            self.scaleCanvas.setEnabled(False)
            # self.scaleCanvas.scene.erase()
            self.threTabLinkButton.setEnabled(True)

            self.editROIButton.setEnabled(True)
            self.editMaskButton.setEnabled(True)

    #############################################################################################
    # Functions for set ROI
    #############################################################################################

    def edit_roi(self):

        self.rectROIButton.setEnabled(True)
        self.circROIButton.setEnabled(True)
        self.applyROIButton.setEnabled(True)

        if self.roiCanvas.scene.ROIs:
            self.resetROIButton.setEnabled(True)
            if self.maskCanvas.scene.Masks and not self.apply_mask_flag:
                self.applyMaskButton.setEnabled(True)
            else:
                self.applyMaskButton.setEnabled(False)
        elif not self.roiCanvas.scene.ROIs and not self.maskCanvas.scene.Masks:
            self.applyMaskButton.setEnabled(False)
            self.resetMaskButton.setEnabled(False)

        if self.applyMaskButton.isEnabled():
            self.resetMaskButton.setEnabled(False)

        self.roiCanvas.setEnabled(True)
        self.maskCanvas.setEnabled(False)
        self.roiCanvas.raise_()
        self.maskCanvas.scene.clearSelection()

        self.rectMaskButton.setEnabled(False)
        self.rectMaskButton.setProperty('Active', False)
        self.rectMaskButton.setStyle(self.rectMaskButton.style())
        self.circMaskButton.setEnabled(False)
        self.circMaskButton.setProperty('Active', False)
        self.circMaskButton.setStyle(self.circMaskButton.style())

    def set_line_roi(self):
        pass

    def set_rect_roi(self):

        self.resetROIButton.setEnabled(True)
        # highlight the line button and gray the rest
        self.rectROIButton.setProperty('Active', True)
        self.rectROIButton.setStyle(self.rectROIButton.style())
        self.circROIButton.setProperty('Active', False)
        self.circROIButton.setStyle(self.circROIButton.style())
        # set drawing flag
        self.roiCanvas.scene.drawRect()

    def set_circ_roi(self):

        self.resetROIButton.setEnabled(True)
        # Highlight circ button and gray the rest
        self.rectROIButton.setProperty('Active', False)
        self.rectROIButton.setStyle(self.rectROIButton.style())
        self.circROIButton.setProperty('Active', True)
        self.circROIButton.setStyle(self.circROIButton.style())
        self.roiCanvas.scene.drawCirc()

    def set_poly_roi(self):
        pass

    def apply_roi(self):

        if not self.roiCanvas.scene.ROIs:
            self.editROIButton.setEnabled(True)
            self.applyROIButton.setEnabled(False)
            self.resetROIButton.setEnabled(False)
            self.rectROIButton.setEnabled(False)
            self.rectROIButton.setProperty('Active', False)
            self.rectROIButton.setStyle(self.rectROIButton.style())
            self.circROIButton.setEnabled(False)
            self.circROIButton.setProperty('Active', False)
            self.circROIButton.setStyle(self.circROIButton.style())

            self.error_msg = QMessageBox()
            self.error_msg.setWindowTitle('TrackingBot')
            self.error_msg.setText('No valid ROI detected')
            self.error_msg.setInformativeText('To apply a ROI, please draw a shape.\n'
                                              'If you do not need a ROI, please directly go to next step by click Threshold button')
            self.error_msg.setIcon(QMessageBox.Warning)
            self.error_msg.setDetailedText('ROIs is empty. \n')
            self.error_msg.exec()
        else:
            for i in range(len(self.roiCanvas.scene.ROIs)):

                if self.roiCanvas.scene.ROIs[i].ROI.rect().isEmpty():
                    self.error_msg = QMessageBox()
                    self.error_msg.setWindowTitle('TrackingBot')
                    self.error_msg.setText('Invalid ROI detected')
                    self.error_msg.setInformativeText('The geometry of ROI is invalid, please draw a new shape.\n'
                                                      'If you do not need a ROI, please directly go to next step by click Threshold button')
                    self.error_msg.setIcon(QMessageBox.Warning)
                    self.error_msg.setDetailedText('ROIs[i].ROI.rect().isEmpty(). \n')
                    self.error_msg.exec()

                    self.reset_roi()
                    return

                else:
                    pass

            self.threshThread.ROIs = self.roiCanvas.scene.ROIs

            self.apply_roi_flag = self.threshThread.apply_roi_flag = self.trackingThread.apply_roi_flag = True

            self.roiCanvas.scene.clearSelection()
            self.roiCanvas.setEnabled(False)

            self.editROIButton.setEnabled(False)
            self.applyROIButton.setEnabled(False)
            self.resetROIButton.setEnabled(True)

            self.rectROIButton.setEnabled(False)
            self.rectROIButton.setProperty('Active', False)
            self.rectROIButton.setStyle(self.rectROIButton.style())

            self.circROIButton.setEnabled(False)
            self.circROIButton.setProperty('Active', False)
            self.circROIButton.setStyle(self.circROIButton.style())

    def reset_roi(self):
        '''
        # reset ROI object list
        # reset ROI index
        # reset all graphics item index
         Reset will be called when 1) Set a invalid ROI
                                    2) click reset button
                                    3) finished tracking and leave tracking tab
                                    4) load a new video
        '''

        self.apply_roi_flag = self.threshThread.apply_roi_flag = self.trackingThread.apply_roi_flag = False

        # reset canvas
        try:
            self.roiCanvas.scene.erase()
        except Exception as e:
            print(e)
        finally:
            self.displayCanvas.scene.erase_roi()
            self.threshThread.roi_canvas = np.zeros((576, 1024), dtype="uint8")
            self.roiCanvas.setEnabled(True)

            self.applyROIButton.setEnabled(False)
            self.editROIButton.setEnabled(True)

            self.rectROIButton.setEnabled(False)
            self.rectROIButton.setProperty('Active', False)
            self.rectROIButton.setStyle(self.rectROIButton.style())

            self.circROIButton.setEnabled(False)
            self.circROIButton.setProperty('Active', False)
            self.circROIButton.setStyle(self.circROIButton.style())

            self.resetROIButton.setEnabled(False)

    #############################################################################################
    # Functions for set mask
    #############################################################################################

    def edit_mask(self):

        self.rectMaskButton.setEnabled(True)
        self.circMaskButton.setEnabled(True)

        self.applyMaskButton.setEnabled(True)

        # have mask and have not applied roi
        if self.maskCanvas.scene.Masks:
            self.resetMaskButton.setEnabled(True)
            if self.roiCanvas.scene.ROIs and not self.apply_roi_flag:
                self.applyROIButton.setEnabled(True)
            else:
                self.applyROIButton.setEnabled(False)
        # no mask and no roi
        elif not self.maskCanvas.scene.Masks and not self.roiCanvas.scene.ROIs:
            self.applyROIButton.setEnabled(False)
            self.resetROIButton.setEnabled(False)

        if self.applyROIButton.isEnabled():
            self.resetROIButton.setEnabled(False)

        self.maskCanvas.setEnabled(True)
        self.roiCanvas.setEnabled(False)
        self.maskCanvas.raise_()
        self.roiCanvas.scene.clearSelection()

        self.rectROIButton.setEnabled(False)
        self.rectROIButton.setProperty('Active', False)
        self.rectROIButton.setStyle(self.rectROIButton.style())
        self.circROIButton.setEnabled(False)
        self.circROIButton.setProperty('Active', False)
        self.circROIButton.setStyle(self.circROIButton.style())

    def set_line_mask(self):
        pass

    def set_rect_mask(self):

        self.resetMaskButton.setEnabled(True)
        # highlight the line button and gray the rest
        self.rectMaskButton.setProperty('Active', True)
        self.rectMaskButton.setStyle(self.rectMaskButton.style())
        self.circMaskButton.setProperty('Active', False)
        self.circMaskButton.setStyle(self.circMaskButton.style())

        self.maskCanvas.scene.drawRect()

    def set_circ_mask(self):

        self.resetMaskButton.setEnabled(True)
        self.maskCanvas.scene.drawCirc()

        # Highlight circ button and gray the rest
        self.rectMaskButton.setProperty('Active', False)
        self.rectMaskButton.setStyle(self.rectMaskButton.style())
        self.circMaskButton.setProperty('Active', True)
        self.circMaskButton.setStyle(self.circMaskButton.style())

    def set_poly_mask(self):
        pass

    def apply_mask(self):

        if not self.maskCanvas.scene.Masks:
            self.editMaskButton.setEnabled(True)
            self.applyMaskButton.setEnabled(False)
            self.resetMaskButton.setEnabled(False)
            self.rectMaskButton.setEnabled(False)
            self.rectMaskButton.setProperty('Active', False)
            self.rectMaskButton.setStyle(self.rectMaskButton.style())
            self.circMaskButton.setEnabled(False)
            self.circMaskButton.setProperty('Active', False)
            self.circMaskButton.setStyle(self.circMaskButton.style())

            self.error_msg = QMessageBox()
            self.error_msg.setWindowTitle('TrackingBot')
            self.error_msg.setText('No valid Mask detected')
            self.error_msg.setInformativeText('To apply a Mask, please draw a shape.\n'
                                              'If you do not need a Mask, please directly go to next step by click Threshold button')
            self.error_msg.setIcon(QMessageBox.Warning)
            self.error_msg.setDetailedText('Masks is empty. \n')
            self.error_msg.exec()

        else:
            for i in range(len(self.maskCanvas.scene.Masks)):

                if self.maskCanvas.scene.Masks[i].Mask.rect().isEmpty():
                    self.error_msg = QMessageBox()
                    self.error_msg.setWindowTitle('TrackingBot')
                    self.error_msg.setText('Invalid Mask detected')
                    self.error_msg.setInformativeText('The geometry of Mask is invalid, please draw a new shape.\n'
                                                      'If you do not need a Mask, please directly go to next step by click Threshold button')
                    self.error_msg.setIcon(QMessageBox.Warning)
                    self.error_msg.setDetailedText('Masks[i].Mask.rect().isEmpty(). \n')
                    self.error_msg.exec()

                    self.reset_mask()
                    return
                # check intersection
                else:
                    n_intersect = 0
                    # a mask need nested inside a roi
                    for i, j in [(i, j) for i in range(len(self.maskCanvas.scene.Masks)) for j in
                                 range(len(self.roiCanvas.scene.ROIs))]:

                        intersect = self.roiCanvas.scene.ROIs[j].ROI.rect().intersects(
                            self.maskCanvas.scene.Masks[i].Mask.rect())
                        if intersect:
                            n_intersect += 1
                    # all mask must nested in this roi, any exception trigger an error
                    if n_intersect < len(self.roiCanvas.scene.ROIs):
                        self.error_msg = QMessageBox()
                        self.error_msg.setWindowTitle('TrackingBot')
                        self.error_msg.setText('Invalid Mask detected')
                        self.error_msg.setInformativeText('Mask zone must in or intersected with one ROI\n'
                                                          'If no ROI defined, please set a ROI first')
                        self.error_msg.setIcon(QMessageBox.Warning)
                        self.error_msg.setDetailedText('Masks[i].intersects(ROIs[j]) is False \n')
                        self.error_msg.exec()

                        self.reset_mask()
                        return
                    else:
                        pass

            self.threshThread.Masks = self.maskCanvas.scene.Masks

            self.apply_mask_flag = self.threshThread.apply_mask_flag = self.trackingThread.apply_mask_flag = True

            self.maskCanvas.scene.clearSelection()
            self.maskCanvas.setEnabled(False)

            self.editMaskButton.setEnabled(False)
            self.applyMaskButton.setEnabled(False)
            self.resetMaskButton.setEnabled(True)

            self.rectMaskButton.setEnabled(False)
            self.rectMaskButton.setProperty('Active', False)
            self.rectMaskButton.setStyle(self.rectMaskButton.style())
            self.circMaskButton.setEnabled(False)
            self.circMaskButton.setProperty('Active', False)
            self.circMaskButton.setStyle(self.circMaskButton.style())

    def reset_mask(self):
        # reset mask object list
        # reset mask index
        # reset all graphics item index

        self.apply_mask_flag = self.threshThread.apply_mask_flag = self.trackingThread.apply_mask_flag = False

        try:
            # reset canvas
            self.maskCanvas.scene.erase()
        except Exception as e:
            print(e)
        finally:
            self.displayCanvas.scene.erase_mask()
            self.threshThread.mask_canvas = np.zeros((576, 1024), dtype="uint8")
            self.maskCanvas.setEnabled(True)

            self.applyMaskButton.setEnabled(False)
            self.editMaskButton.setEnabled(True)

            self.rectMaskButton.setEnabled(False)
            self.rectMaskButton.setProperty('Active', False)
            self.rectMaskButton.setStyle(self.rectMaskButton.style())
            self.circMaskButton.setEnabled(False)
            self.circMaskButton.setProperty('Active', False)
            self.circMaskButton.setStyle(self.circMaskButton.style())

            self.resetMaskButton.setEnabled(False)

    #############################################################################################
    # Functions for threshold control
    #############################################################################################

    def roi_validation(self):

        if self.roiCanvas.scene.ROIs and not self.apply_roi_flag:
            reply = QMessageBox.question(self, 'TrackingBot', 'You have defined ROI(s) but did not apply.\n'
                                                              'Do you want to discard current ROI(s) and proceed?',
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.reset_roi()
                self.mask_validation()

        else:
            self.mask_validation()

    def mask_validation(self):

        if self.maskCanvas.scene.Masks and not self.apply_mask_flag:
            reply = QMessageBox.question(self, 'TrackingBot', 'You have defined Mask(s) but did not apply.\n'
                                                              'Do you want to discard current Mask(s) and proceed?',
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.reset_mask()
                self.enable_threshold()

        elif self.apply_mask_flag and not self.apply_roi_flag:
            self.error_msg = QMessageBox()
            self.error_msg.setWindowTitle('TrackingBot')
            self.error_msg.setText('Invalid Mask detected')
            self.error_msg.setInformativeText('Mask zone must in or intersected with one ROI\n'
                                              'If no ROI defined, please set a ROI first\n'
                                              'If already created a ROI, please click "Apply" button to confirm')
            self.error_msg.setIcon(QMessageBox.Warning)
            self.error_msg.setDetailedText('ROIs[] is empty \n')
            self.error_msg.exec()

        else:
            self.enable_threshold()

    def enable_threshold(self):
        # when enable cali tab, vid been reset, self.playCapture released
        # self.playCapture is closed now
        self.tabWidget.setTabEnabled(2, False)
        self.tabWidget.setTabEnabled(3, True)
        self.tabWidget.setCurrentIndex(3)
        self.backToCaliButton.setEnabled(True)
        self.reset_video()

        self.threshThread.create_roi()
        self.threshThread.create_mask()

        # create final mask image
        combined_mask = cv2.bitwise_xor(self.threshThread.roi_canvas, self.threshThread.mask_canvas)

        # scale this mask to the size of raw video frame
        # so it can be apply on raw video frames directly
        if self.video_prop.width > 1024:  # enlarge the mask to fit
            self._interpolation_flag = cv2.INTER_LINEAR
        elif self.video_prop.width < 1024:  # shrink the mask to fit
            self._interpolation_flag = cv2.INTER_AREA

        scaled_mask = cv2.resize(combined_mask, (int(self.video_prop.width), int(self.video_prop.height)),
                                 interpolation=self._interpolation_flag)

        self.threshThread.final_mask = scaled_mask

        self.displayCanvas.scene.ROIs = self.threshThread.ROIs
        self.displayCanvas.scene.Masks = self.threshThread.Masks

        self.displayCanvas.raise_()

        self.displayCanvas.scene.display_roi()
        self.displayCanvas.scene.display_mask()

        self.invertContrastToggle.setEnabled(True)
        self.blockSizeSlider.setEnabled(True)
        self.blockSizeSpin.setEnabled(True)
        self.offsetSlider.setEnabled(True)
        self.offsetSpin.setEnabled(True)
        self.cntMinSlider.setEnabled(True)
        self.cntMinSpin.setEnabled(True)
        self.cntMaxSlider.setEnabled(True)
        self.cntMaxSpin.setEnabled(True)
        self.previewToggle.setEnabled(True)
        self.applyThreButton.setEnabled(True)
        self.resetThreButton.setEnabled(True)

    def select_cali_tab(self):
        # when enable cali tab, vid been reset, self.playCapture released
        # self.playCapture is closed now
        # No ROI related operation should be done
        self.tabWidget.setTabEnabled(2, True)
        self.tabWidget.setTabEnabled(3, False)
        self.tabWidget.setCurrentIndex(2)
        self.reset_video()
        # keep the thre settings if only back to change rois
        # self.reset_thre_setting()
        self.previewToggle.setChecked(False)
        for i in range(len(self.roiCanvas.scene.ROIs)):
            self.roiCanvas.scene.addItem(self.roiCanvas.scene.ROIs[i].ROI)
        for i in range(len(self.maskCanvas.scene.Masks)):
            self.maskCanvas.scene.addItem(self.maskCanvas.scene.Masks[i].Mask)

    def thresh_vid_control(self):

        if self.video_file[0] == '' or self.video_file[0] is None:
            print('No video is selected')
            return

        if self.status is MainWindow.STATUS_INIT:
            try:
                self.play_thresh_vid()
            except Exception as e:
                error = str(e)
                self.error_msg = QMessageBox()
                self.error_msg.setWindowTitle('Error')
                self.error_msg.setText('An error happened when trying to play video file.')
                self.error_msg.setInformativeText('play_threh_vid() does not execute correctly.')
                self.error_msg.setIcon(QMessageBox.Warning)
                self.error_msg.setDetailedText(error)
                self.error_msg.exec()

        elif self.status is MainWindow.STATUS_PLAYING:
            try:
                self.pause_thresh_vid()
            except Exception as e:
                error = str(e)
                self.error_msg = QMessageBox()
                self.error_msg.setWindowTitle('Error')
                self.error_msg.setText('An error happened when trying to pause video file.')
                self.error_msg.setInformativeText('pause_thresh_vid() does not execute correctly.')
                self.error_msg.setIcon(QMessageBox.Warning)
                self.error_msg.setDetailedText(error)
                self.error_msg.exec()

        elif self.status is MainWindow.STATUS_PAUSE:
            try:
                self.resume_thresh_vid()
            except Exception as e:
                error = str(e)
                self.error_msg = QMessageBox()
                self.error_msg.setWindowTitle('Error')
                self.error_msg.setText('An error happened when trying to resume playing.')
                self.error_msg.setInformativeText('resume_thresh_vid() does not execute correctly.')
                self.error_msg.setIcon(QMessageBox.Warning)
                self.error_msg.setDetailedText(error)
                self.error_msg.exec()

    def play_thresh_vid(self):

        self.threshThread.playCapture.open(self.video_file[0])
        self.threshThread.start()
        self.status = MainWindow.STATUS_PLAYING
        self.set_pause_icon()

    def pause_thresh_vid(self):

        self.threshThread.stop()
        # for camera
        # if self.video_type is MainWindow.VIDEO_TYPE_REAL_TIME:
        #     self.playCapture.release()
        # print(self.playCapture.get(cv2.CAP_PROP_POS_FRAMES))
        self.status = MainWindow.STATUS_PAUSE
        self.set_play_icon()

    def resume_thresh_vid(self):

        self.threshThread.start()
        self.status = MainWindow.STATUS_PLAYING
        self.set_pause_icon()

    def resume_thresh_slider(self):

        # convert current seconds back to frame number
        current_frame = self.threshThread.playCapture.get(cv2.CAP_PROP_FPS) * int(self.threProgressBar.value())
        self.threshThread.playCapture.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

        self.threshThread.start()
        self.status = MainWindow.STATUS_PLAYING
        self.set_pause_icon()

    def stop_thresh_vid(self):

        is_stopped = self.threshThread.is_stopped()
        self.threPlayButton.setEnabled(True)
        # reset when video is paused
        if is_stopped:
            self.threshThread.playCapture.release()
            self.read_video_file(self.video_file[0])
            self.status = MainWindow.STATUS_INIT
        # reset when video still playing
        elif not is_stopped:
            self.threshThread.stop()
            self.threshThread.playCapture.release()
            self.read_video_file(self.video_file[0])
            self.status = MainWindow.STATUS_INIT
            self.set_play_icon()

    def update_detect_cnt(self, max, min):

        if max:
            self.max_size.setText(str(max))
        elif not max:
            self.max_size.setText('-')
        if min:
            self.min_size.setText(str(min))
        elif not min:
            self.min_size.setText('-')

    def update_thre_slider(self, elapse):

        self.threProgressBar.setSliderPosition(int(elapse))
        self.threPosLabel.setText(f"{str(timedelta(seconds=elapse)).split('.')[0]}")

    def update_thre_position(self):
        '''
        when drag slider to new position, update it
        '''

        play_elapse = self.threProgressBar.value()
        self.threPosLabel.setText(f"{str(timedelta(seconds=play_elapse)).split('.')[0]}")

    def apply_object_num(self):

        self.object_num = self.objNumBox.value()
        self.objNumBox.setEnabled(False)
        self.applyObjNumButton.setEnabled(False)

    def set_blocksize_slider(self):

        block_size = self.blockSizeSlider.value()
        # block size must be an odd value
        if block_size % 2 == 0:
            block_size += 1
        if block_size < 3:
            block_size = 3
        # update spin control to same value
        self.blockSizeSpin.setValue(block_size)
        # pass value to thread
        self.threshThread.block_size = block_size

    def set_blocksize_spin(self):

        block_size = self.blockSizeSpin.value()
        if block_size % 2 == 0:
            block_size += 1
        if block_size < 3:
            block_size = 3
        # update slider control to same value
        self.blockSizeSlider.setValue(block_size)
        # pass value to thread
        self.threshThread.block_size = block_size

    def set_offset_slider(self):

        offset = self.offsetSlider.value()
        self.offsetSpin.setValue(offset)
        self.threshThread.offset = offset

    def set_offset_spin(self):

        offset = self.offsetSpin.value()
        self.offsetSlider.setValue(offset)
        self.threshThread.offset = offset

    def set_min_cnt_slider(self):

        min_cnt = self.cntMinSlider.value()
        self.cntMinSpin.setValue(min_cnt)
        self.threshThread.min_contour = min_cnt

    def set_min_cnt_spin(self):

        min_cnt = self.cntMinSpin.value()
        self.cntMinSlider.setValue(min_cnt)
        self.threshThread.min_contour = min_cnt

    def set_max_cnt_slider(self):

        max_cnt = self.cntMaxSlider.value()
        self.cntMaxSpin.setValue(max_cnt)
        self.threshThread.max_contour = max_cnt

    def set_max_cnt_spin(self):

        max_cnt = self.cntMaxSpin.value()
        self.cntMaxSlider.setValue(max_cnt)
        self.threshThread.max_contour = max_cnt

    def set_pause_icon(self):

        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        self.threPlayButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))

    def set_play_icon(self):

        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.threPlayButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))

    def display_threshold_video(self, frame, preview_frame):

        self.threBoxLabel.setPixmap(frame)
        self.previewBoxLabel.setPixmap(preview_frame)

    def enable_thre_preview(self):
        # enable real time preview window of threshold result
        if self.previewToggle.isChecked():
            self.previewBoxLabel.raise_()
        else:
            self.previewBoxLabel.lower()

    def invert_contrast(self):
        # invert contrast of video
        # default white background dark object
        if self.invertContrastToggle.isChecked():
            self.threshThread.invert_contrast = True
        else:
            self.threshThread.invert_contrast = False

    def apply_thre_setting(self):
        '''
        Store current threshold parameter settings and activate next step
        then pass stored settings to tracking thread
        '''

        if self.applyObjNumButton.isEnabled():
            self.error_msg = QMessageBox()
            self.error_msg.setWindowTitle('TrackingBot')
            self.error_msg.setText('Invalid parameter')
            self.error_msg.setInformativeText('Please set Number of Objects\n'
                                              'If already set, please click "Enter" button to confirm.')
            self.error_msg.setIcon(QMessageBox.Warning)
            self.error_msg.setDetailedText('self.object_num is empty \n')
            self.error_msg.exec()
        else:

            self.block_size = self.threshThread.block_size
            self.offset = self.threshThread.offset
            self.min_contour = self.threshThread.min_contour
            self.max_contour = self.threshThread.max_contour
            self.invert_contrast_state = self.threshThread.invert_contrast
            self.trackingThread.valid_mask = self.threshThread.final_mask
            self.applyThreButton.setEnabled(False)
            self.applyObjNumButton.setEnabled(False)
            self.applyThreButton.setEnabled(False)
            self.blockSizeSlider.setEnabled(False)
            self.blockSizeSpin.setEnabled(False)
            self.offsetSlider.setEnabled(False)
            self.offsetSpin.setEnabled(False)
            self.cntMinSlider.setEnabled(False)
            self.cntMinSpin.setEnabled(False)
            self.cntMaxSlider.setEnabled(False)
            self.cntMaxSpin.setEnabled(False)
            self.previewBoxLabel.lower()
            self.previewToggle.setEnabled(False)
            self.previewToggle.setChecked(False)
            self.invertContrastToggle.setEnabled(False)

            self.trackTabLinkButton.setEnabled(True)

    def reset_thre_setting(self):
        '''
        Reset current threshold parameter settings
        '''

        self.applyThreButton.setEnabled(True)
        self.applyObjNumButton.setEnabled(True)
        self.objNumBox.setEnabled(True)
        self.objNumBox.setValue(1)
        self.blockSizeSlider.setEnabled(True)
        self.blockSizeSlider.setValue(11)
        self.blockSizeSpin.setEnabled(True)
        self.blockSizeSpin.setValue(11)
        self.offsetSlider.setEnabled(True)
        self.offsetSlider.setValue(11)
        self.offsetSpin.setEnabled(True)
        self.offsetSpin.setValue(11)
        self.cntMinSlider.setEnabled(True)
        self.cntMinSlider.setValue(1)
        self.cntMinSpin.setEnabled(True)
        self.cntMinSpin.setValue(1)
        self.cntMaxSlider.setEnabled(True)
        self.cntMaxSlider.setValue(100)
        self.cntMaxSpin.setEnabled(True)
        self.cntMaxSpin.setValue(100)

        self.previewToggle.setEnabled(True)
        self.invertContrastToggle.setEnabled(True)

        self.trackTabLinkButton.setEnabled(False)

    #############################################################################################
    # Functions for tracking control
    #############################################################################################

    def enable_tracking(self):
        # when enable cali tab, vid been reset, self.playCapture released
        # self.playCapture is closed now
        # self.playCapture will open when tracking thread started
        self.tabWidget.setTabEnabled(3, False)
        self.tabWidget.setTabEnabled(4, True)
        self.tabWidget.setCurrentIndex(4)
        self.leaveTrackButton.setEnabled(True)
        self.trackStartButton.setEnabled(True)
        # accept parameters from thresholding
        self.trackingThread.obj_num = self.object_num
        self.trackingThread.trackingMethod.obj_num = self.object_num
        self.trackingThread.block_size = self.block_size
        self.trackingThread.offset = self.offset
        self.trackingThread.min_contour = self.min_contour
        self.trackingThread.max_contour = self.max_contour
        self.trackingThread.invert_contrast = self.invert_contrast_state
        self.dataLogThread.obj_num = self.object_num

        self.reset_video()

    def leave_track_tab(self):
        '''
        determine the destination tab according to tracking task status
        :return:
        '''
        # tracking task cancelled, allow back to thresholding process
        if not self.track_fin:
            self.tabWidget.setTabEnabled(3, True)
            self.tabWidget.setTabEnabled(4, False)
            self.tabWidget.setCurrentIndex(3)
            self.trackStartButton.setEnabled(False)
            self.reset_video()
            self.reset_track_results()  # reset data and all status flag

        elif self.track_fin:
            # if data not saved, show reminder before leave
            if not self.export_data_fin or not self.export_graph_fin:
                self.warning_msg = QMessageBox()
                self.warning_msg.setWindowTitle('Warning')
                self.warning_msg.setIcon(QMessageBox.Warning)
                self.warning_msg.setText('Tracking data or graph are not saved.\n'
                                         'Do you want to back to start menu without save the data?')
                self.warning_msg.setStandardButtons(QMessageBox.Cancel | QMessageBox.Yes)
                returnValue = self.warning_msg.exec()
                # if yes, back to start menu and reset all settings
                if returnValue == QMessageBox.Yes:
                    try:
                        self.select_main_menu()
                        self.reset_scale()
                        self.reset_roi()
                        self.reset_mask()
                        self.reset_thre_setting()
                        self.reset_track_results()  # reset data and all status flag

                        self.trackStartButton.setEnabled(False)
                        self.exportDataButton.setEnabled(False)
                        self.exportTraceButton.setEnabled(False)
                        self.exportHeatmapButton.setEnabled(False)
                        self.traceToggle.setEnabled(False)
                        self.traceToggle.setChecked(False)
                        self.heatmapToggle.setEnabled(False)
                        self.heatmapToggle.setChecked(False)
                        self.leaveTrackButton.setText('Back')

                    except Exception as e:
                        error = str(e)
                        self.warning_msg = QMessageBox()
                        self.warning_msg.setWindowTitle('Error')
                        self.warning_msg.setText('An error happened when back to start menu.')
                        self.warning_msg.setInformativeText('leave_track_tab() does not execute correctly.')
                        self.warning_msg.setIcon(QMessageBox.Warning)
                        self.warning_msg.setDetailedText(error)
                        self.warning_msg.exec()
            # if saved, back to start menu and reset all settings
            else:
                self.select_main_menu()
                self.reset_scale()
                self.reset_roi()
                self.reset_mask()
                self.reset_thre_setting()
                self.reset_track_results()  # reset data and all status flag
                self.trackStartButton.setEnabled(False)
                self.exportDataButton.setEnabled(False)
                self.exportTraceButton.setEnabled(False)
                self.exportHeatmapButton.setEnabled(False)
                self.traceToggle.setEnabled(False)
                self.traceToggle.setChecked(False)
                self.heatmapToggle.setEnabled(False)
                self.heatmapToggle.setChecked(False)
                self.leaveTrackButton.setText('Back')

    def tracking_vid_control(self):

        if self.status is MainWindow.STATUS_INIT:
            try:
                self.start_tracking()  # run tracking thread

            except Exception as e:
                error = str(e)
                self.error_msg = QMessageBox()
                self.error_msg.setWindowTitle('Error')
                self.error_msg.setText('An error happened when trying to track video file.')
                self.error_msg.setInformativeText('start_tracking() does not execute correctly.')
                self.error_msg.setIcon(QMessageBox.Warning)
                self.error_msg.setDetailedText(error)
                self.error_msg.exec()

        elif self.status is MainWindow.STATUS_PLAYING:
            self.warning_msg = QMessageBox()
            self.warning_msg.setWindowTitle('Warning')
            self.warning_msg.setIcon(QMessageBox.Warning)
            self.warning_msg.setText('A tracking task is in progress, all data will be lost if stop now.\n'
                                     'Do you want continue to abort current task?')
            self.warning_msg.setStandardButtons(QMessageBox.Cancel | QMessageBox.Yes)
            returnValue = self.warning_msg.exec()

            if returnValue == QMessageBox.Yes:
                try:
                    self.stop_tracking()
                except Exception as e:
                    error = str(e)
                    self.warning_msg = QMessageBox()
                    self.warning_msg.setWindowTitle('Error')
                    self.warning_msg.setText('An error happened when trying to stop tracking.')
                    self.warning_msg.setInformativeText('stop_tracking() does not execute correctly.')
                    self.warning_msg.setIcon(QMessageBox.Warning)
                    self.warning_msg.setDetailedText(error)
                    self.warning_msg.exec()

    def start_tracking(self):
        '''
        read video file and run tracking thread
        :return:
        '''
        self.dataLogThread.df.clear()
        self.dataLogThread.df_archive.clear()
        self.dataLogThread.tracked_object = None
        self.dataLogThread.tracked_index = None
        self.dataLogThread.tracked_elapse = None
        self.trackingThread.playCapture.open(self.video_file[0])
        self.trackingThread.start()
        self.start_tic = time.perf_counter()
        self.status = MainWindow.STATUS_PLAYING

        self.trackStartButton.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
        self.leaveTrackButton.setEnabled(False)

    def stop_tracking(self):
        '''
        cancel and reset tracking progress when stop clicked during ongoing task
        :return:
        '''
        try:
            self.trackingThread.stop()
            self.dataLogThread.stop()
            self.trackingThread.playCapture.release()
            self.reset_track_results()  # also reset all status flags
            self.trackingThread.frame_count = -1
            self.trackingThread.trackingTimeStamp.result_index = -1
            self.trackingThread.video_elapse = 0
            self.trackingThread.is_timeStamp = False
            # reset background frame
            self.trackingBoxLabel.clear()
            self.read_video_file(self.video_file[0])
            self.status = MainWindow.STATUS_INIT
            self.trackStartButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
            self.leaveTrackButton.setEnabled(True)

        except Exception as e:
            error = str(e)
            self.warning_msg = QMessageBox()
            self.warning_msg.setWindowTitle('Error')
            self.warning_msg.setText('An error happened during stop tracking.')
            self.warning_msg.setInformativeText('stop_tracking() does not execute correctly.')
            self.warning_msg.setIcon(QMessageBox.Warning)
            self.warning_msg.setDetailedText(error)
            self.warning_msg.exec()

        finally:
            self.info_msg = QMessageBox()
            self.info_msg.setWindowTitle('TrackingBot')
            self.info_msg.setIcon(QMessageBox.Information)
            self.info_msg.setText('Tracking task cancelled.')
            self.info_msg.exec()

    def exceed_index_alarm(self):
        '''
        cancel and reset tracking progress when object out of index at first frame
        :return:
        '''

        try:
            self.trackingThread.stop()
            self.dataLogThread.stop()
            self.trackingThread.playCapture.release()
            self.reset_track_results()  # also reset all status flags
            # reset background frame
            self.trackingBoxLabel.clear()
            self.read_video_file(self.video_file[0])
            self.status = MainWindow.STATUS_INIT
            self.trackStartButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
            self.leaveTrackButton.setEnabled(True)

        except Exception as e:
            error = str(e)
            self.warning_msg = QMessageBox()
            self.warning_msg.setWindowTitle('Error')
            self.warning_msg.setText('An error happened during stop tracking.')
            self.warning_msg.setInformativeText('index_alarm() does not execute correctly.')
            self.warning_msg.setIcon(QMessageBox.Warning)
            self.warning_msg.setDetailedText(error)
            self.warning_msg.exec()

        finally:
            self.warning_msg = QMessageBox()
            self.warning_msg.setWindowTitle('TrackingBot')
            self.warning_msg.setIcon(QMessageBox.Warning)
            self.warning_msg.setText('Must ensure the detected object number in the first frame '
                                     'not exceed the set value.')
            self.warning_msg.exec()

    def display_tracking_video(self, frame):
        #
        self.trackingBoxLabel.setPixmap(frame)

    def update_track_slider(self, elapse):

        self.trackProgressBar.setSliderPosition(int(elapse))
        self.trackPosLabel.setText(f"{str(timedelta(seconds=elapse)).split('.')[0]}")

    def update_track_vid_position(self):

        play_elapse = self.trackProgressBar.value()
        self.trackPosLabel.setText(f"{str(timedelta(seconds=play_elapse)).split('.')[0]}")

    def update_track_results(self, tracked_objects,expired_id_list,tracked_index,tracked_elapse):
        '''
        pass the list of registered object information,the list of expired id number
        the index of timestamp, ideo time elapsed when timestamp is true, to datalog thread
        '''
        self.dataLogThread.track_results(tracked_objects,
                                         expired_id_list,
                                         tracked_index,
                                         tracked_elapse)
        self.dataLogThread.start()

    def complete_tracking(self):

        self.stop_toc = time.perf_counter()
        total_time = self.stop_toc - self.start_tic
        print(f'Time Cost Total {self.stop_toc - self.start_tic:.5f}')
        self.set_complete_frame()

        self.info_msg = QMessageBox()
        self.info_msg.setWindowTitle('TrackingBot')
        self.info_msg.setIcon(QMessageBox.Information)
        self.info_msg.setText('Tracking finished.')
        self.info_msg.setInformativeText('Total time spent : ' + '{:.2f}'.format(total_time) +'s')
        self.info_msg.exec()

        # set destination to start tab when tracking task finished
        self.track_fin = True
        self.trackStartButton.setEnabled(False)
        self.leaveTrackButton.setEnabled(True)
        self.leaveTrackButton.setText('Menu')

        # enable export data button
        self.exportDataButton.setEnabled(True)
        # must generate graph before export
        self.traceToggle.setEnabled(True)
        # must generate graph before export
        self.heatmapToggle.setEnabled(True)

    def set_complete_frame(self):

        complete_frame = self.last_frame.copy()
        text_frame = self.last_frame.copy()
        font = cv2.FONT_HERSHEY_TRIPLEX
        text = 'Tracking completed'
        # get boundary of the text
        text_size = cv2.getTextSize(text, font, 1, 2)[0]
        # get coords based on boundary
        textX = (self.video_prop.width - text_size[0]) / 2
        textY = (self.video_prop.height + text_size[1]) / 2

        cv2.putText(text_frame, text, (int(textX), int(textY)), font, 1, (226, 137, 4),thickness=4)
        # blend with the original
        opacity = 0.4
        cv2.addWeighted(text_frame, opacity, complete_frame, 1 - opacity, 0, complete_frame)
        # 4:3
        if self.video_prop.height / self.video_prop.width == 0.75:
            scaled = cv2.resize(complete_frame, (768, 576), interpolation=self._interpolation_flag)

        # 16:9
        else:
            scaled = cv2.resize(complete_frame, (1024, 576), interpolation=self._interpolation_flag)

        rgb = cv2.cvtColor(scaled, cv2.COLOR_BGR2RGB)
        cvt = QImage(rgb, rgb.shape[1], rgb.shape[0], rgb.strides[0],
                     QImage.Format_RGB888)
        display = QPixmap.fromImage(cvt)
        self.trackingBoxLabel.setPixmap(display)

    def reset_track_results(self):
        '''
        reset all tracked data when requested to cancel the task
        :return:
        '''
        self.track_fin = False
        self.export_data_fin = False
        self.export_graph_fin = False

        self.dataLogThread.df.clear()
        self.dataLogThread.df_archive.clear()
        self.dataLogThread.tracked_object = None
        self.dataLogThread.tracked_index = None
        self.dataLogThread.tracked_elapse = None
        self.trackingTimeStamp.result_index = -1
        self.trackingThread.trackingMethod.candidate_list.clear()
        self.trackingThread.trackingMethod.candidate_index = 0
        self.trackingThread.trackingMethod.candidate_id = 0
        self.trackingThread.trackingMethod.expired_id.clear()
        self.trace_map = None
        self.heat_map = None

    def export_data(self):

        self.folder_path = QFileDialog.getExistingDirectory(None, 'Select Folder', 'C:/Users/Public/Documents')
        # print(self.folder_path)
        if self.folder_path == '':
            return
        else:
            try:
                # pass path
                self.dataProcessDialog.data_save_path = self.folder_path
                self.dataProcessDialog.video_fps = self.video_prop.fps
                self.dataProcessDialog.object_num = self.object_num
                self.dataProcessDialog.pixel_per_metric = self.pixel_per_metric
                # point thread
                self.dataProcessDialog.dataLogThread = self.dataLogThread
                # start processing data and show progress bar
                self.dataProcessDialog.setStart()

            except Exception as e:
                error = str(e)
                self.error_msg = QMessageBox()
                self.error_msg.setWindowTitle('Error')
                self.error_msg.setText('An error happened when trying to export tracking data.')
                self.error_msg.setInformativeText('export_data() does not execute correctly.\n' + error)
                self.error_msg.setIcon(QMessageBox.Warning)
                self.error_msg.exec()

    def export_data_success(self):

        self.export_data_fin = True

        self.info_msg = QMessageBox()
        self.info_msg.setWindowTitle('TrackingBot')
        self.info_msg.setIcon(QMessageBox.Information)
        self.info_msg.setText('Successfully saved data')
        self.info_msg.addButton('OK', QMessageBox.RejectRole)
        self.info_msg.addButton('Open folder', QMessageBox.AcceptRole)
        returnValue = self.info_msg.exec()
        if returnValue == 1:
            self.open_export_folder()
        else:
            return

    def generate_trace(self):

        if self.traceToggle.isChecked():
            self.heatmapToggle.setChecked(False)
            # first time generate trace map
            if self.trace_map is None:
                try:
                    # point thread
                    self.traceProcessDialog.dataLogThread = self.dataLogThread
                    self.traceProcessDialog.trace_frame = self.last_frame
                    self.traceProcessDialog.video_prop = self.video_prop
                    # start processing and show progress bar
                    self.traceProcessDialog.setStart()

                except Exception as e:
                    error = str(e)
                    self.error_msg = QMessageBox()
                    self.error_msg.setWindowTitle('Error')
                    self.error_msg.setText('An error happened when trying to generate trace map.')
                    self.error_msg.setInformativeText('generate_trace() does not execute correctly.\n' + error)
                    self.error_msg.setIcon(QMessageBox.Warning)
                    self.error_msg.exec()
            else:
                self.trackingBoxLabel.setPixmap(self.trace_map)
        # Toggle off, show complete frame
        else:
            self.set_complete_frame()

    def display_trace(self, trace_map):

        self.trace_map = trace_map
        self.trackingBoxLabel.setPixmap(trace_map)
        # enable export trace button
        self.exportTraceButton.setEnabled(True)

    def raw_trace(self, raw_trace_map):

        self.raw_trace_map = raw_trace_map

    def export_trace(self):

        self.folder_path = QFileDialog.getExistingDirectory(None, 'Select Folder', 'C:/Users/Public/Documents')
        # print(self.folder_path)
        if self.folder_path == '':
            return
        else:
            try:
                # save image to path
                now = datetime.now()
                full_path = self.folder_path + '/TrackingBot export trace map' + now.strftime('%Y-%m-%d-%H%M') + '.png'
                cv2.imwrite(full_path,self.raw_trace_map)
                self.export_trace_success()

            except Exception as e:
                error = str(e)
                self.error_msg = QMessageBox()
                self.error_msg.setWindowTitle('Error')
                self.error_msg.setText('An error happened when trying to export trace map.')
                self.error_msg.setInformativeText('export_trace() does not execute correctly.\n' + error)
                self.error_msg.setIcon(QMessageBox.Warning)
                self.error_msg.exec()

    def export_trace_success(self):

        self.info_msg = QMessageBox()
        self.info_msg.setWindowTitle('TrackingBot')
        self.info_msg.setIcon(QMessageBox.Information)
        self.info_msg.setText('Successfully saved trace map.')
        self.info_msg.addButton('OK', QMessageBox.RejectRole)
        self.info_msg.addButton('Open folder', QMessageBox.AcceptRole)
        returnValue = self.info_msg.exec()
        if returnValue == 1:
            self.open_export_folder()
        else:
            return

    def generate_heatmap(self):

        if self.heatmapToggle.isChecked():

            self.traceToggle.setChecked(False)

            if self.heat_map is None:
                # make it a true/flase flag
                    self.graphProcessDialog.dataLogThread = self.dataLogThread
                    self.graphProcessDialog.video_prop = self.video_prop
                    # start processing and show progress bar
                    self.graphProcessDialog.setStart()
            else:
                self.display_heatmap(self.heat_map)

            self.exportHeatmapButton.setEnabled(True)

        else:
            self.verticalLayoutWidget.lower()
            self.set_complete_frame()

    def display_heatmap(self,heat_map):
        self.heat_map = heat_map
        # init matplotlib fig
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.imshow(self.heat_map, origin='lower', extent=None, cmap=cm.jet)
        ax.invert_yaxis()
        ax.axis('off')

        self.canvas.draw()
        self.verticalLayoutWidget.raise_()

    def export_heatmap(self):

        self.folder_path = QFileDialog.getExistingDirectory(None, 'Select Folder', 'C:/Users/Public/Documents')

        if self.folder_path == '':
            return
        else:
            try:
                now = datetime.now()
                full_path = self.folder_path + '/TrackingBot export heatmap' + now.strftime('%Y-%m-%d-%H%M') + '.png'
                self.figure.savefig(full_path, dpi=300, bbox_inches=None)
                self.export_heatmap_success()

            except Exception as e:
                error= str(e)
                self.error_msg = QMessageBox()
                self.error_msg.setWindowTitle('Error')
                self.error_msg.setText('An error happened when trying to export graph.')
                self.error_msg.setInformativeText('export_graph() does not execute correctly.')
                self.error_msg.setIcon(QMessageBox.Warning)
                self.error_msg.setDetailedText(error)
                self.error_msg.exec()

    def export_heatmap_success(self):

        self.export_graph_fin = True
        self.info_msg = QMessageBox()
        self.info_msg.setWindowTitle('TrackingBot')
        self.info_msg.setIcon(QMessageBox.Information)
        self.info_msg.setText('Successfully saved graph')
        self.info_msg.addButton('OK', QMessageBox.RejectRole)
        self.info_msg.addButton('Open folder', QMessageBox.AcceptRole)
        returnValue = self.info_msg.exec()
        if returnValue == 1:
            self.open_export_folder()
        else:
            return

    def open_export_folder(self):
        '''
        open system directory that data file saved
        for both Win and OS system

        '''
        try:
            os.startfile(self.folder_path)
        except:
            subprocess.Popen(['xdg-open', self.folder_path])


    ###############################################Function for cam threshold###############

    def display_threshold_cam(self, frame, preview_frame):

        self.camBoxLabel.setPixmap(frame)
        self.camPreviewBoxLabel.setPixmap(preview_frame)

    def invert_cam_contrast(self):

        if self.camInvertContrastToggle.isChecked():
            self.threshCamThread.invert_contrast = True
        else:
            self.threshCamThread.invert_contrast = False

    def enable_cam_thre_preview(self):

        if self.camPreviewToggle.isChecked():
            self.camPreviewBoxLabel.show()
            self.camPreviewBoxLabel.raise_()
        else:
            self.camPreviewBoxLabel.lower()

    def update_cam_detect_cnt(self, max, min):

        if max:
            self.cam_max_size.setText(str(max))
        elif not max:
            self.cam_max_size.setText('-')
        if min:
            self.cam_min_size.setText(str(min))
        elif not min:
            self.cam_min_size.setText('-')

    def set_cam_blocksize_slider(self):

        block_size = self.camBlockSizeSlider.value()
        # block size must be an odd value
        if block_size % 2 == 0:
            block_size += 1
        if block_size < 3:
            block_size = 3
        # update spin control to same value
        self.camBlockSizeSpin.setValue(block_size)
        # pass value to thread
        self.threshCamThread.block_size = block_size

    def set_cam_blocksize_spin(self):

        block_size = self.camBlockSizeSpin.value()
        if block_size % 2 == 0:
            block_size += 1
        if block_size < 3:
            block_size = 3
        # update slider control to same value
        self.camBlockSizeSlider.setValue(block_size)
        # pass value to thread
        self.threshCamThread.block_size = block_size

    def set_cam_offset_slider(self):

        offset = self.camOffsetSlider.value()
        self.camOffsetSpin.setValue(offset)
        self.threshCamThread.offset = offset

    def set_cam_offset_spin(self):

        offset = self.camOffsetSpin.value()
        self.camOffsetSlider.setValue(offset)
        self.threshCamThread.offset = offset

    def set_cam_min_cnt_slider(self):

        min_cnt = self.camCntMinSlider.value()
        self.camCntMinSpin.setValue(min_cnt)
        self.threshCamThread.min_contour = min_cnt

    def set_cam_min_cnt_spin(self):

        min_cnt = self.camCntMinSpin.value()
        self.camCntMinSlider.setValue(min_cnt)
        self.threshCamThread.min_contour = min_cnt

    def set_cam_max_cnt_slider(self):

        max_cnt = self.camCntMaxSlider.value()
        self.camCntMaxSpin.setValue(max_cnt)
        self.threshCamThread.max_contour = max_cnt

    def set_cam_max_cnt_spin(self):

        max_cnt = self.camCntMaxSpin.value()
        self.camCntMaxSlider.setValue(max_cnt)
        self.threshCamThread.max_contour = max_cnt

    def apply_cam_object_num(self):

        self.cam_object_num = self.camObjNumBox.value()
        self.camObjNumBox.setEnabled(False)
        self.applyLiveObjNum.setEnabled(False)

    def apply_thre_cam_setting(self):
        '''
        Apply current threshold parameter settings and activate next step
        '''
        if self.applyLiveObjNum.isEnabled():
            self.error_msg = QMessageBox()
            self.error_msg.setWindowTitle('TrackingBot')
            self.error_msg.setText('Invalid parameter')
            self.error_msg.setInformativeText('Please set Number of Objects\n'
                                              'If already set, please click "Enter" button to confirm.')
            self.error_msg.setIcon(QMessageBox.Warning)
            self.error_msg.setDetailedText('self.cam_object_num is empty \n')
            self.error_msg.exec()
        else:
            self.cam_block_size = self.threshCamThread.block_size
            self.cam_offset = self.threshCamThread.offset
            self.cam_min_contour = self.threshCamThread.min_contour
            self.cam_max_contour = self.threshCamThread.max_contour
            self.cam_invert_contrast = self.threshThread.invert_contrast

            self.applyCamThreButton.setEnabled(False)
            self.camPreviewBoxLabel.lower()
            self.camPreviewToggle.setEnabled(False)
            self.camPreviewToggle.setChecked(False)
            self.camInvertContrastToggle.setEnabled(False)
            self.camBlockSizeSlider.setEnabled(False)
            self.camBlockSizeSpin.setEnabled(False)
            self.camOffsetSlider.setEnabled(False)
            self.camOffsetSpin.setEnabled(False)
            self.camCntMinSlider.setEnabled(False)
            self.camCntMinSpin.setEnabled(False)
            self.camCntMaxSlider.setEnabled(False)
            self.camCntMaxSpin.setEnabled(False)

            self.camTrackingStart.setEnabled(True)

    def reset_thre_cam_setting(self):
        '''
        Reset current threshold parameter settings
        '''

        self.applyCamThreButton.setEnabled(True)

        self.applyLiveObjNum.setEnabled(True)
        self.camObjNumBox.setEnabled(True)
        self.camObjNumBox.setValue(1)
        self.camBlockSizeSlider.setEnabled(True)
        self.camBlockSizeSlider.setValue(11)
        self.camBlockSizeSpin.setEnabled(True)
        self.camBlockSizeSpin.setValue(11)
        self.camOffsetSlider.setEnabled(True)
        self.camOffsetSlider.setValue(11)
        self.camOffsetSpin.setEnabled(True)
        self.camOffsetSpin.setValue(11)
        self.camCntMinSlider.setEnabled(True)
        self.camCntMinSlider.setValue(1)
        self.camCntMinSpin.setEnabled(True)
        self.camCntMinSpin.setValue(1)
        self.camCntMaxSlider.setEnabled(True)
        self.camCntMaxSlider.setValue(100)
        self.camCntMaxSpin.setEnabled(True)
        self.camCntMaxSpin.setValue(100)

        self.camPreviewToggle.setEnabled(True)
        self.camInvertContrastToggle.setEnabled(True)

        self.camTrackingStart.setEnabled(False)

    #   ############################################Functions for feedback control

    def cam_tracking_control(self):

        if self.status is MainWindow.STATUS_INIT:
            try:
                self.start_cam_tracking()
                self.status = MainWindow.STATUS_PLAYING
                self.camTrackingStart.setText('STOP')
                self.camTrackingStart.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
            except Exception as e:
                error = str(e)
                self.error_msg = QMessageBox()
                self.error_msg.setWindowTitle('Error')
                self.error_msg.setText('An error happened when trying to start live tracking.')
                self.error_msg.setIcon(QMessageBox.Warning)
                self.error_msg.setDetailedText(error)
                self.error_msg.exec()
        # stop
        elif self.status is MainWindow.STATUS_PLAYING:
            try:
                self.stop_cam_tracking()
                self.status = MainWindow.STATUS_INIT
                self.camTrackingStart.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
            except Exception as e:
                error = str(e)
                self.error_msg = QMessageBox()
                self.error_msg.setWindowTitle('Error')
                self.error_msg.setText('An error happened when trying to stop tracking.')
                self.error_msg.setIcon(QMessageBox.Warning)
                self.error_msg.setDetailedText(error)
                self.error_msg.exec()

    def start_cam_tracking(self):

        self.threshCamThread.stop()
        self.camClockText.setText('-')
        self.camElapseText.setText('-')
        self.cam_max_size.setText('-')
        self.cam_min_size.setText('-')
        self.resetCamThreButton.setEnabled(False)
        self.closeCamButton.setEnabled(False)
        self.trackingCamThread.obj_num = self.cam_object_num
        self.trackingCamThread.trackingMethod.obj_num= self.cam_object_num
        self.trackingCamThread.block_size = self.cam_block_size
        self.trackingCamThread.offset = self.cam_offset
        self.trackingCamThread.min_contour = self.cam_min_contour
        self.trackingCamThread.max_contour = self.cam_max_contour
        self.trackingCamThread.invert_contrast = self.cam_invert_contrast

        time.sleep(1)
        self.trackingCamThread.start()

    def start_cam_recording(self, cam_frame):
        # in the tracking cam thread.run()
        pass

    def display_tracking_cam(self, frame):

        self.camBoxLabel.setPixmap(frame)

    def activate_cam_tracking_log(self, tracked_objects, expired_id_list, tracked_index, tracked_elapse):
        '''
        pass the list of registered object information to datalog thread
        '''

        self.dataLogThread.track_results(tracked_objects,
                                         expired_id_list,
                                         tracked_index,
                                         tracked_elapse)
        self.dataLogThread.start()

    def stop_cam_tracking(self):

        self.warning_msg = QMessageBox()
        self.warning_msg.setWindowTitle('Warning')
        self.warning_msg.setIcon(QMessageBox.Warning)
        self.warning_msg.setText('Do you want to stop tracking?')
        self.warning_msg.setStandardButtons(QMessageBox.Cancel | QMessageBox.Yes)
        returnValue = self.warning_msg.exec()

        if returnValue == QMessageBox.Yes:
            try:
                self.close_camera()
            except Exception as e:
                error = str(e)
                self.warning_msg = QMessageBox()
                self.warning_msg.setWindowTitle('Error')
                self.warning_msg.setText('An error happened when trying to stop tracking.')
                self.warning_msg.setInformativeText('stop_cam_tracking() does not execute correctly.')
                self.warning_msg.setIcon(QMessageBox.Warning)
                self.warning_msg.setDetailedText(error)
                self.warning_msg.exec()
            self.info_msg = QMessageBox()
            self.info_msg.setWindowTitle('TrackingBot')
            self.info_msg.setIcon(QMessageBox.Information)
            self.info_msg.setText('Tracking finished.')
            self.info_msg.exec()
            # allow export data when tracking finished
            self.exportCamData.setEnabled(True)

    def cam_exceed_index_alarm(self):
        '''
        cancel and reset tracking progress when object out of index at first frame
        :return:
        '''

        try:
            self.reload_camera()
            self.reset_cam_track_results()  # also reset all status flags

        except Exception as e:
            error = str(e)
            self.warning_msg = QMessageBox()
            self.warning_msg.setWindowTitle('Error')
            self.warning_msg.setText('An error happened during stop tracking.')
            self.warning_msg.setInformativeText('index_alarm() does not execute correctly.')
            self.warning_msg.setIcon(QMessageBox.Warning)
            self.warning_msg.setDetailedText(error)
            self.warning_msg.exec()

        finally:
            self.warning_msg = QMessageBox()
            self.warning_msg.setWindowTitle('TrackingBot')
            self.warning_msg.setIcon(QMessageBox.Warning)
            self.warning_msg.setText('Must ensure the detected object number in the first frame '
                                     'not exceed the set value.')
            self.warning_msg.exec()

    def reset_cam_track_results(self):
        '''
        reset all tracked data when requested to cancel the task
        :return:
        '''
        self.dataLogThread.df.clear()
        self.dataLogThread.df_archive.clear()
        self.dataLogThread.tracked_object = None
        self.dataLogThread.tracked_index = None
        self.dataLogThread.tracked_elapse = None
        self.trackingTimeStamp.result_index = -1
        self.trackingCamThread.trackingMethod.candidate_list.clear()
        self.trackingCamThread.trackingMethod.candidate_index = 0
        self.trackingCamThread.trackingMethod.candidate_id = 0
        self.trackingCamThread.trackingMethod.expired_id.clear()

    def export_cam_data(self):

        self.folder_path = QFileDialog.getExistingDirectory(None, 'Select Folder', 'C:/Users/Public/Documents')
        if self.folder_path == '':
            return
        else:
            try:
                # pass path
                self.camDataProcessDialog.data_save_path = self.folder_path
                self.camDataProcessDialog.object_num = self.cam_object_num
                # point thread
                self.camDataProcessDialog.dataLogThread = self.dataLogThread
                # start processing data and show progress bar
                self.camDataProcessDialog.setStart()
            except Exception as e:
                error = str(e)
                self.error_msg = QMessageBox()
                self.error_msg.setWindowTitle('Error')
                self.error_msg.setText('An error happened when trying to export tracking data.')
                self.error_msg.setInformativeText('export_cam_data() does not execute correctly.\n' + error)
                self.error_msg.setIcon(QMessageBox.Warning)
                self.error_msg.exec()

    def export_cam_data_success(self):
        self.info_msg = QMessageBox()
        self.info_msg.setWindowTitle('TrackingBot')
        self.info_msg.setIcon(QMessageBox.Information)
        self.info_msg.setText('Successfully saved data')
        self.info_msg.addButton('OK', QMessageBox.RejectRole)
        self.info_msg.addButton('Open folder', QMessageBox.AcceptRole)
        returnValue = self.info_msg.exec()
        if returnValue == 1:
            self.open_export_folder()
        else:
            return
    ###############################################Functions for hardware#################################

    def open_hardware_wizard(self):
        self.hardwareWizard.open_window()
        self.hardwareWizard.get_port()
        self.hardwareWizard.editCamROIButton.setEnabled(True)

    def edit_cam_roi(self):

        self.hardwareWizard.lineCamROIButton.setEnabled(True)
        self.hardwareWizard.rectCamROIButton.setEnabled(True)
        self.hardwareWizard.circCamROIButton.setEnabled(True)

        self.hardwareWizard.applyCamROIButton.setEnabled(True)

        if self.camROICanvas.scene.ROIs:
            self.hardwareWizard.resetCamROIButton.setEnabled(True)

        self.camROICanvas.setEnabled(True)
        self.camROICanvas.raise_()

    def set_cam_line_roi(self):

        # # highlight the line button and gray the rest
        self.hardwareWizard.lineCamROIButton.setProperty('Active', True)
        self.hardwareWizard.lineCamROIButton.setStyle(self.hardwareWizard.lineCamROIButton.style())
        self.hardwareWizard.rectCamROIButton.setProperty('Active', False)
        self.hardwareWizard.rectCamROIButton.setStyle(self.hardwareWizard.rectCamROIButton.style())
        self.hardwareWizard.circCamROIButton.setProperty('Active', False)
        self.hardwareWizard.circCamROIButton.setStyle(self.hardwareWizard.circCamROIButton.style())
        self.camROICanvas.scene.drawLine()

    def set_cam_rect_roi(self):
        self.hardwareWizard.lineCamROIButton.setProperty('Active', False)
        self.hardwareWizard.lineCamROIButton.setStyle(self.hardwareWizard.lineCamROIButton.style())
        self.hardwareWizard.rectCamROIButton.setProperty('Active', True)
        self.hardwareWizard.rectCamROIButton.setStyle(self.hardwareWizard.rectCamROIButton.style())
        self.hardwareWizard.circCamROIButton.setProperty('Active', False)
        self.hardwareWizard.circCamROIButton.setStyle(self.hardwareWizard.circCamROIButton.style())
        self.camROICanvas.scene.drawRect()

    def set_cam_circ_roi(self):
        self.hardwareWizard.lineCamROIButton.setProperty('Active', False)
        self.hardwareWizard.lineCamROIButton.setStyle(self.hardwareWizard.lineCamROIButton.style())
        self.hardwareWizard.rectCamROIButton.setProperty('Active', False)
        self.hardwareWizard.rectCamROIButton.setStyle(self.hardwareWizard.rectCamROIButton.style())
        self.hardwareWizard.circCamROIButton.setProperty('Active', True)
        self.hardwareWizard.circCamROIButton.setStyle(self.hardwareWizard.circCamROIButton.style())
        self.camROICanvas.scene.drawCirc()

    def apply_cam_roi(self):
        if not self.camROICanvas.scene.ROIs:
            self.hardwareWizard.editCamROIButton.setEnabled(True)
            self.hardwareWizard.applyCamROIButton.setEnabled(False)
            self.hardwareWizard.resetCamROIButton.setEnabled(False)
            self.hardwareWizard.lineCamROIButton.setEnabled(False)
            self.hardwareWizard.lineCamROIButton.setProperty('Active', False)
            self.hardwareWizard.lineCamROIButton.setStyle(self.hardwareWizard.lineCamROIButton.style())
            self.hardwareWizard.rectCamROIButton.setEnabled(False)
            self.hardwareWizard.rectCamROIButton.setProperty('Active', False)
            self.hardwareWizard.rectCamROIButton.setStyle(self.hardwareWizard.rectCamROIButton.style())
            self.hardwareWizard.circCamROIButton.setEnabled(False)
            self.hardwareWizard.circCamROIButton.setProperty('Active', False)
            self.hardwareWizard.circCamROIButton.setStyle(self.hardwareWizard.circCamROIButton.style())
            self.error_msg = QMessageBox()
            self.error_msg.setWindowTitle('TrackingBot')
            self.error_msg.setText('No valid ROI detected')
            self.error_msg.setInformativeText('To apply a ROI, please draw a shape.\n')
            self.error_msg.setIcon(QMessageBox.Warning)
            self.error_msg.setDetailedText('ROIs is empty. \n')
            self.error_msg.exec()

        else:
            for i in range(len(self.camROICanvas.scene.ROIs)):
                if self.camROICanvas.scene.ROIs[i].ROI.rect().isEmpty():
                    self.error_msg = QMessageBox()
                    self.error_msg.setWindowTitle('TrackingBot')
                    self.error_msg.setText('Invalid ROI detected')
                    self.error_msg.setInformativeText('The geometry of ROI is invalid, please draw a new shape.\n')
                    self.error_msg.setIcon(QMessageBox.Warning)
                    self.error_msg.setDetailedText('ROIs[i].ROI.rect().isEmpty(). \n')
                    self.error_msg.exec()

                    self.reset_cam_roi()
                    return
                else:
                     pass

            self.controllerThread.ROIs = self.camROICanvas.scene.ROIs
            self.controllerThread.create_roi()
            # self.apply_roi_flag

            self.camROICanvas.scene.clearSelection()
            self.camROICanvas.setEnabled(False)

            self.hardwareWizard.editCamROIButton.setEnabled(False)
            self.hardwareWizard.applyCamROIButton.setEnabled(False)
            self.hardwareWizard.resetCamROIButton.setEnabled(True)
            self.hardwareWizard.lineCamROIButton.setEnabled(False)
            self.hardwareWizard.lineCamROIButton.setProperty('Active', False)
            self.hardwareWizard.lineCamROIButton.setStyle(self.hardwareWizard.lineCamROIButton.style())
            self.hardwareWizard.rectCamROIButton.setEnabled(False)
            self.hardwareWizard.rectCamROIButton.setProperty('Active', False)
            self.hardwareWizard.rectCamROIButton.setStyle(self.hardwareWizard.rectCamROIButton.style())
            self.hardwareWizard.circCamROIButton.setEnabled(False)
            self.hardwareWizard.circCamROIButton.setProperty('Active', False)
            self.hardwareWizard.circCamROIButton.setStyle(self.hardwareWizard.circCamROIButton.style())

    def reset_cam_roi(self):
        # self.apply_roi_flag
        # reset canvas
        try:
            self.camROICanvas.scene.erase()
            self.controllerThread.ROIs = None
            self.controllerThread.ROI_zones.clear()
        except Exception as e:
            print(e)
        finally:
            self.camROICanvas.setEnabled(True)

            self.hardwareWizard.editCamROIButton.setEnabled(True)
            self.hardwareWizard.applyCamROIButton.setEnabled(False)
            self.hardwareWizard.resetCamROIButton.setEnabled(False)
            self.hardwareWizard.lineCamROIButton.setEnabled(False)
            self.hardwareWizard.lineCamROIButton.setProperty('Active', False)
            self.hardwareWizard.lineCamROIButton.setStyle(self.hardwareWizard.lineCamROIButton.style())
            self.hardwareWizard.rectCamROIButton.setEnabled(False)
            self.hardwareWizard.rectCamROIButton.setProperty('Active', False)
            self.hardwareWizard.rectCamROIButton.setStyle(self.hardwareWizard.rectCamROIButton.style())
            self.hardwareWizard.circCamROIButton.setEnabled(False)
            self.hardwareWizard.circCamROIButton.setProperty('Active', False)
            self.hardwareWizard.circCamROIButton.setStyle(self.hardwareWizard.circCamROIButton.style())

    def activate_controller_log(self, tracked_objects,expired_id_list,tracked_index,tracked_elapse):

        self.controllerThread.track_results(tracked_objects,
                                            expired_id_list,
                                            tracked_index,
                                            tracked_elapse)
        self.controllerThread.start()
        self.controllerThread.active_device = self.hardwareWizard.active_device


    #############################################################################################
    # Functions for other operations
    #############################################################################################

    def enable_calibration_help(self, event):

        QWhatsThis.enterWhatsThisMode()
        self.calibrationHelpLabel.setProperty('Active', True)
        self.calibrationHelpLabel.setStyle(self.calibrationHelpLabel.style())
        self.calibrationHelpLabel.setWhatsThis("<h2>Calibration</h2>"
                                               "<br></br>"
                                               "Calibration is to convert the object's coordinates expressed in pixels "
                                               "to real-world units."
                                               ""
                                               "<br></br>"
                                               "<h3>How to use</h3>"
                                               "Click <b>Draw scale</b> to initialize calibration."
                                               "<br></br>"
                                               "<br></br>"
                                               "On the video image, click and hold mouse to draw a scale line across "
                                               "an area or an object with known real-world distance, then release "
                                               "mouse."
                                               "<br></br>"
                                               "<br></br>"
                                               "In the pop-up <b>Input Scale window</b>, type in the real-world "
                                               "distance that the scale line represents, then click OK."
                                               "<br></br>"
                                               "<br></br>"
                                               "TrackingBot automatically converts the scale using the following "
                                               "equation: "
                                               "<br></br>"
                                               "<br></br>"
                                               "<center>Pixel scale = Input Distance / Pixel Distance</center>"
                                               "<br></br>"
                                               "<br></br>"
                                               "and display the result. For instance, Pixel scale is 0.1 means "
                                               "1 pixel in the frame equals 0.1 mm distance in real-world."
                                               "<br></br>"
                                               "<br></br>"
                                               "Click <b>Apply</b> to accept the value, or click <b>Reset</b> to "
                                               "re-draw the line."
                                               "<h3>Note</h3>"
                                               "<ul>"
                                               "<li>To obtain more accurate calibration results, avoid distortion"
                                               "in videos, make sure the camera is set perpendicular to plane of "
                                               "tracking area.</li>"
                                               "<br></br>"
                                               "<li>Varation in calibration results can lead to variation in "
                                               "calculated tracking results. However, it will not affect the raw "
                                               "data. </li>"
                                               "</ul>"
                                               "<br></br>"
                                               "For more details, please read TrackingBot document.")

    def disable_calibration_help(self, event):
        QWhatsThis.leaveWhatsThisMode()
        self.calibrationHelpLabel.setProperty('Active', False)
        self.calibrationHelpLabel.setStyle(self.calibrationHelpLabel.style())

    def enable_roi_help(self, event):
        QWhatsThis.enterWhatsThisMode()
        self.roiHelpLabel.setProperty('Active', True)
        self.roiHelpLabel.setStyle(self.roiHelpLabel.style())
        self.roiHelpLabel.setWhatsThis("<h2>Region of Interest(ROI)</h2>"
                                       "<br></br>"
                                       "A region of interest(ROI) is an area where "
                                       "only objects inside will be detected and tracked."
                                       "All activity outside an ROI will be ignored."
                                       "<br></br>"
                                       "<h3>How to use</h3>"
                                       "To define an ROI, click <b>Edit</b> to activate ROI toolbar,"
                                       "then <b>select a shape</b> and draw on the video preview "
                                       "to cover the area in which you want TrackingBot to track the object."
                                       "<br></br>"
                                       "<br></br>"
                                       "To move an ROI, click on the shape to select it, then keep the mouse "
                                       "button pressed to move it or use the arrow keys on the keyboard to shift "
                                       "it."
                                       "<br></br>"
                                       "<br></br>"
                                       "To resize an ROI, click on the shape to select it, then use the nodes to"
                                       "resize it."
                                       "<br></br>"
                                       "<br></br>"
                                       "To delete an ROI, click on the shape to select it, then left click on "
                                       "the shape and click <b>Delete</b> or press the <b>Delete</b> key on "
                                       "the keyboard. "
                                       "<br></br>"
                                       "<br></br>"
                                       "When finish, click <b>Apply</b> to confirm the setting."
                                       "<br></br>"
                                       "<h3>Note</h3>"
                                       "<ul>"
                                       "<li>Once applied current settings, ROIs become 'read-only' and you cannot "
                                       "operate on it. To edit again, click <b>Reset</b> to delete current settings "
                                       "and create from new.</li>"
                                       "<br></br>"
                                       "<li>Current version only support define single ROI.Define multiple ROIs will "
                                       "be supported in future upgrades.</li>"
                                       "</ul>"
                                       "<br></br>"
                                       "For more details, please read TrackingBot document.")

    def disable_roi_help(self, event):
        QWhatsThis.leaveWhatsThisMode()
        self.roiHelpLabel.setProperty('Active', False)
        self.roiHelpLabel.setStyle(self.roiHelpLabel.style())

    def enable_mask_help(self, event):
        QWhatsThis.enterWhatsThisMode()
        self.maskHelpLabel.setProperty('Active', True)
        self.maskHelpLabel.setStyle(self.maskHelpLabel.style())
        self.maskHelpLabel.setWhatsThis("<h2>Mask</h2>"
                                        "<br></br>"
                                        "A mask is used to exclude an area <i>inside an ROI</i> from being detected "
                                        "and tracked. All activity inside a mask will be ignored."
                                        "<br></br>"
                                        "<br></br>"
                                        "A mask should be placed with cautious and should be considered "
                                        "as the last resort to remove noise for following conditions:"
                                        "<ul>"
                                        "<li>The noise is of similar contrast and size with target "
                                        "object, therefore hard to be eliminated through thresholding and size filter."
                                        "</li>"
                                        "<br></br>"
                                        "<li>The noisy area is small relative to the total area to be tracked, "
                                        "exclude activities in mask area have neglectable effect on the final "
                                        "result.</li> "
                                        "</ul>"
                                        "<h3>How to use</h3>"
                                        "To define a mask, click <b>Edit</b> to activate mask toolbar,"
                                        "then <b>select a shape</b> and draw on the video image "
                                        "to cover the area in which you want TrackingBot to exclude from tracking."
                                        "<br></br>"
                                        "<br></br>"
                                        "<b><i>Important: </i></b> Mask must be placed nested inside an ROI."
                                        "<br></br>"
                                        "<br></br>"
                                        "To move a mask, click on the shape to select it, then keep the mouse "
                                        "button pressed to move it or use the arrow keys on the keyboard to shift it."
                                        "<br></br>"
                                        "<br></br>"
                                        "To resize a mask, click on the shape to select it, then use the nodes to"
                                        "resize it."
                                        "<br></br>"
                                        "<br></br>"
                                        "To delete a mask, click on the shape to select it, then left click on "
                                        "the shape and click <b>Delete</b> or press the <b>Delete</b> key on "
                                        "the keyboard. "
                                        "<br></br>"
                                        "<br></br>"
                                        "When finish, click <b>Apply</b> to confirm the setting."
                                        "<br></br>"
                                        "<h3>Note</h3>"
                                        "<ul>"
                                        "<li>Once applied current settings, masks become 'read-only' and you cannot "
                                        "operate on it. To edit again, click <b>Reset</b> to delete current settings "
                                        "and create from new.</li>"
                                        "<br></br>"
                                        "<li>You can define multiple masks inside one ROI.</li>"
                                        "</ul>"
                                        "<br></br>"
                                        "For more details, please read TrackingBot document.")

    def disable_mask_help(self, event):
        QWhatsThis.leaveWhatsThisMode()
        self.maskHelpLabel.setProperty('Active', False)
        self.maskHelpLabel.setStyle(self.maskHelpLabel.style())

    def enable_blocksize_help(self, event):
        QWhatsThis.enterWhatsThisMode()
        # mention threshold window, size filter
        self.blocksizeHelpLabel.setProperty('Active', True)
        self.blocksizeHelpLabel.setStyle(self.blocksizeHelpLabel.style())
        self.blocksizeHelpLabel.setWhatsThis("<h2>Threshold Strength</h2>"
                                             "<br></br>"
                                             "Threshold strength determines whether a pixel should be "
                                             "converted to white as an object or black as the background."
                                             "<br></br>"
                                             "<h3>How to use</h3>"
                                             "Play the video and move the slider or use the spin box to define a value."
                                             "<br></br>"
                                             "<br></br>"
                                             "Use the <b>Threshold Preview</b> to check the quality of thresholding, "
                                             "adjust the value until the object is fully appears in white and "
                                             "the noise is minimized. Ideally, only target object should be visible "
                                             "in the threshold preview window."
                                             "<br></br>"
                                             "<br></br>"
                                             "To a certain region on the image, the higher the threshold strength "
                                             "value, the more likely it will be classified as an object; the lower "
                                             "the threshold strength value, the less likely it will be classified "
                                             "as an object."
                                             "<br></br>"
                                             "<h3>Note</h3>"
                                             "<ul>"
                                             "<li>Make sure the parameter is optimal throughout the entire video, "
                                             "the changes during the video (i.e. varying illumination/camera position "
                                             "shifts) can cause settings no longer satisfying after change.</li>"
                                             "<br></br>"
                                             "<li>Always combine other parameters in finding the optimal thresholding "
                                             "range.</li>"
                                             "</ul>"
                                             "<br></br>"
                                             "For more details, please read TrackingBot document.")

    def disable_blocksize_help(self, event):
        QWhatsThis.leaveWhatsThisMode()
        self.blocksizeHelpLabel.setProperty('Active', False)
        self.blocksizeHelpLabel.setStyle(self.blocksizeHelpLabel.style())

    def enable_offset_help(self, event):
        QWhatsThis.enterWhatsThisMode()
        self.offsetHelpLabel.setProperty('Active', True)
        self.offsetHelpLabel.setStyle(self.offsetHelpLabel.style())
        # offset determines where to set the threshold relative to the neighbourhood mean
        self.offsetHelpLabel.setWhatsThis("<h2>Noise Filter Strength</h2>"
                                          "<br></br>"
                                          "Noise filter strength determines an offset range relative to the thresh, "
                                          "pixels with intensity fall in this range will be classified as noise and "
                                          "converted to black as background."
                                          "<br></br>"
                                          "<h3>How to use</h3>"
                                          "Play the video and move the slider or use the spin box to define a value."
                                          "<br></br>"
                                          "<br></br>"
                                          "Use the <b>Threshold Preview</b> to check the quality of noise filtering, "
                                          "adjust the value until the object is fully appears in white and "
                                          "the noise is minimized. Ideally, only target object should be visible "
                                          "in the threshold preview window."
                                          "<br></br>"
                                          "<br></br>"
                                          "Under the same threshold strength, the higher the noise filter strength "
                                          "value, the more likely more regions will be classified as noise than an "
                                          "object; the lower the noise filter strength value, the less likely more "
                                          "regions will be classified as noise than an object."
                                          "<br></br>"
                                          "<h3>Note</h3>"
                                          "<ul>"
                                          "<li>Make sure the parameter is optimal throughout the entire video, "
                                          "the changes during the video (i.e. varying illumination/camera position "
                                          "shifts) can cause settings no longer satisfying after change.</li>"
                                          "<br></br>"
                                          "<li>Always combine other parameters in finding the optimal thresholding "
                                          "range.</li>"
                                          "</ul>"
                                          "<br></br>"
                                          "For more details, please read TrackingBot document.")

    def disable_offset_help(self, event):
        QWhatsThis.leaveWhatsThisMode()
        self.offsetHelpLabel.setProperty('Active', False)
        self.offsetHelpLabel.setStyle(self.offsetHelpLabel.style())

    def enable_size_help(self, event):
        QWhatsThis.enterWhatsThisMode()
        self.objectsizeHelpLabel.setProperty('Active', True)
        self.objectsizeHelpLabel.setStyle(self.objectsizeHelpLabel.style())
        self.objectsizeHelpLabel.setWhatsThis("<h2>Object Size</h2>"
                                              "<br></br>"
                                              "Object size determines the detection range. Objects smaller "
                                              "or larger than the set size range will be excluded from detection."
                                              "<br></br>"
                                              "<h3>How to use</h3>"
                                              "<b>Max</b> slider bar and spin box corresponding to the <b>Maximum "
                                              "size</b> (in pixels) of the object, <b>Min</b> slider bar and spin box "
                                              "corresponding to the <b>Minimum size</b> (in pixels) of the object. "
                                              "<br></br>"
                                              "<br></br>"
                                              "Play the video and move the slider or use the spin box to adjust the "
                                              "min and max value."
                                              "<br></br>"
                                              "<br></br>"
                                              "When object size meet the detection range, a thin red contour around "
                                              "object will show up in the video, indicate the object is successfully "
                                              "detected. Adjust the range to ensure all target objects are detected "
                                              "and no noise are detected. Ideally, the shape of contour shape should "
                                              "fully represent the shape of target object."
                                              "<br></br>"
                                              "<h3>Note</h3>"
                                              "<ul>"
                                              "<li>Make sure the object can be detected throughout the entire video, "
                                              "the changes during the video (i.e. varying illumination/camera position "
                                              "shifts) can cause settings no longer satisfying after change.</li>"
                                              "<br></br>"
                                              "<li>Effective detection largely depends on the thresholding quality. "
                                              "Always combine threshold parameters in finding the optimal detection"
                                              "range.</li>"
                                              "</ul>"
                                              "<br></br>"
                                              "For more details, please read TrackingBot document.")

    def disable_size_help(self, event):
        QWhatsThis.leaveWhatsThisMode()
        self.objectsizeHelpLabel.setProperty('Active', False)
        self.objectsizeHelpLabel.setStyle(self.objectsizeHelpLabel.style())

    def enable_trace_help(self, event):
        QWhatsThis.enterWhatsThisMode()
        self.traceHelpLabel.setProperty('Active', True)
        self.traceHelpLabel.setStyle(self.traceHelpLabel.style())
        self.traceHelpLabel.setWhatsThis("<h2>Trace Visualization</h2>"
                                             "<br></br>"
                                             "Trace visualization generate locomotion trajectory for each tracked "
                                             "object of current tracking task."
                                             "<br></br>"
                                             "<br></br>"
                                             "Trace map helps quick inspection of the tracking result."
                                             "<h3>How to use</h3>"
                                             "After tracking task finished, click <b>Trace Visualization </b> toggle"
                                             "to generate trajectory preview."
                                             "<br></br>"
                                             "<br></br>"
                                             "Click <b>Export Trace</b> to save trajectory map as a png image file."
                                             "<br></br>"
                                             "<br></br>"
                                             "For more details, please read TrackingBot document.")

    def disable_trace_help(self, event):
        QWhatsThis.enterWhatsThisMode()
        self.traceHelpLabel.setProperty('Active', False)
        self.traceHelpLabel.setStyle(self.traceHelpLabel.style())

    def enable_heatmap_help(self, event):
        QWhatsThis.enterWhatsThisMode()
        self.heatmapHelpLabel.setProperty('Active', True)
        self.heatmapHelpLabel.setStyle(self.heatmapHelpLabel.style())
        self.heatmapHelpLabel.setWhatsThis("<h2>Heatmap Visualization</h2>"
                                             "<br></br>"
                                             "Heatmap visualization generate heat map for current tracking task. The "
                                             "heat map is a graphical representation of the object's position "
                                             "distribution, in which the frequency of a specific position is "
                                             "represented as a color."
                                             "<br></br>"
                                             "<br></br>"
                                             "Heat map helps find the 'hotspots' of activities and clustering of data "
                                             "points."
                                             "<h3>How to use</h3>"
                                             "After tracking task finished, click <b>Heatmap Visualization </b> toggle"
                                             "to generate heatmap preview."
                                             "<br></br>"
                                             "<br></br>"
                                             "Click <b>Export Heatmap</b> to save heatmap as a png image file."
                                             "<br></br>"
                                             "<br></br>"
                                             "For more details, please read TrackingBot document.")

    def disable_heatmap_help(self, event):
        QWhatsThis.enterWhatsThisMode()
        self.heatmapHelpLabel.setProperty('Active', False)
        self.heatmapHelpLabel.setStyle(self.heatmapHelpLabel.style())

    def enable_cam_blocksize_help(self, event):
        QWhatsThis.enterWhatsThisMode()
        # mention threshold window, size filter
        self.camblocksizeHelpLabel.setProperty('Active', True)
        self.camblocksizeHelpLabel.setStyle(self.camblocksizeHelpLabel.style())
        self.camblocksizeHelpLabel.setWhatsThis("<h2>Threshold Strength</h2>"
                                             "<br></br>"
                                             "Threshold strength determines whether a pixel should be "
                                             "converted to white as an object or black as the background."
                                             "<br></br>"
                                             "<h3>How to use</h3>"
                                             "Play the video and move the slider or use the spin box to define a value."
                                             "<br></br>"
                                             "<br></br>"
                                             "Use the <b>Threshold Preview</b> to check the quality of thresholding, "
                                             "adjust the value until the object is fully appears in white and "
                                             "the noise is minimized. Ideally, only target object should be visible "
                                             "in the threshold preview window."
                                             "<br></br>"
                                             "<br></br>"
                                             "To a certain region on the image, the higher the threshold strength "
                                             "value, the more likely it will be classified as an object; the lower "
                                             "the threshold strength value, the less likely it will be classified "
                                             "as an object."
                                             "<br></br>"
                                             "<h3>Note</h3>"
                                             "<ul>"
                                             "<li>Make sure the parameter is optimal throughout the entire video, "
                                             "the changes during the video (i.e. varying illumination/camera position "
                                             "shifts) can cause settings no longer satisfying after change.</li>"
                                             "<br></br>"
                                             "<li>Always combine other parameters in finding the optimal thresholding "
                                             "range.</li>"
                                             "</ul>"
                                             "<br></br>"
                                             "For more details, please read TrackingBot document.")

    def disable_cam_blocksize_help(self, event):
        QWhatsThis.leaveWhatsThisMode()
        self.camblocksizeHelpLabel.setProperty('Active', False)
        self.camblocksizeHelpLabel.setStyle(self.camblocksizeHelpLabel.style())

    def enable_cam_offset_help(self, event):
        QWhatsThis.enterWhatsThisMode()
        self.camoffsetHelpLabel.setProperty('Active', True)
        self.camoffsetHelpLabel.setStyle(self.camoffsetHelpLabel.style())
        # offset determines where to set the threshold relative to the neighbourhood mean
        self.camoffsetHelpLabel.setWhatsThis("<h2>Noise Filter Strength</h2>"
                                          "<br></br>"
                                          "Noise filter strength determines an offset range relative to the thresh, "
                                          "pixels with intensity fall in this range will be classified as noise and "
                                          "converted to black as background."
                                          "<br></br>"
                                          "<h3>How to use</h3>"
                                          "Play the video and move the slider or use the spin box to define a value."
                                          "<br></br>"
                                          "<br></br>"
                                          "Use the <b>Threshold Preview</b> to check the quality of noise filtering, "
                                          "adjust the value until the object is fully appears in white and "
                                          "the noise is minimized. Ideally, only target object should be visible "
                                          "in the threshold preview window."
                                          "<br></br>"
                                          "<br></br>"
                                          "Under the same threshold strength, the higher the noise filter strength "
                                          "value, the more likely more regions will be classified as noise than an "
                                          "object; the lower the noise filter strength value, the less likely more "
                                          "regions will be classified as noise than an object."
                                          "<br></br>"
                                          "<h3>Note</h3>"
                                          "<ul>"
                                          "<li>Make sure the parameter is optimal throughout the entire video, "
                                          "the changes during the video (i.e. varying illumination/camera position "
                                          "shifts) can cause settings no longer satisfying after change.</li>"
                                          "<br></br>"
                                          "<li>Always combine other parameters in finding the optimal thresholding "
                                          "range.</li>"
                                          "</ul>"
                                          "<br></br>"
                                          "For more details, please read TrackingBot document.")

    def disable_cam_offset_help(self, event):
        QWhatsThis.leaveWhatsThisMode()
        self.camoffsetHelpLabel.setProperty('Active', False)
        self.camoffsetHelpLabel.setStyle(self.camoffsetHelpLabel.style())

    def enable_cam_size_help(self, event):
        QWhatsThis.enterWhatsThisMode()
        self.camobjectsizeHelpLabel.setProperty('Active', True)
        self.camobjectsizeHelpLabel.setStyle(self.camobjectsizeHelpLabel.style())
        self.camobjectsizeHelpLabel.setWhatsThis("<h2>Object Size</h2>"
                                              "<br></br>"
                                              "Object size determines the detection range. Objects smaller "
                                              "or larger than the set size range will be excluded from detection."
                                              "<br></br>"
                                              "<h3>How to use</h3>"
                                              "<b>Max</b> slider bar and spin box corresponding to the <b>Maximum "
                                              "size</b> (in pixels) of the object, <b>Min</b> slider bar and spin box "
                                              "corresponding to the <b>Minimum size</b> (in pixels) of the object. "
                                              "<br></br>"
                                              "<br></br>"
                                              "Play the video and move the slider or use the spin box to adjust the "
                                              "min and max value."
                                              "<br></br>"
                                              "<br></br>"
                                              "When object size meet the detection range, a thin red contour around "
                                              "object will show up in the video, indicate the object is successfully "
                                              "detected. Adjust the range to ensure all target objects are detected "
                                              "and no noise are detected. Ideally, the shape of contour shape should "
                                              "fully represent the shape of target object."
                                              "<br></br>"
                                              "<h3>Note</h3>"
                                              "<ul>"
                                              "<li>Make sure the object can be detected throughout the entire video, "
                                              "the changes during the video (i.e. varying illumination/camera position "
                                              "shifts) can cause settings no longer satisfying after change.</li>"
                                              "<br></br>"
                                              "<li>Effective detection largely depends on the thresholding quality. "
                                              "Always combine threshold parameters in finding the optimal detection"
                                              "range.</li>"
                                              "</ul>"
                                              "<br></br>"
                                              "For more details, please read TrackingBot document.")

    def disable_cam_size_help(self, event):
        QWhatsThis.leaveWhatsThisMode()
        self.camobjectsizeHelpLabel.setProperty('Active', False)
        self.camobjectsizeHelpLabel.setStyle(self.camobjectsizeHelpLabel.style())


class DataProcessDialog(QDialog):

    def __init__(self):
        QDialog.__init__(self)
        self.init_UI()
        self.timesignal = Communicate()
        self.dataLogThread = None
        self.dataExportThread = DataExportThread()
        self.video_fps = None
        self.object_num = None
        self.pixel_per_metric = None
        self.data_save_path = None
        self.graph_save_path = None
        # self.dataExportThread.timesignal.progressStart.connect(self.setStart) # 0%
        self.dataExportThread.timesignal.data_process_fin.connect(self.setFinish) #100%

    def init_UI(self):
        self.setWindowTitle('TrackingBot')
        self.layout = QVBoxLayout()
        self.layout.setAlignment(Qt.AlignCenter)
        message = QLabel(self)
        message.setText('Export data, please wait...')
        message.setStyleSheet("font-size: 14px;qproperty-alignment:AlignCenter;")
        message.setAlignment(Qt.AlignCenter)
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setGeometry(0, 30, 300, 25)
        self.progress_bar.setAlignment(Qt.AlignCenter)
        self.progress_bar.setRange(0, 1)
        self.layout.addWidget(message)
        self.layout.addWidget(self.progress_bar)
        self.setLayout(self.layout)

    def setStart(self):
        self.setModal(True)
        self.show()
        self.dataExportThread.dataLogThread = self.dataLogThread
        self.dataExportThread.data_save_path = self.data_save_path
        self.dataExportThread.video_fps = self.video_fps
        self.dataExportThread.object_num = self.object_num
        self.dataExportThread.pixel_per_metric = self.pixel_per_metric
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setValue(0)
        # start process data
        self.dataExportThread.start()

    def setFinish(self):
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(1)
        time.sleep(1)
        # close progress bar and call success dialog
        self.close()
        self.timesignal.data_export_finish.emit('1')


class CamDataProcessDialog(QDialog):

    def __init__(self):
        QDialog.__init__(self)
        self.init_UI()
        self.timesignal = Communicate()
        self.dataLogThread = None
        self.camDataExportThread = CamDataExportThread()
        self.object_num = None
        self.data_save_path = None
        # self.dataExportThread.timesignal.progressStart.connect(self.setStart) # 0%
        self.camDataExportThread.timesignal.data_process_fin.connect(self.setFinish) #100%

    def init_UI(self):
        self.setWindowTitle('TrackingBot')
        self.layout = QVBoxLayout()
        self.layout.setAlignment(Qt.AlignCenter)
        message = QLabel(self)
        message.setText('Export data, please wait...')
        message.setStyleSheet("font-size: 14px;qproperty-alignment:AlignCenter;")
        message.setAlignment(Qt.AlignCenter)
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setGeometry(0, 30, 300, 25)
        self.progress_bar.setAlignment(Qt.AlignCenter)
        self.progress_bar.setRange(0, 1)
        self.layout.addWidget(message)
        self.layout.addWidget(self.progress_bar)
        self.setLayout(self.layout)

    def setStart(self):
        self.setModal(True)
        self.show()
        self.camDataExportThread.dataLogThread = self.dataLogThread
        self.camDataExportThread.data_save_path = self.data_save_path
        self.camDataExportThread.object_num = self.object_num
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setValue(0)
        # start process data
        self.camDataExportThread.start()

    def setFinish(self):
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(1)
        time.sleep(1)
        # close progress bar and call success dialog
        self.close()
        self.timesignal.cam_data_export_finish.emit('1')


class TraceProcessDialog(QDialog):

    def __init__(self):
        QDialog.__init__(self)
        self.init_UI()
        self.timesignal = Communicate()
        self.dataLogThread = None
        self.traceExportThread = TraceExportThread()
        self.trace_frame = None
        self.video_prop = None

        # self.traceExportThread.timesignal.progressStart.connect(self.setStart) # 0%
        self.traceExportThread.timesignal.trace_process_fin.connect(self.setFinish) #100%
        # pass to main
        self.traceExportThread.timesignal.trace_map.connect(self.display_trace_map)  # emit QPixmap
        self.traceExportThread.timesignal.trace_map_raw.connect(self.raw_trace_map)

    def init_UI(self):
        self.setWindowTitle('TrackingBot')
        self.layout = QVBoxLayout()
        self.layout.setAlignment(Qt.AlignCenter)
        message = QLabel(self)
        message.setText('Generating trajectory, please wait...')
        message.setStyleSheet("font-size: 14px;qproperty-alignment:AlignCenter;")
        message.setAlignment(Qt.AlignCenter)
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setGeometry(0, 30, 300, 25)
        self.progress_bar.setAlignment(Qt.AlignCenter)
        self.progress_bar.setRange(0, 1)
        self.layout.addWidget(message)
        self.layout.addWidget(self.progress_bar)
        self.setLayout(self.layout)

    def setStart(self):
        self.setModal(True)
        self.show()
        self.traceExportThread.dataLogThread = self.dataLogThread
        self.traceExportThread.trace_frame = self.trace_frame
        self.traceExportThread.video_prop = self.video_prop
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setValue(0)
        # start process data
        self.traceExportThread.start()

    def setFinish(self):
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(1)
        time.sleep(1)
        # close progress bar
        self.close()

    def display_trace_map(self, trace_map):
        self.timesignal.trace_map.emit(trace_map)

    def raw_trace_map(self, raw_trace_map):
        self.timesignal.raw_trace_map.emit(raw_trace_map)


class GraphProcessDialog(QDialog):

    def __init__(self):
        QDialog.__init__(self)
        self.init_UI()
        self.timesignal = Communicate()
        self.dataLogThread = None
        self.video_prop = None
        self.graphExportThread = GraphExportThread()

        self.graphExportThread.timesignal.graph_process_fin.connect(self.setFinish) #100%
        self.graphExportThread.timesignal.heat_map.connect(self.display)  # emit object

    def init_UI(self):
        self.setWindowTitle('TrackingBot')
        self.layout = QVBoxLayout()
        self.layout.setAlignment(Qt.AlignCenter)
        message = QLabel(self)
        message.setText('Generating heatmap, please wait...')
        message.setStyleSheet("font-size: 14px;qproperty-alignment:AlignCenter;")
        message.setAlignment(Qt.AlignCenter)
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setGeometry(0, 30, 300, 25)
        self.progress_bar.setAlignment(Qt.AlignCenter)
        self.progress_bar.setRange(0, 1)
        self.layout.addWidget(message)
        self.layout.addWidget(self.progress_bar)
        self.setLayout(self.layout)

    def setStart(self):
        self.setModal(True)
        self.show()
        self.graphExportThread.dataLogThread = self.dataLogThread
        self.graphExportThread.video_prop = self.video_prop

        self.progress_bar.setRange(0, 0)
        self.progress_bar.setValue(0)
        # start process data
        self.graphExportThread.start()

    def setFinish(self):
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(1)
        time.sleep(1)
        # close progress bar
        self.close()

    def display(self,heat_map):
        self.timesignal.heat_map.emit(heat_map)


class Communicate(QObject):
    # cam_signal = pyqtSignal(QImage)
    data_export_finish = pyqtSignal(str)
    cam_data_export_finish = pyqtSignal(str)
    trace_export_finish = pyqtSignal(str)
    trace_map = pyqtSignal(QPixmap)
    raw_trace_map = pyqtSignal(object)
    heat_map = pyqtSignal(object)


class HardwareWizard(QtWidgets.QMainWindow,Ui_HardwireWizardWindow):
    def __init__(self):
        super(HardwareWizard,self).__init__()
        self.setupUi(self)
        self.disconnectPort.setEnabled(False)
        self.active_device = None
        self.comboBox.installEventFilter(self)
        self.comboBox.currentIndexChanged.connect(self.change_port)
        self.connectPort.clicked.connect(self.connect_port)
        self.disconnectPort.clicked.connect(self.disconnect_port)

        self.gateOpenButton.clicked.connect(self.gate_open)
        self.gateCloseButton.clicked.connect(self.gate_close)
        self.lineCamROIButton.setIcon(QtGui.QIcon('icon/line.png'))
        self.lineCamROIButton.setIconSize(QtCore.QSize(25, 25))
        self.rectCamROIButton.setIcon(QtGui.QIcon('icon/rectangle.png'))
        self.rectCamROIButton.setIconSize(QtCore.QSize(25, 25))
        self.circCamROIButton.setIcon(QtGui.QIcon('icon/circle.png'))
        self.circCamROIButton.setIconSize(QtCore.QSize(24, 24))

    def open_window(self):
        self.show()

    def get_port(self):
        self.comboBox.addItem('')
        ports = serial.tools.list_ports.comports()
        available_ports = []

        for p in ports:
            available_ports.append([p.description, p.device])
            # print(str(p.description)) # device name + port name
            # print(str(p.device)) # port name

        for info in available_ports:
            self.comboBox.addItem(info[0])

        print(available_ports)
        return available_ports

    def change_port(self):
        # 1st empty line
        selected_port_index = self.comboBox.currentIndex()-1
        return selected_port_index

    def connect_port(self):
        available_ports = self.get_port()
        selected_port_index = self.change_port()
        print(selected_port_index)

        if available_ports and selected_port_index != -1:
            try:
                # portOpen = True
                self.active_device = serial.Serial(available_ports[selected_port_index][1], 19200, timeout=1)
                # print(f'Connected to port {available_ports[selected_port_index][1]}!')
                time.sleep(0.5)
                # thread start
                print(self.active_device.isOpen())
                self.comboBox.setEnabled(False)
                self.connectPort.setEnabled(False)
                self.disconnectPort.setEnabled(True)
                # print(f'Connected to port {available_ports[selected_port_index][1]}!')
            except Exception as e:
                error = str(e)
                self.error_msg = QMessageBox()
                self.error_msg.setWindowTitle('Error')
                self.error_msg.setText('Cannot connect to selected port.')
                self.error_msg.setInformativeText('Please select a valid port')
                self.error_msg.setIcon(QMessageBox.Warning)
                self.error_msg.setDetailedText(error)
                self.error_msg.exec()

        elif not available_ports:
            self.error_msg = QMessageBox()
            self.error_msg.setWindowTitle('Error')
            self.error_msg.setText('Cannot read available port.')
            self.error_msg.setInformativeText('Please try reload port list by click refresh button .')
            self.error_msg.setIcon(QMessageBox.Warning)
            self.error_msg.exec()

        elif selected_port_index == -1:
            self.error_msg = QMessageBox()
            self.error_msg.setWindowTitle('Error')
            self.error_msg.setText('Please select a valid port.')
            self.error_msg.setInformativeText('selected_port_index is empty.')
            self.error_msg.setIcon(QMessageBox.Warning)
            self.error_msg.exec()

    def disconnect_port(self):
        try:
            self.active_device.close()

        except Exception as e:
            error = str(e)
            self.error_msg = QMessageBox()
            self.error_msg.setWindowTitle('Error')
            self.error_msg.setText('Cannot disconnect from selected port.')
            self.error_msg.setInformativeText('disconnect_port() failed.')
            self.error_msg.setIcon(QMessageBox.Warning)
            self.error_msg.setDetailedText(error)
            self.error_msg.exec()

        if not self.active_device.isOpen():
            print('Connection with port closed')
            print(self.active_device.isOpen())
            # thread stop
            self.comboBox.clear()
            self.comboBox.setEnabled(True)
            self.connectPort.setEnabled(True)
            self.disconnectPort.setEnabled(False)

    def refresh_port(self):
        self.comboBox.clear()
        self.get_port()

    def eventFilter(self,target,event):
        if target == self.comboBox and event.type() == QtCore.QEvent.MouseButtonPress:
            self.refresh_port()

        return False

    def gate_open(self):
        self.active_device.write(f'61'.encode())

    def gate_close(self):
        self.active_device.write(f'60'.encode())


class ControllerThread(QThread):

    def __init__(self):
        QThread.__init__(self)
        self.stopped = False
        self.active_device = None
        self.mutex = QMutex()
        self.ROIs = None
        self.ROI_zones = []
        self.tracked_object = None
        self.expired_id_list = None
        self.tracked_index = None
        self.tracked_elapse = None

    def run(self):

        with QMutexLocker(self.mutex):
            self.stopped = False

        if self.stopped:
            return

        else:
            tic = time.perf_counter()

            if self.ROI_zones:
                # tracked coords on frame, roi coords on canvas, need convert

                for i in range(len(self.tracked_object)):
                    for j in range(len(self.ROI_zones)):
                        condition = self.ROI_zones[j].rect.contains(self.tracked_object[i].pos_prediction[0][0]/1.875,
                                                 self.tracked_object[i].pos_prediction[1][0]/1.875)
                        if condition and self.ROI_zones[j].state is False: # if condition and state is false
                            self.active_device.write(f'{j}1'.encode())
                            self.ROI_zones[j].state = True # change state
                        else:
                            pass
            else:
                # print('No zone')
                pass

            toc = time.perf_counter()

    def stop(self):
        with QMutexLocker(self.mutex):
            self.stopped = True

    def is_stopped(self):
        with QMutexLocker(self.mutex):
            return self.stopped

    def track_results(self,tracked_object,expired_id_list,tracked_index,tracked_elapse):
        '''
        receive the list of registered object information;
        the list of expired id number;
        receive the index of timestamp;
        video time elapsed when time stamp is true
        passed from tracking thread
        '''
        self.tracked_object = tracked_object
        self.expired_id_list = expired_id_list
        self.tracked_index = tracked_index
        self.tracked_elapse = tracked_elapse

    def create_roi(self):
        '''
        extract QGraphicItem from ROI objects
        to read its coordinates
        '''
        for i in range(len(self.ROIs)):
            roi_zone = Indicator(self.ROIs[i].ROI.rect(),False)
            self.ROI_zones.append(roi_zone)


class Indicator(object):
    '''
    Define the ROI object and its properties
    '''
    def __init__(self, rect, state):
        self.rect = rect
        self.state = state


if __name__ == "__main__":
    import traceback
    import sys
    from PyQt5.QtWidgets import  QMessageBox
    sys._excepthook = sys.excepthook

    def exception_hook(exctype, value, traceback):

        traceback_formated = traceback.format_exception(exctype, value, traceback)
        traceback_string = "".join(traceback_formated)
        print(traceback_string, file=sys.stderr)
        error = str(traceback_string)
        crash_msg = QMessageBox()
        crash_msg.setWindowTitle('Error')
        crash_msg.setText('TrackingBot stopped working due to an unexpected error.')
        crash_msg.setIcon(QMessageBox.Warning)
        crash_msg.setDetailedText(error)
        crash_msg.exec()
        timeX = str(time.time())
        with open('C:/Users/Public/Documents/CRASH-' + timeX + '.txt', 'w') as crashLog:
            for i in traceback_string:
                i = str(i)
                crashLog.write(i)
        sys.exit(0)


    sys.excepthook = exception_hook

    app = QtWidgets.QApplication(sys.argv)
    # show launching screen
    splash_pix = QPixmap('icon/splash_screen.png')
    splash = QSplashScreen(splash_pix, Qt.WindowStaysOnTopHint)
    splash.setMask(splash_pix.mask())
    splash.show()

    time.sleep(2)
    # connect subclass with parent class
    window = MainWindow()
    app.setStyleSheet((open('stylesheet.qss').read()))
    app.processEvents()
    window.show()
    splash.finish(window)
    sys.exit(app.exec_())
