# -*- coding: utf-8 -*-

# TrackingBot - A software for video-based animal behavioral tracking and analysis
# Developer: Yutao Bai <hitomiona@gmail.com>
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

import cv2
import numpy as np
import time
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QObject, QMutex, QMutexLocker
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMessageBox
from tracker import TrackingMethod
from datalog import TrackingTimeStamp
from datetime import datetime, timedelta


class TrackingThread(QThread):

    def __init__(self, default_fps=25):
        QThread.__init__(self)
        self.file = ''
        self.obj_num = 1  # default 1
        self.timeSignal = Communicate()
        self.mutex = QMutex()
        self.detection = Detection()
        # obj_num, dist_thresh, max_lost_frames, max_trace_len
        self.trackingMethod = TrackingMethod(self.obj_num, 15, 60, 600)
        self.trackingMethod.timeSignal.index_alarm.connect(self.index_alarm)
        self.trackingTimeStamp = TrackingTimeStamp()
        self.playCapture = cv2.VideoCapture()
        self.video_prop = None
        self.interpolation_flag = cv2.INTER_AREA
        self.scale_aspect = 'widescreen'

        self.stopped = False
        self.fps = default_fps
        self.frame_count = -1  # first frame start from 0
        self.video_elapse = 0
        self.is_timeStamp = False

        self.block_size = 11
        self.offset = 11
        self.min_contour = 1  # default 1
        self.max_contour = 100  # default 100
        self.valid_mask = None

        self.id_list = list(range(1, 100))
        self.obj_id = [format(x, '01d') for x in self.id_list]

        self.invert_contrast = False
        self.apply_roi_flag = False
        self.apply_mask_flag = False

    def run(self):

        with QMutexLocker(self.mutex):
            self.stopped = False

        if self.video_prop.width > 1024:  # shrink
            self.interpolation_flag = cv2.INTER_AREA
        elif self.video_prop.width < 1024:  # enlarge
            self.interpolation_flag = cv2.INTER_LINEAR

        if self.video_prop.height / self.video_prop.width == 0.75:
            self.scale_aspect = 'classic'
        else:
            self.scale_aspect = 'widescreen'

        while True:

            if self.stopped:
                return
            else:
                ret, frame = self.playCapture.read()

                if ret:

                    tic = time.perf_counter()

                    # current position in milliseconds
                    pos_elapse = self.playCapture.get(cv2.CAP_PROP_POS_MSEC)
                    # current position calculated using current frame/fps
                    play_elapse = self.playCapture.get(cv2.CAP_PROP_POS_FRAMES) / self.playCapture.get(cv2.CAP_PROP_FPS)

                    self.timeSignal.updateSliderPos.emit(play_elapse)

                    self.frame_count += 1

                    self.is_timeStamp, self.video_elapse = self.trackingTimeStamp.local_time_stamp(pos_elapse,
                                                                                                   interval=None)

                    if self.invert_contrast:
                        # brighter object, dark background
                        invert_frame = cv2.bitwise_not(frame)

                        if self.apply_roi_flag or self.apply_mask_flag:
                            # if roi defined, apply the mask
                            masked_frame = cv2.bitwise_and(invert_frame, invert_frame, mask=self.valid_mask)

                            thre_frame = self.detection.thresh_video(masked_frame,
                                                                     self.block_size,
                                                                     self.offset)

                            contour_frame, entrant_detected = self.detection.detect_contours(frame,
                                                                                             thre_frame,
                                                                                             self.min_contour,
                                                                                             self.max_contour)

                            self.trackingMethod.identify(entrant_detected, self.min_contour, self.max_contour)

                            # mark indentity of each objects
                            self.trackingMethod.visualize(contour_frame, is_centroid=True,
                                                          is_mark=True, is_trajectory=True)

                            # pass tracking data to datalog thread when local tracking
                            if self.is_timeStamp:
                                self.timeSignal.track_results.emit(self.trackingMethod.candidate_list,
                                                                   self.trackingMethod.expired_id,
                                                                   self.trackingTimeStamp.result_index,
                                                                   self.video_elapse)

                            # scale threshlded frame to match the display window and roi/mask canvas
                            scaled_frame = self.scale_frame(contour_frame, self.interpolation_flag,
                                                            self.scale_aspect)
                            display_frame = self.convert_frame(scaled_frame)

                            # connected to MainWindow.display_tracking_video
                            self.timeSignal.tracking_signal.emit(display_frame)  # QPixmap

                        if not self.apply_roi_flag and not self.apply_mask_flag:

                            thre_frame = self.detection.thresh_video(invert_frame,
                                                                     self.block_size,
                                                                     self.offset)

                            contour_frame, entrant_detected = self.detection.detect_contours(frame,
                                                                                             thre_frame,
                                                                                             self.min_contour,
                                                                                             self.max_contour)

                            self.trackingMethod.identify(entrant_detected, self.min_contour, self.max_contour)

                            self.trackingMethod.visualize(contour_frame, is_centroid=True,
                                                          is_mark=True, is_trajectory=True)

                            if self.is_timeStamp:
                                self.timeSignal.track_results.emit(self.trackingMethod.candidate_list,
                                                                   self.trackingMethod.expired_id,
                                                                   self.trackingTimeStamp.result_index,
                                                                   self.video_elapse)

                            # scale threshlded frame to match the display window and roi/mask canvas
                            scaled_frame = self.scale_frame(contour_frame, self.interpolation_flag,
                                                            self.scale_aspect)
                            display_frame = self.convert_frame(scaled_frame)

                            # connected to MainWindow.display_tracking_video
                            self.timeSignal.tracking_signal.emit(display_frame)  # QPixmap

                    elif not self.invert_contrast:

                        if self.apply_roi_flag or self.apply_mask_flag:
                            # if roi defined, apply the mask
                            masked_frame = cv2.bitwise_and(frame, frame, mask=self.valid_mask)

                            thre_frame = self.detection.thresh_video(masked_frame,
                                                                     self.block_size,
                                                                     self.offset)

                            contour_frame, entrant_detected = self.detection.detect_contours(frame,
                                                                                             thre_frame,
                                                                                             self.min_contour,
                                                                                             self.max_contour)

                            self.trackingMethod.identify(entrant_detected, self.min_contour, self.max_contour)

                            self.trackingMethod.visualize(contour_frame, is_centroid=True,
                                                          is_mark=True, is_trajectory=True)

                            if self.is_timeStamp:
                                self.timeSignal.track_results.emit(self.trackingMethod.candidate_list,
                                                                   self.trackingMethod.expired_id,
                                                                   self.trackingTimeStamp.result_index,
                                                                   self.video_elapse)

                            scaled_frame = self.scale_frame(contour_frame, self.interpolation_flag,
                                                            self.scale_aspect)
                            display_frame = self.convert_frame(scaled_frame)

                            self.timeSignal.tracking_signal.emit(display_frame)  # QPixmap

                        if not self.apply_roi_flag and not self.apply_mask_flag:

                            thre_frame = self.detection.thresh_video(frame,
                                                                     self.block_size,
                                                                     self.offset)

                            contour_frame, entrant_detected = self.detection.detect_contours(frame,
                                                                                             thre_frame,
                                                                                             self.min_contour,
                                                                                             self.max_contour)

                            self.trackingMethod.identify(entrant_detected, self.min_contour, self.max_contour)

                            # mark indentity of each objects
                            self.trackingMethod.visualize(contour_frame, is_centroid=True,
                                                          is_mark=True, is_trajectory=True)

                            # pass tracking data to datalog thread when local tracking
                            if self.is_timeStamp:
                                self.timeSignal.track_results.emit(self.trackingMethod.candidate_list,
                                                                   self.trackingMethod.expired_id,
                                                                   self.trackingTimeStamp.result_index,
                                                                   self.video_elapse)

                            # scale threshlded frame to match the display window and roi/mask canvas
                            scaled_frame = self.scale_frame(contour_frame, self.interpolation_flag,
                                                            self.scale_aspect)
                            display_frame = self.convert_frame(scaled_frame)

                            # connected to MainWindow.display_tracking_video
                            self.timeSignal.tracking_signal.emit(display_frame)  # QPixmap

                    toc = time.perf_counter()

                elif not ret:
                    # video finished
                    self.timeSignal.track_reset_alarm.emit('1')  # complete_tracking()
                    self.timeSignal.track_reset.emit('1')  # reset video()
                    self.frame_count = -1
                    self.trackingTimeStamp.result_index = -1
                    self.video_elapse = 0
                    self.is_timeStamp = False
                    return

    def stop(self):
        with QMutexLocker(self.mutex):
            self.stopped = True

    def set_fps(self, video_fps):
        self.fps = video_fps

    def scale_frame(self, frame, interpolation, aspect):
        '''
        scale video frame to display window size
        :param frame:
        :return:
        '''

        # 4:3
        if aspect == 'classic':
            scaled_img = cv2.resize(frame, (768, 576), interpolation=interpolation)

        # 16:9
        elif aspect == 'widescreen':
            scaled_img = cv2.resize(frame, (1024, 576), interpolation=interpolation)

        return scaled_img

    def convert_frame(self, frame):
        '''
        convert image to QImage
        :param frame:
        :return:
        '''

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_cvt = QImage(frame_rgb, frame_rgb.shape[1], frame_rgb.shape[0], frame_rgb.strides[0],
                           QImage.Format_RGB888)
        frame_display = QPixmap.fromImage(frame_cvt)
        return frame_display

    def index_alarm(self):
        self.timeSignal.exceed_index_alarm.emit('1')

    def ini_palette(self):
        pass


class TrackingCamThread(QThread):

    def __init__(self):
        QThread.__init__(self)

        self.stopped = False

        self.timeSignal = Communicate()
        self.mutex = QMutex()
        self.detection = Detection()
        self.obj_num = 1  # default 1
        self.trackingMethod = TrackingMethod(self.obj_num, 15, 60, 600)
        self.trackingMethod.timeSignal.index_alarm.connect(self.index_alarm)
        self.trackingTimeStamp = TrackingTimeStamp()
        self.cam_prop = None
        self.interpolation_flag = cv2.INTER_AREA
        self.scale_aspect = 'widescreen'

        self.block_size = 11
        self.offset = 11
        self.min_contour = 1
        self.max_contour = 100
        self.invert_contrast = False

        # create a list of numbers to mark subject indentity
        self.id_list = list(range(1, 100))
        # the elements in this list needs to be in string format
        self.obj_id = [format(x, '01d') for x in self.id_list]

        self.fps = 25
        self.video_elapse = 0
        self.is_timeStamp = False
        self.frame_count = -1  # first frame start from 0

    def run(self):

        with QMutexLocker(self.mutex):
            self.stopped = False
        try:
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        except Exception as e:
            error = str(e)
            self.error_msg = QMessageBox()
            self.error_msg.setWindowTitle('Error')
            self.error_msg.setText('Failed to open camera.')
            self.error_msg.setInformativeText('TrackingCamThread.run() failed\n'
                                              'Please make sure camera is connected with computer.\n')
            self.error_msg.setIcon(QMessageBox.Warning)
            self.error_msg.setDetailedText(error)
            self.error_msg.exec()

        if self.cam_prop.width > 1024:  # shrink
            self.interpolation_flag = cv2.INTER_AREA
        elif self.cam_prop.width < 1024:  # enlarge
            self.interpolation_flag = cv2.INTER_LINEAR

        if self.cam_prop.height / self.cam_prop.width == 0.75:
            self.scale_aspect = 'classic'
        else:
            self.scale_aspect = 'widescreen'

        now = datetime.now()
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame_size = (int(self.cam_prop.width), int(self.cam_prop.height))
        export = cv2.VideoWriter('C:/Users/Public/Videos' + '/TrackingBot recording '
                                 + now.strftime('%Y-%m-%d-%H%M') + '.mp4', fourcc, self.fps, frame_size, True)

        # stream start time
        start_delta = time.perf_counter()

        while True:
            tic = time.perf_counter()
            if self.stopped:
                self.cap.release()
                return
            else:
                ret, frame = self.cap.read()
                if ret:
                    # get current date and time
                    clock_time = self.trackingTimeStamp.update_clock()
                    self.timeSignal.update_clock.emit(clock_time)

                    self.frame_count += 1

                    export.write(frame)

                    # absolute time elapsed after start capturing
                    end_delta = time.perf_counter()

                    elapse_delta = timedelta(milliseconds=(end_delta - start_delta) * 1000)

                    self.is_timeStamp, self.video_elapse = self.trackingTimeStamp.liveTimeStamp(elapse_delta,
                                                                                                interval=None)

                    self.timeSignal.update_elapse.emit(self.video_elapse)

                    if self.invert_contrast:
                        invert_cam = cv2.bitwise_not(frame)

                        thre_cam = self.detection.thresh_video(invert_cam,
                                                               self.block_size,
                                                               self.offset)

                        contour_cam, entrant_detected = self.detection.detect_contours(frame,
                                                                                       thre_cam,
                                                                                       self.min_contour,
                                                                                       self.max_contour)

                        self.trackingMethod.identify(entrant_detected, self.min_contour, self.max_contour)

                        self.trackingMethod.visualize(contour_cam, is_centroid=True,
                                                      is_mark=True, is_trajectory=True)

                        if self.is_timeStamp:
                            self.timeSignal.cam_track_results.emit(self.trackingMethod.candidate_list,
                                                                   self.trackingMethod.expired_id,
                                                                   self.trackingTimeStamp.result_index,
                                                                   self.video_elapse)

                        # scale threshlded frame to match the display window and roi/mask canvas
                        scaled_frame = self.scale_frame(contour_cam, self.interpolation_flag,
                                                        self.scale_aspect)
                        display_frame = self.convert_frame(scaled_frame)

                        self.timeSignal.cam_tracking_signal.emit(display_frame)
                        # time.sleep(1/25)

                    elif not self.invert_contrast:
                        thre_cam = self.detection.thresh_video(frame,
                                                               self.block_size,
                                                               self.offset)

                        contour_cam, entrant_detected = self.detection.detect_contours(frame,
                                                                                       thre_cam,
                                                                                       self.min_contour,
                                                                                       self.max_contour)

                        self.trackingMethod.identify(entrant_detected, self.min_contour, self.max_contour)

                        self.trackingMethod.visualize(contour_cam, is_centroid=True,
                                                      is_mark=True, is_trajectory=True)

                        if self.is_timeStamp:
                            self.timeSignal.cam_track_results.emit(self.trackingMethod.candidate_list,
                                                                   self.trackingMethod.expired_id,
                                                                   self.trackingTimeStamp.result_index,
                                                                   self.video_elapse)

                        scaled_frame = self.scale_frame(contour_cam, self.interpolation_flag,
                                                        self.scale_aspect)
                        display_frame = self.convert_frame(scaled_frame)

                        self.timeSignal.cam_tracking_signal.emit(display_frame)
                        # time.sleep(1/25)

                    toc = time.perf_counter()

                elif not ret:
                    # call cam_reload
                    self.timeSignal.cam_reload.emit('1')
                    self.cap.release()
                    self.frame_count = -1
                    self.trackingTimeStamp.result_index = -1
                    self.video_elapse = 0
                    self.is_timeStamp = False
                    self.error_msg = QMessageBox()
                    self.error_msg.setWindowTitle('Error')
                    self.error_msg.setText('No camera frame returned.')
                    self.error_msg.setInformativeText('cv2.VideoCapture() does not return frame\n'
                                                      'Please make sure camera is working and try to reload camera.\n')
                    self.error_msg.setIcon(QMessageBox.Warning)
                    self.error_msg.exec()

    def stop(self):
        with QMutexLocker(self.mutex):
            self.stopped = True

    def set_fps(self, video_fps):
        self.fps = video_fps

    def scale_frame(self, frame, interpolation, aspect):
        '''
        scale video frame to display window size
        :param frame:
        :return:
        '''

        # 4:3
        if aspect == 'classic':
            scaled_img = cv2.resize(frame, (768, 576), interpolation=interpolation)

        # 16:9
        elif aspect == 'widescreen':
            scaled_img = cv2.resize(frame, (1024, 576), interpolation=interpolation)

        return scaled_img

    def convert_frame(self, frame):
        '''
        convert frame to QImage
        :param frame:
        :return:
        '''

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_cvt = QImage(frame_rgb, frame_rgb.shape[1], frame_rgb.shape[0], frame_rgb.strides[0],
                           QImage.Format_RGB888)
        frame_display = QPixmap.fromImage(frame_cvt)
        return frame_display

    def index_alarm(self):
        self.timeSignal.exceed_index_alarm.emit('1')


class Communicate(QObject):
    updateSliderPos = pyqtSignal(float)
    track_results = pyqtSignal(list, list, int, str)
    tracking_signal = pyqtSignal(QPixmap)  # display_tracking_video
    track_reset = pyqtSignal(str)  # reset video()
    track_reset_alarm = pyqtSignal(str)  # complete_tracking()
    exceed_index_alarm = pyqtSignal(str)
    update_clock = pyqtSignal(str)
    update_elapse = pyqtSignal(str)
    cam_tracking_signal = pyqtSignal(QImage)
    cam_track_results = pyqtSignal(list, list, int, str)
    cam_reload = pyqtSignal(str)


class Detection():
    '''
    adaptive thresholding and contour filtering
    '''

    def __init__(self):
        super().__init__()

    ## video thresholding
    def thresh_video(self, frame, block_size, offset):
        """
        This function retrieves a video frame and preprocesses it for object tracking.
        The code 1) blurs image to reduce noise
                 2) converts it to greyscale
                 3) returns a thresholded version of the original image.
                 4) perform morphological operation to closing small holes inside objects
        Parameters
        ----------
        frame : source image containing all three colour channels
        block_size: int(optional), default = blocksize_ini
        offset: int(optional), default = offset_ini
        """
        frame = cv2.GaussianBlur(frame, (5, 5), 1)
        # vid = cv2.blur(vid, (5, 5))
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thresh_frame = cv2.adaptiveThreshold(gray_frame,
                                             255,
                                             cv2.ADAPTIVE_THRESH_MEAN_C,
                                             cv2.THRESH_BINARY_INV,
                                             block_size,
                                             offset)

        # Morphology operation
        # Dilation followed by erosion to closing small holes inside the foreground objects
        kernel = np.ones((5, 5), np.uint8)
        morph_frame = cv2.morphologyEx(thresh_frame, cv2.MORPH_CLOSE, kernel)

        return morph_frame

    def detect_contours(self, frame, thresh_frame, cnt_min, cnt_max):

        """
        frame : original video source for drawing and visualize contours
        thresh_frame : the frame after threshold
        cnt_min: minimum contour area threshold used to identify object of interest
        cnt_max: maximum contour area threshold used to identify object of interest

        :return
        contours: list
            a list of all detected contours that pass the area based threshold criterion
        entrant_detection: a list of (2,1) array, dtype=float
            individual's location detected on current frame
        """

        contours, hierarchy = cv2.findContours(thresh_frame.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        contour_frame = frame.copy()

        # list of detected centroids
        entrant_detection = []

        # list of area of contours
        cnt_area_list = []

        # contours from mask item that need to be excluded
        mask_cnt = []
        # when  object contour intersect with two mask contours simutaneously
        # a h1 contour can have 2 child that both in h2 level
        mask_cnt_sibling = []

        for cnt in range(len(contours)):
            # conditons to find inner cnt of mask
            # hierarchy[0,i,0] == -1 and hierarchy[0,i,1] == -1 and hierarchy[0,i,2] == -1 and
            if hierarchy[0, cnt, 3] != -1 and hierarchy[0, cnt, 1] == -1:
                mask_cnt.append(cnt)

            if hierarchy[0, cnt, 3] != -1 and hierarchy[0, cnt, 1] != -1:
                mask_cnt_sibling.append(cnt)

        # exclude contours that belong to mask
        for cnt in sorted(mask_cnt, reverse=True):
            del contours[cnt]  # inner cnt
            del contours[cnt - 1]  # outer cnt, parent of inner cnt

        # compute areas of all contours after exclude the mask contours
        for cnt in range(len(contours)):
            cnt_area = cv2.contourArea(contours[cnt])
            cnt_area_list.append(cnt_area)

        for i in sorted(range(len(cnt_area_list)), reverse=True):
            if cnt_area_list[i] < cnt_min or cnt_area_list[i] > cnt_max:
                del contours[i]
            else:
                cv2.drawContours(contour_frame, contours, i, (0, 0, 255), 2, cv2.LINE_8)
                ## calculate the centroid of current contour
                M = cv2.moments(contours[i])
                if M['m00'] != 0:
                    cx = M['m10'] / M['m00']
                    cy = M['m01'] / M['m00']
                else:
                    cx = 0
                    cy = 0
                ## update current position to new centroid
                centroids = np.array([[cx], [cy]])

                entrant = EntrantProperty(centroids, cnt_area_list[i])
                entrant_detection.append(entrant)

        return contour_frame, entrant_detection  # , contours # , entrant_detection, pos_archive


class EntrantProperty(object):
    def __init__(self, pos_detected, cnt_area):
        self.pos_detected = pos_detected
        self.cnt_area = cnt_area
