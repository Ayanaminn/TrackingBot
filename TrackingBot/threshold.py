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
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QObject, QMutex, QMutexLocker
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMessageBox
from datalog import TrackingTimeStamp
import time
from datetime import timedelta


class ThreshVidThread(QThread):

    def __init__(self, default_fps=25):
        QThread.__init__(self)

        self.mutex = QMutex()
        self.timeSignal = Communicate()
        self.detection = Detection()
        self.playCapture = cv2.VideoCapture()
        self.video_prop = None
        self.interpolation_flag = cv2.INTER_AREA
        self.scale_aspect = 'widescreen'

        self.roi_canvas = np.zeros((576, 1024), dtype='uint8')
        self.mask_canvas = np.zeros((576, 1024), dtype='uint8')
        self.final_mask = None # combine roi and mask
        self.stopped = False
        self.fps = default_fps

        self.block_size = 11
        self.offset = 11
        self.min_contour = 1
        self.max_contour = 100
        self.invert_contrast = False
        self.apply_roi_flag = False
        self.apply_mask_flag = False

        self.ROIs = []
        self.Masks = []

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

                    # get and update video elapse time
                    play_elapse = self.playCapture.get(cv2.CAP_PROP_POS_FRAMES) / self.playCapture.get(cv2.CAP_PROP_FPS)
                    self.timeSignal.updateSliderPos.emit(play_elapse)

                    if self.invert_contrast:
                        # brighter object, darker background
                        invert_frame = cv2.bitwise_not(frame)

                        if self.apply_roi_flag or self.apply_mask_flag:
                            # if roi defined, apply the mask
                            masked_frame = cv2.bitwise_and(invert_frame, invert_frame, mask=self.final_mask)

                            thre_frame = self.detection.thresh_video(masked_frame,
                                                                     self.block_size,
                                                                     self.offset)

                            contour_frame, max_detect_cnt,min_detect_cnt = self.detection.detect_contours(frame,
                                                                              thre_frame,
                                                                              self.min_contour,
                                                                              self.max_contour)

                            # scale threshlded frame to match the display window and roi/mask canvas
                            scaled_frame = self.scale_frame(contour_frame, self.interpolation_flag, self.scale_aspect)

                            # convert to QImage
                            display_frame = self.convert_frame(scaled_frame)
                            preview_frame = self.convert_preview_frame(thre_frame)

                            # connected to MainWindow.displayThresholdVideo
                            self.timeSignal.thresh_signal.emit(display_frame, preview_frame)  # QPixmap

                            self.timeSignal.detect_cnt.emit(max_detect_cnt, min_detect_cnt)

                            time.sleep(1 / self.fps)

                        elif not self.apply_roi_flag and not self.apply_mask_flag:

                            thre_frame = self.detection.thresh_video(invert_frame,
                                                                     self.block_size,
                                                                     self.offset)

                            contour_frame, max_detect_cnt,min_detect_cnt = self.detection.detect_contours(frame,
                                                                              thre_frame,
                                                                              self.min_contour,
                                                                              self.max_contour)

                            # scale threshlded frame to match the display window and roi/mask canvas
                            scaled_frame = self.scale_frame(contour_frame, self.interpolation_flag, self.scale_aspect)

                            # convert to QImage
                            display_frame = self.convert_frame(scaled_frame)
                            preview_frame = self.convert_preview_frame(thre_frame)

                            # connected to MainWindow.displayThresholdVideo
                            self.timeSignal.thresh_signal.emit(display_frame, preview_frame)  # QPixmap

                            self.timeSignal.detect_cnt.emit(max_detect_cnt, min_detect_cnt)

                            time.sleep(1 / self.fps)

                    elif not self.invert_contrast:

                        if self.apply_roi_flag or self.apply_mask_flag:
                            # if roi defined, apply the mask
                            masked_frame = cv2.bitwise_and(frame, frame, mask=self.final_mask)

                            thre_frame = self.detection.thresh_video(masked_frame,
                                                                     self.block_size,
                                                                     self.offset)

                            contour_frame, max_detect_cnt,min_detect_cnt = self.detection.detect_contours(frame,
                                                                              thre_frame,
                                                                              self.min_contour,
                                                                              self.max_contour)

                            # scale threshlded frame to match the display window and roi/mask canvas
                            scaled_frame = self.scale_frame(contour_frame, self.interpolation_flag, self.scale_aspect)

                            # convert to QImage
                            display_frame = self.convert_frame(scaled_frame)
                            preview_frame = self.convert_preview_frame(thre_frame)

                            # connected to MainWindow.displayThresholdVideo
                            self.timeSignal.thresh_signal.emit(display_frame, preview_frame)  # QPixmap

                            self.timeSignal.detect_cnt.emit(max_detect_cnt, min_detect_cnt)

                            time.sleep(1 / self.fps)

                        if not self.apply_roi_flag and not self.apply_mask_flag:

                            thre_frame = self.detection.thresh_video(frame,
                                                                     self.block_size,
                                                                     self.offset)

                            contour_frame, max_detect_cnt,min_detect_cnt = self.detection.detect_contours(frame,
                                                                              thre_frame,
                                                                              self.min_contour,
                                                                              self.max_contour)

                            # scale threshlded frame to match the display window and roi/mask canvas
                            scaled_frame = self.scale_frame(contour_frame, self.interpolation_flag, self.scale_aspect)

                            # convert to QImage
                            display_frame = self.convert_frame(scaled_frame)
                            preview_frame = self.convert_preview_frame(thre_frame)

                            # connected to MainWindow.displayThresholdVideo
                            self.timeSignal.thresh_signal.emit(display_frame, preview_frame)  # QPixmap

                            self.timeSignal.detect_cnt.emit(max_detect_cnt, min_detect_cnt)

                            time.sleep(1 / self.fps)

                elif not ret:
                    # video finished
                    # connected to MainWindow.resetVideo
                    self.timeSignal.thresh_reset.emit('1')

                    return

    def stop(self):
        with QMutexLocker(self.mutex):
            self.stopped = True

    def is_stopped(self):
        with QMutexLocker(self.mutex):
            return self.stopped

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

    def convert_preview_frame(self, frame):
        '''
        convert frame to QImage
        :param frame:
        :return:
        '''

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_cvt = QImage(frame_rgb, frame_rgb.shape[1], frame_rgb.shape[0], frame_rgb.strides[0],
                           QImage.Format_RGB888)
        frame_scaled = frame_cvt.scaled(320, 180, Qt.KeepAspectRatio)
        frame_display = QPixmap.fromImage(frame_scaled)
        return frame_display

    def create_roi(self):
        '''
        create roi on roi canvas based on parameters of all valid roi(s)
        :return:
        '''

        if self.apply_roi_flag and self.ROIs:

            for i in range(len(self.ROIs)):

                if self.ROIs[i].type == 'rect':  # rectangle shape ROIs

                    raw_rect = self.ROIs[i].ROI.rect()
                    # Map item coords to thresh canvas (label widget) coords
                    valid_rect = self.ROIs[i].ROI.mapRectToScene(raw_rect)
                    # image slice indices must be integer
                    x1, y1, x2, y2 = valid_rect.getCoords()
                    cv2.rectangle(self.roi_canvas,
                                  (int(x1), int(y1)),
                                  (int(x2), int(y2)), 255, -1)

                elif self.ROIs[i].type == 'circ':  # ellipse shape ROIs

                    raw_rect = self.ROIs[i].ROI.rect()
                    valid_rect = self.ROIs[i].ROI.mapRectToScene(raw_rect)

                    x, y, w, h = valid_rect.getRect()
                    center = (int(x + w / 2), int(y + h / 2))
                    axis_major = int(h / 2)
                    axis_minor = int(w / 2)
                    cv2.ellipse(self.roi_canvas, center, (axis_minor, axis_major),
                                0, 0, 360, 255, -1)

                else: # save for line or polygons
                    pass

        else:
            # print('No valid ROI or flag is false')
            pass

    def create_mask(self):
        '''
        create mask on mask canvas based on parameters of all valid mask(s)
        :return:
        '''

        if self.apply_mask_flag and self.Masks:

            for i in range(len(self.Masks)):

                if self.Masks[i].type == 'rect':  # rectangle shape ROIs

                    raw_rect = self.Masks[i].Mask.rect()
                    valid_rect = self.Masks[i].Mask.mapRectToScene(raw_rect)
                    # image slice indices must be integer

                    x1, y1, x2, y2 = valid_rect.getCoords()
                    cv2.rectangle(self.mask_canvas,
                                  (int(x1), int(y1)),
                                  (int(x2), int(y2)), 255, -1)

                elif self.Masks[i].type == 'circ':  # ellipse shape ROIs

                    raw_rect = self.Masks[i].Mask.rect()
                    valid_rect = self.Masks[i].Mask.mapRectToScene(raw_rect)
                    # image slice indices must be integer

                    x, y, w, h = valid_rect.getRect()
                    center = (int(x + w / 2), int(y + h / 2))
                    axis_major = int(h / 2)
                    axis_minor = int(w / 2)
                    cv2.ellipse(self.roi_canvas, center, (axis_minor, axis_major),
                                0, 0, 360, 0, -1)

                else:  # save for line or polygons
                    pass

        else:
            pass


class ThreshCamThread(QThread):

    def __init__(self):
        QThread.__init__(self)

        self.timeSignal = Communicate()
        self.mutex = QMutex()
        self.detection = Detection()
        self.trackingTimeStamp = TrackingTimeStamp()
        self.cam_prop = None
        self.video_elapse = 0

        self.block_size = 11
        self.offset = 11
        self.min_contour = 1
        self.max_contour = 100

        self.stopped = False
        self.invert_contrast = False
        self.interpolation_flag = cv2.INTER_AREA
        self.scale_aspect = 'widescreen'

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
            self.error_msg.setInformativeText('ThreshCamThread.run() failed\n'
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
        # stream start time
        start_delta = time.perf_counter()

        while True:
            if self.stopped:
                self.cap.release()
                return
            else:
                ret, frame = self.cap.read()
                if ret:

                    # get current date and time
                    clock_time = self.trackingTimeStamp.update_clock()
                    self.timeSignal.update_clock.emit(clock_time)

                    # absolute time elapsed after start capturing
                    end_delta = time.perf_counter()
                    # elapse_delta = timedelta(seconds=end_delta - start_delta).total_seconds()
                    elapse_delta = timedelta(milliseconds=(end_delta - start_delta) * 1000)

                    _, self.video_elapse = self.trackingTimeStamp.liveTimeStamp(elapse_delta,interval=None)

                    self.timeSignal.update_elapse.emit(self.video_elapse)

                    if self.invert_contrast:
                        invert_cam = cv2.bitwise_not(frame)

                        thre_cam = self.detection.thresh_video(invert_cam,
                                                               self.block_size,
                                                               self.offset)

                        contour_cam, max_detect_cnt,min_detect_cnt = self.detection.detect_contours(frame,
                                                                        thre_cam,
                                                                        self.min_contour,
                                                                        self.max_contour)

                        # scale threshlded frame to match the display window and roi/mask canvas
                        scaled_cam = self.scale_frame(contour_cam,self.interpolation_flag,self.scale_aspect)

                        # convert to QImage
                        display_cam = self.convert_frame(scaled_cam)
                        preview_cam = self.convert_preview_frame(thre_cam)

                        # connected to MainWindow.display_threshold_cam
                        self.timeSignal.cam_thresh_signal.emit(display_cam, preview_cam)  # QPixmap

                        self.timeSignal.cam_detect_cnt.emit(max_detect_cnt, min_detect_cnt)

                    elif not self.invert_contrast:

                        thre_cam = self.detection.thresh_video(frame,
                                                               self.block_size,
                                                               self.offset)

                        contour_cam, max_detect_cnt,min_detect_cnt = self.detection.detect_contours(frame,
                                                                        thre_cam,
                                                                        self.min_contour,
                                                                        self.max_contour)


                        # scale threshlded frame to match the display window and roi/mask canvas
                        scaled_cam = self.scale_frame(contour_cam, self.interpolation_flag,self.scale_aspect)
                        # convert to QImage
                        display_cam = self.convert_frame(scaled_cam)
                        preview_cam = self.convert_preview_frame(thre_cam)

                        # connected to MainWindow.display_threshold_cam
                        self.timeSignal.cam_thresh_signal.emit(display_cam,preview_cam)# QPixmap

                        self.timeSignal.cam_detect_cnt.emit(max_detect_cnt, min_detect_cnt)


                elif not ret:
                    # call reloadCamera() to try reload camera
                    self.timeSignal.cam_reload.emit('1')
                    self.error_msg = QMessageBox()
                    self.error_msg.setWindowTitle('Error')
                    self.error_msg.setText('No camera frame returned.')
                    self.error_msg.setInformativeText('cv2.VideoCapture() does not return frame\n'
                                                      'Please make sure camera is working and try to reload camera.\n')
                    self.error_msg.setIcon(QMessageBox.Warning)
                    self.error_msg.exec()
                    return

    def stop(self):
        with QMutexLocker(self.mutex):
            self.stopped = True

    def is_stopped(self):
        with QMutexLocker(self.mutex):
            return self.stopped

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

    def convert_preview_frame(self, frame):
        '''
        convert frame to QImage
        :param frame:
        :return:
        '''

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_cvt = QImage(frame_rgb, frame_rgb.shape[1], frame_rgb.shape[0], frame_rgb.strides[0],
                           QImage.Format_RGB888)
        frame_scaled = frame_cvt.scaled(320, 180, Qt.KeepAspectRatio)
        frame_display = QPixmap.fromImage(frame_scaled)
        return frame_display


class Detection():
    '''
    adaptive thresholding and contour filtering
    '''

    def __init__(self):
        super().__init__()

    def thresh_video(self, vid, block_size, offset):
        """
        This function retrieves a video frame and preprocesses it for object tracking.
        The code 1) blurs image to reduce noise
                 2) converts it to greyscale
                 3) returns a thresholded version of the original image.
                 4) perform morphological operation to closing small holes inside objects
        Parameters
        ----------
        vid : source image containing all three colour channels
        block_size: int(optional), default = blocksize_ini
        offset: int(optional), default = offset_ini
        """
        vid = cv2.GaussianBlur(vid, (5, 5), 1)
        # vid = cv2.blur(vid, (5, 5))
        vid_gray = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
        vid_th = cv2.adaptiveThreshold(vid_gray,
                                       255,
                                       cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY_INV,
                                       block_size,
                                       offset)

        ## Dilation followed by erosion to closing small holes inside the foreground objects
        kernel = np.ones((5, 5), np.uint8)
        vid_closing = cv2.morphologyEx(vid_th, cv2.MORPH_CLOSE, kernel)

        return vid_closing


    def detect_contours(self, vid, vid_th, cnt_min, cnt_max):
        """
        vid : original video source for drawing and visualize contours
        vid_detect : the masked video for detect contours
        min_th: minimum contour area threshold used to identify object of interest
        max_th: maximum contour area threshold used to identify object of interest

        :return
        contours: list
            a list of all detected contours that pass the area based threshold criterion
        pos_archive: a list of (2,1) array, dtype=float
            individual's location on previous frame
        pos_detection: a list of (2,1) array, dtype=float
            individual's location detected on current frame
        """

        contours, hierarchy = cv2.findContours(vid_th.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        vid_draw = vid.copy()

        ## initialize contour number

        ## roll current position to past
        ## clear current position to accept updated value
        pos_detection = []

        # list of area of contours
        cnt_area_list = []

        filtered_cnt = []

        max_detect_cnt = None
        min_detect_cnt = None

        # contours need to be excluded
        mask_cnt = []
        # when  object contour intersect with two mask contours simultaneously
        # a h1 contour can have 2 child that both in h2 level
        mask_cnt_sibling = []
        del mask_cnt[:]
        del mask_cnt_sibling[:]

        for cnt in range(len(contours)):
            # inner cnt of mask zone
            if hierarchy[0, cnt, 3] != -1 and hierarchy[0, cnt, 1] == -1:
                mask_cnt.append(cnt)

            if hierarchy[0, cnt, 3] != -1 and hierarchy[0, cnt, 1] != -1:
                mask_cnt_sibling.append(cnt)

        for cnt in sorted(mask_cnt, reverse=True):
            del contours[cnt]  # inner cnt
            del contours[cnt - 1] # outer cnt, parent of inner cnt

        # compute areas of all contours after exclude the mask contours
        for cnt in range(len(contours)):
            cnt_area = cv2.contourArea(contours[cnt])
            cnt_area_list.append(cnt_area)

        for i in sorted(range(len(cnt_area_list)), reverse=True):
            if cnt_area_list[i] < cnt_min or cnt_area_list[i] > cnt_max:
                del contours[i]

            # draw contour if meet the threshold
            else:

                filtered_cnt.append(cnt_area_list[i])
                cv2.drawContours(vid_draw, contours, i, (0, 0, 255), 2, cv2.LINE_8)
                # calculate the centroid of current contour
                M = cv2.moments(contours[i])
                if M['m00'] != 0:
                    cx = M['m10'] / M['m00']
                    cy = M['m01'] / M['m00']
                else:
                    cx = 0
                    cy = 0

                centroids = np.array([[cx], [cy]])
                pos_detection.append(centroids)
                i += 1

        # update real time cnt range
        sorted_cnt = sorted(filtered_cnt)

        if sorted_cnt:
            max_detect_cnt = sorted_cnt[-1]
            min_detect_cnt = sorted_cnt[0]

        else:
            max_detect_cnt = None
            min_detect_cnt = None

        return vid_draw, max_detect_cnt, min_detect_cnt #,pos_detection,


class Communicate(QObject):
    # thresh_signal = pyqtSignal(QImage)
    thresh_signal = pyqtSignal(QPixmap,QPixmap)
    cam_thresh_signal = pyqtSignal(QPixmap,QPixmap)
    cam_reload = pyqtSignal(str)
    update_clock = pyqtSignal(str)
    update_elapse = pyqtSignal(str)
    thresh_reset = pyqtSignal(str)
    updateSliderPos = pyqtSignal(float)
    detect_cnt = pyqtSignal(object,object)
    cam_detect_cnt = pyqtSignal(object, object)
