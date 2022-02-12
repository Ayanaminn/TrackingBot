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

import numpy as np
import cv2
from PyQt5.QtCore import pyqtSignal, QObject
from PyQt5.QtWidgets import QMessageBox
from kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


class Candidate(object):
    """This class register properties of every detected centroids(object)
    Attributes:
        None
    """

    def __init__(self, pos_prediction, candidate_size, candidate_index, candidate_id, lost_sample):
        """Initialize variables used by TrackList class
        Args:
            candidate_size: candidate contour size
            candidate_index:total count of candidate been detected, including expired candidate
            candidate_id: the assigned object id, associated with identity
            lost_sample:flag if sample is lost
        Return:
            None
        """

        # Apply Kalman filter
        self.KF = KalmanFilter(1, 1, 1, 1, 0.1, 0.1)
        # Convert the input to an array
        self.pos_prediction = np.asarray(pos_prediction)
        self.candidate_size = candidate_size
        self.candidate_index = candidate_index
        self.candidate_id = candidate_id
        # number of frames a registered object undetected
        self.lost_frames = 0
        self.lost_sample = lost_sample
        # trace path
        self.trace = []


class TrackingMethod(object):
    """This class update detected centroids(objects) and make assignment
    Attributes:
        None
    """

    def __init__(self, obj_num, dist_thresh, max_lost_frames, max_trace_len):
        """Initialize variable used by Tracker class
        Args:
             obj_num: number of objects in video
            dist_thresh: cost (distance) threshold
                         between prediction and detection coordinates
                         as condition to un_assign (delete) the object
                         and register a new object
            max_lost_frames: maximum allowed frames for
                                   the track object being undetected
                                   as the threshold to un_assign (delete) the object
            max_trace_len: trace path history length
        Return:
            None
        """

        self.obj_num = obj_num
        # cost (distance) threshold
        # between prediction and detection coordinates
        # as condition to un_assign (delete) the registration
        self.dist_thresh = dist_thresh
        self.max_lost_frames = max_lost_frames
        self.max_trace_len = max_trace_len
        # an array to hold registered centroids(objects)
        self.candidate_list = []
        # init candidate index and occupy 0, so that first id index will be 1
        self.candidate_index = 0
        self.candidate_id = 0
        # hold assignment index
        self.assignment = []
        # hold expired id number
        self.expired_id = []
        # first frame out of index alarm
        self.timeSignal = Communicate()

    def identify(self, entrant, cnt_min, cnt_max):
        '''
        Create registration array if no centroids(object) found
        Apply hungarian algorithm to differentiate tracked centroids
        from new detected centroids.
        Cost function is the euclidean distances between centroids
        detected and predicted
        :param entrant: list of (2,1) array
        :param cnt_min: minimum contour size threshold
        :param cnt_max: maximum contour size threshold
        :return:
        '''
        C = len(self.candidate_list)
        E = len(entrant)

        # If no object is registered OR lost all candidate and re-pickup
        if C == 0:
            if not self.expired_id:
                for i in range(E):
                    # mandate C = E in the first frame
                    if self.candidate_index < self.obj_num:
                        # first id index is 1 and last index = object number
                        self.candidate_index += 1
                        self.candidate_id += 1
                        object = Candidate(entrant[i].pos_detected, entrant[i].cnt_area,
                                           self.candidate_index, self.candidate_id, lost_sample=False)
                        self.candidate_list.append(object)
                    else:

                        # candidates in 1st frame exceeds setting target
                        self.timeSignal.index_alarm.emit('1')
                        break
            # lost all previous candidates
            elif self.expired_id:
                for i in range(E):
                    # mandate C = E when pick up
                    if self.candidate_index < self.obj_num:
                        try:
                            id = self.expired_id.pop()
                            self.candidate_id = id
                            self.candidate_index += 1
                            object = Candidate(entrant[i].pos_detected, entrant[i].cnt_area,
                                               self.candidate_index, self.candidate_id, lost_sample=False)
                            self.candidate_list.append(object)
                        except Exception as e:
                            error = str(e)
                            self.warning_msg = QMessageBox()
                            self.warning_msg.setWindowTitle('Error')
                            self.warning_msg.setText('An error happened during tracking.')
                            self.warning_msg.setIcon(QMessageBox.Warning)
                            self.warning_msg.setDetailedText(error)
                            self.warning_msg.exec()
                            pass
                    # rest are noise
                    else:
                        pass

        else:
            cost = np.zeros(shape=(C, E))  # Cost matrix
            # Calculate cost
            for i in range(C):
                for j in range(E):
                    try:
                        # the ith registered centroids and jth detected centroids
                        diff = self.candidate_list[i].pos_prediction - entrant[j].pos_detected
                        dist = np.sqrt(diff[0][0] * diff[0][0] +
                                       diff[1][0] * diff[1][0])
                        # a list of (C,E) array, each element value
                        # represent the distance between ith and jth centroid
                        # so that the min distance represent optimal assignment
                        cost[i][j] = dist

                    except Exception as e:
                        error = str(e)
                        self.warning_msg = QMessageBox()
                        self.warning_msg.setWindowTitle('Error')
                        self.warning_msg.setText('An error happened during tracking.')
                        self.warning_msg.setInformativeText('cost matrix does not compute correctly.')
                        self.warning_msg.setIcon(QMessageBox.Warning)
                        self.warning_msg.setDetailedText(error)
                        self.warning_msg.exec()

            # average the squared ERROR
            cost = (0.5) * cost
            candidate_index, assigned_index = linear_sum_assignment(cost)

            # init/reset assignment list
            self.assignment.clear()

            for _ in range(C):
                self.assignment.append(-1)

            for i in range(len(candidate_index)):
                self.assignment[candidate_index[i]] = assigned_index[i]

            expired_candidate = []

            for i in range(len(self.assignment)):
                # if E>C or E=C
                # all assignment[i] != -1
                if not self.candidate_list[i].lost_sample:

                    if self.assignment[i] != -1:

                        self.candidate_list[i].lost_sample = False
                        # reset lost time
                        self.candidate_list[i].lost_frames = 0
                        # validation of assignment:
                        if cost[i][self.assignment[i]] > self.dist_thresh:
                            self.assignment[i] = -1
                            self.candidate_list[i].lost_frames += 1
                            self.candidate_list[i].lost_sample = True

                    # ONLY when E < C
                    else:
                        self.candidate_list[i].lost_frames += 1
                        self.candidate_list[i].lost_sample = True

                # object was lost but now re-detected
                else:
                    if self.assignment[i] != -1:
                        # accept
                        if cnt_min < self.candidate_list[i].candidate_size < cnt_max:
                            self.candidate_list[i].lost_sample = False
                            # reset lost time
                            self.candidate_list[i].lost_frames = 0
                        # reject
                        else:
                            self.candidate_list[i].lost_frames += 1
                            self.candidate_list[i].lost_sample = True
                    # still lost
                    else:
                        self.candidate_list[i].lost_frames += 1
                        self.candidate_list[i].lost_sample = True

            # If registered centroids(objects) are not assigned for long time,
            # mark it expired and send to delete queue, keep the lost id
            for i in range(C):
                if self.candidate_list[i].lost_frames > self.max_lost_frames:
                    # candidate[i] expired, keep its id
                    self.expired_id.append(self.candidate_list[i].candidate_id)
                    expired_candidate.append(i)

            # when expired candidate in the queue
            # delete it to accept new detection
            if len(expired_candidate) > 0:

                for i in sorted(expired_candidate, reverse=True):
                    del self.candidate_list[i]
                    del self.assignment[i]
                    self.candidate_index -= 1

            for j in range(E):
                # if E > C
                if j not in self.assignment:
                    # AND an object(s) just expired
                    # accept, keep C <= set number
                    if self.candidate_index < self.obj_num:
                        # need further test for robustness
                        try:
                            id = self.expired_id.pop()
                            # inherit id
                            self.candidate_id = id
                            self.candidate_index += 1
                            object = Candidate(entrant[j].pos_detected, entrant[j].cnt_area,
                                               self.candidate_index, self.candidate_id, lost_sample=False)
                            self.candidate_list.append(object)
                        except Exception as e:
                            error = str(e)
                            self.warning_msg = QMessageBox()
                            self.warning_msg.setWindowTitle('Error')
                            self.warning_msg.setText('An error happened during tracking.')
                            self.warning_msg.setIcon(QMessageBox.Warning)
                            self.warning_msg.setDetailedText(error)
                            self.warning_msg.exec()
                            pass
                    else:
                        # E[j] is noise
                        # ignore E[j]
                        pass
                # all E assigned
                else:
                    # do nothing
                    pass

            # update kalman state according to assignment
            for i in range(len(self.assignment)):

                self.candidate_list[i].KF.predict()
                # if assigned, update
                if self.assignment[i] != -1:

                    self.candidate_list[i].pos_prediction = self.candidate_list[i].KF.update(
                        entrant[self.assignment[i]].pos_detected, 1)

                    self.candidate_list[i].candidate_size = entrant[self.assignment[i]].cnt_area
                # do not update until re-assigned or expired
                else:
                    pass

                # if recorded trajectory counts is over max setting
                if len(self.candidate_list[i].trace) > self.max_trace_len:
                    # clear the counts
                    for j in range(len(self.candidate_list[i].trace) - self.max_trace_len):
                        del self.candidate_list[i].trace[j]

                # record trajectory counts to display
                self.candidate_list[i].trace.append(self.candidate_list[i].pos_prediction)

                self.candidate_list[i].KF.previousState = self.candidate_list[i].pos_prediction

    def visualize(self, video, is_centroid = True, is_mark = True,
                  is_trajectory=True):
        """visualize the indentity of tracked objects with marks and trajectories
        Args:
            video: the video source to displayed on
            id_marks: the list of numerical or text that used as id mark to represent
                      the identity of the object
        Return:
            None
        """
        trace_colors = [[86, 94, 219], [86, 194, 219], [86, 219, 145], [127, 219, 86],
                 [219, 211, 86], [219, 111, 86], [219, 86, 160], [178, 86, 219]]

        # trace_paired = [[180, 120, 31], [44, 160, 51], [28, 26, 227], [0, 127, 255], [154, 61, 106],
        #                 [40, 89, 177], [227, 206, 166],[138, 223, 178], [153, 154, 251], [111, 191, 253],
        #                 [214, 178, 202], [153, 255, 255]]

        for i in range(len(self.candidate_list)):

            if self.candidate_list[i].candidate_index <= self.obj_num:

                if is_centroid:
                    # display centroid
                    cv2.circle(video,
                               tuple([int(x) for x in self.candidate_list[i].pos_prediction]),
                               1, (41, 255, 255), -1, cv2.LINE_AA)
                if is_mark:
                    # display id mark
                    cv2.putText(video,
                                str(self.candidate_list[i].candidate_id),
                                tuple([int(x) for x in self.candidate_list[i].pos_prediction]),
                                cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)

                if is_trajectory:
                    # display the trajectory (line style)
                    if len(self.candidate_list[i].trace) > 1:
                        for j in range(len(self.candidate_list[i].trace) - 1):
                            x = int(self.candidate_list[i].trace[j][0][0])
                            y = int(self.candidate_list[i].trace[j][1][0])
                            x1 = int(self.candidate_list[i].trace[j + 1][0][0])
                            y1 = int(self.candidate_list[i].trace[j + 1][1][0])
                            cv2.line(video, (x,y), (x1,y1),
                                     (trace_colors[i % len(self.candidate_list) % 8][0],
                                      trace_colors[i % len(self.candidate_list) % 8][1],
                                      trace_colors[i % len(self.candidate_list) % 8][2]), 1)
                    ## display the trajectory (circle style)
                    # if (len(self.registration[i].trajectory) > 1):
                    #     for j in range(len(self.registration[i].trajectory)):
                    #         x = int(self.registration[i].trajectory[j][0][0])
                    #         y = int(self.registration[i].trajectory[j][1][0])
                    #         cv2.circle(contour_vid, (x, y), 1, (0,255,0), -1)
                    else:
                        pass


class Communicate(QObject):
    index_alarm = pyqtSignal(str)
