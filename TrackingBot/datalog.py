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
import pandas as pd
import cv2
import time
from datetime import datetime, timedelta
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QObject, QMutex, QMutexLocker
from scipy.spatial import cKDTree
from scipy.ndimage.filters import gaussian_filter


class DataLogThread(QThread):

    def __init__(self):
        QThread.__init__(self)
        self.stopped = False
        self.mutex = QMutex()

        self.df = []
        self.df_archive = [] # final data

        self.id_list = list(range(1, 100))
        # the elements in this list needs to be in string format
        self.obj_id = [format(x, '01d') for x in self.id_list]
        self.obj_num = 1
        self.tracked_object = None
        self.expired_id_list = None
        self.tracked_index = None
        self.tracked_elapse = None

    def run(self):
        tic = time.perf_counter()
        with QMutexLocker(self.mutex):
            self.stopped = False
        if self.stopped:
            return
        else:
            if self.expired_id_list:

                for j in range(len(self.expired_id_list)):

                    expired_candidate = ExpiredCandidate('NaN', 0, 'NaN', self.expired_id_list[j], lost_sample=True)
                    self.tracked_object.append(expired_candidate)

                for i in range(len(self.tracked_object)):
                    if i < self.obj_num:
                        if not self.tracked_object[i].lost_sample and self.tracked_object[i].lost_sample is not None:
                            self.df.append([self.tracked_index,
                                            self.tracked_elapse,
                                            self.tracked_object[i].candidate_id,
                                            self.tracked_object[i].pos_prediction[0][0],
                                            self.tracked_object[i].pos_prediction[1][0]])
                        elif self.tracked_object[i].lost_sample:
                            self.df.append([self.tracked_index,
                                            self.tracked_elapse,
                                            self.tracked_object[i].candidate_id,
                                            'NaN',
                                            'NaN'])
            else:
                for i in range(len(self.tracked_object)):
                    if i < self.obj_num:
                        if not self.tracked_object[i].lost_sample and self.tracked_object[i].lost_sample is not None:
                            self.df.append([self.tracked_index,
                                            self.tracked_elapse,
                                            self.tracked_object[i].candidate_id,
                                            self.tracked_object[i].pos_prediction[0][0],
                                            self.tracked_object[i].pos_prediction[1][0]])
                        elif self.tracked_object[i].lost_sample:
                            self.df.append([self.tracked_index,
                                            self.tracked_elapse,
                                            self.tracked_object[i].candidate_id,
                                            'NaN',
                                            'NaN'])

            if len(self.df) >= 100:
                self.df_archive.extend(self.df.copy())
                del self.df[:]

        toc = time.perf_counter()

    def track_results(self,tracked_object,expired_id_list,tracked_index,tracked_elapse):
        '''
        the list of expired id number;
        the index of timestamp;
        video time elapsed when time stamp is true
        '''
        self.tracked_object = tracked_object
        self.expired_id_list = expired_id_list
        self.tracked_index = tracked_index
        self.tracked_elapse = tracked_elapse

    def stop(self):
        if len(self.df) != 0:
            self.df_archive.extend(self.df.copy())
            del self.df[:]
        with QMutexLocker(self.mutex):

            self.stopped = True


class DataExportThread(QThread):

    def __init__(self):
        QThread.__init__(self)
        self.timesignal = Communicate()
        self.dataLogThread = None

        self.video_fps = None
        self.object_num = None
        self.pixel_per_metric = None
        self.data_save_path = None
        self.graph_save_path = None

    def run(self):
        self.convert_data()

    def convert_data(self):
        # pay attention to dtype
        tic = time.perf_counter()
        df = pd.DataFrame(np.array(self.dataLogThread.df_archive),
                                      columns=['Result(Frame)', 'Video elapse', 'Subject', 'pos_x', 'pos_y'])

        # convert data type for each parameter
        df['Result(Frame)'] = df['Result(Frame)'].astype(int)
        df['Video elapse'] = df['Video elapse'].astype(str)
        df['Subject'] = 'Subject ' + df['Subject'].astype(str)
        df['pos_x'] = df['pos_x'].astype(float)
        df['pos_y'] = df['pos_y'].astype(float)

        # Splitting dataframe into multiple dataframes of each detected subject
        df_list = [d for _, d in df.groupby(['Subject'])]
        # pix * (mm/pix)
        for i in df_list:
            dx = i['pos_x'] - i['pos_x'].shift(1)
            dy = i['pos_y'] - i['pos_y'].shift(1)
            i['Distance moved (mm)'] = (np.sqrt(dx ** 2 + dy ** 2)) * self.pixel_per_metric
            i['Distance moved (mm)'] = i['Distance moved (mm)'].astype(float)
            # Return cumulative sum over DataFrame column.
            i['Accumulate Distance moved (mm)'] = i['Distance moved (mm)'].cumsum(axis = 0)
            i['Accumulate Distance moved (mm)'] = i['Accumulate Distance moved (mm)'].astype(float)
            # i['Velocity (mm/s)'] = i['Distance moved (mm)']/(1/self.video_fps)
        # concatenate all sub-dataframes and maintain index order
        result = pd.concat(df_list, sort=False).sort_index()

        # for 1 second time binned dataframe
        # pass

        toc = time.perf_counter()
        self.save_data(result)

    def save_data(self,df_raw):
        now = datetime.now()
        full_path = self.data_save_path + '/TrackingBot export ' + now.strftime('%Y-%m-%d-%H%M') + '.xlsx'

        with pd.ExcelWriter(full_path, engine='xlsxwriter') as writer:
            df_raw.to_excel(writer, sheet_name='Raw_data', index=False)
        # time.sleep(3)
        # Emit signal to update progress bar value
        self.timesignal.data_process_fin.emit('1')


class CamDataExportThread(QThread):

    def __init__(self):
        QThread.__init__(self)
        self.timesignal = Communicate()
        self.dataLogThread = None

        self.object_num = None
        self.data_save_path = None

    def run(self):
        self.convert_data()

    def convert_data(self):
        # pay attention to dtype!!!
        # otherwise can not perform calculation betwteen different datatype
        # such as str and float
        # print(f'df archive {self.dataLogThread.df_archive}')
        df = pd.DataFrame(np.array(self.dataLogThread.df_archive),
                                      columns=['Result(Frame)', 'Video elapse', 'Subject', 'pos_x', 'pos_y'])

        # convert data type for each parameter
        df['Result(Frame)'] = df['Result(Frame)'].astype(int)
        df['Video elapse'] = df['Video elapse'].astype(str)
        df['Subject'] = 'Subject ' + df['Subject'].astype(str)
        df['pos_x'] = df['pos_x'].astype(float)
        df['pos_y'] = df['pos_y'].astype(float)

        # Splitting dataframe into multiple dataframes of each detected subject
        df_list = [d for _, d in df.groupby(['Subject'])]
        # pix * (mm/pix)
        for i in df_list:
            dx = i['pos_x'] - i['pos_x'].shift(1)
            dy = i['pos_y'] - i['pos_y'].shift(1)
            i['Distance moved (pix)'] = (np.sqrt(dx ** 2 + dy ** 2))
            i['Distance moved (pix)'] = i['Distance moved (pix)'].astype(float)
            # Return cumulative sum over DataFrame column.
            i['Accumulate Distance moved (pix)'] = i['Distance moved (pix)'].cumsum(axis = 0)
            i['Accumulate Distance moved (pix)'] = i['Accumulate Distance moved (pix)'].astype(float)
        # concatenate all sub-dataframes and maintain index order
        result = pd.concat(df_list, sort=False).sort_index()

        self.save_data(result)

    def save_data(self,df_raw):
        now = datetime.now()
        full_path = self.data_save_path + '/TrackingBot export ' + now.strftime('%Y-%m-%d-%H%M') + '.xlsx'

        with pd.ExcelWriter(full_path, engine='xlsxwriter') as writer:
            df_raw.to_excel(writer, sheet_name='Raw_data', index=False)
        # time.sleep(3)
        # Emit signal to update progress bar value
        self.timesignal.data_process_fin.emit('1')


class TraceExportThread(QThread):

    def __init__(self):
        QThread.__init__(self)
        self.timesignal = Communicate()
        self.dataLogThread = None
        self.trace_frame = None
        self.video_prop = None
        self.trace_colors = [[86, 94, 219], [86, 194, 219], [86, 219, 145], [127, 219, 86],
                 [219, 211, 86], [219, 111, 86], [219, 86, 160], [178, 86, 219]]

    def run(self):
        self.generate_trace()

    def generate_trace(self):
        trace_map = self.trace_frame.copy()
        df = pd.DataFrame(np.array(self.dataLogThread.df_archive),
                          columns=['Result(Frame)', 'Video elapse', 'Subject', 'pos_x', 'pos_y'])

        # convert data type for each parameter
        df['Result(Frame)'] = df['Result(Frame)'].astype(int)
        df['Video elapse'] = df['Video elapse'].astype(str)
        df['Subject'] = 'Subject ' + df['Subject'].astype(str)
        df['pos_x'] = df['pos_x'].astype(float)
        df['pos_y'] = df['pos_y'].astype(float)

        # Splitting dataframe into multiple dataframes of each detected subject
        df_list = [d for _, d in df.groupby(['Subject'])]

        for i in range(len(df_list)):
            df_list[i] = df_list[i][['pos_x','pos_y']].copy()
            df_list[i]['coord'] = df_list[i].values.tolist()

            coord_list = df_list[i]['coord'].tolist()

            for j in range(len(coord_list) - 1): # number of elements = range -1

                x = coord_list[j][0]
                y = coord_list[j][1]
                x1 = coord_list[j+1][0]
                y1 = coord_list[j+1][1]

                if pd.isnull(x) or pd.isnull(y) or pd.isnull(x1) or pd.isnull(y1):
                    pass
                else:
                    cv2.line(trace_map, (int(x), int(y)), (int(x1), int(y1)),
                             (self.trace_colors[i % len(df_list) % 8][0],
                              self.trace_colors[i % len(df_list) % 8][1],
                              self.trace_colors[i % len(df_list) % 8][2]), 1)

        self.convert_trace(trace_map)
        self.timesignal.trace_map_raw.emit(trace_map)

    def convert_trace(self, trace_map):
        # 4:3
        if self.video_prop.height / self.video_prop.width == 0.75:
            scaled_trace_map = cv2.resize(trace_map, (768, 576))

        # 16:9
        else:
            scaled_trace_map = cv2.resize(trace_map, (1024, 576))

        trace_map_rgb = cv2.cvtColor(scaled_trace_map, cv2.COLOR_BGR2RGB)
        trace_map_cvt = QImage(trace_map_rgb, trace_map_rgb.shape[1], trace_map_rgb.shape[0], trace_map_rgb.strides[0],
                           QImage.Format_RGB888)
        trace_map_display = QPixmap.fromImage(trace_map_cvt)

        self.timesignal.trace_process_fin.emit('1') # TraceProcessDialog.setFinish
        self.timesignal.trace_map.emit(trace_map_display) # TraceProcessDialog.display


class GraphExportThread(QThread):

    def __init__(self):
        QThread.__init__(self)
        self.timesignal = Communicate()
        self.dataLogThread = None

        self.video_prop = None
        self.neighbours = 16
        self.resolution = 500
        self.sigma = 6
        self.data_save_path = None
        self.graph_save_path = None

    def run(self):
        self.generate_plot()

    def generate_plot(self):
        df = pd.DataFrame(np.array(self.dataLogThread.df_archive),
                          columns=['Result(Frame)', 'Video elapse', 'Subject', 'pos_x', 'pos_y'])
        df_x = df['pos_x'].astype(float)
        df_y = df['pos_y'].astype(float)

        heat_map, _, _ = np.histogram2d(df_x, df_y, bins=[np.arange(0, self.video_prop.width, 1),
                                                           np.arange(0, self.video_prop.height, 1)])

        heat_map = gaussian_filter(heat_map, sigma=self.sigma)
        heat_map = heat_map.T

        self.timesignal.graph_process_fin.emit('1')  # GraphProcessDialog.setFinish
        self.timesignal.heat_map.emit(heat_map)  # GraphProcessDialog.display

    def data_coord2view_coord(self, pos,resolution, pos_min, pos_max):
        dp = pos_max - pos_min
        dv = (pos - pos_min) / dp * resolution
        return dv

    def kNN2DDens(self,xv, yv, resolution, neighbours, dim=2):
        # Create the tree
        tree = cKDTree(np.array([xv, yv]).T)
        # Find the closest nnmax-1 neighbors (first entry is the point itself)
        grid = np.mgrid[0:resolution, 0:resolution].T.reshape(resolution ** 2, dim)
        dists = tree.query(grid, neighbours)
        # Inverse of the sum of distances to each grid point.
        inv_sum_dists = 1. / dists[0].sum(1)

        # Reshape
        im = inv_sum_dists.reshape(resolution, resolution)
        return im


class VideoExportThread(QThread):
    '''
    This class is used to store and export live tracking video
    '''

    def __init__(self):
        QThread.__init__(self)
        self.mutex = QMutex()
        self.stopped = False
        self.cam_prop = None
        self.cam_frame = None # frame to be write
        self.codec = 'H264'

    def run(self):
        with QMutexLocker(self.mutex):
            self.stopped = False

    def stop(self):
        with QMutexLocker(self.mutex):
            self.stopped = True

    def camera_frame(self, cam_frame):
        self.cam_frame = cam_frame


class Communicate(QObject):
    # data_progress_start = pyqtSignal(int)
    data_process_fin = pyqtSignal(int)
    trace_process_fin = pyqtSignal(int)
    trace_map = pyqtSignal(QPixmap)
    trace_map_raw = pyqtSignal(object)
    graph_process_fin = pyqtSignal(int)
    heat_map = pyqtSignal(object)


class ExpiredCandidate(object):

    def __init__(self, position,candidate_size, candidate_index, candidate_id,lost_sample):
        self.position = position
        self.candidate_size = candidate_size
        self.candidate_index = candidate_index
        self.candidate_id = candidate_id
        self.lost_sample = lost_sample


class TrackingTimeStamp(object):
    '''
    This class is used to store and export tracking data
    '''

    def __init__(self):

        self.df = []
        # first frame start from 0
        self.result_index = -1
        self.result_index_label = 'Result'

    def update_clock(self):
        # get current time and clock
        get_clock = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        return get_clock

    def local_time_stamp(self, local_elapse, interval=None):
        """Create time stamp when tracking local video files
           and store data based on time stamp mark
        Args:
            local_elapse: time elapsed between each frame by read current position of video files in milisec

        Return:
            self.is_stamp: store data if time mark condition is true
            video_elapse: absolute time elapsed between each frame
                          display while video playing
        """
        self.is_stamp = False
        self.is_min = 1
        is_stampSec = local_elapse % 1000
        is_stampMin = local_elapse % 60000

        # store data every frame
        if interval == None:
            if is_stampSec == 0:
                self.result_index += 1
                # avoid format error when at each second
                video_elapse = f"{str(timedelta(milliseconds=local_elapse)).split('.')[0]}.000"
                self.is_stamp = True
            else:
                self.result_index += 1
                video_elapse = f"{str(timedelta(milliseconds=local_elapse)).split('.')[0]}.{str(timedelta(milliseconds=local_elapse)).split('.')[1][:-3]}"
                self.is_stamp = True
            return self.is_stamp, video_elapse

    def liveTimeStamp(self, live_elapse, interval=None):
        """Create time stamp when tracking live video source
           and store data based on time stamp mark
        Args:
            live_elapse: time elapsed between each frame by read current position of video files in milisec

        Return:
            self.is_stamp: store data if time mark condition is true
            video_elapse: absolute time elapsed between each frame
                          display while video playing
        """
        self.is_stamp = False
        self.is_min = 1  # count how many mintues passed

        # store data every frame
        if interval == None:
            self.result_index += 1
            # video_elapse = f"{str(time.strftime('%H:%M:%S', time.gmtime(live_elapse)))}"
            video_elapse = f"{str(live_elapse).split('.')[0]}.{str(live_elapse).split('.')[1][:-3]}"
            self.is_stamp = True

            return self.is_stamp, video_elapse

    def liveDataFrame(self,clock, video_elapse,tracked_object, id_marks):

        for i in range(len(tracked_object)):
            self.df.append([self.result_index,clock,video_elapse,tracked_object[i].pos_prediction[0][0],
                            tracked_object[i].pos_prediction[1][0], id_marks[i]])

        dataframe = pd.DataFrame(np.array(self.df),
                                 columns=[self.result_index_label, 'Recording Time', 'Time elapsed(s)','pos_x', 'pos_y',
                                          'Object'])

        return dataframe

    def exportData(self, dataframe, save_path):
        is_export = input('Export data in .csv file? Y/N')
        if is_export == 'Y':
            try:
                self.dataToCSV(dataframe, save_path)
                # print(dataframe)
            except Exception as e:
                print(e)
        elif is_export == 'N':
            exit()

    def dataToCSV(self, dataframe, save_path):
        """
        save dataframe as .csv file
        Args:
            dataframe: returned dataframe from local/liveDateFrame function
            save_path: save path and file name of .csv file
        """

        raw_data = dataframe

        raw_data.to_csv(save_path, sep=',', index=False)

    def dataConvert(self, save_path, obj_num, metric):
        """
        calculate basic behavior parameter and convert unit from pixel to scaled metric
        Args:
            save_path: load saved raw data
            obj_num: number of tracking objects
            metric: pixel_per_metric obtained from calibration module
        """

        df = pd.read_csv(save_path)

        # calculate coordinate pixel changes of each object between each frame
        dx = df['pos_x'] - df['pos_x'].shift(obj_num)
        dy = df['pos_y'] - df['pos_y'].shift(obj_num)

        # calculate distance moved between each frame and convert
        # unit using scaled metric obtained from calibration module
        df['Distance (mm)'] = (np.sqrt(dx ** 2 + dy ** 2)) / metric

        df.to_csv(save_path, index=False)



