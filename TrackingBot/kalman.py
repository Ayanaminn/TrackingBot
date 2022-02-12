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


class KalmanFilter(object):

    def __init__(self, dt, u_x, u_y, std_acc, x_std_meas, y_std_meas):
        """
        :param dt: sampling time (time for 1 cycle)
        :param u_x: acceleration in x-direction
        :param u_y: acceleration in y-direction
        :param std_acc: process noise magnitude
        :param x_std_meas: standard deviation of the measurement in x-direction
        :param y_std_meas: standard deviation of the measurement in y-direction
        """
        # Define sampling time/delta time/time for one loop(step)
        self.dt = dt

        # Intial State
        self.state = np.array([[0], [0], [0], [0]])

        # Define the State Transition Matrix
        self.F = np.array([[1, 0, self.dt, 0],
                            [0, 1, 0, self.dt],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])

        # Initial Covariance Matrix
        self.P = np.identity(self.F.shape[0])

        # Define the Control Input Matrix
        self.B = np.array([[(self.dt ** 2) / 2, 0],
                            [0, (self.dt ** 2) / 2],
                            [self.dt, 0],
                            [0, self.dt]])

        # Define the  control input
        self.u = np.array([[u_x], [u_y]])

        # Initial Process Noise Covariance
        self.Q = np.array([[(self.dt ** 4) / 4, 0, (self.dt ** 3) / 2, 0],
                            [0, (self.dt ** 4) / 4, 0, (self.dt ** 3) / 2],
                            [(self.dt ** 3) / 2, 0, self.dt ** 2, 0],
                            [0, (self.dt ** 3) / 2, 0, self.dt ** 2]]) * std_acc ** 2

        # Define Measurement Mapping Matrix
        self.H = np.array([[1, 0, 0, 0],[0, 1, 0, 0]])

        # Initial Measurement Noise Covariance
        self.R = np.array([[x_std_meas ** 2, 0],
                            [0, y_std_meas ** 2]])

        # Initial last result to hold previous prediction
        self.previousState = np.array([[0], [255]])

    def predict(self):
        '''
        Predict state vector u and variance of uncertainty P (covariance).
        :return:vector of predicted state estimate
        '''

        # Predict state
        self.state = np.dot(self.F, self.state) + np.dot(self.B, self.u)

        # Calculate error covariance prediction
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

        self.previousState = self.state  # same last predicted result

        return self.state[0:2]

    def update(self, z, flag):
        '''
        Correct or update state vector x and variance of uncertainty P (covariance)
        :param z: vector of observation/measurements
        :param flag:if "true" , update prediction
        :return: predicted state vector x
        '''
        if not flag:  # update using previous prediction
            self.z = self.previousState
        else:  # update using new detection
            self.z = z

        # Measurement residual
        self.residual = self.z - np.dot(self.H, self.state)

        # Measurement prediction covariance
        self.S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R

        # Calculate the Kalman Gain
        self.K = np.dot(self.P,np.dot(self.H.T, np.linalg.inv(self.S)))

        # Update the predicted state
        self.state = self.state + np.dot(self.K, self.residual)

        # Update error covariance matrix
        self.P = np.dot(np.identity(self.P.shape[0]) -
                              np.dot(self.K, self.H), self.P)

        return self.state[0:2]
