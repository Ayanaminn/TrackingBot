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

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QInputDialog, QGraphicsScene, QGraphicsView, QGraphicsLineItem, \
    QGraphicsRectItem, QGraphicsEllipseItem,QMenu
from PyQt5.QtGui import QPainter, QPainterPath, QPen,QBrush,QColor,QTransform
from PyQt5.QtCore import Qt, QPoint, QPointF,QLineF, QRectF, pyqtSignal, QObject
import math


class Calibration(QGraphicsView):

    def __init__(self, parent=None):
        QGraphicsView.__init__(self, parent)
        self.setEnabled(False)
        self.lower()
        self.setGeometry(QtCore.QRect(0, 0, 1024, 576))
        self.scene = CalibrationScene()
        self.setScene(self.scene)
        self.setAlignment(Qt.AlignTop)
        self.setAlignment(Qt.AlignLeft)
        self.setCursor(Qt.CrossCursor)
        # force trasparent to override application style
        self.setStyleSheet("background-color: rgba(0,0,0,0%)")
        rcontent = self.contentsRect()
        self.setSceneRect(0, 0, rcontent.width(), rcontent.height())


class CalibrationScene(QGraphicsScene):

    def __init__(self, parent=None):
        QGraphicsScene.__init__(self, parent)

        self.inputDialog = ScaleInput()
        self.erase_flag = False

        self.begin = QPointF()
        self.end = QPointF()
        self.pen = QPen(Qt.yellow, 2)

        self.new_line = None
        self.arrow = None

        self.lines = []

    def mousePressEvent(self, event):
        self.lines.clear()
        self.erase_flag = False
        self.begin = self.end = event.scenePos()

        # if there are no items at this position.
        if self.itemAt(event.scenePos(), QtGui.QTransform()) is None:
            self.new_line = QGraphicsLineItem()
            self.new_line.setPen(self.pen)
            self.addItem(self.new_line)

            line = QLineF(self.begin.x(), self.begin.y(), self.end.x(), self.end.y())
            self.new_line.setLine(line)
            # init arrow item
            self.arrow_head = ArrowPath(origin=self.begin, destination=self.end, pen=self.pen)
            self.arrow_tail = ArrowPath(origin=self.begin, destination=self.end, pen=self.pen)
            self.addItem(self.arrow_head)
            self.addItem(self.arrow_tail)

        self.update()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        self.end = event.scenePos()

        # set boundary and 1 pixel border margin
        if self.end.x() < 1:
            self.end.setX(1)
        elif self.end.x() > 1023:
            self.end.setX(1023)
        if self.end.y() < 1:
            self.end.setY(1)
        elif self.end.y() > 575:
            self.end.setY(575)

        if self.new_line is not None:
            line = QLineF(self.begin.x(), self.begin.y(), self.end.x(), self.end.y())
            self.new_line.setLine(line)

            self.arrow_head.setDestination(self.end)
            # opposite direction
            self.arrow_tail.setOrigin(self.end)
        self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):

        self.arrow_head.setDestination(self.end)
        self.arrow_tail.setOrigin(self.end)

        self.lines.append(self.new_line)

        # reset
        self.begin = self.end = QPoint()
        self.new_line = None
        self.update()
        super().mouseReleaseEvent(event)

        # ask for metric input
        self.inputDialog.getValue()

    def erase(self):
        self.erase_flag = True
        self.lines.clear()
        self.clear()
        self.update()


class ArrowPath(QtWidgets.QGraphicsPathItem):
    '''
    coordinates at the origin of QGraphicsView
    '''

    def __init__(self, origin, destination, pen):
        super(ArrowPath, self).__init__()

        self.origin_pos = origin
        self.destination_pos = destination

        self._arrow_height = 5
        self._arrow_width = 4

        self.pen = pen

    def setOrigin(self, origin_pos):
        self.origin_pos = origin_pos

    def setDestination(self, destination_pos):
        self.destination_pos = destination_pos

    def directionPath(self):
        path = QtGui.QPainterPath(self.origin_pos)
        path.lineTo(self.destination_pos)
        return path

    # calculates the point of triangles where the arrow should be drawn
    def arrowCalc(self, origin_point=None, des_point=None):

        try:
            originPoint, desPoint = origin_point, des_point

            if origin_point is None:
                originPoint = self.origin_pos

            if desPoint is None:
                desPoint = self.destination_pos

            dx, dy = originPoint.x() - desPoint.x(), originPoint.y() - desPoint.y()

            leng = math.sqrt(dx ** 2 + dy ** 2)
            normX, normY = dx / leng, dy / leng  # normalize

            # perpendicular vector
            perpX = -normY
            perpY = normX

            leftX = desPoint.x() + self._arrow_height * normX + self._arrow_width * perpX
            leftY = desPoint.y() + self._arrow_height * normY + self._arrow_width * perpY

            rightX = desPoint.x() + self._arrow_height * normX - self._arrow_height * perpX
            rightY = desPoint.y() + self._arrow_height * normY - self._arrow_width * perpY

            leftPoint = QtCore.QPointF(leftX, leftY)
            rightPoint = QtCore.QPointF(rightX, rightY)

            return QtGui.QPolygonF([leftPoint, desPoint, rightPoint])

        except (ZeroDivisionError, Exception):
            return None

    def paint(self, painter: QtGui.QPainter, option, widget=None):

        painter.setRenderHint(painter.Antialiasing)
        painter.setPen(self.pen)
        path = self.directionPath()
        self.setPath(path)

        # change path.PointAtPercent() value to move arrow on the line
        triangle_source = self.arrowCalc(path.pointAtPercent(0.1), self.origin_pos)

        if triangle_source is not None:
            painter.drawPolyline(triangle_source)


class ScaleInput(QInputDialog):

    def __init__(self, parent=None):
        QInputDialog.__init__(self, parent)

        self.scale = Communicate()
        self.get_scale = pyqtSignal(str)
        self.scale_value = 1

    def getValue(self):

        num, ok = self.getDouble(self, 'Input Scale',
                              'Enter actual scale (mm):', 1, 1, 1001, flags=Qt.WindowSystemMenuHint)

        if num and ok:
            self.scale_value = num
            # convertScale()
            self.scale.get_scale.emit('1')

        else:  # cancel and reset
            self.scale_value = 1
            # resetScale()
            self.scale.reset_scale.emit('1')


class Communicate(QObject):
    get_scale = pyqtSignal(str)
    reset_scale = pyqtSignal(str)


class DefineROI(QGraphicsView):

    def __init__(self, parent=None):
        QGraphicsView.__init__(self, parent)
        self.setEnabled(False)
        self.lower()
        self.setGeometry(QtCore.QRect(0, 0, 1024, 576))
        self.scene = ROIScene()
        self.setScene(self.scene)
        self.setAlignment(Qt.AlignTop)
        self.setAlignment(Qt.AlignLeft)
        self.setCursor(Qt.CrossCursor)
        # force trasparent to override application style
        self.setStyleSheet("background-color: rgba(0,0,0,0%)")
        rcontent = self.contentsRect()
        self.setSceneRect(0, 0, rcontent.width(), rcontent.height())


class ROIScene(QGraphicsScene):

    def __init__(self, parent=None):
        QGraphicsScene.__init__(self, parent)

        self.erase_flag = False
        self.line_flag = False
        self.rect_flag = False
        self.circ_flag = False

        self.begin = QPointF()
        self.end = QPointF()

        self.new_line = None
        self.new_rect = None
        self.new_circ = None

        self.ROIs = []
        self.ROI_index = 1

    def mousePressEvent(self, event):

        '''
        Draw QGraphicsItem(s) according to current shape flag
        :param event:
        :return:
        '''

        self.erase_flag = False

        # Only enable create new item when no item is in selected state
        if not self.selectedItems():

            if self.itemAt(event.scenePos(), QtGui.QTransform()) is None:

                # this is scene coordinate
                self.begin = self.end = event.scenePos()

                if self.line_flag:

                        self.new_line = QGraphicsLineItem()
                        self.new_line.setPen(self.pen)
                        self.addItem(self.new_line)

                        line = QLineF(self.begin.x(), self.begin.y(), self.end.x(), self.end.y())
                        self.new_line.setLine(line)

                elif self.rect_flag:

                        self.new_rect = RectROI()
                        self.addItem(self.new_rect)
                        rect = QRectF(self.begin, self.end).normalized()
                        self.new_rect.setRect(rect)

                elif self.circ_flag:
                        self.new_circ = EllipseROI()
                        self.addItem(self.new_circ)
                        circ = QRectF(self.begin, self.end).normalized()
                        self.new_circ.setRect(circ)

        else:
            pass

        self.update()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        '''
        Update QGraphicsItem coordinates along with mouse movement
        :param event:
        :return:
        '''

        self.end = event.scenePos()

        # set 1 pixel border margin
        if self.end.x() < 1:
            self.end.setX(1)
        elif self.end.x() > 1023:
            self.end.setX(1023)
        if self.end.y() < 1:
            self.end.setY(1)
        elif self.end.y() > 575:
            self.end.setY(575)

        if self.line_flag:
            if self.new_line is not None:
                line = QLineF(self.begin.x(), self.begin.y(), self.end.x(), self.end.y())
                self.new_line.setLine(line)

        elif self.rect_flag:
            if self.new_rect is not None:
                rect = QRectF(self.begin, self.end).normalized()
                self.new_rect.setRect(rect)

        elif self.circ_flag:
            if self.new_circ is not None:
                rect = QRectF(self.begin, self.end).normalized()
                self.new_circ.setRect(rect)

        self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        '''
        Accept created QGraphicsItem as an ROI object
        :param event:
        :return:
        '''

        if self.begin != self.end:

            if self.line_flag:

                if self.new_line is not None:
                    # new item
                    self.new_line.index = self.ROI_index
                    roi = ROI(self.new_line,self.ROI_index,'line')
                    self.ROIs.append(roi)
                    self.ROI_index += 1

            elif self.rect_flag:
                if self.new_rect is not None:
                    # new item
                    self.new_rect.index = self.ROI_index
                    # create roi object
                    roi = ROI(self.new_rect,self.ROI_index,'rect')
                    self.ROIs.append(roi)
                    self.ROI_index += 1

            elif self.circ_flag:
                if self.new_circ is not None:
                    # new item
                    self.new_circ.index = self.ROI_index
                    roi = ROI(self.new_circ,self.ROI_index,'circ')
                    self.ROIs.append(roi)
                    self.ROI_index += 1

            # reset
            self.begin = self.end = QPointF()
            self.new_line = None
            self.new_rect = None
            self.new_circ = None

        else:

            self.begin = self.end = QPointF()
            self.new_line = None
            self.new_rect = None
            self.new_circ = None

        self.update()
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):

        if event.key() == QtCore.Qt.Key_Delete:
            self.deleteItems()
        else:
            super().keyPressEvent(event)
        self.update()

    def contextMenuEvent(self, event):
        '''
        right click menu
        '''
        # Check it item exists on event position
        item = self.itemAt(event.scenePos(),QTransform())

        if item:
            if item.isSelected():
                menu = QMenu()
                action = menu.addAction('Delete')
                action.triggered.connect(self.deleteItems)
                menu.exec_(event.screenPos())
                return
            else:
                pass

    def deleteItems(self):
        for i in range(len(self.selectedItems())):
            # remove from list
            del self.ROIs[i]
        for i in self.selectedItems():
            # remove from scene
            self.removeItem(i)

    def erase(self):
        self.erase_flag = True
        self.line_flag = False
        self.rect_flag = False
        self.circ_flag = False

        for i in self.ROIs:
            self.removeItem(i.ROI)

        self.ROIs.clear()
        self.ROI_index = 1
        self.clear()
        self.update()

    def drawLine(self):
        self.line_flag = True
        self.rect_flag = False
        self.circ_flag = False

    def drawRect(self):
        self.line_flag = False
        self.rect_flag = True
        self.circ_flag = False

    def drawCirc(self):
        self.line_flag = False
        self.rect_flag = False
        self.circ_flag = True


class ROI(object):
    '''
    Define the ROI object and its properties
    '''
    def __init__(self, ROI, index, type):
        self.ROI = ROI
        self.roi_index = index
        self.type = type


class DefineMask(QGraphicsView):

    def __init__(self, parent=None):
        QGraphicsView.__init__(self, parent)
        self.setEnabled(False)
        self.lower()
        self.setGeometry(QtCore.QRect(0, 0, 1024, 576))
        self.scene = MaskScene()
        self.setScene(self.scene)
        self.setAlignment(Qt.AlignTop)
        self.setAlignment(Qt.AlignLeft)
        self.setCursor(Qt.CrossCursor)
        # force trasparent to override application style
        self.setStyleSheet("background-color: rgba(0,0,0,0%)")
        rcontent = self.contentsRect()
        self.setSceneRect(0, 0, rcontent.width(), rcontent.height())


class MaskScene(QGraphicsScene):

    def __init__(self, parent=None):
        QGraphicsScene.__init__(self, parent)

        self.erase_flag = False
        self.line_flag = False
        self.rect_flag = False
        self.circ_flag = False

        self.begin = QPointF()
        self.end = QPointF()

        self.new_line = None
        self.new_rect = None
        self.new_circ = None

        # list for all shapes for global indexing
        self.Masks = []
        self.Mask_index = 1

    def mousePressEvent(self, event):

        self.erase_flag = False

        # this is main window coordinate
        # Only enable create new item when no item is in selected state
        if not self.selectedItems():
            # if there are no items at this position.
            if self.itemAt(event.scenePos(), QtGui.QTransform()) is None:

                # this is scene coordinate
                self.begin = self.end = event.scenePos()

                if self.line_flag:

                        self.new_line = QGraphicsLineItem()
                        self.new_line.setPen(self.pen)
                        self.addItem(self.new_line)

                        line = QLineF(self.begin.x(), self.begin.y(), self.end.x(), self.end.y())
                        self.new_line.setLine(line)

                elif self.rect_flag:

                        self.new_rect = RectMask()
                        self.addItem(self.new_rect)
                        rect = QRectF(self.begin, self.end).normalized()
                        self.new_rect.setRect(rect)

                elif self.circ_flag:
                        self.new_circ = EllipseMask()
                        self.addItem(self.new_circ)
                        circ = QRectF(self.begin, self.end).normalized()
                        self.new_circ.setRect(circ)

        else:
            pass

        self.update()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):

        self.end = event.scenePos()

        # set 1 pixel border margin
        if self.end.x() < 1:
            self.end.setX(1)
        elif self.end.x() > 1023:
            self.end.setX(1023)
        if self.end.y() < 1:
            self.end.setY(1)
        elif self.end.y() > 575:
            self.end.setY(575)

        if self.line_flag:
            if self.new_line is not None:
                line = QLineF(self.begin.x(), self.begin.y(), self.end.x(), self.end.y())
                self.new_line.setLine(line)

        elif self.rect_flag:
            if self.new_rect is not None:
                rect = QRectF(self.begin, self.end).normalized()
                self.new_rect.setRect(rect)

        elif self.circ_flag:
            if self.new_circ is not None:
                rect = QRectF(self.begin, self.end).normalized()
                self.new_circ.setRect(rect)

        self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):

        if self.begin != self.end:

            if self.line_flag:

                if self.new_line is not None:
                    # new item
                    self.new_line.index = self.Mask_index
                    mask = Mask(self.new_line,self.Mask_index,'line')
                    self.Masks.append(mask)
                    self.Mask_index += 1

            elif self.rect_flag:

                if self.new_rect is not None:
                    # new item
                    self.new_rect.index = self.Mask_index
                    # create mask object
                    mask = Mask(self.new_rect,self.Mask_index,'rect')
                    self.Masks.append(mask)
                    self.Mask_index += 1

            elif self.circ_flag:
                if self.new_circ is not None:
                    # new item
                    self.new_circ.index = self.Mask_index
                    mask = Mask(self.new_circ,self.Mask_index,'circ')
                    self.Masks.append(mask)
                    self.Mask_index += 1

            # reset
            self.begin = self.end = QPointF()
            self.new_line = None
            self.new_rect = None
            self.new_circ = None

        else:

            self.begin = self.end = QPointF()
            self.new_line = None
            self.new_rect = None
            self.new_circ = None

        self.update()
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):

        if event.key() == QtCore.Qt.Key_Delete:
            self.deleteItems()
        else:
            super().keyPressEvent(event)
        self.update()

    def contextMenuEvent(self, event):
        '''
        right click menu on selected item
        '''
        # Check it item exists on event position
        item = self.itemAt(event.scenePos(), QTransform())

        if item:
            if item.isSelected():
                menu = QMenu()
                action = menu.addAction('Delete')
                action.triggered.connect(self.deleteItems)
                menu.exec_(event.screenPos())
                return
            else:
                pass

    def deleteItems(self):

        for i in range(len(self.selectedItems())):
            # remove from list
            del self.Masks[i]
        for i in self.selectedItems():
            # remove from scene
            self.removeItem(i)

    def erase(self):
        self.erase_flag = True
        self.line_flag = False
        self.rect_flag = False
        self.circ_flag = False

        for i in self.Masks:
            self.removeItem(i.Mask)

        self.Masks.clear()
        self.Mask_index = 1
        self.clear()
        self.update()

    def drawLine(self):
        self.line_flag = True
        self.rect_flag = False
        self.circ_flag = False

    def drawRect(self):
        self.line_flag = False
        self.rect_flag = True
        self.circ_flag = False

    def drawCirc(self):
        self.line_flag = False
        self.rect_flag = False
        self.circ_flag = True


class Mask(object):
    '''
    Define the Mask object and its properties
    '''

    def __init__(self, Mask, index, type):
        self.Mask = Mask
        self.mask_index = index
        self.type = type


class DisplayROI(QGraphicsView):
    '''
    used to for display roi and masks in thresholding process as an visual reference only
    '''
    def __init__(self, parent=None):
        QGraphicsView.__init__(self, parent)
        self.setEnabled(False)
        self.lower()
        self.setGeometry(QtCore.QRect(0, 0, 1024, 576))
        self.scene = DisplayScene()
        self.setScene(self.scene)
        self.setAlignment(Qt.AlignTop)
        self.setAlignment(Qt.AlignLeft)
        self.setCursor(Qt.CrossCursor)
        # force trasparent to override application style
        self.setStyleSheet("background-color: rgba(0,0,0,0%)")
        rcontent = self.contentsRect()
        self.setSceneRect(0, 0, rcontent.width(), rcontent.height())


class DisplayScene(QGraphicsScene):

    def __init__(self, parent=None):
        QGraphicsScene.__init__(self, parent)
        self.ROIs = []
        self.Masks = []

        self.roi_pen = QPen(Qt.green,1)
        self.mask_pen = QPen(Qt.black, 1)
        self.roi_brush = QBrush(QColor(255, 255, 255, 0))
        self.mask_brush = QBrush(QColor(0, 0, 0))

    def display_roi(self):
        for i in range(len(self.ROIs)):
            self.ROIs[i].ROI.setPen(self.roi_pen)
            self.ROIs[i].ROI.setBrush(self.roi_brush)
            self.addItem(self.ROIs[i].ROI)

    def display_mask(self):
        for i in range(len(self.Masks)):
            self.Masks[i].Mask.setPen(self.mask_pen)
            self.Masks[i].Mask.setBrush(self.mask_brush)
            self.addItem(self.Masks[i].Mask)

    def erase_roi(self):
        self.ROIs.clear()
        self.clear()
        self.update()

    def erase_mask(self):
        self.Masks.clear()
        self.clear()
        self.update()


class RectROI(QGraphicsRectItem):

    nodeTopLeft = 1
    nodeTopMiddle = 2
    nodeTopRight = 3
    nodeMiddleLeft = 4
    nodeMiddleRight = 5
    nodeBottomLeft = 6
    nodeBottomMiddle = 7
    nodeBottomRight = 8

    # Handle node size (equals radius if node shape is circular)
    nodeSize = +8.0
    # Node center offset against shape boundary, minus towards inside, positive towards outside
    nodeSpace = -4.0

    nodeCursors = {
        nodeTopLeft: Qt.SizeFDiagCursor,
        nodeTopMiddle: Qt.SizeVerCursor,
        nodeTopRight: Qt.SizeBDiagCursor,
        nodeMiddleLeft: Qt.SizeHorCursor,
        nodeMiddleRight: Qt.SizeHorCursor,
        nodeBottomLeft: Qt.SizeBDiagCursor,
        nodeBottomMiddle: Qt.SizeVerCursor,
        nodeBottomRight: Qt.SizeFDiagCursor,
    }

    def __init__(self):
        QGraphicsRectItem.__init__(self)

        self.nodes = {}
        self.node_selected = None
        self.resize_start_pos = None
        self.origin_rect = None
        self.setFlag(self.ItemIsSelectable, True)
        self.setFlag(self.ItemSendsGeometryChanges, True)
        self.setFlag(self.ItemIsFocusable, True)
        self.setFlag(self.ItemIsMovable, True)

        self.setAcceptHoverEvents(True)
        self.index = 0
        # step size when use keyboard to move item
        self.key_x_offset = 1
        self.key_y_offset = 1

    def matchNodePos(self, pos):
        """
        Match input mouse pos with node pos, if match:
        Returns the resize node index.
        """
        # key=node index, value=node coords
        for k, v, in self.nodes.items():
            if v.contains(pos):
                return k
        return None

    def hoverMoveEvent(self, event):
        """
        Executed when the mouse moves over the shape.
        """
        self.setCursor(Qt.SizeAllCursor)
        # when item in selected state , keep track mouse pos to match node pos
        if self.isSelected():
            # update and pass mouse position and pass to node match function
            is_node = self.matchNodePos(event.pos())
            # if mouse pos matches node pos, change cursor to reflect
            cursor = Qt.SizeAllCursor if is_node is None else self.nodeCursors[is_node]
            self.setCursor(cursor)

        super().hoverMoveEvent(event)

    def effectBoundingRect(self):
        """
        Calculate the effective bounding rect that incorporated nodes.
        """
        boundary = self.nodeSize + self.nodeSpace
        return self.rect().adjusted(-boundary, -boundary, boundary, boundary)

    def updateNodePos(self):
        """
        Update nodes position for the item
        """
        s = self.nodeSize
        b = self.effectBoundingRect()
        # update node position in handles dictionary
        self.nodes[self.nodeTopLeft] = QRectF(b.left(), b.top(), s, s)
        self.nodes[self.nodeTopMiddle] = QRectF(b.center().x() - s / 2, b.top(), s, s)
        self.nodes[self.nodeTopRight] = QRectF(b.right() - s, b.top(), s, s)
        self.nodes[self.nodeMiddleLeft] = QRectF(b.left(), b.center().y() - s / 2, s, s)
        self.nodes[self.nodeMiddleRight] = QRectF(b.right() - s, b.center().y() - s / 2, s, s)
        self.nodes[self.nodeBottomLeft] = QRectF(b.left(), b.bottom() - s, s, s)
        self.nodes[self.nodeBottomMiddle] = QRectF(b.center().x() - s / 2, b.bottom() - s, s, s)
        self.nodes[self.nodeBottomRight] = QRectF(b.right() - s, b.bottom() - s, s, s)

    def hoverLeaveEvent(self, event):
        pass

    def mousePressEvent(self, event):
        """
        Executed when the mouse is pressed on the item.
        """
        self.updateNodePos()
        # if pressed pos matches a node pos, return that node index
        # other wise return None
        self.node_selected = self.matchNodePos(event.pos())

        if self.node_selected:
            self.resize_start_pos = event.pos()
            self.origin_rect = self.effectBoundingRect()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """
        Executed when the mouse is being moved over the item while being pressed.
        """

        # if mouse is pressed on a node and moving
        if self.node_selected is not None:

            pos = event.pos()
            self.interactiveResize(pos)
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """
        Executed when the mouse is released from the item.
        """
        self.node_selected = None
        self.resize_start_pos = None
        self.origin_rect = None
        self.update()
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        '''
        Enable move item by keyboard operation
        :param event:
        :return:
        '''
        if self.isSelected():
            if event.key() == QtCore.Qt.Key_Right:
                self.setPos(self.x()+self.key_x_offset,self.y())
            elif event.key() == QtCore.Qt.Key_Left:
                self.setPos(self.x()-self.key_x_offset,self.y())
            elif event.key() == QtCore.Qt.Key_Down:
                self.setPos(self.x(),self.y()+self.key_y_offset)
            elif event.key() == QtCore.Qt.Key_Up:
                self.setPos(self.x(),self.y()-self.key_y_offset)

        super().keyPressEvent(event)

    def interactiveResize(self, mousePos):
        """
        Perform shape interactive resize.
        """
        offset = self.nodeSize + self.nodeSpace
        effectRect = self.effectBoundingRect()
        rect = self.rect()
        diff = QPointF(0, 0)

        self.prepareGeometryChange()

        if self.node_selected == self.nodeTopLeft:

            # start point
            fromX = self.origin_rect.left()
            fromY = self.origin_rect.top()
            # end/new point
            toX = fromX + mousePos.x() - self.resize_start_pos.x()
            toY = fromY + mousePos.y() - self.resize_start_pos.y()
            diff.setX(toX - fromX)
            diff.setY(toY - fromY)
            # update effect rect coords
            effectRect.setLeft(toX)
            effectRect.setTop(toY)
            # update true rect coords
            rect.setLeft(effectRect.left() + offset)
            rect.setTop(effectRect.top() + offset)
            self.setRect(rect)

        elif self.node_selected == self.nodeTopMiddle:

            fromY = self.origin_rect.top()
            toY = fromY + mousePos.y() - self.resize_start_pos.y()

            diff.setY(toY - fromY)
            effectRect.setTop(toY)
            rect.setTop(effectRect.top() + offset)
            self.setRect(rect)

        elif self.node_selected == self.nodeTopRight:

            fromX = self.origin_rect.right()
            fromY = self.origin_rect.top()
            toX = fromX + mousePos.x() - self.resize_start_pos.x()
            toY = fromY + mousePos.y() - self.resize_start_pos.y()
            diff.setX(toX - fromX)
            diff.setY(toY - fromY)
            effectRect.setRight(toX)
            effectRect.setTop(toY)
            rect.setRight(effectRect.right() - offset)
            rect.setTop(effectRect.top() + offset)
            self.setRect(rect)

        elif self.node_selected == self.nodeMiddleLeft:

            fromX = self.origin_rect.left()
            toX = fromX + mousePos.x() - self.resize_start_pos.x()
            diff.setX(toX - fromX)
            effectRect.setLeft(toX)
            rect.setLeft(effectRect.left() + offset)
            self.setRect(rect)

        elif self.node_selected == self.nodeMiddleRight:

            fromX = self.origin_rect.right()
            toX = fromX + mousePos.x() - self.resize_start_pos.x()
            diff.setX(toX - fromX)
            effectRect.setRight(toX)
            rect.setRight(effectRect.right() - offset)
            self.setRect(rect)

        elif self.node_selected == self.nodeBottomLeft:

            fromX = self.origin_rect.left()
            fromY = self.origin_rect.bottom()
            toX = fromX + mousePos.x() - self.resize_start_pos.x()
            toY = fromY + mousePos.y() - self.resize_start_pos.y()
            diff.setX(toX - fromX)
            diff.setY(toY - fromY)
            effectRect.setLeft(toX)
            effectRect.setBottom(toY)
            rect.setLeft(effectRect.left() + offset)
            rect.setBottom(effectRect.bottom() - offset)
            self.setRect(rect)

        elif self.node_selected == self.nodeBottomMiddle:

            fromY = self.origin_rect.bottom()
            toY = fromY + mousePos.y() - self.resize_start_pos.y()
            diff.setY(toY - fromY)
            effectRect.setBottom(toY)
            rect.setBottom(effectRect.bottom() - offset)
            self.setRect(rect)

        elif self.node_selected == self.nodeBottomRight:

            fromX = self.origin_rect.right()
            fromY = self.origin_rect.bottom()
            toX = fromX + mousePos.x() - self.resize_start_pos.x()
            toY = fromY + mousePos.y() - self.resize_start_pos.y()
            diff.setX(toX - fromX)
            diff.setY(toY - fromY)
            effectRect.setRight(toX)
            effectRect.setBottom(toY)
            rect.setRight(effectRect.right() - offset)
            rect.setBottom(effectRect.bottom() - offset)
            self.setRect(rect)

        self.updateNodePos()

    def shape(self):
        """
        Returns the shape of this item as a QPainterPath in local coordinates.
        """
        path = QPainterPath()
        path.addRect(self.rect())
        if self.isSelected():
            for shape in self.nodes.values():
                path.addEllipse(shape)

        return path

    def paint(self, painter, option, widget=None):
        """
        Paint the node in the graphic view.
        """
        painter.setPen(QPen(Qt.green, 1))
        # rgba, alpha channel =0 is full transperant
        painter.setBrush(QBrush(QColor(204, 255, 204, 30)))
        painter.drawRect(self.rect())

        if self.isSelected():
            painter.setBrush(QBrush(QColor(204, 255, 204, 40)))
            for node, rect in self.nodes.items():
                # smooth the outline
                painter.setRenderHint(QPainter.Antialiasing)
                painter.setPen(QPen(QColor(0, 255, 0, 255), 1.0, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
                if self.node_selected is None or node == self.node_selected:
                    painter.drawEllipse(rect)

            painter.drawRect(self.rect())


class EllipseROI(QGraphicsEllipseItem):

    nodeTopLeft = 1
    nodeTopMiddle = 2
    nodeTopRight = 3
    nodeMiddleLeft = 4
    nodeMiddleRight = 5
    nodeBottomLeft = 6
    nodeBottomMiddle = 7
    nodeBottomRight = 8

    # Handle node size (equals radius if node shape is circular)
    nodeSize = +8.0
    # Node center offset against shape boundary, minus towards inside, positive towards outside
    nodeSpace = -4.0

    nodeCursors = {
        nodeTopLeft: Qt.SizeFDiagCursor,
        nodeTopMiddle: Qt.SizeVerCursor,
        nodeTopRight: Qt.SizeBDiagCursor,
        nodeMiddleLeft: Qt.SizeHorCursor,
        nodeMiddleRight: Qt.SizeHorCursor,
        nodeBottomLeft: Qt.SizeBDiagCursor,
        nodeBottomMiddle: Qt.SizeVerCursor,
        nodeBottomRight: Qt.SizeFDiagCursor,
    }

    def __init__(self):
        QGraphicsEllipseItem.__init__(self)

        self.nodes = {}
        self.node_selected = None
        self.resize_start_pos = None
        self.origin_rect = None

        self.setFlag(self.ItemIsSelectable, True)
        self.setFlag(self.ItemSendsGeometryChanges, True)
        self.setFlag(self.ItemIsFocusable, True)
        self.setFlag(self.ItemIsMovable, True)

        self.setAcceptHoverEvents(True)
        self.index = 0
        # step size when use keyborad to move item
        self.key_x_offset = 1
        self.key_y_offset = 1

    def matchNodePos(self, pos):
        """
        Match input mouse pos with node pos, if match:
        Returns the resize node index.
        """
        # key=node index, value=node coords
        for k, v, in self.nodes.items():
            if v.contains(pos):
                return k
        return None

    def hoverMoveEvent(self, event):
        """
        Executed when the mouse moves over the shape.
        """
        self.setCursor(Qt.SizeAllCursor)
        # when item in selected state , keep track mouse pos to match node pos
        if self.isSelected():
            # update and pass mouse position and pass to node match function
            is_node = self.matchNodePos(event.pos())
            # if mouse pos matches node pos, change cursor to reflect
            cursor = Qt.SizeAllCursor if is_node is None else self.nodeCursors[is_node]
            self.setCursor(cursor)

        super().hoverMoveEvent(event)

    def effectBoundingRect(self):
        """
        Calculate the effective bounding rect that incorporated nodes.
        """
        boundary = self.nodeSize + self.nodeSpace
        return self.rect().adjusted(-boundary, -boundary, boundary, boundary)

    def updateNodePos(self):
        """
        Update nodes position for the item
        """
        s = self.nodeSize
        b = self.effectBoundingRect()
        # update node position in handles dictionary
        self.nodes[self.nodeTopLeft] = QRectF(b.left(), b.top(), s, s)
        self.nodes[self.nodeTopMiddle] = QRectF(b.center().x() - s / 2, b.top(), s, s)
        self.nodes[self.nodeTopRight] = QRectF(b.right() - s, b.top(), s, s)
        self.nodes[self.nodeMiddleLeft] = QRectF(b.left(), b.center().y() - s / 2, s, s)
        self.nodes[self.nodeMiddleRight] = QRectF(b.right() - s, b.center().y() - s / 2, s, s)
        self.nodes[self.nodeBottomLeft] = QRectF(b.left(), b.bottom() - s, s, s)
        self.nodes[self.nodeBottomMiddle] = QRectF(b.center().x() - s / 2, b.bottom() - s, s, s)
        self.nodes[self.nodeBottomRight] = QRectF(b.right() - s, b.bottom() - s, s, s)

    def hoverLeaveEvent(self, event):
        pass

    def mousePressEvent(self, event):
        """
        Executed when the mouse is pressed on the item.
        """
        self.updateNodePos()
        # if pressed pos matches a node pos, return that node index
        # other wise return None
        self.node_selected = self.matchNodePos(event.pos())

        if self.node_selected:
            self.resize_start_pos = event.pos()
            self.origin_rect = self.effectBoundingRect()

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """
        Executed when the mouse is being moved over the item while being pressed.
        """

        # if mouse is pressed on a node and moving
        if self.node_selected is not None:

            pos = event.pos()
            self.interactiveResize(pos)
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """
        Executed when the mouse is released from the item.
        """
        self.node_selected = None
        self.resize_start_pos = None
        self.origin_rect = None
        self.update()
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        '''
        Enable move item by keyboard operation
        :param event:
        :return:
        '''
        if self.isSelected():
            if event.key() == QtCore.Qt.Key_Right:
                self.setPos(self.x()+self.key_x_offset,self.y())
            elif event.key() == QtCore.Qt.Key_Left:
                self.setPos(self.x()-self.key_x_offset,self.y())
            elif event.key() == QtCore.Qt.Key_Down:
                self.setPos(self.x(),self.y()+self.key_y_offset)
            elif event.key() == QtCore.Qt.Key_Up:
                self.setPos(self.x(),self.y()-self.key_y_offset)

        super().keyPressEvent(event)

    def interactiveResize(self, mousePos):
        """
        Perform shape interactive resize.
        """
        offset = self.nodeSize + self.nodeSpace
        effectRect = self.effectBoundingRect()
        rect = self.rect()
        diff = QPointF(0, 0)

        self.prepareGeometryChange()

        if self.node_selected == self.nodeTopLeft:

            # start point
            fromX = self.origin_rect.left()
            fromY = self.origin_rect.top()
            # end/new point
            toX = fromX + mousePos.x() - self.resize_start_pos.x()
            toY = fromY + mousePos.y() - self.resize_start_pos.y()
            diff.setX(toX - fromX)
            diff.setY(toY - fromY)
            # update effect rect coords
            effectRect.setLeft(toX)
            effectRect.setTop(toY)
            # update true rect coords
            rect.setLeft(effectRect.left() + offset)
            rect.setTop(effectRect.top() + offset)
            self.setRect(rect)

        elif self.node_selected == self.nodeTopMiddle:

            fromY = self.origin_rect.top()
            toY = fromY + mousePos.y() - self.resize_start_pos.y()

            diff.setY(toY - fromY)
            effectRect.setTop(toY)
            rect.setTop(effectRect.top() + offset)
            self.setRect(rect)

        elif self.node_selected == self.nodeTopRight:

            fromX = self.origin_rect.right()
            fromY = self.origin_rect.top()
            toX = fromX + mousePos.x() - self.resize_start_pos.x()
            toY = fromY + mousePos.y() - self.resize_start_pos.y()
            diff.setX(toX - fromX)
            diff.setY(toY - fromY)
            effectRect.setRight(toX)
            effectRect.setTop(toY)
            rect.setRight(effectRect.right() - offset)
            rect.setTop(effectRect.top() + offset)
            self.setRect(rect)

        elif self.node_selected == self.nodeMiddleLeft:

            fromX = self.origin_rect.left()
            toX = fromX + mousePos.x() - self.resize_start_pos.x()
            diff.setX(toX - fromX)
            effectRect.setLeft(toX)
            rect.setLeft(effectRect.left() + offset)
            self.setRect(rect)

        elif self.node_selected == self.nodeMiddleRight:

            fromX = self.origin_rect.right()
            toX = fromX + mousePos.x() - self.resize_start_pos.x()
            diff.setX(toX - fromX)
            effectRect.setRight(toX)
            rect.setRight(effectRect.right() - offset)
            self.setRect(rect)

        elif self.node_selected == self.nodeBottomLeft:

            fromX = self.origin_rect.left()
            fromY = self.origin_rect.bottom()
            toX = fromX + mousePos.x() - self.resize_start_pos.x()
            toY = fromY + mousePos.y() - self.resize_start_pos.y()
            diff.setX(toX - fromX)
            diff.setY(toY - fromY)
            effectRect.setLeft(toX)
            effectRect.setBottom(toY)
            rect.setLeft(effectRect.left() + offset)
            rect.setBottom(effectRect.bottom() - offset)
            self.setRect(rect)

        elif self.node_selected == self.nodeBottomMiddle:

            fromY = self.origin_rect.bottom()
            toY = fromY + mousePos.y() - self.resize_start_pos.y()
            diff.setY(toY - fromY)
            effectRect.setBottom(toY)
            rect.setBottom(effectRect.bottom() - offset)
            self.setRect(rect)

        elif self.node_selected == self.nodeBottomRight:

            fromX = self.origin_rect.right()
            fromY = self.origin_rect.bottom()
            toX = fromX + mousePos.x() - self.resize_start_pos.x()
            toY = fromY + mousePos.y() - self.resize_start_pos.y()
            diff.setX(toX - fromX)
            diff.setY(toY - fromY)
            effectRect.setRight(toX)
            effectRect.setBottom(toY)
            rect.setRight(effectRect.right() - offset)
            rect.setBottom(effectRect.bottom() - offset)
            self.setRect(rect)

        self.updateNodePos()

    def shape(self):
        """
        Returns the shape of this item as a QPainterPath in local coordinates.
        """
        path = QPainterPath()
        path.addRect(self.rect())
        if self.isSelected():
            for shape in self.nodes.values():
                path.addEllipse(shape)

        return path

    def paint(self, painter, option, widget=None):
        """
        Paint the node in the graphic view.
        """
        painter.setPen(QPen(Qt.green, 1))
        # rgba, alpha channel =0 is full transperant
        painter.setBrush(QBrush(QColor(204, 255, 204, 30)))
        painter.drawEllipse(self.rect())

        if self.isSelected():
            painter.setPen(QPen(QColor(0, 255, 0, 255), 1, Qt.DashLine))
            painter.setBrush(QBrush(QColor(255, 255, 255, 0)))
            painter.drawRect(self.rect())

            painter.setBrush(QBrush(QColor(204, 255, 204, 40)))

            for node, rect in self.nodes.items():
                # smooth the outline
                painter.setRenderHint(QPainter.Antialiasing)
                painter.setPen(QPen(QColor(0, 255, 0, 255), 1.0, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
                if self.node_selected is None or node == self.node_selected:
                    painter.drawEllipse(rect)

            painter.drawEllipse(self.rect())


class RectMask(QGraphicsRectItem):

    nodeTopLeft = 1
    nodeTopMiddle = 2
    nodeTopRight = 3
    nodeMiddleLeft = 4
    nodeMiddleRight = 5
    nodeBottomLeft = 6
    nodeBottomMiddle = 7
    nodeBottomRight = 8

    # Handle node size (equals radius if node shape is circular)
    nodeSize = +8.0
    # Node center offset against shape boundary, minus towards inside, positive towards outside
    nodeSpace = -4.0

    nodeCursors = {
        nodeTopLeft: Qt.SizeFDiagCursor,
        nodeTopMiddle: Qt.SizeVerCursor,
        nodeTopRight: Qt.SizeBDiagCursor,
        nodeMiddleLeft: Qt.SizeHorCursor,
        nodeMiddleRight: Qt.SizeHorCursor,
        nodeBottomLeft: Qt.SizeBDiagCursor,
        nodeBottomMiddle: Qt.SizeVerCursor,
        nodeBottomRight: Qt.SizeFDiagCursor,
    }

    def __init__(self):
        QGraphicsRectItem.__init__(self)

        self.nodes = {}
        self.node_selected = None
        self.resize_start_pos = None
        self.origin_rect = None
        self.setFlag(self.ItemIsSelectable, True)
        self.setFlag(self.ItemSendsGeometryChanges, True)
        self.setFlag(self.ItemIsFocusable, True)
        self.setFlag(self.ItemIsMovable, True)

        self.setAcceptHoverEvents(True)
        self.index = 0
        # step size when use keyborad to move item
        self.key_x_offset = 1
        self.key_y_offset = 1

    def matchNodePos(self, pos):
        """
        Match input mouse pos with node pos, if match:
        Returns the resize node index.
        """
        # key=node index, value=node coords
        for k, v, in self.nodes.items():
            if v.contains(pos):
                return k
        return None

    def hoverMoveEvent(self, event):
        """
        Executed when the mouse moves over the shape.
        """
        self.setCursor(Qt.SizeAllCursor)
        # when item in selected state , keep track mouse pos to match node pos
        if self.isSelected():
            # update and pass mouse position and pass to node match function
            is_node = self.matchNodePos(event.pos())
            # if mouse pos matches node pos, change cursor to reflect
            cursor = Qt.SizeAllCursor if is_node is None else self.nodeCursors[is_node]
            self.setCursor(cursor)

        super().hoverMoveEvent(event)

    def effectBoundingRect(self):
        """
        Calculate the effective bounding rect that incorporated nodes.
        """
        boundary = self.nodeSize + self.nodeSpace
        return self.rect().adjusted(-boundary, -boundary, boundary, boundary)

    def updateNodePos(self):
        """
        Update nodes position for the item
        """
        s = self.nodeSize
        b = self.effectBoundingRect()
        # update node position in handles dictionary
        self.nodes[self.nodeTopLeft] = QRectF(b.left(), b.top(), s, s)
        self.nodes[self.nodeTopMiddle] = QRectF(b.center().x() - s / 2, b.top(), s, s)
        self.nodes[self.nodeTopRight] = QRectF(b.right() - s, b.top(), s, s)
        self.nodes[self.nodeMiddleLeft] = QRectF(b.left(), b.center().y() - s / 2, s, s)
        self.nodes[self.nodeMiddleRight] = QRectF(b.right() - s, b.center().y() - s / 2, s, s)
        self.nodes[self.nodeBottomLeft] = QRectF(b.left(), b.bottom() - s, s, s)
        self.nodes[self.nodeBottomMiddle] = QRectF(b.center().x() - s / 2, b.bottom() - s, s, s)
        self.nodes[self.nodeBottomRight] = QRectF(b.right() - s, b.bottom() - s, s, s)

    def hoverLeaveEvent(self, event):
        pass

    def mousePressEvent(self, event):
        """
        Executed when the mouse is pressed on the item.
        """
        self.updateNodePos()
        # if pressed pos matches a node pos, return that node index
        # other wise return None
        self.node_selected = self.matchNodePos(event.pos())

        if self.node_selected:
            self.resize_start_pos = event.pos()
            self.origin_rect = self.effectBoundingRect()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """
        Executed when the mouse is being moved over the item while being pressed.
        """

        # if mouse is pressed on a node and moving
        if self.node_selected is not None:

            pos = event.pos()
            self.interactiveResize(pos)
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """
        Executed when the mouse is released from the item.
        """
        self.node_selected = None
        self.resize_start_pos = None
        self.origin_rect = None
        self.update()
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        '''
        Enable move item by keyboard operation
        :param event:
        :return:
        '''
        if self.isSelected():
            if event.key() == QtCore.Qt.Key_Right:
                self.setPos(self.x()+self.key_x_offset,self.y())
            elif event.key() == QtCore.Qt.Key_Left:
                self.setPos(self.x()-self.key_x_offset,self.y())
            elif event.key() == QtCore.Qt.Key_Down:
                self.setPos(self.x(),self.y()+self.key_y_offset)
            elif event.key() == QtCore.Qt.Key_Up:
                self.setPos(self.x(),self.y()-self.key_y_offset)

        super().keyPressEvent(event)

    def interactiveResize(self, mousePos):
        """
        Perform shape interactive resize.
        """
        offset = self.nodeSize + self.nodeSpace
        effectRect = self.effectBoundingRect()
        rect = self.rect()
        diff = QPointF(0, 0)

        self.prepareGeometryChange()

        if self.node_selected == self.nodeTopLeft:

            # start point
            fromX = self.origin_rect.left()
            fromY = self.origin_rect.top()
            # end/new point
            toX = fromX + mousePos.x() - self.resize_start_pos.x()
            toY = fromY + mousePos.y() - self.resize_start_pos.y()
            diff.setX(toX - fromX)
            diff.setY(toY - fromY)
            # update effect rect coords
            effectRect.setLeft(toX)
            effectRect.setTop(toY)
            # update true rect coords
            rect.setLeft(effectRect.left() + offset)
            rect.setTop(effectRect.top() + offset)
            self.setRect(rect)

        elif self.node_selected == self.nodeTopMiddle:

            fromY = self.origin_rect.top()
            toY = fromY + mousePos.y() - self.resize_start_pos.y()

            diff.setY(toY - fromY)
            effectRect.setTop(toY)
            rect.setTop(effectRect.top() + offset)
            self.setRect(rect)

        elif self.node_selected == self.nodeTopRight:

            fromX = self.origin_rect.right()
            fromY = self.origin_rect.top()
            toX = fromX + mousePos.x() - self.resize_start_pos.x()
            toY = fromY + mousePos.y() - self.resize_start_pos.y()
            diff.setX(toX - fromX)
            diff.setY(toY - fromY)
            effectRect.setRight(toX)
            effectRect.setTop(toY)
            rect.setRight(effectRect.right() - offset)
            rect.setTop(effectRect.top() + offset)
            self.setRect(rect)

        elif self.node_selected == self.nodeMiddleLeft:

            fromX = self.origin_rect.left()
            toX = fromX + mousePos.x() - self.resize_start_pos.x()
            diff.setX(toX - fromX)
            effectRect.setLeft(toX)
            rect.setLeft(effectRect.left() + offset)
            self.setRect(rect)

        elif self.node_selected == self.nodeMiddleRight:

            fromX = self.origin_rect.right()
            toX = fromX + mousePos.x() - self.resize_start_pos.x()
            diff.setX(toX - fromX)
            effectRect.setRight(toX)
            rect.setRight(effectRect.right() - offset)
            self.setRect(rect)

        elif self.node_selected == self.nodeBottomLeft:

            fromX = self.origin_rect.left()
            fromY = self.origin_rect.bottom()
            toX = fromX + mousePos.x() - self.resize_start_pos.x()
            toY = fromY + mousePos.y() - self.resize_start_pos.y()
            diff.setX(toX - fromX)
            diff.setY(toY - fromY)
            effectRect.setLeft(toX)
            effectRect.setBottom(toY)
            rect.setLeft(effectRect.left() + offset)
            rect.setBottom(effectRect.bottom() - offset)
            self.setRect(rect)

        elif self.node_selected == self.nodeBottomMiddle:

            fromY = self.origin_rect.bottom()
            toY = fromY + mousePos.y() - self.resize_start_pos.y()
            diff.setY(toY - fromY)
            effectRect.setBottom(toY)
            rect.setBottom(effectRect.bottom() - offset)
            self.setRect(rect)

        elif self.node_selected == self.nodeBottomRight:

            fromX = self.origin_rect.right()
            fromY = self.origin_rect.bottom()
            toX = fromX + mousePos.x() - self.resize_start_pos.x()
            toY = fromY + mousePos.y() - self.resize_start_pos.y()
            diff.setX(toX - fromX)
            diff.setY(toY - fromY)
            effectRect.setRight(toX)
            effectRect.setBottom(toY)
            rect.setRight(effectRect.right() - offset)
            rect.setBottom(effectRect.bottom() - offset)
            self.setRect(rect)

        self.updateNodePos()

    def shape(self):
        """
        Returns the shape of this item as a QPainterPath in local coordinates.
        """
        path = QPainterPath()
        path.addRect(self.rect())
        if self.isSelected():
            for shape in self.nodes.values():
                path.addEllipse(shape)

        return path

    def paint(self, painter, option, widget=None):
        """
        Paint the node in the graphic view.
        """
        painter.setPen(QPen(Qt.red, 1))
        # rgba, alpha channel =0 is full transperant
        painter.setBrush(QBrush(QColor(0, 0, 0, 255)))
        painter.drawRect(self.rect())

        if self.isSelected():
            painter.setBrush(QBrush(QColor(0, 0, 0, 40)))
            for node, rect in self.nodes.items():
                # smooth the outline
                painter.setRenderHint(QPainter.Antialiasing)
                painter.setPen(QPen(QColor(255, 0, 0, 255), 1.0, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
                if self.node_selected is None or node == self.node_selected:
                    painter.drawEllipse(rect)

            painter.drawRect(self.rect())


class EllipseMask(QGraphicsEllipseItem):

    nodeTopLeft = 1
    nodeTopMiddle = 2
    nodeTopRight = 3
    nodeMiddleLeft = 4
    nodeMiddleRight = 5
    nodeBottomLeft = 6
    nodeBottomMiddle = 7
    nodeBottomRight = 8

    # Handle node size (equals radius if node shape is circular)
    nodeSize = +8.0
    # Node center offset against shape boundary, minus towards inside, positive towards outside
    nodeSpace = -4.0

    nodeCursors = {
        nodeTopLeft: Qt.SizeFDiagCursor,
        nodeTopMiddle: Qt.SizeVerCursor,
        nodeTopRight: Qt.SizeBDiagCursor,
        nodeMiddleLeft: Qt.SizeHorCursor,
        nodeMiddleRight: Qt.SizeHorCursor,
        nodeBottomLeft: Qt.SizeBDiagCursor,
        nodeBottomMiddle: Qt.SizeVerCursor,
        nodeBottomRight: Qt.SizeFDiagCursor,
    }

    def __init__(self):
        QGraphicsEllipseItem.__init__(self)

        self.nodes = {}
        self.node_selected = None
        self.resize_start_pos = None
        self.origin_rect = None

        self.setFlag(self.ItemIsSelectable, True)
        self.setFlag(self.ItemSendsGeometryChanges, True)
        self.setFlag(self.ItemIsFocusable, True)
        self.setFlag(self.ItemIsMovable, True)

        self.setAcceptHoverEvents(True)
        self.index = 0
        # step size when use keyborad to move item
        self.key_x_offset = 1
        self.key_y_offset = 1

    def matchNodePos(self, pos):
        """
        Match input mouse pos with node pos, if match:
        Returns the resize node index.
        """
        # key=node index, value=node coords
        for k, v, in self.nodes.items():
            if v.contains(pos):
                return k
        return None

    def hoverMoveEvent(self, event):
        """
        Executed when the mouse moves over the shape.
        """
        self.setCursor(Qt.SizeAllCursor)
        # when item in selected state , keep track mouse pos to match node pos
        if self.isSelected():
            # update and pass mouse position and pass to node match function
            is_node = self.matchNodePos(event.pos())
            # if mouse pos matches node pos, change cursor to reflect
            cursor = Qt.SizeAllCursor if is_node is None else self.nodeCursors[is_node]
            self.setCursor(cursor)

        super().hoverMoveEvent(event)

    def effectBoundingRect(self):
        """
        Calculate the effective bounding rect that incorporated nodes.
        """
        boundary = self.nodeSize + self.nodeSpace
        return self.rect().adjusted(-boundary, -boundary, boundary, boundary)

    def updateNodePos(self):
        """
        Update nodes position for the item
        """
        s = self.nodeSize
        b = self.effectBoundingRect()
        # update node position in handles dictionary
        self.nodes[self.nodeTopLeft] = QRectF(b.left(), b.top(), s, s)
        self.nodes[self.nodeTopMiddle] = QRectF(b.center().x() - s / 2, b.top(), s, s)
        self.nodes[self.nodeTopRight] = QRectF(b.right() - s, b.top(), s, s)
        self.nodes[self.nodeMiddleLeft] = QRectF(b.left(), b.center().y() - s / 2, s, s)
        self.nodes[self.nodeMiddleRight] = QRectF(b.right() - s, b.center().y() - s / 2, s, s)
        self.nodes[self.nodeBottomLeft] = QRectF(b.left(), b.bottom() - s, s, s)
        self.nodes[self.nodeBottomMiddle] = QRectF(b.center().x() - s / 2, b.bottom() - s, s, s)
        self.nodes[self.nodeBottomRight] = QRectF(b.right() - s, b.bottom() - s, s, s)

    def hoverLeaveEvent(self, event):
        pass

    def mousePressEvent(self, event):
        """
        Executed when the mouse is pressed on the item.
        """
        self.updateNodePos()
        # if pressed pos matches a node pos, return that node index
        # other wise return None
        self.node_selected = self.matchNodePos(event.pos())

        if self.node_selected:
            self.resize_start_pos = event.pos()
            self.origin_rect = self.effectBoundingRect()

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """
        Executed when the mouse is being moved over the item while being pressed.
        """

        # if mouse is pressed on a node and moving
        if self.node_selected is not None:

            pos = event.pos()
            self.interactiveResize(pos)
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """
        Executed when the mouse is released from the item.
        """
        self.node_selected = None
        self.resize_start_pos = None
        self.origin_rect = None
        self.update()
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        '''
        Enable move item by keyboard operation
        :param event:
        :return:
        '''
        if self.isSelected():
            if event.key() == QtCore.Qt.Key_Right:
                self.setPos(self.x()+self.key_x_offset,self.y())
            elif event.key() == QtCore.Qt.Key_Left:
                self.setPos(self.x()-self.key_x_offset,self.y())
            elif event.key() == QtCore.Qt.Key_Down:
                self.setPos(self.x(),self.y()+self.key_y_offset)
            elif event.key() == QtCore.Qt.Key_Up:
                self.setPos(self.x(),self.y()-self.key_y_offset)

        super().keyPressEvent(event)

    def interactiveResize(self, mousePos):
        """
        Perform shape interactive resize.
        """
        offset = self.nodeSize + self.nodeSpace
        effectRect = self.effectBoundingRect()
        rect = self.rect()
        diff = QPointF(0, 0)

        self.prepareGeometryChange()

        if self.node_selected == self.nodeTopLeft:

            # start point
            fromX = self.origin_rect.left()
            fromY = self.origin_rect.top()
            # end/new point
            toX = fromX + mousePos.x() - self.resize_start_pos.x()
            toY = fromY + mousePos.y() - self.resize_start_pos.y()
            diff.setX(toX - fromX)
            diff.setY(toY - fromY)
            # update effect rect coords
            effectRect.setLeft(toX)
            effectRect.setTop(toY)
            # update true rect coords
            rect.setLeft(effectRect.left() + offset)
            rect.setTop(effectRect.top() + offset)
            self.setRect(rect)

        elif self.node_selected == self.nodeTopMiddle:

            fromY = self.origin_rect.top()
            toY = fromY + mousePos.y() - self.resize_start_pos.y()

            diff.setY(toY - fromY)
            effectRect.setTop(toY)
            rect.setTop(effectRect.top() + offset)
            self.setRect(rect)

        elif self.node_selected == self.nodeTopRight:

            fromX = self.origin_rect.right()
            fromY = self.origin_rect.top()
            toX = fromX + mousePos.x() - self.resize_start_pos.x()
            toY = fromY + mousePos.y() - self.resize_start_pos.y()
            diff.setX(toX - fromX)
            diff.setY(toY - fromY)
            effectRect.setRight(toX)
            effectRect.setTop(toY)
            rect.setRight(effectRect.right() - offset)
            rect.setTop(effectRect.top() + offset)
            self.setRect(rect)

        elif self.node_selected == self.nodeMiddleLeft:

            fromX = self.origin_rect.left()
            toX = fromX + mousePos.x() - self.resize_start_pos.x()
            diff.setX(toX - fromX)
            effectRect.setLeft(toX)
            rect.setLeft(effectRect.left() + offset)
            self.setRect(rect)

        elif self.node_selected == self.nodeMiddleRight:

            fromX = self.origin_rect.right()
            toX = fromX + mousePos.x() - self.resize_start_pos.x()
            diff.setX(toX - fromX)
            effectRect.setRight(toX)
            rect.setRight(effectRect.right() - offset)
            self.setRect(rect)

        elif self.node_selected == self.nodeBottomLeft:

            fromX = self.origin_rect.left()
            fromY = self.origin_rect.bottom()
            toX = fromX + mousePos.x() - self.resize_start_pos.x()
            toY = fromY + mousePos.y() - self.resize_start_pos.y()
            diff.setX(toX - fromX)
            diff.setY(toY - fromY)
            effectRect.setLeft(toX)
            effectRect.setBottom(toY)
            rect.setLeft(effectRect.left() + offset)
            rect.setBottom(effectRect.bottom() - offset)
            self.setRect(rect)

        elif self.node_selected == self.nodeBottomMiddle:

            fromY = self.origin_rect.bottom()
            toY = fromY + mousePos.y() - self.resize_start_pos.y()
            diff.setY(toY - fromY)
            effectRect.setBottom(toY)
            rect.setBottom(effectRect.bottom() - offset)
            self.setRect(rect)

        elif self.node_selected == self.nodeBottomRight:

            fromX = self.origin_rect.right()
            fromY = self.origin_rect.bottom()
            toX = fromX + mousePos.x() - self.resize_start_pos.x()
            toY = fromY + mousePos.y() - self.resize_start_pos.y()
            diff.setX(toX - fromX)
            diff.setY(toY - fromY)
            effectRect.setRight(toX)
            effectRect.setBottom(toY)
            rect.setRight(effectRect.right() - offset)
            rect.setBottom(effectRect.bottom() - offset)
            self.setRect(rect)

        self.updateNodePos()

    def shape(self):
        """
        Returns the shape of this item as a QPainterPath in local coordinates.
        """
        path = QPainterPath()
        path.addRect(self.rect())
        if self.isSelected():
            for shape in self.nodes.values():
                path.addEllipse(shape)

        return path

    def paint(self, painter, option, widget=None):
        """
        Paint the node in the graphic view.
        """
        painter.setPen(QPen(Qt.red, 1))
        # rgba, alpha channel =0 is full transperant
        painter.setBrush(QBrush(QColor(0, 0, 0, 255)))
        painter.drawEllipse(self.rect())

        if self.isSelected():
            painter.setPen(QPen(QColor(255, 0, 0, 255), 1, Qt.DashLine))
            painter.setBrush(QBrush(QColor(255, 255, 255, 0)))
            painter.drawRect(self.rect())

            painter.setBrush(QBrush(QColor(0, 0, 0, 40)))

            for node, rect in self.nodes.items():
                # smooth the outline
                painter.setRenderHint(QPainter.Antialiasing)
                painter.setPen(QPen(QColor(255, 0, 0, 255), 1.0, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
                if self.node_selected is None or node == self.node_selected:
                    painter.drawEllipse(rect)

            painter.drawEllipse(self.rect())