#!/usr/bin/env python

import argparse
import requests
import tempfile
import numpy as np
from scipy import misc, ndimage
from matplotlib import pyplot
from transform import regularize


class Camera(object):
    def __init__(self, url):
        self.url = url

    def capture(self):
        with tempfile.NamedTemporaryFile() as f:
            f.write(requests.get(self.url).content)
            f.seek(0)
            img = misc.imread(f.name, True)
        return img


class Window(object):
    def __init__(self, camera):
        self.camera = camera
        figure = pyplot.figure()
        self.raw_axes = figure.add_subplot(121)
        self.target_axes = figure.add_subplot(122)
        self.raw_data = None
        self.raw_im = None
        self.target_data = None
        self.target_im = None
        self.target_bounds = None
        self.target_vertices = None
        self.pressed_index = None

    def show(self):
        self.update()
        pyplot.show()

    def update(self):
        self.raw_data = self.camera.capture()
        if self.raw_im is None:
            self.raw_im = self.raw_axes.imshow(self.raw_data)
            self.raw_axes.set_autoscale_on(False)
        else:
            self.raw_im.set_data(self.raw_data)

        if self.target_bounds is None:
            h, w = self.raw_data.shape
            self.target_vertices = self.raw_axes.scatter([w/4, 3*w/4, 3*w/4, w/4],
                                                         [h/4, h/4, 3*h/4, 3*h/4])
            self.target_bounds = pyplot.Polygon(self.target_vertices.get_offsets(), fill=False)
            self.raw_axes.add_patch(self.target_bounds)
            self.connect()
            self.show_target_image()

    def show_target_image(self):

        pts = [(y,x) for x, y in self.target_vertices.get_offsets()]
        self.target_data, transformed_pts = regularize(self.raw_data, pts)
        if self.target_im is None:
            self.target_im = self.target_axes.imshow(self.target_data)
        else:
            self.target_im.set_data(self.target_data)
            self.target_im.figure.canvas.draw()

    def connect(self):
        self.press_cid = self.target_vertices.figure.canvas.mpl_connect(
                'button_press_event', self.on_press)
        self.release_cid = self.target_vertices.figure.canvas.mpl_connect(
                'button_release_event', self.on_release)
        self.press_cid = self.target_vertices.figure.canvas.mpl_connect(
                'motion_notify_event', self.on_motion)

    def on_press(self, event):
        if event.inaxes != self.target_vertices.axes:
            return
        picked, info = self.target_vertices.contains(event)
        if picked:
            self.pressed_index = info['ind'][0]

    def on_release(self, event):
        self.pressed_index = None
        self.target_vertices.figure.canvas.draw()
        self.show_target_image()

    def on_motion(self, event):
        if self.pressed_index is None:
            return
        if event.inaxes != self.target_vertices.axes:
            return
        pts = self.target_vertices.get_offsets()
        pts[self.pressed_index] = (event.xdata, event.ydata)
        self.target_bounds.set_xy(pts)
        self.target_vertices.figure.canvas.draw()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read text from an webcam.')
    parser.add_argument('url', help='A url for the webcam image.')
    args = parser.parse_args()
    camera = Camera(args.url)
    window = Window(camera)
    window.show()
