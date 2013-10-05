
import numpy as np
from scipy import ndimage

def isolate(im, pts):
    pts = np.array(pts)
    pts = pts[pts[:, 0].argsort()]
    min_r = min(pts[:, 0])
    max_r = max(pts[:, 0])
    min_c = min(pts[:, 1])
    max_c = max(pts[:, 1])
    im = im[min_r:max_r, min_c:max_c]
    pts = [(r - min_r, c - min_c) for r, c in pts]
    return im, pts

def align_top(im, pts):
    pts = np.array(pts)
    height_change = pts[1][0] - pts[0][0]
    angle = np.arctan2(pts[1][0] - pts[0][0], pts[1][1] - pts[0][1])
    R = np.matrix([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    im = ndimage.interpolation.rotate(im, np.rad2deg(angle))
    pts_mat = np.matrix(pts).T
    pts_mat = R * pts_mat
    pts = [(int(pt[0, 0] + height_change), int(pt[0, 1])) for pt in pts_mat.T]
    return im, pts

def straighten_left(im, pts):
    # Transforms a parallelogram to a rectangle
    # Input parallelogram base should be horizontal
    pts = np.array(pts)
    pts = pts[pts[:, 1].argsort()]
    slope = float(pts[1][1] - pts[0][1]) / (pts[1][0] - pts[0][0])
    transform = lambda pt: (pt[0], pt[1] - slope * pt[0])
    inverse_transform = lambda pt: (pt[0], pt[1] + slope * pt[0])
    im = ndimage.interpolation.geometric_transform(im, inverse_transform)
    pts = [map(int, transform(pt)) for pt in pts]
    return im, pts
    

def regularize(im, pts):
    pts = np.array(pts)
    im, pts = isolate(im, pts)
    im, pts = align_top(im, pts)
    im, pts = isolate(im, pts)
    im, pts = straighten_left(im, pts)
    im, pts = isolate(im, pts)
    return im, pts
