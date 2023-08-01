from gridfinder_finalized import *
import math
import numpy as np
import cv2


def rotate_lines_numpy(lines, angle, center, useShape=True):
    # WARNING: BUGGY #TODO: fix bug and use to speed algorithm
    flattened_pts = np.array(lines)[:, :, :].reshape(-1, 1, 2)
    rotated_pts = rotate_points_numpy(flattened_pts, angle, center, useShape=useShape)
    rotated_lines = rotated_pts.reshape(-1, 2, 2)
    rotated_lines = ([(tuple(line[1]), tuple(line[0])) for line in rotated_lines])
    return rotated_lines

def rotate_lines_center(lines, angle, center, useShape=True):
    if useShape:
        center = (center[0] // 2, center[1] // 2)
    return [rotate_line(line, angle, center) for line in lines]


def rotate_line(line, angle, center):
    (x1, y1), (x2, y2) = line
    cx, cy = center
    x1, y1 = rotate_point((x1, y1), angle, (cx, cy))
    x2, y2 = rotate_point((x2, y2), angle, (cx, cy))
    return (x1, y1), (x2, y2)


def rotate_points(points, angle, center):
    return [rotate_point(point, angle, center) for point in points]


def rotate_point(point, angle, center):
    angle = math.radians(angle % 360)
    x, y = point
    cx, cy = center
    x, y = x - cx, y - cy
    x, y = x * math.cos(angle) - y * math.sin(angle), x * math.sin(angle) + y * math.cos(angle)
    x, y = x + cx, y + cy
    return int(x), int(y)

def rotate_point_numpy(point, angle, center):
    angle = math.radians(angle % 360)
    x, y = point
    cx, cy = center
    x, y = x - cx, y - cy
    rotation_matrix = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    x, y = np.matmul(rotation_matrix, np.array([x, y]))
    x, y = x + cx, y + cy
    return int(x), int(y)

    # TODO: consider filtering lines outside extrema

def is_outside(line, extrema):
    (x1, y1), (x2, y2) = line
    left_x, right_x, top_y, bot_y = extrema
    if x1 > left_x and x2 > left_x:
        return False
    if x1 < right_x and x2 < right_x:
        return False
    if y1 > top_y and y2 > top_y:
        return False
    if y1 < bot_y and y2 < bot_y:
        return False
    return True

def line_struct(img, scale=SHARPEN_KERNAL_PERCENT, fuzz=1, axis=0):
    size = img.shape
    horizontal_size = int(size[axis] * scale)
    shape = [fuzz, fuzz]
    shape[axis] = horizontal_size
    shape = tuple(shape)
    structure = cv2.getStructuringElement(cv2.MORPH_RECT, shape)
    return structure