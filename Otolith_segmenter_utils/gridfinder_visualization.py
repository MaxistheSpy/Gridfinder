import cv2
import numpy as np
from PIL import Image
def show(*imgs):
    if imgs is None:
        print("nothing")
        return
    Image.fromarray(np.hstack(imgs)).imshow()

def draw_points(img, points):
    ret_img = img.copy()
    for point in points:
        cv2.circle(ret_img, point, 10, 100, -1)
    return ret_img

def visualize_lines_color(img, lines, color=255):
    line_image = np.copy(img) * 0
    if not is_color(img):
        line_image = cv2.cvtColor(line_image, cv2.COLOR_GRAY2BGR)  # creating a blank to draw lines on
    draw_cart_lines_color(line_image, lines, color)
    return line_image


def draw_cart_lines_color(img, lines, color):
    for line in lines:
        draw_cart_line_color(img, line, color)


def draw_cart_line_color(img, line, color):
    (x1, y1), (x2, y2) = line
    cv2.line(img, (x1, y1), (x2, y2), color, 4)

def draw_intersections(img, intersections):
    for intersection in intersections:
        cv2.circle(img, intersection, 5, 100, -1)
    return img

def draw_intersections_color(img, intersections, color=255):
    ret_img = to_color(img.copy())
    for intersection in intersections:
        cv2.circle(ret_img, intersection, 10, color, -1)
    return ret_img


def draw_points_color(img, points, color=255, append=False):
    if not append:
        ret_img = to_color(img.copy())
    if append:
        ret_img = to_color(img)
    for point in points:
        cv2.circle(ret_img, point, 10, color, -1)
    return ret_img


def is_color(img):
    return len(img.shape) == 3


def overlay(first, second):
    first = first.copy()
    second = second.copy()
    if not is_color(first):
        first = cv2.cvtColor(first, cv2.COLOR_GRAY2BGR)  # creating a blank to draw lines on
    if not is_color(second):
        second = cv2.cvtColor(second, cv2.COLOR_GRAY2BGR)  # creating a blank to draw lines on
    return first + second


def to_grey(img):
    if is_color(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def to_color(img):
    if not is_color(img):
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img
