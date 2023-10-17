import math
import cv2
import imutils
import numpy
import numpy as np
from PIL import Image

LINE_KERNEL_PERCENT = 50 / 2136
SHARPEN_KERNAL_PERCENT = .025
KERNAL_SIZE_TINY = 1
KERNAL_SIZE_MID = 5
KERNAL_SIZE_LARGE = 7
LINE_MIN_LENGTH_PERCENT = 1 / 4
HOLDER_MAX_BRIGHTNESS = 65
# ATTEMPT_DOWNSCALING = False #Implement

### The one function needed to be called from outside this file
def pair_otoliths_to_grid(volume, otolith_points_3D):
    otolith_points_2D = otolith_points_3D[:, :2]  # remove z coord
    # otolith_points_2D = list(map(tuple,otolith_points_2D.tolist()))
    volume_dims = volume.shape
    z_height = volume_dims[0]
    img_dims = volume_dims[1:]
    slice_radius = int(.025 * z_height)
    middle_slice = int(z_height * .5)
    slices = volume[range(middle_slice - slice_radius, middle_slice + slice_radius), :, :]
    preprocessed, skew_angle = process_image_stack(slices)
    horsontal_set, vertical_set, orientation_angle = process_lines(preprocessed)
    rotations = skew_angle + orientation_angle
    intersections = get_intersections(horsontal_set, vertical_set)
    regions = find_ordered_squares(intersections)
    points_by_region = fill_squares(intersections, otolith_points_2D, rotations, img_dims)
    otolith_points_2D_rotated = rotate_points_numpy(otolith_points_2D, rotations, img_dims)
    point_idx_map = {n: i for i, n in enumerate(otolith_points_2D_rotated)}
    otoliths_by_region = [[point_idx_map[n] for n in region] for region in points_by_region]
    return otoliths_by_region, regions, rotations


########################################################################################################################
#Implement function currently unused
def cut_img_stack(img_stack, cut_position=.5,cut_width=.025):
    volume_dims = img_stack.shape
    z_height = volume_dims[0]
    img_dims = volume_dims[1:]
    slice_radius = int(cut_width * z_height)
    middle_slice = int(z_height * cut_position)
    slices = img_stack[range(middle_slice - slice_radius, middle_slice + slice_radius), :, :]
    return slices

def straights(img, scale=SHARPEN_KERNAL_PERCENT, fuzz=1, iter=1, doMask=False, sharpen=False):
    vertical = veritcal_sharpen(img, scale, fuzz, iter, sharpen)
    horizontal = horizontal_sharpen(img, scale, fuzz, iter, sharpen)
    full = tops(np.bitwise_or(vertical, horizontal))
    if doMask:
        return mask_img(img, binarize(full))
    return full


def tops(img):
    top_vals = (img == img.max())
    return (top_vals * 255).astype(dtype=np.uint8)


def draw_cart_lines(line_image, lines):
    for line in lines:
        if len(line) == 2:
            (x1, y1), (x2, y2) = line
        else:
            x1, y1, x2, y2 = line
        cv2.line(line_image, (x1, y1), (x2, y2), 255, 5)
    return line_image


def mask_img(img, mask, anti=False):
    if anti:
        mask = cv2.bitwise_not(mask)
    return cv2.bitwise_and(img, img, mask=mask)


def binarize(img):
    binary_img = (img.astype(dtype=bool) * 255).astype(dtype=np.uint8)
    return binary_img


def closing(img, kern=3, iter=1):
    if type(kern) is not np.ndarray:
        kernel = np.ones((kern, kern), np.uint8)
    else:
        kernel = kern
    closed_img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=iter)
    return closed_img


def opening(img, kern, iter):
    if type(kern) is not np.ndarray:
        kernel = np.ones((kern, kern), np.uint8)
    else:
        kernel = kern
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=iter)


# process image Stack
def apply_batch(function, stack):
    result = list(map(function, stack))
    return np.array(result)


def merge(img):
    return np.average(img, 0).astype(np.uint8)

#TODO: consider speeding algo by raycasting to check instead of rotating full image
def find_rot(img):
    rotations = [imutils.rotate(img, n) for n in range(0, 90)]
    rotation_variance = [np.var(numpy.mean(n, axis=0)) for n in rotations]
    rotation = (np.argmax(rotation_variance))
    return rotation


def straighten(img, angle=None):
    if angle == None:  # TODO: Consider removing
        angle = find_rot(img)
    return imutils.rotate(img, angle)


# TODO: fix this travesty of a function
def blur_and_line(img, sharpen=False):
    close = closing(img, KERNAL_SIZE_TINY, 1)
    ret = opening(close, KERNAL_SIZE_MID, 2)
    ret = closing(ret, 7, 3)
    ret = closing(ret, KERNAL_SIZE_MID, 1)
    ret = cv2.inRange(ret, 1, 255)
    full = straights(ret, doMask=True, sharpen=sharpen)
    return full


def img_cleanup(img, rotate=False, sharpen=False):
    clean = blur_and_line(otolith_threshold(img), sharpen=sharpen)
    if rotate:
        angle = find_rot(clean)
        straightened = imutils.rotate(clean, angle)
        return straightened
    return clean


# todo: look this over once more
def otolith_threshold(img):
    over_threshold = cv2.inRange(img, HOLDER_MAX_BRIGHTNESS, 257)
    over_threshold_blurred = cv2.medianBlur(over_threshold, KERNAL_SIZE_MID)

    dilation_kernel = np.ones((KERNAL_SIZE_MID, KERNAL_SIZE_MID), np.uint8)
    dilated_mask = cv2.dilate(over_threshold_blurred, dilation_kernel, iterations=1)
    masked = mask_img(img, dilated_mask, True)
    return masked


# Process Lines
# Get Intersections
def convert_PP_to_SEL(line):
    (x1, y1), (x2, y2) = line
    if y1 == y2:
        return (x1, x2), y1, 1
    if x1 == x2:
        return x1, (y1, y2), 0


def search_for_intersection(active_horizontal_lines, vertical_line):
    intersections = []
    for active_line in active_horizontal_lines:
        if active_line[1] in range(min(vertical_line[1]), max(vertical_line[1])):
            intersections.append((vertical_line[0], active_line[1]))
    return intersections


# Find Squares
def make_grid(intersections):
    xs = [point[0] for point in intersections]
    ys = [point[1] for point in intersections]
    uxs = sorted(np.unique(xs))
    uys = sorted(np.unique(ys))
    grid = [[(x, y) if ((x, y) in intersections) else None for x in uxs] for y in uys]
    return grid


def get_square(grid, x, y):
    if grid[y][x] is None:
        return None
    if grid[y][x + 1] is None:
        return None
    if grid[y + 1][x] is None:
        return None
    if grid[y + 1][x + 1] is None:
        return None
    top_left = grid[y][x]
    bottom_right = grid[y + 1][x + 1]
    return (top_left, bottom_right)


def get_squares(grid):
    squares = []
    for y in range(len(grid) - 1):
        for x in range(len(grid[y]) - 1):
            square = get_square(grid, x, y)
            if square is not None:
                squares.append(square)
    return squares


def process_image_stack(stack):
    cleaned_stack = apply_batch(img_cleanup, stack)
    cleaned_avg = merge(cleaned_stack)
    skew = find_rot(cleaned_avg)
    straight_slices = apply_batch(lambda x: straighten(x, skew), stack)
    straight_tidy_slices = apply_batch(img_cleanup, straight_slices)
    preprocessed_img = merge(straight_tidy_slices)
    return preprocessed_img, skew


def process_lines(img):
    vertical_blur = extract_vertical(img)
    horizontal_blur = extract_horizontal(img)
    horizontal_lines = to_lines(vertical_blur, 1)
    vertical_lines = to_lines(horizontal_blur, 0)
    boarders, boarder_idx, extrema = get_boarders(horizontal_lines, vertical_lines)
    boarder_left, boarder_right, boarder_top, boarder_bot = boarders
    (idx_left, idx_right, idx_top, idx_bot) = boarder_idx
    (left, right, top, bot) = extrema

    # trimming
    horizontal_lines[idx_top], horizontal_lines[idx_bot] = boarder_top, boarder_bot
    vertical_lines[idx_left], vertical_lines[idx_right] = boarder_left, boarder_right
    horizontal_trimmed = [trim_line(line, right, left, bot, top) for line in horizontal_lines]
    vertical_trimmed = [trim_line(line, right, left, bot, top) for line in vertical_lines]

    # filtering
    horizontal_filtered = remove_short_lines(horizontal_trimmed, img.shape[1] * LINE_MIN_LENGTH_PERCENT)
    vertical_filtered = remove_short_lines(vertical_trimmed, img.shape[0] * LINE_MIN_LENGTH_PERCENT)

    line_cleaned_img = visualize_lines(img, horizontal_trimmed + vertical_trimmed)
    trimmed_img = line_cleaned_img[top:bot, left:right]

    # debug
    # print(vertical_lines)
    # show(vertical_blur, horizontal_blur)
    # show(visualize_lines(img, boarders), visualize_lines(img, vertical_lines))
    # show(visualize_lines(img, vertical_trimmed), visualize_lines(img, horizontal_trimmed),visualize_lines(img, horizontal_lines), visualize_lines(img, vertical_lines))
    # show(visualize_lines(img, horizontal_filtered), visualize_lines(img, vertical_filtered))

    # rotation
    upright_angle = find_upright_angle(trimmed_img)
    # WARNING: while I would like to use numpy rotations, they are not working properly
    horizontal_upright = rotate_lines_right(horizontal_filtered, upright_angle, img, axis=1)
    vertical_upright = rotate_lines_right(vertical_filtered, upright_angle, img, axis=0)
    if upright_angle % 180 == 90:
        horizontal_upright, vertical_upright = vertical_upright, horizontal_upright
    return horizontal_upright, vertical_upright, upright_angle


def find_squares(intersections):
    grid = make_grid(intersections)
    squares = get_squares(grid)
    return squares


def find_ordered_squares(intersections):
    squares = find_squares(intersections)
    squares = sorted(squares, key=lambda x: (x[0][1], x[0][1]))
    return squares


def veritcal_sharpen(img, scale=SHARPEN_KERNAL_PERCENT, fuzz=1, iter=1, sharpen=False):
    rows, cols = img.shape
    verticalsize = int(rows * scale)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (fuzz, verticalsize))
    return opening(img, kern=kernel, iter=iter)


def horizontal_sharpen(img, scale=SHARPEN_KERNAL_PERCENT, fuzz=1, iter=1, sharpen=False):
    rows, cols = img.shape
    horizontal_size = int(cols * scale)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, fuzz))
    return opening(img, kern=kernel, iter=iter)


def sharp_struct(exact_size=None, axis=0, img=None, scale=LINE_KERNEL_PERCENT):
    if exact_size is None:
        exact_size = int(img.shape[axis] * scale)
    sharp = np.repeat([[-1], [2], [-1]], exact_size, 1)
    if axis == 0:
        return sharp.T
    return sharp


def extract_axis(img, axis):
    # create sharpening kernal
    binary_img = tops(img)
    struct_length = int(img.shape[axis] * LINE_KERNEL_PERCENT)
    struct = sharp_struct(exact_size=struct_length, axis=axis)

    # apply sharpening kernal
    ret = closing(binary_img, struct, iter=1)
    ret = opening(ret, struct, iter=4)
    ret = closing(ret, struct, iter=10)
    ret = opening(ret, struct, iter=10)
    return ret


def extract_horizontal(img):
    return extract_axis(img, axis=0)


def extract_vertical(img):
    return extract_axis(img, axis=1)


def extract_orthogonal(img):
    return extract_vertical(img), extract_horizontal(img)


def to_lines(img, axis=0):
    num_labels, label_ids, values, centroids = cv2.connectedComponentsWithStats(img, 8)
    parts = []
    for i in range(1, num_labels):
        w, h = values[i, cv2.CC_STAT_WIDTH], values[i, cv2.CC_STAT_HEIGHT]
        x1, y1 = values[i, cv2.CC_STAT_LEFT], values[i, cv2.CC_STAT_TOP]
        x2, y2 = (x1 + w, y1 + h)
        (X, Y) = centroids[i]
        if axis == 0:
            parts.append(((int(X), y1), (int(X), y2)))
        else:
            parts.append(((x1, int(Y)), (x2, int(Y))))
    return parts


def visualize_lines(img, lines):
    line_image = np.copy(img) * 0  # creating a blank to draw lines on
    draw_cart_lines(line_image, lines)
    return line_image


def get_boarders(horisontal_lines, vertical_lines):
    heights = ([x[0][1] for x in horisontal_lines])
    distances = ([x[0][0] for x in vertical_lines])
    idx_bot, idx_top = np.argmin(heights), np.argmax(heights)
    idx_left, idx_right = np.argmin(distances), np.argmax(distances)
    heights = sorted(heights)
    distances = sorted(distances)
    left, right = distances[0], distances[-1]
    top, bot = heights[0], heights[-1]
    second_top, second_bot = sorted(heights)[1], sorted(heights)[-2]
    second_left, second_right = sorted(distances)[1], sorted(distances)[-2]
    boarder_left = ((left, second_top), (left, second_bot))
    boarder_right = ((right, second_top), (right, second_bot))
    boarder_top = ((second_left, top), (second_right, top))
    boarder_bot = ((second_left, bot), (second_right, bot))
    boarders = (boarder_left, boarder_right, boarder_top, boarder_bot)
    boarder_idxs = (idx_left, idx_right, idx_top, idx_bot)
    extrema = (left, right, top, bot)
    return boarders, boarder_idxs, extrema


def find_upright_angle(img):
    angles = [n for n in range(0, 360, 90)]
    rotations = [imutils.rotate(img, angle) for angle in angles]
    rotation_runs = [max(first_zero_run_length_2D(rotation)) for rotation in rotations]
    rotation_angle = angles[np.argmax(rotation_runs)]
    return rotation_angle


def clamp(n, low_val, high_val):
    return max(min(high_val, n), low_val)


def trim_point(point, x_high, x_low, y_high, y_low):
    x, y = point
    x, y = clamp(x, x_low, x_high), clamp(y, y_low, y_high)
    return x, y


def trim_line(line, x_high, x_low, y_high, y_low):
    (x1, y1), (x2, y2) = line
    x1, y1 = trim_point((x1, y1), x_high, x_low, y_high, y_low)
    x2, y2 = trim_point((x2, y2), x_high, x_low, y_high, y_low)
    return (x1, y1), (x2, y2)


def is_point_in_square(point, square):
    (x1, y1), (x2, y2) = square
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])
    x, y = point
    if x in range(x1, x2) and y in range(y1, y2):
        return True
    return False


def get_line_distance(line, axis=None):
    (x1, y1), (x2, y2) = line
    if axis is None:
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    if axis == 0:
        return abs(x1 - x2)
    if axis == 1:
        return abs(y1 - y2)


def first_zero_run_length(vector):
    i = 0
    while i < len(vector) and vector[i] == 0:
        i += 1
    return i


def first_zero_run_length_2D(array):
    return np.apply_along_axis(first_zero_run_length, 1, array)


def remove_short_lines(lines, length=10):
    return [line for line in lines if get_line_distance(line) > length]


# function that rotates a set of lines by angle degrees, by visualizing them into a matrix then rotating that usign imutils then converting them back to lines using to_lines

def rotate_lines_right(lines, angle, img, axis):
    axis = (axis - ((angle // 90) % 2))
    matrix = visualize_lines(img.copy() * 0, lines)
    rotated_matrix = imutils.rotate(matrix, angle)
    rotated_lines = to_lines(rotated_matrix, axis)
    return rotated_lines


# use a rotation matrix to rotate line around a centeral point
def rotate_points_numpy(points, angle, center, useShape=True):
    if useShape:
        center = (center[0] // 2, center[1] // 2)
    angle = math.radians(angle % 360)
    cx, cy = center
    points = np.array(points)
    points = points - np.array([cx, cy])
    rotation_matrix = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    points = np.matmul(points, rotation_matrix)
    points = points + np.array([cx, cy])
    return list(map(tuple,points.astype(int).tolist()))


def get_intersections(h_lines, v_lines):
    # TODO: Clean up this mess of a function
    h_lines = [convert_PP_to_SEL(line) for line in h_lines]
    v_lines = [convert_PP_to_SEL(line) for line in v_lines]
    # Make dictionaries
    h_dict = {}
    for line in h_lines:
        for n in range(2):
            x = line[0][n]
            if x not in h_dict:
                h_dict[x] = []
            h_dict[x] = h_dict[x] + [line]
    v_dict = {}
    for line in v_lines:
        x = line[0]
        if x not in v_dict:
            v_dict[x] = []
        v_dict[x] = v_dict[x] + [line]

    # Find intersections
    x_cords_in_order = sorted(list(h_dict.keys()) + list(v_dict.keys()))
    currently_active = []
    intersections = []
    toggle = lambda active_line_list, point: active_line_list.remove(
        point) if point in active_line_list else active_line_list.append(point)
    for x in x_cords_in_order:
        x_is_horizontal = x in h_dict
        x_is_vertical = x in v_dict
        if x_is_vertical and x_is_horizontal:
            intersections.append((x, h_dict[x][1]))
        if x_is_vertical:
            for line in v_dict[x]:
                intersections += search_for_intersection(currently_active, line)
        if x in h_dict:
            # map(lambda x: toggle(currently_active, x), h_dict[x])
            for line in h_dict[x]:
                toggle(currently_active, line)
                # TODO: consider compressing this into one function that reads if the line is vertical or horizontal and then does the appropriate thing
    return intersections


def fill_squares(intersections, points, rotation=0, dims=None):
    if rotation != 0:
        points = rotate_points_numpy(points, rotation, dims)
    squares = find_ordered_squares(intersections)
    filled_squares = [[point for point in points if is_point_in_square(point, sq)] for sq in squares]
    return filled_squares


## DEBUG
#import_volume = np.load("/home/max/Projects/test/junk/holderv.npy")
#otolith_positions = np.load("./otolith_pos.npy")
#square_to_points_out, squares_out, rotations_out = pair_otoliths_to_grid(import_volume, otolith_positions)
#print(list(enumerate(square_to_points_out)), square_to_points_out, squares_out, rotations_out)
# print(square_to_points)
# show(-lsiv, lsiv + pointed_dg)
# horsontal_set_raw, vertical_set_raw, rotated_angle = process_lines(preprocessed)
# line_set_raw = horsontal_set_raw + vertical_set_raw
# image = visualize_lines(preprocessed, line_set_raw)
# rotted_lines = rotate_lines_numpy(line_set_raw, 180, preprocessed.shape)
# show(image, visualize_lines(preprocessed, rotted_lines))
