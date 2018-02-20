import cv2
import numpy as np

import derivative_analysis
import greyscale_img
import segmentation_analysis

def segment_xy_deriv_filter(xy_deriv, segment_border):
    matrix = np.full(xy_deriv.shape, 255)

    for i in range(0, xy_deriv.shape[0]):
        for j in range(0, xy_deriv.shape[1]):
            if xy_deriv[i,j] == 0 and segment_border[i,j] == True:
                matrix[i,j] = 0

    return matrix

def segment_point_of_interest_map(points, segments, threshold=30):
    h = {}    # segment -> number of points of points_of_interest
    matrix = np.full(segments.shape, 255)

    for i in range(0, len(points)):
        s = segments[points[i][0],points[i][1]]
        if s not in h:
            h[s] = 0

        h[s] += 1

    for i in range(0, segments.shape[0]):
        for j in range(0, segments.shape[1]):
            s = segments[i,j]
            if s in h and h[s] >= threshold:
                matrix[i,j] = 0

    return matrix

def bounded_segments(xy_deriv, segments, segment_border):
    h = {}

    for i in range(0, xy_deriv.shape[0]):
        for j in range(0, xy_deriv.shape[1]):
            if segment_border[i,j] == True:
                s = segments[i,j]
                if s not in h:
                    h[s] = (0,0)

                h[s] = (h[s][0], h[s][1]+1)
                if xy_deriv[i,j] == 0:
                    h[s] = (h[s][0]+1,h[s][1])

    return h

# threshold = percentage of a segment of a border that have pixel == 0
def fill_bounded_segments(h, segments, threshold=0.75):
    bounded_segments = {}

    for k,v in h.items():
        if v[0]/v[1] >= threshold:
            bounded_segments[k] = None

    matrix = np.full(segments.shape, 255)

    for i in range(0, segments.shape[0]):
        for j in range(0, segments.shape[1]):
            s = segments[i,j]
            if s in bounded_segments:
                matrix[i,j] = 0

    return matrix

def fill_over_fill_img(over_fill_segments):
    matrix = np.zeros(over_fill_segments.shape)

    # left to right
    for i in range(0, over_fill_segments.shape[0]):
        for j in range(0, over_fill_segments.shape[1]):
            if over_fill_segments[i,j] == 0:
                break
            matrix[i,j] = 255

    # right to left
    for i in range(0, over_fill_segments.shape[0]):
        for j in range(over_fill_segments.shape[1]-1, -1, -1):
            if over_fill_segments[i,j] == 0:
                break
            matrix[i,j] = 255

    # top to bottom
    for j in range(0, over_fill_segments.shape[1]):
        for i in range(0, over_fill_segments.shape[0]):
            if over_fill_segments[i,j] == 0:
                break
            matrix[i,j] = 255

    # bottom to top
    for j in range(0, over_fill_segments.shape[1]):
        for i in range(over_fill_segments.shape[0]-1, -1, -1):
            if over_fill_segments[i,j] == 0:
                break
            matrix[i,j] = 255

    return matrix

## return segment that has a border touching outer white space should be pruned
def overestimated_segments(overestimation, segments):
    prune = {}

    # left to right
    for i in range(0, overestimation.shape[0]):
        for j in range(0, overestimation.shape[1]):
            if overestimation[i,j] == 0:
                prune[segments[i,j]] = None
                break

    # right to left
    for i in range(0, overestimation.shape[0]):
        for j in range(overestimation.shape[1]-1, -1, -1):
            if overestimation[i,j] == 0:
                prune[segments[i,j]] = None
                break

    # top to bottom
    for j in range(0, overestimation.shape[1]):
        for i in range(0, overestimation.shape[0]):
            if overestimation[i,j] == 0:
                prune[segments[i,j]] = None
                break

    # bottom to top
    for j in range(0, overestimation.shape[1]):
        for i in range(overestimation.shape[0]-1, -1, -1):
            if overestimation[i,j] == 0:
                prune[segments[i,j]] = None
                break

    return prune

def prune_segments(overestimation, segments, prune_hash):
    matrix = np.full(overestimation.shape, 255)

    for i in range(0, overestimation.shape[0]):
        for j in range(0, overestimation.shape[1]):
            if overestimation[i,j] == 0 and segments[i,j] not in prune_hash:
                matrix[i,j] = 0

    return matrix

def analysis():
    img = cv2.imread("./images/birdo.jpg", cv2.IMREAD_UNCHANGED)
    grey_img = greyscale_img.generate(img)

    segments = segmentation_analysis.run_slic(img)
    boundaries = segmentation_analysis.run_find_boundaries(segments)
    slic_avg_img, _ = segmentation_analysis.slic_averager(grey_img, segments)

    x_deriv_color = derivative_analysis.x_derivative(img, c_threshold=85, greyscale_img=False)
    x_deriv_filter = segment_xy_deriv_filter(x_deriv_color, boundaries)
    y_deriv_color = derivative_analysis.y_derivative(img, c_threshold=85, greyscale_img=False)
    y_deriv_filter = segment_xy_deriv_filter(y_deriv_color, boundaries)

    xy_deriv_color = derivative_analysis.combine_deriv(x_deriv_color, y_deriv_color)

    x_filter = derivative_analysis.x_deriv_filter(x_deriv_filter)
    y_filter = derivative_analysis.y_deriv_filter(y_deriv_filter)
    xy_filter = derivative_analysis.xy_deriv_filter(x_filter, y_filter)

    bound_segments_dict = bounded_segments(xy_deriv_color, segments, boundaries)
    details = fill_bounded_segments(bound_segments_dict, segments, threshold=0.75)
    over_fill_segments = fill_bounded_segments(bound_segments_dict, segments, threshold=0.15)

    fill_over_fill_segments = fill_over_fill_img(over_fill_segments)
    overestimated = overestimated_segments(fill_over_fill_segments, segments)
    pruned_img = prune_segments(fill_over_fill_segments, segments, overestimated)

    #poi_color_img, points = derivative_analysis.points_of_interest(xy_deriv_color)
    #segment_poi_map = segment_point_of_interest_map(points, segments)

    #cv2.imwrite("./outputs/segment_poi_map.png", segment_poi_map)
    #cv2.imwrite("./outputs/birdo_xy_deriv_color_filtered.png", xy_deriv_color_filtered)
    cv2.imwrite("./outputs/birdo_xy_filter.png", xy_filter)
    cv2.imwrite("./outputs/birdo_filled_segments.png", fill_over_fill_segments)
    cv2.imwrite("./outputs/birdo_pruned_img.png", pruned_img)
    cv2.imwrite("./outputs/birdo_test.png", derivative_analysis.combine_deriv(pruned_img, details))

if __name__ == '__main__':
    analysis()
