import cv2
import numpy as np
import os

## preprocess the image ##
def first_step(img_path):
    img = cv2.imread(img_path)
    height, width, _ = img.shape
    if height > 700 or width > 700:
        ResizeFactor = 2              ## Higher the value more reduction in size
    else :
        ResizeFactor = 1

    imgResize = cv2.resize(img, (int(width / ResizeFactor), int(height / ResizeFactor)))
    imgContour = imgResize.copy()
    shape = imgResize.shape
    blackImg = np.zeros(shape, np.uint8)
    edge_gray = cv2.cvtColor(blackImg, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.cvtColor(imgResize, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(gray_img, (3, 3), 1) 
    imgCanny = cv2.Canny(gray_img, 50, 500)

    return imgResize, blackImg, gray_img, imgBlur, imgCanny


def Pre_Process(img):
    kernel = np.ones((3, 3), np.uint8)
    imgThreshold = cv2.erode(img, kernel, iterations=0)
    return imgThreshold


def getContours(img,imgResize):
    
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:

        area = cv2.contourArea(cnt)
        if area > 2000:
            approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
            cv2.drawContours(imgResize, approx, -1, (255, 0, 0), 3)
            point = []
            for i in range(4):
                point.append(approx[i][0])
            image_dim = 500
            points = np.float32([point[0], point[3], point[1], point[2]])
            required_size = np.float32([[0, 0], [image_dim, 0], [0, image_dim], [image_dim, image_dim]])
            warp_image = cv2.getPerspectiveTransform(points, required_size)
            warped_image = cv2.warpPerspective(imgResize, warp_image, (image_dim, image_dim))
            cv2.imshow("warped", warped_image)
            
            return warped_image


def find_corners(contour):
    top_left = [10000, 10000]
    top_right = [0, 10000]
    bottom_right = [0, 0]
    bottom_left = [10000, 0]
    mean_x = np.mean(contour[:, :, 0])
    mean_y = np.mean(contour[:, :, 1])

    for j in range(len(contour)):
        x, y = contour[j][0]
        if x > mean_x: 
            if y > mean_y:  
                bottom_right = [x, y]
            else:
                top_right = [x, y]
        else:
            if y > mean_y:  
                bottom_left = [x, y]
            else:
                top_left = [x, y]
    return [top_left, top_right, bottom_right, bottom_left]


def look_for_corners(img_lines, display=False):
    if display:
        img_contours = cv2.cvtColor(img_lines.copy(), cv2.COLOR_GRAY2BGR)
    else:
        img_contours = None

    contours, _ = cv2.findContours(img_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_contours = []
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    biggest_area = cv2.contourArea(contours[0])

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > biggest_area:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4:
                best_contours.append(approx)
        if display:
            cv2.drawContours(img_contours, [cnt], 0, (0, 255, 0), 1)

    if not best_contours:
        if not display:
            return None
        else:
            return None, img_lines, img_contours
    corners = []
    for best_contour in best_contours:
        corners.append(find_corners(best_contour))

    if not display:
        return corners
    else:
        for best_contour in best_contours:
            cv2.drawContours(img_contours, [best_contour], 0, (0, 0, 255), 1)
            for corner in corners:
                for point in corner:
                    x, y = point
                    cv2.circle(img_contours, (x, y), 10, (255, 0, 0), 1)
        return corners, img_lines, img_contours


def crop_from_points(img, corners):
    cnt = np.array([
        corners[0],
        corners[1],
        corners[2],
        corners[3]
    ])

    image_size = 25
    src_pts = np.float32([corners[0], corners[1], corners[2], corners[3]])
    dst_pts = np.float32([[0, 0], [image_size, 0], [0, image_size], [image_size, image_size]])

    ##  perspective transformation matrix ##
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    ## warp the image ##
    warped = cv2.warpPerspective(img, M, (image_size, image_size))

    # transformation_data = {
    #     'matrix': M,
    #     'original_shape': (height, width)
    # }

    return warped


def build_sudoku(sudoku_image, test=False, alpha=1, beta=0):
    edge = cv2.cvtColor(sudoku_image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("test.png", edge)

    ## get height and width of sudoku box ##
    h, w = sudoku_image.shape[0], sudoku_image.shape[1]

    ## specify size of the outer border of sudoku ## 
    sudoku_border = 6

    ## The inner border size ##
    border = 6

    ## divide the sudoku into 9 equal sizes ##
    x = w / 9
    y = h / 9

    ## cell number ##
    cell_num = 0

    for i in range(9):
        for j in range(9):
            # We get the position of each case #
            top = int(round(y * i + border))
            left = int(round(x * j + border))
            right = int(round(x * (j + 1) - border))
            bottom = int(round(y * (i + 1) - border))
            if i == 0:
                top += sudoku_border
            if i == 8:
                bottom -= sudoku_border
            if j == 0:
                left += sudoku_border
            if j == 8:
                right -= sudoku_border

            point = [
                [[left, top]],
                [[right, top]],
                [[left, bottom]],
                [[right, bottom]]
            ]

            ## Crop out the preprocessed case (edges) ##
            square = crop_from_points(edge, point)
            # square = cv2.addWeighted(square, alpha, np.zeros(square.shape, square.dtype), 0, beta)
            if cell_num < 81:
                cv2.imwrite("temp_dir_for_cell/square" + str(cell_num) + ".png", square)
                cell_num += 1
            if test is True:
                if i == 0 and j == 0:
                    cv2.imshow('square', square)
    
    return square


def houghlines(img,blackImg):
    lines = cv2.HoughLines(imgCanny, 1, np.pi / 180, 192)
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            cv2.line(blackImg, pt1, pt2, (255, 255, 255), 3, cv2.LINE_AA)

def color_splice(gray_img):
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.adaptiveThreshold(gray_img, 250, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
    # _, edges = cv2.threshold(imgGray, 150, 255, cv2.THRESH_BINARY_INV)
    return edges


def get_cells(img_path):
    imgResize, blackImg, gray_img, imgBlur, imgCanny = first_step(img_path)
    spliced_img = color_splice(gray_img)
    warped = getContours(spliced_img,imgResize)
    build_sudoku(warped, test=False)
