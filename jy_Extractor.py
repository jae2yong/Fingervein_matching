
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import numpy as np
import math
from skimage.morphology import skeletonize, thin
from skimage import img_as_ubyte
from Extractor import extract_minutiae_features
from fastdtw import fastdtw

# 이미지 파일 경로 설정

def _meshgrid(xgv, ygv):
    x = np.outer(np.ones_like(ygv), xgv)
    y = np.outer(ygv, np.ones_like(xgv))
    return x, y
def _conv(src, kernel):
    anchor = (kernel.shape[1] - kernel.shape[1] // 2 - 1, kernel.shape[0] - kernel.shape[0] // 2 - 1)
    flipped = cv2.flip(kernel, 0)
    result = cv2.filter2D(src, -1, flipped, anchor=anchor, borderType=cv2.BORDER_REPLICATE)
    return result

def get_imgs_roi(img_file):
    images = os.listdir(img_file)
    for i, image in enumerate(images):
        print(i)
        print(image)
        
        img_raw = cv2.imread(os.path.join(img_file, image), 0)
        img_raw = cv2.resize(img_raw, (664, 250), cv2.INTER_LINEAR)
        #print(img_raw.shape)
        img_raw, img_edge = edge_detection(img_raw)
        img_raw, img_Blur = first_filter(img_raw)
        img_raw, img_Blur_edge = edge_detection(img_Blur)


        img_Blur_edge_polar = pixel_polarization(img_Blur_edge, img_raw, 25) 
        img_Blur_edge_polar_midd, middle_left, middle_right, w1, w2= positioning_middle_point(img_raw, img_Blur_edge_polar, 100)
        img_Blur_edge_polar_midd_rotated, rotated_img = rotation_correction(img_raw, img_Blur_edge_polar_midd, middle_right, middle_left, w1, w2)
        # roi저장
        new_file = './roi_600_2_all_320240'
        save_root = os.path.join(new_file, image)
        roi_img = roi(rotated_img, img_Blur_edge_polar_midd_rotated, w1, w2, save_root)
        cv2.imshow('roi_img', roi_img)
        cv2.waitKey(0)
        resized_roi_img = img_resized_enhance(roi_img, save_root)

def first_filter(img): ##optional
    #均值滤波
    img_Blur=cv2.blur(img,(5,5))
    '''
    #GaussianBlur
    img_GaussianBlur=cv2.GaussianBlur(img,(7,7),0)
    #bilateralFilter
    img_bilateralFilter=cv2.bilateralFilter(img,40,75,75)
    '''
    return img, img_Blur

def edge_detection(img):
    x = cv2.Sobel(img,cv2.CV_16S,1,0)
    y = cv2.Sobel(img,cv2.CV_16S,0,1)
    absX = cv2.convertScaleAbs(x)# 转回uint8
    absY = cv2.convertScaleAbs(y)
    img_edge = cv2.addWeighted(absX,0.5,absY,0.5,0)
    
    #cv2.imshow("absX", absX)
    #cv2.imshow("absY", absY)
    #cv2.imshow("Result", img_edge)
    #cv2.waitKey(0)
    return img, img_edge

def pixel_polarization(img_edge, img, threshold):
    # Check if img_edge is a NumPy array
    if isinstance(img_edge, np.ndarray):
        # Perform element-wise comparison and thresholding
        img_edge = np.where(img_edge > threshold, 255, 0)
    else:
        # Handle the case where img_edge is not a NumPy array
        raise ValueError("img_edge should be a NumPy array")

    # Rest of your code here
    img_edge_polar = img_edge
    return img_edge_polar

def positioning_middle_point(img, dst, point_pixel):
    h, w = img.shape
    w1 = w // 5  
    w2 = (w // 5) * 4 
    print("roi width: ", h, w1, w2)

    low_l = False
    high_l = False
    lower_left = 0
    higher_left = 0
    
    while (not low_l or not high_l) and w1 < (w // 2):
        for i, pix in enumerate(dst[:, w1]):
            if i+1 < (h // 2) and not low_l:
                if pix == 255:
                    low_l = True
                    lower_left = i
            elif i+1 > (h // 2) and not high_l:
                h_h = int(h * (3/2) - (i+1))
                if dst[h_h, w1] == 255:
                    high_l = True
                    higher_left = h_h
        if not low_l or not high_l:
            w1 = w1 + 2

    middle_left = (lower_left + higher_left) // 2

    low_r = False
    high_r = False
    lower_right = 0
    higher_right = 0
    
    while (not low_r or not high_r) and w2 > (w // 2):
        for i, pix in enumerate(dst[:, w2]):
            if i+1 < (h // 2) and not low_r:
                if pix == 255:
                    low_r = True
                    lower_right = i
            elif i+1 > (h // 2) and not high_r:
                h_h = int(h * (3/2) - (i+1))
                if dst[h_h, w2] == 255:
                    high_r = True
                    higher_right = h_h
        if not low_r or not high_r:
            w2 = w2 - 2

    middle_right = (lower_right + higher_right) // 2
    return dst, middle_left, middle_right, w1, w2

def rotation_correction(img, dst, middle_right, middle_left, w1, w2):  ### 필요한지 모르겠음
    # tangent_value = float(middle_right - middle_left) / float(w2 - w1)
    # rotation_angle = np.arctan(tangent_value)/math.pi*180
    # (h,w) = img.shape
    # center = (w // 2,h // 2)
    #
    # #M = cv2.getRotationMatrix2D(center, rotation_angle, 1)
    # M = [[1. 0. 0.][0. 1. 0.]]
    # print("Dimensions of dst:", dst.shape)
    # print("Dimensions of img:", img.shape)
    # print("Transformation matrix M:")
    # print(M)
    # rotated_dst = cv2.warpAffine(dst,M,(w,h))
    # rotated_img = cv2.warpAffine(img,M,(w,h))
    # '''
    # fig = plt.figure(figsize = (16, 16))
    # ax1 = fig.add_subplot(1, 3, 1)
    # ax2 = fig.add_subplot(1, 3, 2)
    # ax3 = fig.add_subplot(1, 3, 3)
    # ax1.imshow(img, cmap = plt.cm.gray)
    # ax2.imshow(rotated_dst, cmap = plt.cm.gray)
    # ax3.imshow(rotated_img, cmap = plt.cm.gray)
    # plt.show()
    # '''
    # return rotated_dst, rotated_img
    return dst, img

def roi(rotated_img, rotated_edge, w1, w2, url):
    h, w = rotated_edge.shape
    print("rotated :", h, w)
    r = range(0, h)
    r1 = range(0, h // 2)
    r2 = range(h // 2, h - 1)
    c = range(0, w)
    c1 = range(0, w // 2)
    c2 = range(w // 2, w-1)
    leftest_edge = w1
    rightest_edge = w2
    highest_edge = (rotated_edge[r1][:,c].sum(axis=1).argmax())
    lowest_edge = (rotated_edge[r2][:,c].sum(axis=1).argmax() + (h // 2))
    print( " w : " , w, "h : " , h)
    print("highest_edge" , highest_edge, "lowest_edge : ", lowest_edge)
    print("leftest_edge", leftest_edge, "rightest_edge", rightest_edge)
    cv2.rectangle(rotated_img,  (leftest_edge, rightest_edge),(lowest_edge, highest_edge), (0,255,0), 2)
    #cv2.rectangle(rotated_img, (leftest_edge, rightest_edge), (lowest_edge, highest_edge), (0, 255, 0), 2)
    #cv2.imshow("Image with Rectangle", rotated_img)
    #cv2.waitKey(0)
    rotated_croped_img = rotated_img[highest_edge
                                     : lowest_edge-30, highest_edge : rightest_edge]
    #new = rotated_img[highest_edge : 250, 0 : 664]
    #print("rotated_croped_img.shape", rotated_croped_img.shape)
    cv2.imshow('rotated_croped_img', rotated_croped_img)
    #cv2.waitKey(0)
    #rotated_croped_img = rotated_img[h : lowest_edge, leftest_edge : w]
    return rotated_croped_img

def img_resized_enhance(img, url):
    #尺度归一化
    print("img_resized_enhance : ", img.shape)
    #resized_img = cv2.resize(img, (320, 240), cv2.INTER_LINEAR) #双线性插值
    resized_img = img
    norm_resized_img = resized_img
    # nom
    norm_resized_img = cv2.normalize(resized_img, norm_resized_img, 0, 255, cv2.NORM_MINMAX)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_resized_img = clahe.apply(norm_resized_img)

    # plt.figure(figsize = (30, 30))
    # plt.subplot(2, 2, 1), plt.title('image')
    # plt.imshow(img, cmap = plt.cm.gray)
    # plt.subplot(2, 2, 2), plt.title('resized_img')
    # plt.imshow(resized_img, cmap = plt.cm.gray)
    # plt.subplot(2, 2, 3), plt.title('norm_resized_img')
    # plt.imshow(norm_resized_img, cmap = plt.cm.gray)
    # plt.subplot(2, 2, 4), plt.title('CLAHE')
    # plt.imshow(clahe_resized_img, cmap = plt.cm.gray)
    # plt.show()
    new_file = './roi_600_2_all_320240'
    print('saving...')
    #保存前一定要创建文件夹
    url_st = url[-11:]

    #cv2.imshow('test', clahe_resized_img)
    #cv2.imwrite(url_st, clahe_resized_img)
    #cv2.imwrite('test.bmp', clahe_resized_img)
    print('done')
    return clahe_resized_img

def extract_minutiae(image, isEndpoint, isBipoint):
    print("image : " , image)

    # image = cv2.cvtColor(image)
    curve = draw_curve_vein(image)
    #cv2.imshow('curve.jpg', curve)
    #cv2.waitKey(0)
    line = skeleton(curve)
    #cv2.imshow('line.jpg', line)
    #cv2.waitKey(0)
    smooth = smooth_image(line)
    #cv2.imshow('sm.jpg', smooth)
    #cv2.waitKey(0)
    dispImg, result, minutiae = feature_extractor(smooth, isEndpoint, isBipoint)

    return curve, result, dispImg, minutiae

def draw_curve_vein(finger):
    # finger = cv2.imread('/media/nguyendung/NguyenDung/FingerVein/VERA/VERA-fingervein/cropped/bf/001-M/001_R_2.png', cv2.IMREAD_GRAYSCALE)
    print("finger : " ,finger)
    finger = cv2.imread(finger, cv2.IMREAD_GRAYSCALE)
    print("finger : " ,finger)
    #cv2.imshow('test', finger)
    #cv2.waitKey(0)
    mask = np.ones(finger.shape, np.uint8)  # Locus space
    result = MaxCurvature(finger, mask, 8)
    _, max_val = cv2.minMaxLoc(result)[:2]
    print(max_val)
    result = ((result * 255.0 / max_val) * 20).astype(np.uint8)
   
    # cv2.imwrite("R2.png", result)
    return result

def MaxCurvature(_src, _mask, sigma):
    src = _src.copy()

    src = src.astype(np.float32) / 255.0

    mask = _mask

    sigma2 = np.power(sigma, 2)
    sigma4 = np.power(sigma, 4)

    # Construct filter kernels
    winsize = math.ceil(4 * sigma)
    X, Y = _meshgrid(range(-winsize, winsize), range(-winsize, winsize))

    # Construct h
    X2 = np.power(X, 2)
    Y2 = np.power(Y, 2)
    X2Y2 = X2 + Y2

    expXY = np.exp(-X2Y2 / (2 * sigma2))
    h = (1 / (2 * np.pi * sigma2)) * expXY

    # Construct hx
    Xsigma2 = -X / sigma2
    hx = np.multiply(h, Xsigma2)

    # Construct hxx
    temp = ((X2 - sigma2) / sigma4)
    hxx = np.multiply(h, temp)

    # Construct hy
    hy = hx.T

    # Construct hyy
    hyy = hxx.T

    # Construct hxy
    XY = np.multiply(X, Y)
    hxy = np.multiply(h, XY / sigma4)

    fx = -_conv(src, hx)
    fxx = _conv(src, hxx)
    fy = _conv(src, hy)
    fyy = _conv(src, hyy)
    fxy = -_conv(src, hxy)

    f1 = 0.5 * math.sqrt(2.0) * (fx + fy)
    f2 = 0.5 * math.sqrt(2.0) * (fx - fy)
    f11 = 0.5 * fxx + fxy + 0.5 * fyy
    f22 = 0.5 * fxx - fxy + 0.5 * fyy

    img_h = src.shape[0]
    img_w = src.shape[1]

    k1 = np.zeros(src.shape, dtype=np.float32)
    k2 = np.zeros(src.shape, dtype=np.float32)
    k3 = np.zeros(src.shape, dtype=np.float32)
    k4 = np.zeros(src.shape, dtype=np.float32)

    # Iterate over the image 
    for x in range(img_w):
        for y in range(img_h):
            p = (y, x)
            if mask[p] > 0:
                k1[p] = fxx[p] / np.power(1 + np.power(fx[p], 2), 1.5)
                k2[p] = fyy[p] / np.power(1 + np.power(fy[p], 2), 1.5)
                k3[p] = f11[p] / np.power(1 + np.power(f1[p], 2), 1.5)
                k4[p] = f22[p] / np.power(1 + np.power(f2[p], 2), 1.5)

    # Scores
    Wr = 0
    Vt = np.zeros(src.shape, dtype=np.float32)
    pos_end = 0

    # Continue converting the rest of the C++ code as above...
    # Horizontal direction
    for y in range(img_h):
        Wr = 0
        for x in range(img_w):
            p = (y, x)
            bla = k1[p] > 0

            if bla:
                Wr += 1

            if Wr > 0 and (x == img_w - 1 or not bla):
                pos_end = x if x == img_w - 1 else x - 1
                pos_start = pos_end - Wr + 1  # Start pos of concave

                pos_max = 0
                max_val = float('-inf')
                for i in range(pos_start, pos_end + 1):
                    value = k1[(y, i)]
                    if value > max_val:
                        pos_max = i
                        max_val = value

                Scr = k1[(y, pos_max)] * Wr
                Vt[(y, pos_max)] += Scr
                Wr = 0

    # Vertical direction
    for x in range(img_w):
        Wr = 0
        for y in range(img_h):
            p = (y, x)
            bla = k2[p] > 0

            if bla:
                Wr += 1

            if Wr > 0 and (y == img_h - 1 or not bla):
                pos_end = y if y == img_h - 1 else y - 1
                pos_start = pos_end - Wr + 1  # Start pos of concave

                pos_max = 0
                max_val = float('-inf')
                for i in range(pos_start, pos_end + 1):
                    value = k2[(i, x)]
                    if value > max_val:
                        pos_max = i
                        max_val = value

                Scr = k2[(pos_max, x)] * Wr
                Vt[(pos_max, x)] += Scr
                Wr = 0

    pos_x_end = 0
    pos_y_end = 0

    # Direction \ .
    for start in range(img_h + img_w - 1):
        # Initial values
        if start < img_w:
            x = start
            y = 0
        else:
            x = 0
            y = start - img_w + 1

        done = False
        Wr = 0

        while not done:
            p = (y, x)
            bla = k3[p] > 0
            if bla:
                Wr += 1

            if Wr > 0 and (y == img_h - 1 or x == img_w - 1 or not bla):
                if y == img_h - 1 or x == img_w - 1:
                    # Reached edge of image
                    pos_x_end = x
                    pos_y_end = y
                else:
                    pos_x_end = x - 1
                    pos_y_end = y - 1

                pos_x_start = pos_x_end - Wr + 1
                pos_y_start = pos_y_end - Wr + 1

                rect = np.s_[pos_y_start:pos_y_end+1, pos_x_start:pos_x_end+1]
                dd = k3[rect]
                d = np.diagonal(dd)

                max_val = float('-inf')
                pos_max = 0
                for i in range(len(d)):
                    value = d[i]
                    if value > max_val:
                        pos_max = i
                        max_val = value

                pos_x_max = pos_x_start + pos_max
                pos_y_max = pos_y_start + pos_max
                Scr = k3[(pos_y_max, pos_x_max)] * Wr

                Vt[(pos_y_max, pos_x_max)] += Scr
                Wr = 0

            if x == img_w - 1 or y == img_h - 1:
                done = True
            else:
                x += 1
                y += 1


    # Direction /
    for start in range(img_h + img_w - 1):
        # Initial values
        if start < img_w:
            x = start
            y = img_h - 1
        else:
            x = 0
            y = img_w + img_h - start - 2

        done = False
        Wr = 0

        while not done:
            p = (y, x)
            bla = k4[p] > 0
            if bla:
                Wr += 1

            if Wr > 0 and (y == 0 or x == img_w - 1 or not bla):
                if y == 0 or x == img_w - 1:
                    # Reached edge of image
                    pos_x_end = x
                    pos_y_end = y
                else:
                    pos_x_end = x - 1
                    pos_y_end = y + 1

                pos_x_start = pos_x_end - Wr + 1
                pos_y_start = pos_y_end + Wr - 1

                rect = np.s_[pos_y_end:pos_y_start+1, pos_x_start:pos_x_end+1]
                roi = k4[rect]
                dd = np.flip(roi, 0)
                d = np.diagonal(dd)

                max_val = float('-inf')
                pos_max = 0
                for i in range(len(d)):
                    value = d[i]
                    if value > max_val:
                        pos_max = i
                        max_val = value

                pos_x_max = pos_x_start + pos_max
                pos_y_max = pos_y_start - pos_max

                Scr = k4[(pos_y_max, pos_x_max)] * Wr

                if pos_y_max < 0:
                    pos_y_max = 0

                Vt[(pos_y_max, pos_x_max)] += Scr
                Wr = 0

            if x == img_w - 1 or y == 0:
                done = True
            else:
                x += 1
                y -= 1

    Cd1 = np.zeros(src.shape, np.float32)
    Cd2 = np.zeros(src.shape, np.float32)
    Cd3 = np.zeros(src.shape, np.float32)
    Cd4 = np.zeros(src.shape, np.float32)
    for x in range(2, src.shape[1] - 3):
        for y in range(2, src.shape[0] - 3):
            p = (y, x)
            Cd1[p] = min(max(Vt[(y, x + 1)], Vt[(y, x + 2)]), max(Vt[(y, x - 1)], Vt[(y, x - 2)]))
            Cd2[p] = min(max(Vt[(y + 1, x)], Vt[(y + 2, x)]), max(Vt[(y - 1, x)], Vt[(y - 2, x)]))
            Cd3[p] = min(max(Vt[(y - 1, x - 1)], Vt[(y - 2, x - 2)]), max(Vt[(y + 1, x + 1)], Vt[(y + 2, x + 2)]))
            Cd4[p] = min(max(Vt[(y - 1, x + 1)], Vt[(y - 2, x + 2)]), max(Vt[(y + 1, x - 1)], Vt[(y + 2, x - 2)]))

    # Connection of vein centres
    veins = np.zeros(src.shape, np.float32)
    for x in range(src.shape[1]):
        for y in range(src.shape[0] - 3):
            p = (y, x)
            veins[p] = max(max(Cd1[p], Cd2[p]), max(Cd3[p], Cd4[p]))

    dst = veins.copy()
    return dst

def skeleton(img):
    _, binary_img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
    binary_img = binary_img / 255.0
    skeleton = skeletonize(binary_img)
    skeleton = img_as_ubyte(skeleton)
    return skeleton

def smooth_image(skeleton_image):

    kernel_size = 3
    structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    dilated_image = cv2.dilate(skeleton_image, structuring_element, iterations=4)
    smoothed_image = cv2.erode(dilated_image, structuring_element, iterations=4)
    return smoothed_image

def feature_extractor(img, isEndpoint, isBipoint):
    # img = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)
    DispImg, result, FeaturesTerminations, FeaturesBifurcations, FeaturesCross = extract_minutiae_features(img, isEndpoint, isBipoint, spuriousMinutiaeThresh=10, 
                                                                                                            invertImage=False, showResult=True, saveResult=False)
    minutiae = []
    if isEndpoint:
        minutiae += FeaturesTerminations
    if isBipoint:
        minutiae += (FeaturesBifurcations + FeaturesCross)
    
    return DispImg, result, minutiae

def knn_matchingScore(img1, minutiae1, img2, minutiae2):
    # Create an ORB object
    orb = cv2.ORB_create()

    # Compute ORB descriptors for the minutiae points in fingerprint1
    keypoints1 = [cv2.KeyPoint(minutia.locY, minutia.locX, size=20) for minutia in minutiae1]
    _, descriptors1 = orb.compute(img1, keypoints1)
    print(keypoints1)
    # Compute ORB descriptors for the minutiae points in fingerprint2
    keypoints2 = [cv2.KeyPoint(minutia.locY, minutia.locX, size=20) for minutia in minutiae2]
    _, descriptors2 = orb.compute(img2, keypoints2)
    print(keypoints2)

    matches = cv2.FlannBasedMatcher(dict(algorithm = 6,
                       table_number = 6, # 12
                       key_size = 6,     # 20
                       multi_probe_level = 1), #2
                dict()).knnMatch(descriptors1, descriptors2, k=2)
    
    match_points = []

    for match in matches:
        if len(match) >= 2:
            p, q = match
            if p.distance < 0.3*q.distance:
                match_points.append(p)

    print(len(match_points))

    keypoints = 0
    if len(keypoints1) <= len(keypoints2):
        keypoints = len(keypoints1)            
    else:
        keypoints = len(keypoints2)

    # if (len(match_points) / keypoints)>0.95:
    print("% match: ", len(match_points) / keypoints * 100)
    
    result = cv2.drawMatches(img1, keypoints1, img2, 
                            keypoints2, match_points, None) 
    # result = cv2.resize(result, None, fx=2.5, fy=2.5)
    #cv2.imshow("result", result)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

def BFMatcher_matchingScore(img1, minutiae1, img2, minutiae2):
    # Create an ORB object
    orb = cv2.ORB_create()

    # Compute ORB descriptors for the minutiae points in fingerprint1
    keypoints1 = [cv2.KeyPoint(minutia.locY, minutia.locX, size=20) for minutia in minutiae1]
    _, descriptors1 = orb.compute(img1, keypoints1)

    # Compute ORB descriptors for the minutiae points in fingerprint2
    keypoints2 = [cv2.KeyPoint(minutia.locY, minutia.locX, size=20) for minutia in minutiae2]
    _, descriptors2 = orb.compute(img2, keypoints2)

    # Matching between descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(descriptors1, descriptors2), key= lambda match:match.distance)

    distance_threshold = 25  # You can adjust this value based on your needs
    good_matches = [match for match in matches if match.distance < distance_threshold]

    # Calculate the matching score
    matching_score = len(good_matches) / len(matches)

    # Print the matching score
    print("Matching Score: ", matching_score * 100, "%")

    img3 = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, flags=2, outImg=None)
    plt.imshow(img3)
    #plt.text((img3.shape[1] / 2) - 20, -20, "Matching score: {}".format(round(matching_score * 100)), color='green', fontsize=12)
    #Hide axes and display the plot
    plt.axis('off')
    plt.show()
    return round(matching_score * 100), img3

get_imgs_roi(r'C:\Users\CVlab\Documents\01_2023\01_2023_BiosyntheticData\FE_finger_vein2\Test2\001_L_1', r'C:\Users\CVlab\Documents\01_2023\01_2023_BiosyntheticData\FE_finger_vein2\Test2\001_L_1')

# _, _, smooth1, minutiae1 = extract_minutiae(r'C:\Users\CVlab\Documents\01_2023\01_2023_BiosyntheticData\FE_finger_vein2\test5.bmp', False, True)
# _, _, smooth2, minutiae2 = extract_minutiae(r'C:\Users\CVlab\Documents\01_2023\01_2023_BiosyntheticData\FE_finger_vein2\test6.bmp', False, True)
# knn_matchingScore(smooth1, minutiae1, smooth2, minutiae2)
#BFMatcher_matchingScore(smooth1, minutiae1, smooth2, minutiae2)
from scipy.spatial.distance import euclidean
#def DTW(img1, minutiae1, img2, minutiae2):
def DTW(img1, minutiae1, img2, minutiae2):

    # 이미지 1의 특징점 (X, Y 좌표) 데이터 예시
    keypoints1 = [cv2.KeyPoint(minutia.locY, minutia.locX, size=20) for minutia in minutiae1]
    keypoints2 = [cv2.KeyPoint(minutia.locY, minutia.locX, size=20) for minutia in minutiae2]
    print("keypoint1 : ", keypoints1)
    # DTW 거리 계산 함수
    # DTW 거리 계산을 위한 특징점 X, Y 좌표 시계열 생성
    x1 = np.array([point.pt[0] for point in keypoints1])
    y1 = np.array([point.pt[1] for point in keypoints1])
    x2 = np.array([point.pt[0] for point in keypoints2])
    y2 = np.array([point.pt[1] for point in keypoints2])

    # X 좌표와 Y 좌표에 대해 각각 DTW 거리 계산
    distance_x, _ = fastdtw(x1, x2)
    distance_y, _ = fastdtw(y1, y2)

    # X 좌표와 Y 좌표 간의 DTW 거리를 합산하여 매칭 점수 계산
    total_distance = distance_x + distance_y

    # 이미지 1과 이미지 2를 수평으로 연결
    combined_image = np.hstack(( smooth1,  smooth2))

    # DTW 거리 정보를 이미지에 표시
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined_image, f'DTW Distance: {total_distance:.2f}', (20, 30), font, 1, (0, 255, 0), 2)

    # 결과 이미지를 화면에 표시
    cv2.imshow('DTW Distance', combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# _, _, smooth1, minutiae1 = extract_minutiae(r'C:\Users\CVlab\Documents\01_2023\01_2023_BiosyntheticData\FE_finger_vein2\001_L_1.bmp', False, True)
# _, _, smooth2, minutiae2 = extract_minutiae(r'C:\Users\CVlab\Documents\01_2023\01_2023_BiosyntheticData\FE_finger_vein2\001_L_2.bmp', False, True)
#     # 결과 출력
#     #print(f"DTW 매칭 점수: {dtw_distance}")
# DTW(smooth1, minutiae1, smooth2, minutiae2)