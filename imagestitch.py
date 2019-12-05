import numpy as np
import random
import cv2
import skimage
import matplotlib.pyplot as plt

def load(path, scale):
    # imageleft = cv2.imread(path+'uttower_left.jpg', scale)
    # imageright = cv2.imread(path+'uttower_right.jpg', scale)
    imageleft = cv2.imread(path+'bbb_left.jpg', scale)
    imageright = cv2.imread(path+'bbb_right.jpg', scale)
    # iml = skimage.img_as_float64(imageleft)
    # imr = skimage.img_as_float64(imageright)
    return imageleft, imageright

def detect(image, index):
    sift = cv2.xfeatures2d.SIFT_create(400)
    kp, des = sift.detectAndCompute(image,None)
    img=cv2.drawKeypoints(image,kp,image, flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    cv2.imwrite("p2/siftimagebbb" + str(index) + ".png", img)
    return kp, des

def findclosest(descriptors):
    dl = descriptors[0]
    dr = descriptors[1]
    for i in range(len(dl)):
        dl[i] = (dl[i] - dl[i].mean()) / dl[i].var()
    for i in range(len(dr)):
        dr[i] = (dr[i] - dr[i].mean()) / dr[i].var()
    distance = np.zeros((len(dl), len(dr)))
    for i in range(len(dl)):
        for j in range(len(dr)):
            distance[i][j] = np.linalg.norm(dl[i] - dr[j])
    return distance

def selectmatches(distance, keypoints):
    thres = 0.8
    pairs = []
    for i in range(len(distance)):
        ind = np.argpartition(distance[i], 2)[:2]
        s1 = distance[i][ind[0]]
        s2 = distance[i][ind[1]]
        if (s1/s2) < thres:
            pairs.append((keypoints[0][i], keypoints[1][ind[0]]))
    return pairs

def getcoords(pairs):
    coord = np.zeros((len(pairs), 4))
    for i in range(len(pairs)):
        p = pairs[i]
        coord[i][0] = p[0].pt[0]
        coord[i][1] = p[0].pt[1]
        coord[i][2] = p[1].pt[0]
        coord[i][3] = p[1].pt[1]
    return coord

def fitmodel(data):
    num = len(data)
    A = np.zeros((2*num, 9))
    for i in range(num):
        A[2*i][3] = -data[i][0]
        A[2*i][4] = -data[i][1]
        A[2*i][5] = -1
        A[2*i][6] = data[i][3] * data[i][0]
        A[2*i][7] = data[i][3] * data[i][1]
        A[2*i][8] = data[i][3]
        A[2*i+1][0] = data[i][0]
        A[2*i+1][1] = data[i][1]
        A[2*i+1][2] = 1
        A[2*i+1][6] = -data[i][2] * data[i][0]
        A[2*i+1][7] = -data[i][2] * data[i][1]
        A[2*i+1][8] = -data[i][2]
    _, _, v = np.linalg.svd(A)
    h = np.reshape(v[8], (3, 3))
    h = h / np.linalg.norm(v[8])
    h = (1/h.item(8))*h
    return h

def calculatedist(point, h):
    p1 = np.array([point[0], point[1], 1])
    p2 = np.array([point[2], point[3], 1])
    ep2 = h.dot(p1.T)
    ep2 = (1.0/ep2[2])*ep2
    return np.linalg.norm(ep2 - p2)

def ransac(coords):
    NUM_TRIALS = 100
    THRES = 5
    best, bestinliers, bestcount, besterror, bestindex = None, None, -1, 0, None
    for trial in range(NUM_TRIALS):
        setindex = random.choices(range(len(coords)), k=4)
        subset = coords[setindex]
        h = fitmodel(subset)
        inliers = []
        indexes = []
        all_error = 0
        for i in range(len(coords)):
            d = calculatedist(coords[i], h)
            if d < THRES:
                inliers.append(coords[i])
                indexes.append(i)
                all_error += d
        if len(inliers) > bestcount:
            bestcount = len(inliers)
            best = h
            bestinliers = inliers
            besterror = all_error
            bestindex = indexes
    best = fitmodel(bestinliers)
    return best, bestcount, bestinliers, besterror, bestindex

def warpimages(img1, img2, h):
    h1,w1 = img1.shape[0], img1.shape[1]
    h2,w2 = img2.shape[0], img2.shape[1]
    corner1 = np.array([[0,0,w1,w1], [0,h1,h1,0], [1,1,1,1]], np.float32)
    corner2 = np.array([[0,0,w2,w2], [0,h2,h2,0]], np.float32)
    sc1 = h.dot(corner1)
    sc1 = ((1/sc1[2])*sc1)[0:2, :]
    allcorner = np.concatenate((sc1, corner2), axis=1)
    xmin, ymin = (allcorner.min(axis=1) - 0.5).astype('int')
    xmax, ymax = (allcorner.max(axis=1) + 0.5).astype('int')
    ht = np.array([[1,0,-xmin],[0,1,-ymin],[0,0,1]])

    result = cv2.warpPerspective(img1, ht.dot(h), (xmax-xmin, ymax-ymin))

    for i in range(-ymin, -ymin+h2):
        for j in range(-xmin, -xmin+w2):
            if result[i][j][0] == 0 and result[i][j][1] == 0 and result[i][j][2] == 0:
                result[i][j] = img2[i+ymin][j+xmin]
            else:
                result[i][j][0] = (int(result[i][j][0]) + int(img2[i+ymin][j+xmin][0]))/2
                result[i][j][1] = (int(result[i][j][1]) + int(img2[i+ymin][j+xmin][1]))/2
                result[i][j][2] = (int(result[i][j][2]) + int(img2[i+ymin][j+xmin][2]))/2
    return result

def main():
    #images = [left, right]
    images = load('p2/', 0)
    descriptors = []
    keypoints = []
    for i in range(len(images)):
        kp, des = detect(images[i], i)
        keypoints.append(kp)
        descriptors.append(des)
    distance = findclosest(descriptors)
    pairs = selectmatches(distance, keypoints)
    coord = getcoords(pairs)
    h, count, inliers, error, index = ransac(coord)
    print (count)
    print (error/count)

    originimage = load('p2/', 1)
    cimage = []
    for i in range(2):
        cimage.append(cv2.cvtColor(originimage[i], cv2.COLOR_BGR2RGB))
    result = warpimages(cimage[0], cimage[1], h)
    plt.imshow(result)
    plt.show()

    mimage = np.array([])
    nk1, nk2 = [], []
    matches = []
    ii = 0
    for t in index:
        nk1.append(pairs[t][0])
        nk2.append(pairs[t][1])
        match = cv2.DMatch(ii, ii, 0)
        matches.append(match)
        ii += 1
    mimage = cv2.drawMatches(images[0], nk1, images[1], nk2, matches, mimage)
    plt.imshow(mimage)
    plt.show()

if __name__ == '__main__':
    main()
