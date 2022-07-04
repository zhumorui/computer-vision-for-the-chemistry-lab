import numpy as np
from collections import Counter
from sklearn.cluster import KMeans
import numpy as np
import cv2


# Calculate Color change between two different images.
# Input Channel should be RGB not BGR. 
def cal_color_change(img1,img2):
    assert img1.shape == img2.shape, "Image Size doesn't match"
    
    # Image should use RGB channel not BGR
    # Convert int type to float
    img1 = img1.astype(float)
    img2 = img2.astype(float)

    rmean = (img1[:,:,0] + img2[:,:,0])/2
    r = img1[:,:,0] - img2[:,:,0]
    g = img1[:,:,1] - img2[:,:,1]
    b = img1[:,:,2] - img2[:,:,2]
    distance = np.sqrt(((2 + rmean / 256) * r ** 2 + 4 * g ** 2 + (2 + (255 - rmean) / 256) * b ** 2))
    mean_distance = np.mean(distance)

    return mean_distance


def rgb_to_hex(rgb_color):
    hex_color = "#"
    for i in rgb_color:
        i = int(i)
        hex_color += ("{:02x}".format(i))
    return hex_color


def main_color_analysis(img,n_clusters=3):
    img = prep_image(img)
    clf = KMeans(n_clusters)
    color_labels = clf.fit_predict(img)
    center_colors = clf.cluster_centers_
    counts = Counter(color_labels)
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [rgb_to_hex(ordered_colors[i]) for i in counts.keys()]
    return counts, hex_colors


def prep_image(raw_img):
    modified_img = cv2.resize(raw_img, (900, 600), interpolation = cv2.INTER_AREA)
    modified_img = modified_img.reshape(modified_img.shape[0]*modified_img.shape[1], 3)
    return modified_img