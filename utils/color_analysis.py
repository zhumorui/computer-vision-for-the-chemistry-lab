







import numpy as np




# Calculate Color change between two different images.
# Input should be RGB images not BGR images!!!
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