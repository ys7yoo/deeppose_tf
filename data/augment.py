# data augmentation functions

import numpy as np
import cv2 as cv


def flip_image_and_joints(image, joints, is_valid_joints, symmetric_joints=None, bbox=None):
    W = image.shape[1]

    # flip image
    image_flipped=cv.flip(image,1)

    # flip joints
    joints_flipped = joints.copy()
    #joints_flipped[:,0] = W + 1 - joints_flipped[:,0]
    joints_flipped[:,0] = W - 1 - joints_flipped[:,0]
    #print(joint_flipped)
        
    is_valid_joints_flipped = is_valid_joints.copy()
    #print(is_valid_joints_flipped)

    # swap symmetric joints!
    for i, j in symmetric_joints:
        joints_flipped[i], joints_flipped[j] = joints_flipped[j].copy(), joints_flipped[i].copy()
        is_valid_joints_flipped[i], is_valid_joints_flipped[j] = is_valid_joints_flipped[j].copy(), is_valid_joints_flipped[i].copy()
        
    #print(joints_flipped)
    #print(is_valid_joints_flipped)
    
    if bbox is not None:    
        # flip bbox
        bbox_flipped = bbox.copy()
        #bbox_flipped[0] = W + 1 - (bbox_flipped[0] + bbox_flipped[2])
        bbox_flipped[0] = W - 1 - (bbox_flipped[0] + bbox_flipped[2])
        #print(bbox_flipped)
        
        return image_flipped, joints_flipped, is_valid_joints_flipped, bbox_flipped
 
    return image_flipped, joints_flipped, is_valid_joints_flipped



def rotate_image_and_joints(image, joints, is_valid_joints, center, angle):
    """ 
    rotate image & points by the given angle
    modified code from https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
    """
    # grab the dimensions of the image
    (h, w) = image.shape[:2]
    
    # determine the center    
#     (cX, cY) = (w // 2, h // 2)
    (cX, cY) = center
 
    # grab the rotation matrix (applying the negative of the angle to rotate clockwise)
    M = cv.getRotationMatrix2D((cX, cY), -angle, 1.0)
    # print(M)

    image_rotated = cv.warpAffine(image, M, (w, h))
        
    """
    # grab the sine and cosine (i.e., the rotation components of the matrix)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    print(M)
    
    
    # perform the actual rotation on the given image
    image_rotated = cv.warpAffine(image, M, (nW, nH))
    """

    # get the size of the rotated image
    H, W, C = image_rotated.shape
    #print(W,H)

    # rotate joints
    n= joints.shape[0]    
    joints_rotated = np.array(np.c_[ joints.copy(), np.ones(n)] * np.mat(M).transpose())
    
    # check valid joints
    is_valid_joints_rotated = is_valid_joints.copy()
    for i in range(n):
        x = joints_rotated[i,0]
        y = joints_rotated[i,1]
        #print(x,y)

        if not np.all(is_valid_joints_rotated[i,:]):
            # original point is not valid
            continue
        if (x<0) or (x>=W) or (y<0) or (y>=H):
            is_valid_joints_rotated[i,:] = 0    
            #print('invalid')
            print('[Warning] During rotation, joint {} becomes invalid {}.'.format(joints[i,:], joints_rotated[i,:]))
    return image_rotated, joints_rotated, is_valid_joints_rotated


