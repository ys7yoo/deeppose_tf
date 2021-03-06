from __future__ import division
from __future__ import print_function

from data import augment

from chainer.dataset import dataset_mixin
import csv
import cv2 as cv
import json
import logging
import numpy as np
import os
import math
import warnings
from tqdm import tqdm

# from matplotlib import pyplot as plt

class PoseDataset(dataset_mixin.DatasetMixin):

    def __init__(self, csv_fn, img_dir, im_size, fliplr=False,
                 rotate=False, rotate_range=None,
                 bbox_extension_range=None,
                 shift=None,
                 min_dim=0, coord_normalize=True, gcn=True,
                 fname_index=0,
                 joint_index=1, joint_index_end=29, symmetric_joints=None, ignore_label=-1,
                 should_return_bbox=False,
                 should_downscale_images=False,
                 downscale_height=480):
        for key, val in locals().items():
            setattr(self, key, val)
        self.symmetric_joints = json.loads(symmetric_joints)
        self.images = None
        self.downscale_factor = None
        self.joints = None
        self.info = None
        self.load_images()
        logging.info('{} is ready'.format(csv_fn))

    @staticmethod
    def get_valid_joints(joints, is_valid_joints):
        valid_joints = []
        for i, joint in enumerate(joints):
            if is_valid_joints is not None \
                    and (is_valid_joints[i][0] == 0 or is_valid_joints[i][1] == 0):
                continue
            valid_joints.append(joint)
        return np.asarray(valid_joints)

    @staticmethod
    def calc_joint_center(joints):
        x_center = (np.min(joints[:, 0]) + np.max(joints[:, 0])) / 2
        y_center = (np.min(joints[:, 1]) + np.max(joints[:, 1])) / 2
        return [x_center, y_center]

    @staticmethod
    def calc_joints_bbox(joints):
        lt = np.min(joints, axis=0)
        rb = np.max(joints, axis=0)
        x, y = lt
        w = rb[0] - lt[0]
        h = rb[1] - lt[1]
        return x, y, w, h

    def augmentByRotation(self, degrees, removeInvalidJoints=True):
        numImages = len(self)

        for i in range (numImages):
            # get joints & info
            image_id, joints = self.joints[i]
            is_valid_joints, bbox = self.info[i]
            #print(image_id, joints, is_valid_joints, bbox)
            #print(bbox)

            # get image
            image = self.images[image_id]

            # find center 
            if joints.shape[0] == 14: # full model
                # check needed joints are valid

                # center = center of torso
                center_hip = np.mean(joints[2:4,:],axis=0)
                #print(center_hip)
                center =  (center_hip + joints[12,:]) * 0.5
                #print(center)
            else:
                # center = neck
                # check needed joints are valid

                center= joints[6,:] # neck
                #print(center)

            for degree in degrees:
                # rotate image and joints 
                image_rotated, joints_rotated, is_valid_joints_rotated = augment.rotate_image_and_joints(image, joints, is_valid_joints, center, degree)


                # check all the joints are inside the image
                if removeInvalidJoints and not np.all(is_valid_joints_rotated):
                    print('Not all points are valid. Skip the rotated image.')
                    continue

                # re-calc bbox
                valid_joints_rotated = self.get_valid_joints(joints_rotated, is_valid_joints_rotated)
                bbox_rotated = np.array(self.calc_joints_bbox(valid_joints_rotated))
                
                # add to the self
                image_id_rotated = image_id+'_R{}'.format(degree)
                self.images[image_id_rotated] = image_rotated

                self.joints.append((image_id_rotated, joints_rotated))
                self.info.append((is_valid_joints_rotated, bbox_rotated))



    def augmentByFlip(self):
        numImages = len(self)

        for i in range (numImages):
            # get joints & info
            image_id, joints = self.joints[i]
            is_valid_joints, bbox = self.info[i]
            #print(image_id, joints, is_valid_joints, bbox)
            #print(bbox)

            # get image
            image = self.images[image_id]

            image_flipped, joints_flipped, is_valid_joints_flipped, bbox_flipped = augment.flip_image_and_joints(image, joints, is_valid_joints, self.symmetric_joints, bbox)


            # below code is moved to tools/augment.py
            """
            H, W, C = image.shape
            #print(W,H,C)

            # flip image
            image_flipped=cv.flip(image,1)
            #plt.imshow(image_flipped)

            # flip joints
            joints_flipped = joints.copy()
            #joints_flipped[:,0] = W + 1 - joints_flipped[:,0]
            joints_flipped[:,0] = W - 1 - joints_flipped[:,0]

            is_valid_joints_flipped = is_valid_joints.copy()

            # swap symmetric joints
            for i, j in self.symmetric_joints:
                joints_flipped[i], joints_flipped[j] = joints_flipped[j].copy(), joints_flipped[i].copy()
                is_valid_joints_flipped[i], is_valid_joints_flipped[j] = is_valid_joints_flipped[j].copy(), is_valid_joints_flipped[i].copy()

            # flip bbox
            bbox_flipped = bbox.copy()
            # bbox_flipped[0] = W + 1 - (bbox_flipped[0] + bbox_flipped[2])
            bbox_flipped[0] = W - 1 - (bbox_flipped[0] + bbox_flipped[2])
            """

            # add to the self
            image_id_flipped = image_id+'_FLR'
            self.images[image_id_flipped] = image_flipped

            self.joints.append((image_id_flipped, joints_flipped))
            self.info.append((is_valid_joints_flipped, bbox_flipped))


    def load_images(self):
        self.images = dict()
        self.joints = list()
        self.info = list()
        self.downscale_factor = dict()
        print('Reading dataset from {}'.format(self.csv_fn))
        #if self.should_downscale_images:
        #    print('Downscale images to the height {}px'.format(self.downscale_height))
        for person_num, line in tqdm(enumerate(csv.reader(open(self.csv_fn)))):

            ##### for DEBUG
            #print(line[self.fname_index])
            #####

            # uncomment a breakpoint here for debugging
            # import pdb; pdb.set_trace()

            image_id = line[self.fname_index]
            img_path = os.path.join(self.img_dir, image_id)
            if image_id in self.images:
                print("[WARNING] duplicated image: {} in {}".format(image_id,person_num))
                image = self.images[image_id]
            else:
                if not os.path.exists(img_path):
                    raise IOError('File not found: {}'.format(img_path))
                image = cv.imread(img_path)  # HWC BGR image
                if image is None:
                    raise IOError('Cannot open image: {}'.format(img_path))
                    #print('Cannot open image:')
                    #print(line[self.fname_index])
                #try:
                #    image.shape
                #except AttributeError:
                #    print('Cannot open image:')
                #    print(line[self.fname_index])
                if self.should_downscale_images and image.shape[0] > self.downscale_height:
                    print('Downscale {} to the height {}px'.format(image_id, self.downscale_height))
                    self.downscale_factor[image_id] = float(image.shape[0]) / self.downscale_height
                    image = cv.resize(image, None, fx=1.0 / self.downscale_factor[image_id],
                                      fy=1.0 / self.downscale_factor[image_id])
            ##### for DEBUG
            # print(line[self.joint_index:])
            #####
            # uncomment a breakpoint here for debugging
            # import pdb; pdb.set_trace()

            # get (x,y) coordinate of joints
            # QUICK FIX - get first 28 numbers 
            coords = [float(c) for c in line[self.joint_index:self.joint_index_end]]
            #coords = [float(c) for c in line[self.joint_index:self.joint_index+14*2]]
            # coords = coords[:14*2]
            # 
            joints = np.array(list(zip(coords[0::2], coords[1::2])))

            # get valid joint info
            #valids = [int(c) for c in line[self.joint_index+14*2:]]
            #if len(valids)>0:
            #    is_valid_joints = np.array(list(zip(valids, valids)))
            #else:
            if True:
                # generate valid joint info

                # is_valid_joints[i] = 0 if we need to ignore the i-th joint
                is_valid_joints = [0 if v == self.ignore_label else 1 for v in joints.flatten()]
                is_valid_joints = np.array(list(zip(is_valid_joints[0::2], is_valid_joints[1::2])))
                for i_joint, (a, b) in enumerate(is_valid_joints):
                    if a == 0 or b == 0:
                        is_valid_joints[i_joint, :] = 0

            if not np.any(is_valid_joints): # no valid joints
                #print("No valid joint in {}".format(image_id))
                print("No valid joint in {}{}{}".format(image_id,joints,is_valid_joints))
                #should_skip_joints = True
                continue

            if not np.all(is_valid_joints):
                # print('person {} contains non-valid joints'.format(person_num))
                print('[{}] {} contains non-valid joints'.format(person_num, img_path))
                print(is_valid_joints[:,0])
                print(joints)

                #print('valid joints:')
                #print(self.get_valid_joints(joints, is_valid_joints))

            if image_id in self.downscale_factor:
                joints /= self.downscale_factor[image_id]

            # clip joints to fit the image
            should_skip_joints = False
            image_shape = image.shape
            for i_joint in range(joints.shape[0]):
                if np.all(is_valid_joints[i_joint]):
                    if joints[i_joint][0] - image_shape[1] > 3 or \
                       joints[i_joint][1] - image_shape[0] > 3:
                        #warnings.warn('Skipping joint with incorrect joints coordinates. They are out of the image.\n'
                        print('Skipping joint with incorrect joints coordinates. They are out of the image.\n'
                                         'image: {}, joint: {}, im.shape: {}'.format(img_path, joints[i_joint], image_shape[:2]))
                        should_skip_joints = True
                        break
                    else:
                        joints[i_joint][0] = np.clip(joints[i_joint][0], 0, image_shape[1])
                        joints[i_joint][1] = np.clip(joints[i_joint][1], 0, image_shape[0])

            if should_skip_joints:
                continue

            valid_joints = self.get_valid_joints(joints, is_valid_joints)
            bbox = np.array(self.calc_joints_bbox(valid_joints))

            # uncomment a breakpoint here for debugging
            # import pdb; pdb.set_trace()

            # Ignore small label regions smaller than min_dim
            if bbox[2] < self.min_dim or bbox[3] < self.min_dim:
                print('Skip small person in {}'.format(image_id))
                continue

            ###################################################
            # now save image and joints

            # store image to dict
            self.images[image_id] = image

            self.joints.append((image_id, joints))
            self.info.append((is_valid_joints, bbox))
            #print('stored {},{},{},{}'.format(image_id,joints, is_valid_joints, bbox))

        print('{} images loaded'.format(len(self)))
        print('joints shape:', self.joints[0][1].shape)


    def to_csv(self, filename):
        """
        Write to a text file in csv format
        """

        with open(filename,'w') as f:
            for imageIdx in range (len(self)):
                image_id, joints = self.joints[imageIdx]
                is_valid_joints, bbox = self.info[imageIdx]

                line = [image_id] + joints.reshape(1,-1)[0].tolist() + is_valid_joints[:,0].tolist()

                for item in line[:-1]: 
                    f.write("{}, ".format(item))
                # last item
                f.write("{}".format(line[-1]))
                f.write("\n")

    def __len__(self):
        return len(self.joints)

    def apply_fliplr(self, image, joints, is_valid_joints):
        joints = joints.copy()
        is_valid_joints = is_valid_joints.copy()

        image = cv.flip(image, 1)
        joints[:, 0] = (image.shape[1] - 1) - joints[:, 0]
        for i, j in self.symmetric_joints:
            joints[i], joints[j] = joints[j].copy(), joints[i].copy()
            is_valid_joints[i], is_valid_joints[j] = is_valid_joints[j].copy(), is_valid_joints[i].copy()
        return image, joints, is_valid_joints

    def apply_zoom(self, image, joints, fx, fy):
        assert fx is not None and fy is not None
        # TODO: cubic interpolation?
        zoomed_image = cv.resize(image, None, fx=fx, fy=fy)
        zoomed_joints = joints * np.array([fx, fy])
        zoomed_joints[:, 0] = np.clip(zoomed_joints[:, 0], 0, zoomed_image.shape[1] - 1)
        zoomed_joints[:, 1] = np.clip(zoomed_joints[:, 1], 0, zoomed_image.shape[0] - 1)
        return zoomed_image, zoomed_joints.astype(np.int32)

    def apply_cropping(self, image, joints, bbox, bbox_extension_range=None, shift=None):
        """
        Randomly enlarge the bounding box of joints and randomly shift the box.
        Crop using the resultant bounding box.
        """
        x, y, w, h = bbox

        if bbox_extension_range is not None:
            if not (1.0 <= bbox_extension_range[0] <= bbox_extension_range[1]):
                raise ValueError('Must be 1.0 <= crop_pad_inf <= crop_pad_sup')
            # bounding rect extending
            inf, sup = bbox_extension_range
            r = sup - inf
            pad_w_r = np.random.rand() * r + inf  # inf~sup
            pad_h_r = np.random.rand() * r + inf  # inf~sup
            pad_x = (w * pad_w_r - w) / 2
            pad_y = (h * pad_h_r - h) / 2
            x -= pad_x
            y -= pad_y
            w *= pad_w_r
            h *= pad_h_r

        if shift is not None and bbox_extension_range is not None:
            # if bbox_extension_range is None or (bbox_extension_range[0] - 1.0) / 2 < shift:
            #     raise ValueError('Shift must be <= (bbox_extension_min - 1.0) / 2')
            if shift < 0.0 or shift > 1.0:
                raise ValueError('Shift must be from 0 to 1')
            shift_x = min(pad_x / bbox[2], shift)
            shift_y = min(pad_y / bbox[3], shift)
            # shifting
            shift_x_pix = shift_x * bbox[2] * (2 * np.random.rand() - 1)
            shift_y_pix = shift_y * bbox[3] * (2 * np.random.rand() - 1)
            x += shift_x_pix
            y += shift_y_pix

        # clipping
        int_x, int_y = int(x), int(y)
        x_diff, y_diff = x - int_x, y - int_y
        x, y = int_x, int_y
        w, h = int(math.ceil(w + x_diff)), int(math.ceil(h + y_diff))

        x = np.clip(x, 0, image.shape[1] - 1)
        y = np.clip(y, 0, image.shape[0] - 1)
        w = np.clip(w, 1, image.shape[1] - x)
        h = np.clip(h, 1, image.shape[0] - y)
        image = image[y:y + h, x:x + w]
        
        check_bounds(joints, x, y, x + w, y + h)
        joints[:, 0] = np.clip(joints[:, 0], x, x + w - 1)
        joints[:, 1] = np.clip(joints[:, 1], y, y + h - 1)

        # joint shifting
        bbox_origin = np.array([x, y])  # position of the crop ont the original image
        joints -= bbox_origin
        bbox = np.array([0, 0, w, h], dtype=int)
        return image, joints, bbox, bbox_origin


    def crop_reshape(self, image, joints, bbox):
        joints = np.array(joints)
        x_min, y_min, w, h = bbox
        image = image[y_min:y_min + h, x_min:x_min + w]
        assert y_min == 0
        assert x_min == 0
        joints -= np.array([x_min, y_min])
        assert image.shape[:2] == (h, w)

        fx, fy = self.im_size / w, self.im_size / h
        image, joints = self.apply_zoom(image, joints, fx, fy)
        bbox = np.array([0, 0, image.shape[1], image.shape[0]], dtype=int)
        return image, joints, bbox

    @staticmethod
    def apply_coord_normalization(image, joints):
        joints = np.array(joints, dtype=np.float32)
        h, w = image.shape[:2]
        check_bounds(joints, 0, 0, w, h)
        # TODO: exclude the upper bound of the bbox
        joints[:, 0] /= w
        joints[:, 1] /= h
        joints -= 0.5
        bbox = np.array([-0.5, -0.5, 1.0, 1.0], dtype=np.float32)
        return joints, bbox

    @staticmethod
    def apply_gcn(image):
        """
        Global contrast normalization.
        Make RGB pixels zero-mean and divide by std.
        """
        image = image.astype(np.float32)
        image -= image.reshape(-1, 3).mean(axis=0)
        image /= image.reshape(-1, 3).std(axis=0) + 1e-5
        return image

    
    def get_image_and_joints(self, imageIdx):
        image=self.get_original_image(imageIdx)    
        image_id, joints = self.joints[imageIdx]
        is_valid_joints, bbox = self.info[imageIdx]
        # print(image_id, joints, is_valid_joints, bbox)
        
        return image, joints, is_valid_joints
        #plt.imshow(image[:,:,::-1])
        #plt.plot(joints[:,0], joints[:,1], 'or')


    def get_original_image(self, i):
        """
        Args:
          i: index of the example
        Returns:
          image: HWC BGR original image (not cropped)
        """
        img_id, joints = self.joints[i]
        image = self.images[img_id]
        return np.array(image)

    def get_example(self, i, gcn=None, bbox_extension_range=None, shift=None):
        """
        Args:
          i: example index
        Return:
          image: bounding box crop in HWC BGR format
          joints:          float32 array [num_joints x 2] of the joints coordinates
          is_valid_joints: int32   array [num_joints x 2] with 1-s on the positions of valid joints
          misc: dict of other optional information
            bbox: bbox of the returned crop inside the original image (after its downsampling if enabled))
            orig_tightest_bbox: bbox of the tightest crop (for bbox_extension_range=(1,1))
              on the original image (after its downsampling if enabled)
        """
        if gcn is None:
            gcn = self.gcn
        if bbox_extension_range is None:
            bbox_extension_range = self.bbox_extension_range
        if shift is None:
            shift = self.shift

        img_id, joints = self.joints[i]
        # print(img_id)
        image = self.images[img_id]
        is_valid_joints, orig_bbox = self.info[i]

        # WARNING! Some vars can be changed by methods in-place!!!
        joints = np.array(joints)
        image = np.array(image)
        # print(image.shape)   # (213, 170, 3)
        bbox = np.array(orig_bbox)

        is_valid_joints = is_valid_joints.astype(np.bool)
        for a, b in is_valid_joints:
            if a != b:
                raise ValueError('Both coordinates of the Joint must be either valid or non valid')

        valid_joints = joints[is_valid_joints].reshape(-1, 2)
        assert valid_joints.shape[0] < joints.shape[0] or np.all(is_valid_joints)
        assert valid_joints.shape[1] == joints.shape[1] == 2

        if self.rotate:
            raise NotImplementedError

        # import pdb; pdb.set_trace()   # for DEBUG

        #print(image.shape)      # (480, 360, 3)    (H, W, C)
        #print(valid_joints)
        #print(bbox)             # [ 99.4047619  135.11904762 107.97619048 127.26190476]
        image, valid_joints, bbox, bbox_origin = self.apply_cropping(image, valid_joints, bbox,
                                                        bbox_extension_range=bbox_extension_range,
                                                        shift=shift)
        #print(image.shape)      # (256, 217, 3) 
        #print(valid_joints)
        #print(bbox)             # [  0   0 217 256]
        check_bounds(valid_joints, *bbox, exclude_upper_bound=True)

        crop_bbox = np.array(bbox)
        # shift bbox to its original position
        crop_bbox[:2] += bbox_origin


        ## crop and reshape to (227, 227, 3) for AlexNet
        image, valid_joints, bbox = self.crop_reshape(image, valid_joints, bbox)
        #print(image.shape)      # (227, 227, 3)
        #print(valid_joints)
        #print(bbox)             # [  0   0 227 227]
        check_bounds(valid_joints, *bbox, exclude_upper_bound=True)

        if self.fliplr and np.random.randint(0, 2) == 1:
            #print('fliplr')
            joints[is_valid_joints] = valid_joints.reshape(-1)
            image, joints, is_valid_joints = self.apply_fliplr(image, joints, is_valid_joints)
            valid_joints = joints[is_valid_joints].reshape(-1, 2)
            check_bounds(valid_joints, *bbox, exclude_upper_bound=True)
        if self.coord_normalize:
            #print('apply coordinate normalization')
            valid_joints, bbox = self.apply_coord_normalization(image, valid_joints)
            #print(valid_joints)
            check_bounds(valid_joints, *bbox)
        if gcn:
            #print('apply gcn')
            image = self.apply_gcn(image)

        image = np.asarray(image, dtype=np.float32)  # HWC BGR
        valid_joints = np.asarray(valid_joints, dtype=np.float32)
        joints[is_valid_joints] = valid_joints.reshape(-1)
        joints[~is_valid_joints] = 0.0
        check_bounds(joints, *bbox)

        misc = None
        if self.should_return_bbox:
            misc = dict(bbox=crop_bbox, orig_tightest_bbox=np.array(orig_bbox), image_id=img_id)
        return image, joints, is_valid_joints, misc


def check_bounds(joints, x, y, w, h, exclude_upper_bound=False):
    """
    Function for a sanity check
    """
    eps = 1e-7

    assert joints[:, 0].max() <= x + w - exclude_upper_bound + eps, \
        'max={} > {}'.format(joints[:, 0].max(), x + w - exclude_upper_bound)
    assert joints[:, 1].max() <= y + h - exclude_upper_bound + eps, \
        'max={} > {}'.format(joints[:, 1].max(), y + h - exclude_upper_bound)
    assert joints[:, 0].min() >= x - eps, 'min={} < {}'.format(joints[:, 0].min(), x)
    assert joints[:, 1].min() >= y - eps, 'min={} < {}'.format(joints[:, 1].min(), y)
