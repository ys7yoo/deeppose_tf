from model import alexnet

import tensorflow as tf
from tensorflow.core.framework import summary_pb2

import numpy as np
import copy
import math
import time
from tqdm import tqdm

import cv2 as cv # for flipping images


#import evaluation
from evaluation.utils import *
#from evaluation.lsp import *
from evaluation import lsp
from evaluation import mpii
from evaluation import met
from evaluation.pcp import *


## training options
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def training_convnet(net, loss_op, fc_lr, conv_lr, optimizer_type='adam',
                     trace_gradients=False):
    with net.graph.as_default():
        print('Creating optimizer {}'.format(optimizer_type))
        if optimizer_type == 'adam':
            conv_optimizer = tf.train.AdamOptimizer(conv_lr)
            fc_optimizer = tf.train.AdamOptimizer(fc_lr)
        elif optimizer_type == 'adagrad':
            conv_optimizer = tf.train.AdagradOptimizer(conv_lr,
                                                       initial_accumulator_value=0.0001)
            fc_optimizer = tf.train.AdagradOptimizer(fc_lr,
                                                     initial_accumulator_value=0.0001)
        elif optimizer_type == 'sgd':
            conv_optimizer = tf.train.GradientDescentOptimizer(conv_lr)
            fc_optimizer = tf.train.GradientDescentOptimizer(fc_lr)
        elif optimizer_type == 'momentum':
            conv_optimizer = tf.train.MomentumOptimizer(conv_lr, momentum=0.9)
            fc_optimizer = tf.train.MomentumOptimizer(fc_lr, momentum=0.9)
        else:
            raise ValueError('Unknown optimizer type {}'.format(optimizer_type))

        print('Conv LR: {}, FC LR: {}'.format(conv_lr, fc_lr))

        conv_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'conv')
        fc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'fc')

        assert len(conv_vars) + len(fc_vars) == \
            len(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)),\
            'You dont train all the variables'

        grads = tf.gradients(loss_op, conv_vars + fc_vars)
        conv_grads = grads[:len(conv_vars)]
        fc_grads = grads[len(conv_vars):]
        assert len(conv_grads) == len(conv_vars)
        assert len(fc_grads) == len(fc_vars)

        with tf.name_scope('grad_norms'):
            for v, grad in zip(conv_vars + fc_vars, grads):
                if grad is not None:
                    grad_norm_op = tf.nn.l2_loss(grad, name=format(v.name[:-2]))
                    tf.add_to_collection('grads', grad_norm_op)
                    if trace_gradients:
                        tf.summary.scalar(grad_norm_op.name, grad_norm_op)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            conv_tran_op = conv_optimizer.apply_gradients(zip(conv_grads, conv_vars), name='conv_train_op')
        fc_tran_op = fc_optimizer.apply_gradients(zip(fc_grads, fc_vars),
                                                  global_step=net.global_iter_counter, name='fc_train_op')
        return tf.group(conv_tran_op, fc_tran_op)



##

def create_regression_net(n_joints=14, optimizer_type=None,
                          init_snapshot_path=None, is_resume=False,
                          reset_iter_counter=False,
                          reset_moving_averages=False,
                          gpu_memory_fraction=None,
                          net_type='Alexnet'):
    with tf.Graph().as_default():
        if net_type == 'Alexnet':
            net = alexnet.Alexnet(init_model=None,
                                        num_classes=1,
                                        im_shape=(227, 227, 3),
                                        device_id='/gpu:0',
                                        num_layers_to_init=0,
                                        random_init_type=alexnet.Alexnet.RandomInitType.GAUSSIAN,
                                        use_batch_norm=False,
                                        gpu_memory_fraction=gpu_memory_fraction)
            with tf.variable_scope('fc_regression') as scope:
                num_inputs = int(net.fc7_dropout.get_shape()[1])
                w, b = net.get_fc_weights(layer_index=99, net_data=None,
                                          num_inputs=num_inputs,
                                          num_outputs=n_joints * 2,
                                          wights_std=0.01,
                                          bias_init_value=0.0)
                net.fc_regression = tf.nn.xw_plus_b(net.fc7_dropout, w, b, name=scope.name)
        else:
            # Other network architectures can be defined here.
            raise ValueError('unknown net_type {}'.format(net_type))

        with tf.name_scope('pose_input'):
            joints_gt = tf.placeholder(tf.float32, shape=(None, n_joints, 2), name='joints_gt')
            joints_is_valid = tf.placeholder(tf.int32, shape=(None, n_joints, 2),
                                             name='joints_is_valid')

        joints_gt_flat = tf.reshape(joints_gt, shape=[-1, n_joints * 2])
        joints_is_valid_flat = tf.cast(tf.reshape(joints_is_valid, shape=[-1, n_joints * 2]),
                                       tf.float32)

        diff = tf.subtract(joints_gt_flat, net.fc_regression)
        diff_valid = tf.multiply(diff, joints_is_valid_flat)

        num_valid_joints = tf.reduce_sum(joints_is_valid_flat,
                                         axis=1) / tf.constant(2.0, dtype=tf.float32)
        pose_loss_op = tf.reduce_mean(
            tf.reduce_sum(tf.square(diff_valid), axis=1) / num_valid_joints,
            name='joint_euclidean_loss')

        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
        loss_with_decay_op = pose_loss_op + tf.constant(0.0005, name='weight_decay') * l2_loss

        with tf.variable_scope('lr'):
            conv_lr_pl = tf.placeholder(tf.float32, tuple(), name='conv_lr')
            fc_lr_pl = tf.placeholder(tf.float32, tuple(), name='fc_lr')

        # with tf.name_scope('norms'):
        #     conv5w_norm = tf.nn.l2_loss(net.graph.get_tensor_by_name('conv5/weight:0'), name='conv5w')
        #     conv5b_norm = tf.nn.l2_loss(net.graph.get_tensor_by_name('conv5/bias:0'), name='conv5b')
        #     tf.add_to_collection('norms', conv5w_norm)
        #     tf.add_to_collection('norms', conv5b_norm)
        #     tf.scalar_summary([conv5w_norm.name, conv5b_norm.name], [conv5w_norm, conv5b_norm])

        tf.summary.scalar('loss_with_decay', loss_with_decay_op)
        tf.summary.scalar('loss', pose_loss_op)
        tf.summary.scalar('conv_lr', conv_lr_pl)
        tf.summary.scalar('fc_lr', fc_lr_pl)

        net.sess.run(tf.global_variables_initializer())
        if init_snapshot_path is not None:
            if not is_resume:
                if net_type == 'Alexnet':
                    print('restoring from'+init_snapshot_path)
                    net.restore_from_snapshot(init_snapshot_path, 7,
                                              restore_iter_counter=False)
                else:
                    raise ValueError('unknown net type {}'.format(net_type))
            else:
                print('Restoring everything from snapshot and resuming from '+init_snapshot_path)
                saver = tf.train.Saver(max_to_keep=None)
                saver.restore(net.sess, init_snapshot_path)

        train_op = network_spec.training_convnet(net, pose_loss_op, fc_lr=fc_lr_pl,
                                                       conv_lr=conv_lr_pl,
                                                       optimizer_type=optimizer_type,
                                                       trace_gradients=True)
        start = time.time()
        uninit_vars = [v for v in tf.global_variables()
                       if not tf.is_variable_initialized(v).eval(session=net.sess)]
        print('uninit vars:', [v.name for v in uninit_vars])
        print('Elapsed time for finding uninitialized variables: {:.2f}s'.format(time.time() - start))
        start = time.time()
        if reset_iter_counter:
            uninit_vars.append(net.global_iter_counter)
        setup_moving_averages(net.graph, reset=reset_moving_averages, track=False)

        net.sess.run(tf.variables_initializer(uninit_vars))
        print('Elapsed time to init them: {:.2f}s'.format(time.time() - start))

        return net, loss_with_decay_op, pose_loss_op, train_op


def setup_moving_averages(graph, reset=False, track=False):
    with graph.as_default():
        movin_avg_vars = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if
                          'moving_' in v.name]
        if track:
            tf.summary.scalar(['moving/' + v.name for v in movin_avg_vars],
                              [tf.nn.l2_loss(v) for v in movin_avg_vars])
        if reset:
            print('Resetting moving averages')
            tf.variables_initializer(movin_avg_vars)


def batch2feeds(batch):
    """
    Args:
      batch: a batch that is returned by dataset_iterator,
      which is a list of tuples (image, img_joints_gt, img_joints_is_valid)
    Return:
        images: float32 array [batch_size x H x W x C]
        joints_gt: joints coordinates float32 array [batch_size x n_joints x 2]
        joints_is_valid: is joint valid float32 array [batch_size x n_joints x 2]
        misc: list of misc data for examples
    """
    images, joints_gt, joints_is_valid, misc = zip(*batch)
    images = np.asarray(images)
    joints_gt = np.asarray(joints_gt)
    joints_is_valid = np.asarray(joints_is_valid)
    return images, joints_gt, joints_is_valid, misc


def batch2feeds_flip(batch):
    """
    Args:
      batch: a batch that is returned by dataset_iterator,
      which is a list of tuples (image, img_joints_gt, img_joints_is_valid)
    Return:
        images: float32 array [batch_size x H x W x C]
        joints_gt: joints coordinates float32 array [batch_size x n_joints x 2]
        joints_is_valid: is joint valid float32 array [batch_size x n_joints x 2]
        misc: list of misc data for examples
    """

    images, joints_gt, joints_is_valid, misc = zip(*batch)

#    import pdb; pdb.set_trace()

    #print ('generate flipped images')

    num_images = len(images)
    images_flipped = list()
    joints_gt_flipped = joints_gt


    # cleaner code using zip
    for image, joint in zip(images, joints_gt_flipped):
        # flip image (horizontally)
        images_flipped.append(cv.flip(image,1))
        # flip joints by inverting x coordinate
        joint[:,0]  = -joint[:,0]

    """
    for i in range(num_images):
        image = images[i]
    #for i, image in tqdm(enumerate(images), total=num_images):
        # flip image (horizontally)
        images_flipped.append(cv.flip(image,1))
        # flip joints by inverting x coordinate
        joints_gt_flipped[i][:,0]  = -joints_gt_flipped[i][:,0]
    """

    images = np.asarray(images)
    joints_gt = np.asarray(joints_gt)
    joints_is_valid = np.asarray(joints_is_valid)

#    import pdb; pdb.set_trace()

    # append flipped images
    images = np.append(images, images_flipped, axis=0)
    joints_gt = np.append(joints_gt, joints_gt_flipped, axis=0)
    joints_is_valid = np.append(joints_is_valid, joints_is_valid, axis=0)


    return images, joints_gt, joints_is_valid, misc


def fill_joint_feed_dict(net, batch_feeds,
                         conv_lr=None, fc_lr=None,
                         phase='test', train_keep_prob=0.4):
    """Fills the feed_dict for training the given step.

    A feed_dict takes the form of:
    feed_dict = {
        <placeholder>: <tensor of values to be passed for placeholder>,
        ....
    }

    Args:
      batch_loader: BatchLoader, that provides batches of the data
      images_pl: The images placeholder, from placeholder_inputs().
      labels_pl: The labels placeholder, from placeholder_inputs().

    Returns:
      feed_dict: The feed dictionary mapping from placeholders to values.
    """
    if phase not in ['train', 'test']:
        raise ValueError('phase must be "train" or "test"')
    if phase == 'train':
        keep_prob = train_keep_prob
        is_phase_train = True
    else:
        keep_prob = 1.0
        is_phase_train = False

    if len(batch_feeds) != 3:
        raise ValueError('feeds must contain only 3 elements: images, joints_gt, joints_is_valid')
    images, joints_gt, joints_is_valid = batch_feeds

    feed_dict = {
        net.x: images,
        'pose_input/joints_gt:0': joints_gt,
        'pose_input/joints_is_valid:0': joints_is_valid,
        'input/is_phase_train:0': is_phase_train,
        'lr/conv_lr:0': conv_lr,
        'lr/fc_lr:0': fc_lr
    }

    try:
        keep_prob_pl = net.graph.get_tensor_by_name('input/dropout_keep_prob:0')
        dropout_params = {keep_prob_pl: keep_prob}
    except KeyError:
        dropout_params = {'fc6/keep_prob_pl:0': keep_prob,
                          'fc7/keep_prob_pl:0': keep_prob}
    feed_dict.update(dropout_params)
    return feed_dict


def calculate_metric(gt_joints, predicted_joints, orig_bboxes, dataset_name, metric_name='PCP'):
    if metric_name not in ['PCP', 'RelaxedPCP', 'PCKh']:
        raise ValueError('Unknown metric {}'.format(metric_name))

    if gt_joints.shape != predicted_joints.shape:
        raise ValueError('GT joints and predicted joints shape mismatch! {} != {}'.format(
            gt_joints.shape, predicted_joints.shape))
    if orig_bboxes.ndim != 2 or orig_bboxes.shape[1] != 4 or orig_bboxes.shape[0] != gt_joints.shape[0]:
        raise ValueError('Incorrect orig_bboxes shape {}'.format(orig_bboxes.shape))

    gt_joints = gt_joints.copy()
    predicted_joints = predicted_joints.copy()

    predicted_joints = np.clip(predicted_joints, -0.5, 0.5)
    # convert joints
    for i in range(gt_joints.shape[0]):
        #gt_joints[i, ...] = evaluation.utils.project_joint_onto_original_image(gt_joints[i],
        gt_joints[i, ...] = project_joint_onto_original_image(gt_joints[i],orig_bboxes[i])
        #predicted_joints[i, ...] = evaluation.utils.project_joint_onto_original_image(
        predicted_joints[i, ...] = project_joint_onto_original_image(
            predicted_joints[i], orig_bboxes[i])

    print(dataset_name)

    if dataset_name == 'mpii':
        gt_joints = mpii.convert2canonical(gt_joints)
        predicted_joints = mpii.convert2canonical(predicted_joints)
    elif dataset_name == 'lsp':
        gt_joints = lsp.convert2canonical(gt_joints)
        predicted_joints = lsp.convert2canonical(predicted_joints)
    else: # for MET with 8 point model
        gt_joints = met.convert2canonical(gt_joints)
        predicted_joints = met.convert2canonical(predicted_joints)


#    gt_joints = evaluation.__dict__[dataset_name].convert2canonical(gt_joints)
#    predicted_joints = evaluation.__dict__[dataset_name].convert2canonical(predicted_joints)

#   QUICK FIX FOR LSP
#    gt_joints = convert2canonical(gt_joints)
#    predicted_joints = convert2canonical(predicted_joints)


#    from evaluation.pcp import *
    if metric_name == 'RelaxedPCP':
#        full_scores = evaluation.pcp.eval_relaxed_pcp(gt_joints, predicted_joints)
        full_scores = eval_relaxed_pcp(gt_joints, predicted_joints)
    elif metric_name == 'PCP':
#        full_scores = evaluation.pcp.eval_strict_pcp(gt_joints, predicted_joints)
        full_scores = eval_strict_pcp(gt_joints, predicted_joints)
    elif metric_name == 'PCKh':
#        full_scores = evaluation.pcp.eval_pckh(dataset_name, gt_joints, predicted_joints)
        full_scores = eval_pckh(dataset_name, gt_joints, predicted_joints)
    else:
        raise ValueError()
    return full_scores


def print_scores(global_step, score_per_stick, score_per_part, part_names, tag_prefix, score_name):
    print ('Step\t {}\t {}/m{}\t {:.3f}'.format(global_step, tag_prefix, score_name,
                                           np.mean(score_per_part)))
    print ('Step {} {}/parts_{}:'.format(global_step, tag_prefix, score_name))
    print ('\t'.join(part_names))
    print ('\t'.join(['{:.3f}'.format(val) for val in score_per_part]))

    # print 'Step {} {}/full_{}:'.format(global_step, tag_prefix, score_name)
    # print '\t'.join(evaluation.pcp.CANONICAL_STICK_NAMES)
    # print '\t'.join(['{:.3f}'.format(val) for val in score_per_stick])


def print_pckh(dataset_name, global_step, score_per_joint, tag_prefix):
    print ('Step\t {}\t {}/mPCKh\t {:.3f}'.format(global_step, tag_prefix,
                                             np.mean(score_per_joint)))
    # print '\t'.join(evaluation.__dict__[dataset_name].CANONICAL_JOINT_NAMES)
    # print '\t'.join(['{:.3f}'.format(val) for val in score_per_joint])

    pckh_symmetric_joints, joint_names = \
        evaluation.pcp.average_pckh_symmetric_joints(dataset_name, score_per_joint)
    print ('Step\t {}\t {}/mSymmetricPCKh\t {:.3f}'.format(global_step, tag_prefix,
                                             np.mean(pckh_symmetric_joints)))
    print ('Step {} {}/parts_SymmetricPCKh:'.format(global_step, tag_prefix))
    print ('\t'.join(joint_names))
    print ('\t'.join(['{:.3f}'.format(val) for val in pckh_symmetric_joints]))


def create_sumamry(tag, value):
    """
    Create a summary for logging via tf.train.SummaryWriter
    """
    x = summary_pb2.Summary.Value(tag=tag, simple_value=value)
    return summary_pb2.Summary(value=[x])

# separated into two functions!
def predict(net, pose_loss_op, test_dataset, batch_size, dataset_name, tag_prefix='test'):

    # moved test iterator to here
    # test_it = copy.copy(test_iterator)
    from chainer import iterators
    test_it = iterators.MultiprocessIterator(test_dataset, batch_size=batch_size,
                                             repeat=False, shuffle=False,
                                             n_processes=1, n_prefetch=1)
    # http://docs.chainer.org/en/stable/reference/generated/chainer.iterators.MultiprocessIterator.html


    num_test_examples = len(test_it.dataset)
    num_batches = int(math.ceil(num_test_examples / test_it.batch_size))
    num_joints = int(int(net.fc_regression.get_shape()[1]) / 2)

    gt_joints = list()
    gt_joints_is_valid = list()
    predicted_joints = list()
    orig_bboxes = list()

    total_loss = 0.0

    print(len(test_it.dataset))
    for i, batch in tqdm(enumerate(test_it), total=num_batches):

        # uncomment the following breakpoint for debugging
        # import pdb; pdb.set_trace()

        feeds = batch2feeds(batch)    # (?, 227, 227, 3), (?, 14, 2), (?, 14, 2)
        feed_dict = fill_joint_feed_dict(net,
                                         feeds[:3],
                                         conv_lr=0.0,
                                         fc_lr=0.0,
                                         phase='test')

        pred_j, batch_loss_value = net.sess.run([net.fc_regression, pose_loss_op], feed_dict=feed_dict)
        total_loss += batch_loss_value * len(batch)
        predicted_joints.append(pred_j.reshape(-1, num_joints, 2))

        gt_joints.append(feeds[1])
        gt_joints_is_valid.append(feeds[2])
        orig_bboxes.append(np.vstack([x['bbox'] for x in feeds[3]]))

    avg_loss = total_loss / num_test_examples
    global_step = net.sess.run(net.global_iter_counter)

    print('Step {} {}/pose_loss = {:.3f}'.format(global_step, tag_prefix, avg_loss))

    gt_joints = np.vstack(gt_joints)
    gt_joints_is_valid = np.vstack(gt_joints_is_valid)
    predicted_joints = np.vstack(predicted_joints)
    orig_bboxes = np.vstack(orig_bboxes)

    return avg_loss, global_step, gt_joints, gt_joints_is_valid, predicted_joints, orig_bboxes

def calc_pcp(global_step, gt_joints, gt_joints_is_valid, predicted_joints, orig_bboxes, dataset_name, tag_prefix='test', summary_writer=None):
    assert predicted_joints.shape[0] == gt_joints.shape[0] == orig_bboxes.shape[0] # == num_test_examples
    assert predicted_joints.shape[1] == gt_joints.shape[1] # == num_joints
    assert predicted_joints.shape[2] == gt_joints.shape[2] == 2
    assert orig_bboxes.shape[1] == 4

    # QUICK FIX
    #if not np.all(gt_joints_is_valid):
    #    raise ValueError('For testing All Ground Truth joints must be valid!')

    # import pdb; pdb.set_trace()

    pcp_per_stick = calculate_metric(gt_joints, predicted_joints, orig_bboxes,
                                     dataset_name=dataset_name,
                                     metric_name='PCP')
    pcp_per_part, part_names = evaluation.pcp.average_pcp_left_right_limbs(pcp_per_stick)
    print_scores(global_step, pcp_per_stick, pcp_per_part, part_names, tag_prefix, 'PCP')

    relaxed_pcp_per_stick = calculate_metric(gt_joints, predicted_joints, orig_bboxes,
                                             dataset_name=dataset_name, metric_name='RelaxedPCP')
    relaxed_pcp_per_part, part_names = evaluation.pcp.average_pcp_left_right_limbs(relaxed_pcp_per_stick)
    print_scores(global_step, relaxed_pcp_per_stick, relaxed_pcp_per_part, part_names, tag_prefix, 'RelaxedPCP')

    pckh_per_joint = calculate_metric(gt_joints, predicted_joints, orig_bboxes,
                                      dataset_name=dataset_name, metric_name='PCKh')
    pckh_symmetric_joints, joint_names = \
        evaluation.pcp.average_pckh_symmetric_joints(dataset_name, pckh_per_joint)
    print_pckh(dataset_name, global_step, pckh_per_joint, tag_prefix)


    if summary_writer is not None:
        summary_writer.add_summary(create_sumamry('{}/mPCP'.format(tag_prefix), np.mean(pcp_per_part)),
                                   global_step=global_step)
        # summary_writer.add_summary(create_sumamry('{}/mRelaxedPCP'.format(tag_prefix),
        #                            np.mean(relaxed_pcp_per_part)), global_step=global_step)
        summary_writer.add_summary(create_sumamry('{}/PCKh'.format(tag_prefix),
                                       np.mean(pckh_per_joint)), global_step=global_step)
        summary_writer.add_summary(create_sumamry('{}/symPCKh'.format(tag_prefix),
                                                              np.mean(pckh_symmetric_joints)),
                                   global_step=global_step)

        summary_writer.add_summary(create_sumamry('{}/pose_loss'.format(tag_prefix), avg_loss),
                                   global_step=global_step)

    return pcp_per_stick, None



# Keep the original code for compatibility

def evaluate_pcp(net, pose_loss_op, test_iterator, summary_writer, dataset_name, tag_prefix='test'):
    test_it = copy.copy(test_iterator)
    num_test_examples = len(test_it.dataset)
    num_batches = int(math.ceil(num_test_examples / test_it.batch_size))
    num_joints = int(int(net.fc_regression.get_shape()[1]) / 2)
    gt_joints = list()
    gt_joints_is_valid = list()
    predicted_joints = list()
    orig_bboxes = list()
    total_loss = 0.0

    print(len(test_it.dataset))
    for i, batch in tqdm(enumerate(test_it), total=num_batches):
        feeds = batch2feeds(batch)
        feed_dict = fill_joint_feed_dict(net,
                                         feeds[:3],
                                         conv_lr=0.0,
                                         fc_lr=0.0,
                                         phase='test')
        pred_j, batch_loss_value = net.sess.run([net.fc_regression, pose_loss_op], feed_dict=feed_dict)
        total_loss += batch_loss_value * len(batch)
        gt_joints.append(feeds[1])
        gt_joints_is_valid.append(feeds[2])
        predicted_joints.append(pred_j.reshape(-1, num_joints, 2))
        orig_bboxes.append(np.vstack([x['bbox'] for x in feeds[3]]))

    avg_loss = total_loss / num_test_examples
    gt_joints = np.vstack(gt_joints)
    gt_joints_is_valid = np.vstack(gt_joints_is_valid)
    predicted_joints = np.vstack(predicted_joints)
    orig_bboxes = np.vstack(orig_bboxes)
    assert predicted_joints.shape[0] == gt_joints.shape[0] == orig_bboxes.shape[0] == num_test_examples
    assert predicted_joints.shape[1] == gt_joints.shape[1] == num_joints
    assert predicted_joints.shape[2] == gt_joints.shape[2] == 2
    assert orig_bboxes.shape[1] == 4
    if not np.all(gt_joints_is_valid):
        print('[WARNING] For testing, all Ground Truth joints must be valid!')
        #raise ValueError('For testing All Ground Truth joints must be valid!')
    global_step = net.sess.run(net.global_iter_counter)
    print('Step {} {}/pose_loss = {:.3f}'.format(global_step, tag_prefix, avg_loss))

    pcp_per_stick = calculate_metric(gt_joints, predicted_joints, orig_bboxes,
                                     dataset_name=dataset_name,
                                     metric_name='PCP')
    #pcp_per_part, part_names = evaluation.pcp.average_pcp_left_right_limbs(pcp_per_stick)
    #print_scores(global_step, pcp_per_stick, pcp_per_part, part_names, tag_prefix, 'PCP')

    #relaxed_pcp_per_stick = calculate_metric(gt_joints, predicted_joints, orig_bboxes,
    #                                         dataset_name=dataset_name, metric_name='RelaxedPCP')
    #relaxed_pcp_per_part, part_names = evaluation.pcp.average_pcp_left_right_limbs(relaxed_pcp_per_stick)
    #print_scores(global_step, relaxed_pcp_per_stick, relaxed_pcp_per_part, part_names, tag_prefix, 'RelaxedPCP')

    #pckh_per_joint = calculate_metric(gt_joints, predicted_joints, orig_bboxes,
    #                                  dataset_name=dataset_name, metric_name='PCKh')
    #pckh_symmetric_joints, joint_names = \
    #    evaluation.pcp.average_pckh_symmetric_joints(dataset_name, pckh_per_joint)
    #print_pckh(dataset_name, global_step, pckh_per_joint, tag_prefix)

    if summary_writer is not None:
        summary_writer.add_summary(create_sumamry('{}/mPCP'.format(tag_prefix), np.mean(pcp_per_stick)), global_step=global_step)
        #summary_writer.add_summary(create_sumamry('{}/mPCP'.format(tag_prefix), np.mean(pcp_per_part)),
        #                           global_step=global_step)
        # summary_writer.add_summary(create_sumamry('{}/mRelaxedPCP'.format(tag_prefix),
        #                            np.mean(relaxed_pcp_per_part)), global_step=global_step)
        #summary_writer.add_summary(create_sumamry('{}/PCKh'.format(tag_prefix),
        #                               np.mean(pckh_per_joint)), global_step=global_step)
        #summary_writer.add_summary(create_sumamry('{}/symPCKh'.format(tag_prefix),
        #                                                      np.mean(pckh_symmetric_joints)),
        #                           global_step=global_step)

        summary_writer.add_summary(create_sumamry('{}/pose_loss'.format(tag_prefix), avg_loss),
                                   global_step=global_step)
    return pcp_per_stick, None
