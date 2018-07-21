from __future__ import division

from data import dataset
from model import regressionnet

from chainer import iterators
import os
import time
import tensorflow as tf
import copy
from tqdm import tqdm
import numpy as np
import math
import pprint
import datetime

###########################################
# set cmd options
import argparse
import os.path


#from config import *
# copied to here
import os

# Full path to the project root
ROOT_DIR = os.path.expanduser('~/src/deeppose')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'out')

# put data in /tmp
LSP_DATASET_ROOT = os.path.expanduser('/var/data/lsp')
LSP_EXT_DATASET_ROOT = os.path.expanduser('/var/data/lsp_ext')

MPII_DATASET_ROOT = os.path.expanduser('/var/data/mpii')

MET_DATASET_ROOT = os.path.expanduser('/var/data/MET3')

#default location (old)
#LSP_DATASET_ROOT = os.path.expanduser('~/data/lsp')
#LSP_EXT_DATASET_ROOT = os.path.expanduser('~/data/lsp_ext')
#MPII_DATASET_ROOT = os.path.expanduser('~/data/mpii')




def cast_path(value):
    path = None
    if value == '' or value.lower() == 'none':
        pass
    else:
        path = value
    return path


def cast_num_workers(value):
    value = int(value)
    if value < 1:
        raise ValueError('Num workers must be positive number')
    return value


def cast_downscale_height(value):
    value = int(value)
    if value < 227:
        raise ValueError('Image downscale height must be at least 227 px')
    return value


def get_arguments(argv):
    parser = argparse.ArgumentParser()

    # Basic parameters
    parser.add_argument('--max_iter', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--snapshot_step', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=1701)
    parser.add_argument('--ignore_label', type=float, default=-1)
    parser.add_argument('--dataset_name', type=str, choices=['met', 'lsp', 'mpii'], default='met')
    parser.add_argument(
        '--train_csv_fn', type=str,
        default=os.path.join(LSP_EXT_DATASET_ROOT, 'train_joints.csv'))
    parser.add_argument(
        '--test_csv_fn', type=str,
        default=os.path.join(LSP_EXT_DATASET_ROOT, 'test_joints.csv'))
    parser.add_argument(
        '--val_csv_fn', type=str,
        default='')
    parser.add_argument(
        '--img_path_prefix', type=str,
        default='')
    parser.add_argument('--o_dir', type=str, default=OUTPUT_DIR)
    parser.add_argument(
        '--test_step', type=int, default=100,
        help='Perform test every step iterations')
    parser.add_argument(
        '--log_step', type=int, default=1,
        help='Show loss value per this iterations')

    # Data argumentation settings
    parser.add_argument(
        '--im_size', type=int, default=227,
        help='Resize input image into this big')
    parser.add_argument(
        '--fliplr', action='store_true', default=False,
        help=('Flip image\'s left and right for data augmentation'))
    parser.add_argument(
        '--rotate', action='store_true', default=False,
        help=('Randomly rotate images for data augmentation'))
    parser.add_argument(
        '--rotate_range', type=int, default=10,
        help=('The max angle(degree) of rotation for data augmentation'))
    parser.add_argument(
        '--shift', type=float, default=0.0,
        help=('Max shift. Randomly shift bounding box for data augmentation. '
              'The value is the fraction of the bbox width and height.'))
    parser.add_argument(
        '--bbox_extension_min', type=float, default=None,
        help=('The min multiplier for joints bounding box.'))
    parser.add_argument(
        '--bbox_extension_max', type=float, default=None,
        help=('The max multiplier for joints bounding box.'))
    parser.add_argument(
        '--min_dim', type=int, default=6,
        help='Minimum dimension of a person')
    parser.add_argument(
        '--coord_normalize', action='store_true', default=True,
        help=('Perform normalization to all joint coordinates'))
    parser.add_argument(
        '--gcn', action='store_true', default=False,
        help=('Perform global contrast normalization for each input image'))

    # Data configuration
    parser.add_argument('--n_joints', type=int, default=14, help='Number of joints per person')
    parser.add_argument(
        '--fname_index', type=int, default=0,
        help='the index of image file name in a csv line')
    parser.add_argument(
        '--joint_index', type=int, default=1,
        help='the start index of joint values in a csv line')
    parser.add_argument(
        '--joint_index_end', type=int, default=1,
        help='the end index of joint values in a csv line')
    parser.add_argument(
        '--symmetric_joints', type=str, default='[[8, 9], [7, 10], [6, 11], [2, 3], [1, 4], [0, 5]]',
        help='Symmetric joint ids in JSON format')
    # flic_swap_joints = [(2, 4), (1, 5), (0, 6)]
    # lsp_swap_joints = [(8, 9), (7, 10), (6, 11), (2, 3), (1, 4), (0, 5)]
    # mpii_swap_joints = [(12, 13), (11, 14), (10, 15), (2, 3), (1, 4), (0, 5)]

    parser.add_argument('--should_downscale_images', action='store_true', default=False,
                        help='Downscale all images when loading to $downscale_height, rescale gt joints appropriately.')
    parser.add_argument('--downscale_height', type=cast_downscale_height, default=480,
                        help='Downscale images to this height if their height is bigger than this value. '
                             '(default=480px)')

    # Optimization settings
    parser.add_argument('--conv_lr', type=float, default=0.0005)
    parser.add_argument('--fc_lr', type=float, default=0.0005)
    parser.add_argument('--fix_conv_iter', type=int, default=0)
    parser.add_argument('--optimizer', type=str, choices=['adam', 'adagrad', 'momentum', 'sgd'], default='adagrad', )
    parser.add_argument('--resume', action='store_true', default=False, help='If you want to resume training from the snapshot. '
                                                                             'Should not be used if you want to initialize only several layers from the snapshot.')
    parser.add_argument('-s', '--snapshot', type=cast_path, help='snapshot path to use as initialization or to resume training', default=None)
    parser.add_argument('--workers', type=cast_num_workers, default=1)
    parser.add_argument('--reset_iter_counter', action='store_true', default=False)
    parser.add_argument('--reset_moving_averages', action='store_true', default=False)

    parser.add_argument('--net_type', type=str, default=None,
                        help='Type of the network architecture. For ex.: Alexnet')
    args = parser.parse_args(argv)
    return args


from model.regressionnet import evaluate_pcp, create_sumamry

def evaluate(net, pose_loss_op, test_iterator, summary_writer, tag='test/pose_loss'):
    test_it = copy.copy(test_iterator)
    total_loss = 0.0
    cnt = 0
    num_batches = int(math.ceil(len(test_it.dataset) / test_it.batch_size))
    print(len(test_it.dataset))
    for batch in tqdm(test_it, total=num_batches):
        feed_dict = regressionnet.fill_joint_feed_dict(net,
                                                       regressionnet.batch2feeds(batch)[:3],
                                                       conv_lr=0.0,
                                                       fc_lr=0.0,
                                                       phase='test')
        global_step, loss_value = net.sess.run([net.global_iter_counter, pose_loss_op],
                                               feed_dict=feed_dict)
        total_loss += loss_value * len(batch)
        cnt += len(batch)
    avg_loss = total_loss / len(test_it.dataset)
    print ('Step {} {} = {:.3f}'.format(global_step, tag, avg_loss))
    summary_writer.add_summary(create_sumamry(tag, avg_loss),
                               global_step=global_step)
    assert cnt == 1000, 'cnt = {}'.format(cnt)


def train_loop(net, saver, loss_op, pose_loss_op, train_op, dataset_name, train_iterator, test_iterator,
               val_iterator=None,
               max_iter=None,
               test_step=None,
               snapshot_step=None,
               log_step=1,
               batch_size=None,
               conv_lr=None,
               fc_lr=None,
               fix_conv_iter=None,
               output_dir='results',
               ):

    summary_step = 50

    with net.graph.as_default():
        summary_writer = tf.summary.FileWriter(output_dir, net.sess.graph)
        summary_op = tf.summary.merge_all()
        fc_train_op = net.graph.get_operation_by_name('fc_train_op')
    global_step = None

    for step in range(max_iter + 1):

        # test, snapshot
        if step % test_step == 0 or step + 1 == max_iter or step == fix_conv_iter:
            global_step = net.sess.run(net.global_iter_counter)
            evaluate_pcp(net, pose_loss_op, test_iterator, summary_writer,
                         dataset_name=dataset_name,
                         tag_prefix='test')
            if val_iterator is not None:
                evaluate_pcp(net, pose_loss_op, val_iterator, summary_writer,
                             dataset_name=dataset_name,
                             tag_prefix='val')

        if step % snapshot_step == 0 and step > 1:
            checkpoint_prefix = os.path.join(output_dir, 'checkpoint')
            assert global_step is not None
            saver.save(net.sess, checkpoint_prefix, global_step=global_step)
        if step == max_iter:
            break

        # training
        start_time = time.time()

        #feed_dict = regressionnet.fill_joint_feed_dict(net,
        #                                               regressionnet.batch2feeds_flip(train_iterator.next())[:3],
        #                                               conv_lr=conv_lr,
        #                                               fc_lr=fc_lr,
        #                                               phase='train')


        feed_dict = regressionnet.fill_joint_feed_dict(net,
                                                       regressionnet.batch2feeds(train_iterator.next())[:3],
                                                       conv_lr=conv_lr,
                                                       fc_lr=fc_lr,
                                                       phase='train')
        if step < fix_conv_iter:
            feed_dict['lr/conv_lr:0'] = 0.0

        if step < fix_conv_iter:
            cur_train_op = fc_train_op
        else:
            cur_train_op = train_op

        if step % summary_step == 0:
            global_step, summary_str, _, loss_value = net.sess.run(
                [net.global_iter_counter,
                 summary_op,
                 cur_train_op,
                 pose_loss_op],
                feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, global_step=global_step)
        else:
            global_step, _, loss_value = net.sess.run(
                [net.global_iter_counter, cur_train_op, pose_loss_op],
                feed_dict=feed_dict)

        duration = time.time() - start_time

        if step % log_step == 0 or step + 1 == max_iter:
            print('Step %d: train/pose_loss = %.2f (%.3f s, %.2f im/s)'
                  % (global_step, loss_value, duration,
                     batch_size // duration))


def main(argv):
    """
    Run training of the Deeppose stg-1
    """
    args = get_arguments(argv)
    if not os.path.exists(args.o_dir):
        os.makedirs(args.o_dir)
    suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    with open(os.path.join(args.o_dir, 'params.dump_{}.txt'.format(suffix)), 'w') as f:
        f.write('{}\n'.format(pprint.pformat(args)))

    net, loss_op, pose_loss_op, train_op = regressionnet.create_regression_net(
        n_joints=args.n_joints,
        init_snapshot_path=args.snapshot,
        is_resume=args.resume,
        reset_iter_counter=args.reset_iter_counter,
        reset_moving_averages=args.reset_moving_averages,
        optimizer_type=args.optimizer,
        #gpu_memory_fraction=0.32,  # Set how much GPU memory to reserve for the network
        #gpu_memory_fraction=0.95,  # increased
        net_type=args.net_type)
    with net.graph.as_default():
        saver = tf.train.Saver(max_to_keep=None)

    print ('args.resume: {}\nargs.snapshot: {}'.format(args.resume, args.snapshot))
    bbox_extension_range = (args.bbox_extension_min, args.bbox_extension_max)
    if bbox_extension_range[0] is None or bbox_extension_range[1] is None:
        bbox_extension_range = None
        test_bbox_extension_range = None
    else:
        test_bbox_extension_range = (bbox_extension_range[1], bbox_extension_range[1])

    ###########################################################################
    # prepare training data
    train_dataset = dataset.PoseDataset(
        args.train_csv_fn, args.img_path_prefix, args.im_size,
        fliplr=args.fliplr,
        rotate=args.rotate,
        rotate_range=args.rotate_range,
        shift=args.shift,
        bbox_extension_range=bbox_extension_range,
        min_dim=args.min_dim,
        coord_normalize=args.coord_normalize,
        gcn=args.gcn,
        fname_index=args.fname_index,
        joint_index=args.joint_index, joint_index_end=29,
        symmetric_joints=args.symmetric_joints,
        ignore_label=args.ignore_label,
        should_downscale_images=args.should_downscale_images,
        downscale_height=args.downscale_height
    )
    print ('training dataset size: {}'.format(len(train_dataset)))

    # augment the training data set by horizontal flip (2018. 6. 25)
    train_dataset.augmentByRotation((-5,5))
    #train_dataset.augmentByRotation((-10,-5,5,10))
    print ('augment training dataset by rotation: {}'.format(len(train_dataset)))

    # augment the training data set by horizontal flip (2018. 6. 21)
    train_dataset.augmentByFlip()
    print ('augment training dataset by horizontal flip: {}'.format(len(train_dataset)))

    ###########################################################################
    # prepare training data
    test_dataset = dataset.PoseDataset(
        args.test_csv_fn, args.img_path_prefix, args.im_size,
        fliplr=False, rotate=False,
        shift=None,
        bbox_extension_range=test_bbox_extension_range,
        coord_normalize=args.coord_normalize,
        gcn=args.gcn,
        fname_index=args.fname_index,
        joint_index=args.joint_index, joint_index_end=29,
        symmetric_joints=args.symmetric_joints,
        ignore_label=args.ignore_label,
        should_return_bbox=True,
        should_downscale_images=args.should_downscale_images,
        downscale_height=args.downscale_height
    )


    ###########################################################################
    # prepare iterators

    np.random.seed(args.seed)
    train_iterator = iterators.MultiprocessIterator(train_dataset, args.batch_size,
                                                    n_processes=args.workers, n_prefetch=3)
    test_iterator = iterators.MultiprocessIterator(test_dataset, args.batch_size,
                                                   repeat=False, shuffle=False,
                                                   n_processes=1, n_prefetch=1)

    #val_iterator = None
    val_iterator = train_iterator    # By default, training set is the validation set
    if args.val_csv_fn is not None and args.val_csv_fn != '':
        small_train_dataset = dataset.PoseDataset(
            args.val_csv_fn,
            args.img_path_prefix, args.im_size,
            fliplr=False, rotate=False,
            shift=None,
            bbox_extension_range=test_bbox_extension_range,
            coord_normalize=args.coord_normalize,
            gcn=args.gcn,
            fname_index=args.fname_index,
            joint_index=args.joint_index,
            symmetric_joints=args.symmetric_joints,
            ignore_label=args.ignore_label,
            should_return_bbox=True,
            should_downscale_images=args.should_downscale_images,
            downscale_height=args.downscale_height
        )
        val_iterator = iterators.MultiprocessIterator(
            small_train_dataset, args.batch_size,
            repeat=False, shuffle=False,
            n_processes=1, n_prefetch=1)

    train_loop(net, saver, loss_op, pose_loss_op, train_op, args.dataset_name,
               train_iterator, test_iterator,
               val_iterator=val_iterator,
               max_iter=args.max_iter,
               test_step=args.test_step,
               log_step=args.log_step,
               snapshot_step=args.snapshot_step,
               batch_size=args.batch_size,
               conv_lr=args.conv_lr,
               fc_lr=args.fc_lr,
               fix_conv_iter=args.fix_conv_iter,
               output_dir=args.o_dir
               )

if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
