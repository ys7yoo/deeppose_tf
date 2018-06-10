import numpy as np
"""
Person-Centric (PC) Annotations.
The canonical joint order for MET dataset:
0 Head top
1 Neck
2 Right shoulder (from person's perspective)
3 Right elbow
4 Right wrist
#5 Right hip
#6 Right knee
#7 Right ankle
5 Left shoulder
6 Left elbow
7 Left wrist
#11 Left hip
#12 Left knee
#13 Left ankle
"""

NUM_JOINTS = 8
CANONICAL_JOINT_NAMES = ['Head', 'Neck', 
                         'R Shoulder', 'R elbow', 'R wrist',
                         #'R hip', 'R knee', 'R ankle',
                         'L shoulder', 'L elbow', 'L wrist'] 
                         #'L hip','L knee', 'L ankle']


def joints2sticks(joints):
    """
    Args:
        joints: array of joints in the canonical order.
      The canonical joint order:
        0 Head top
        1 Neck
        2 Right shoulder (from person's perspective)
        3 Right elbow
        4 Right wrist
        5 Left shoulder
        6 Left elbow
        7 Left wrist

    Returns:
        sticks: array of sticks in the canonical order.
      The canonical part stick order:
        0 Head
        1 Right Upper Arm
        2 Right Lower Arm
        3 Left Upper Arm
        4 Left Lower Arm
    """
    assert joints.shape == (NUM_JOINTS, 2)
    stick_n = 5  # number of stick
    sticks = np.zeros((stick_n, 4), dtype=np.float32)
    sticks[0, :] = np.hstack([joints[0, :], joints[1, :]])  # Head
    sticks[1, :] = np.hstack([joints[2, :], joints[3, :]])  # Left U.arms
    sticks[2, :] = np.hstack([joints[3, :], joints[4, :]])  # Left L.arms
    sticks[3, :] = np.hstack([joints[5, :], joints[6, :]])  # Right U.arms
    sticks[4, :] = np.hstack([joints[6, :], joints[7, :]])  # Right L.arms
    return sticks


def convert2canonical(joints):
    """
    Convert joints to evaluation structure.
    Permute joints according to the canonical joint order.
    """
    assert joints.shape[1:] == (NUM_JOINTS, 2), 'MET must contain 8 joints per person'
    # convert to the canonical joint order
    joint_order = [7,  # Head top
                   6,  # Neck
                   2,   # Right shoulder
                   1,   # Right elbow
                   0,   # Right wrist
                   3,   # Left shoulder
                   4,  # Left elbow
                   5]  # Left wrist
    canonical = [dict() for _ in range(joints.shape[0])]
    for i in range(joints.shape[0]):
        canonical[i]['joints'] = joints[i, joint_order, :]
        canonical[i]['sticks'] = joints2sticks(canonical[i]['joints'])
    return canonical


