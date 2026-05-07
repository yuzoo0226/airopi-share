import albumentations as A

import numpy as np
import cv2

# mind the slashes
JOINTS_TOPIC = "/hsrb/joint_states"
IMAGE_HEAD_TOPIC = "/hsrb/head_rgbd_sensor/rgb/image_rect_color/compressed"
IMAGE_HAND_TOPIC = "/hsrb/hand_camera/image_raw/compressed"

DEPTH_HEAD_TOPIC = "/hsrb/head_rgbd_sensor/depth_registered/image_rect_raw"

ARM_TOPIC = "/hsrb/arm_trajectory_controller/command"
HEAD_TOPIC = "/hsrb/head_trajectory_controller/command"
GRIPPER_OPEN_TOPIC = "/hsrb/gripper_controller/command"
GRIPPER_CLOSE_TOPIC = "/hsrb/gripper_controller/grasp/goal"
BASE_TOPIC = "/hsrb/command_velocity"
BASE_TOPIC2 = "/hsrb/omni_base_controller/command"

CONTROL_TOPIC = "/control_mode"

# determines the content and order of values in vector
JOINT_NAMES = [
    "hand_motor_joint",
    "arm_flex_joint",
    "arm_lift_joint",
    "arm_roll_joint",
    "wrist_flex_joint",
    "wrist_roll_joint",
    # "head_pan_joint",
    # "head_tilt_joint",
]

JOINT_RANGE = [
    (-0.2, 1.24),  # ?
    (-2.617, 0),
    (0, 0.69),
    (-1.919, 3.665),
    (-1.919, 1.221),
    (-1.919, 3.665),
]

# determines the content and order of values in vector
ACTION_NAMES = [
    "arm_flex_joint",
    "arm_lift_joint",
    "arm_roll_joint",
    "wrist_flex_joint",
    "wrist_roll_joint",
    "base_x",
    "base_y",
    "base_t",
    "gripper",
    "head_pan_joint",
    "head_tilt_joint",
]

ACTION_RANGE = [
    (-2.617, 0),
    (0, 0.69),
    (-1.919, 3.665),
    (-1.919, 1.221),
    (-1.919, 3.665),
    (-0.3, 0.3),
    (-0.3, 0.3),
    (-1.0, 1.0),
    (0.0, 1.0),
    (-3.839, 1.745),
    (-1.570, 0.523),
]


def dict2vec(act_dict):
    return np.asarray([act_dict[name] if name in act_dict else 0 for name in ACTION_NAMES], dtype=np.float32)


HOME_POSE = dict2vec({
    "wrist_flex_joint": -0.5 * np.pi,
    "gripper": 1,
    "head_tilt_joint": -0.2 * np.pi,
})

LOOK_DOWN_POSE = dict2vec({
    "wrist_flex_joint": -0.5 * np.pi,
    "arm_roll_joint": -0.5 * np.pi,
    "gripper": 1,
    "head_tilt_joint": -0.4 * np.pi,
})

tilt = 0.08
FLOOR_PICK_POSE = dict2vec({
    "arm_flex_joint": (-0.5 - tilt) * np.pi,
    "arm_lift_joint": 0.5,
    "wrist_flex_joint": (-0.5 + tilt) * np.pi,
    "gripper": 1,
    "head_tilt_joint": -0.4 * np.pi,
})

FLOOR_PICK_POSE2 = dict2vec({
    "arm_flex_joint": -0.5 * np.pi,
    "arm_lift_joint": 0.2,
    "wrist_flex_joint": -0.25 * np.pi,
    "gripper": 1,
    "head_tilt_joint": -0.2 * np.pi,
})

SIDE_PICK_POSE = dict2vec({
    "arm_flex_joint": -0.5 * np.pi,
    "arm_lift_joint": 0.43,
    "gripper": 1,
})

DEFAULT_CAMERA_INFO_HEAD = dict(
    K=[536.0731084002158, 0.0, 328.3979612150148, 0.0, 536.8475975592386, 243.0627964250878, 0.0, 0.0, 1.0],
    D=[0.06019986631604691, -0.2156954597119431, -0.0006445663064014923, 0.002549600335155992, 0.1413441734129954]
)

DEFAULT_CAMERA_INFO_HAND = dict(
    K=[218.56578498479448, 0.0, 322.71813084582493, 0.0, 217.51070337632643, 266.3125602819701, 0.0, 0.0, 1.0],
    D=[-0.13043778002804507, 0.015626710289458456, 0.0006325662756056388, -0.00042837589843374926, -0.0008662741644224998]
)


DEFAULT_TRANSFORM_HEAD = dict()
DEFAULT_TRANSFORM_HAND = dict(
    rotate=90,
    K=DEFAULT_CAMERA_INFO_HAND["K"],
    D=DEFAULT_CAMERA_INFO_HAND["D"],
)


def create_transform(image_size=224, rotate=0, K=None, D=None):
    transforms = [A.CenterCrop(480, 480), A.Resize(image_size, image_size)]

    if rotate > 0:
        transforms.append(A.Rotate((rotate, rotate), p=1.0))

    image_transform = A.Compose(transforms)

    def fn(image):
        if K is not None and D is not None:
            image = cv2.undistort(
                image,
                np.asarray(K).reshape(3, 3),
                np.asarray(D)
            )

        image = image_transform(image=image)["image"]
        return image

    return fn