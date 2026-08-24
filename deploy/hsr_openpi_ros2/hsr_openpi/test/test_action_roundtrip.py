"""The dataset's action definition and the deployment node must be exact inverses.

`rosbag2_to_lerobot.py` writes

    action[0:5]  = arm_command  - state[0:5]
    action[5]    = gripper_command
    action[6:8]  = head_command  - state[6:8]
    action[8:11] = cmd_vel

and `OpenpiPolicy._delta_to_absolute` reconstructs the command as

    command = action + [state[:5], 0, state[6:8], 0, 0, 0]

If those two ever drift apart the policy still trains happily and then commands
nonsense on the robot, with nothing in the logs to say why - so pin the contract
down here.

    python -m pytest deploy/hsr_openpi_ros2/hsr_openpi/test/test_action_roundtrip.py
"""

import numpy as np
import pytest

from hsr_openpi.openpi_policy import OpenpiPolicy

# Same layout as gz_world / rosbag2_to_lerobot:
#   0..4  arm_lift, arm_flex, arm_roll, wrist_flex, wrist_roll   (relative)
#   5     gripper                                                (absolute)
#   6..7  head_pan, head_tilt                                    (relative)
#   8..10 base vx, vy, wz                                        (velocity)
ARM = slice(0, 5)
GRIPPER = 5
HEAD = slice(6, 8)
BASE = slice(8, 11)


def encode(state: np.ndarray, arm_cmd, gripper_cmd, head_cmd, base_cmd) -> np.ndarray:
    """What the converter writes into action.relative."""
    action = np.zeros(11, dtype=np.float32)
    action[ARM] = np.asarray(arm_cmd, np.float32) - state[ARM]
    action[GRIPPER] = gripper_cmd
    action[HEAD] = np.asarray(head_cmd, np.float32) - state[6:8]
    action[BASE] = base_cmd
    return action


def test_roundtrip_recovers_the_commands():
    rng = np.random.default_rng(0)
    for _ in range(100):
        state = rng.uniform(-2.0, 2.0, size=8).astype(np.float32)
        arm_cmd = rng.uniform(-2.0, 2.0, size=5).astype(np.float32)
        gripper_cmd = np.float32(rng.uniform(0.0, 1.2))
        head_cmd = rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
        base_cmd = rng.uniform(-0.3, 0.3, size=3).astype(np.float32)

        action = encode(state, arm_cmd, gripper_cmd, head_cmd, base_cmd)
        recovered = OpenpiPolicy._delta_to_absolute(action, state)

        np.testing.assert_allclose(recovered[ARM], arm_cmd, rtol=0, atol=1e-6)
        np.testing.assert_allclose(recovered[GRIPPER], gripper_cmd, rtol=0, atol=1e-6)
        np.testing.assert_allclose(recovered[HEAD], head_cmd, rtol=0, atol=1e-6)
        np.testing.assert_allclose(recovered[BASE], base_cmd, rtol=0, atol=1e-6)


def test_absolute_dimensions_ignore_the_state():
    """gripper and base must not pick up the measured joint values."""
    action = np.zeros(11, dtype=np.float32)
    action[GRIPPER] = 0.42
    action[BASE] = [0.1, -0.2, 0.3]

    quiet = np.zeros(8, dtype=np.float32)
    busy = np.arange(8, dtype=np.float32)

    a = OpenpiPolicy._delta_to_absolute(action, quiet)
    b = OpenpiPolicy._delta_to_absolute(action, busy)

    np.testing.assert_allclose(a[GRIPPER], b[GRIPPER])
    np.testing.assert_allclose(a[BASE], b[BASE])
    # ...while the relative dimensions must follow the state
    assert not np.allclose(a[ARM], b[ARM])
    assert not np.allclose(a[HEAD], b[HEAD])


def test_hand_motor_state_index_is_not_added_to_the_gripper_action():
    """state[5] is hand_motor; adding it would double count the gripper."""
    state = np.zeros(8, dtype=np.float32)
    state[5] = 1.0
    action = np.zeros(11, dtype=np.float32)
    action[GRIPPER] = 0.3
    assert OpenpiPolicy._delta_to_absolute(action, state)[GRIPPER] == pytest.approx(0.3)


def test_gripper_separation_mapping_is_monotonic_and_inverts():
    from hsr_openpi.pick_task import motor_for_separation

    # measured on hsrb4s: 1.20 -> 0.1349 m, 0.00 -> 0.0044 m
    assert motor_for_separation(0.1349) == pytest.approx(1.2, abs=1e-3)
    assert motor_for_separation(0.0044) == pytest.approx(0.0, abs=1e-3)
    widths = np.linspace(0.005, 0.135, 25)
    motors = [motor_for_separation(w) for w in widths]
    assert all(b >= a for a, b in zip(motors, motors[1:]))
    # and it stays inside the joint limit
    assert motor_for_separation(1.0) == pytest.approx(1.2)
    assert motor_for_separation(-1.0) == pytest.approx(0.0)
