"""Spawn / remove / teleport models in an Ignition Gazebo world from Python.

`ros_gz` only ships a `create` executable, which costs a process launch per
spawn; for a thousand-episode collection run the Ignition transport services are
called directly through the `ign` CLI instead:

    /world/<world>/create    ignition.msgs.EntityFactory -> Boolean
    /world/<world>/remove    ignition.msgs.Entity        -> Boolean
    /world/<world>/set_pose  ignition.msgs.Pose          -> Boolean

The object library below is deliberately simple (boxes and cylinders of varying
size, mass, friction and colour): the point is that the policy has to locate a
*randomly placed, randomly shaped* object from the cameras, not that the meshes
are realistic. Every object is narrow enough for the HSR gripper, whose fingers
span 13.5 cm fully open.
"""

from __future__ import annotations

import dataclasses
import math
import pathlib
import subprocess
import tempfile
from typing import Dict, List, Optional, Sequence, Tuple

# Measured on hsrb4s in Ignition Fortress (see docs/ros2_pick_task_ja.md):
#   with arm_flex = wrist_flex = -pi/2 (palm pointing straight down) the palm sits
#   at (0.474, 0.078, 0.194 + arm_lift) in base_footprint, and the finger tips are
#   0.049 m (fully open) to 0.094 m (closed) below the palm.
GRASP_FIX_NOTE = """The Ignition HSR gripper cannot hold anything.

hand_motor_joint and the finger joints are driven kinematically by
hsrb_gz_ros2_control (mimic joints), so the fingers close straight through
whatever is between them: commanding a full close around a static 9 cm box still
reaches hand_motor_joint = 0.0. Sustained grip force is therefore impossible,
and the object is never picked up no matter how the closing is tuned (position
command, grasp action / force control, or any squeeze value).

Like gazebo_grasp_fix in Gazebo Classic, the object is instead *welded* to the
hand once a geometric grasp condition holds. Fortress's DetachableJoint only
attaches at spawn time (its attach_topic does not re-attach after a detach), so
attaching means removing the object and respawning it, with the plugin, at its
current pose. Note that `hand_palm_link` is a massless frame that SDF conversion
folds into its parent - the weld has to target `hand_motor_dummy_link`, which
sits at the same origin and does exist as a Gazebo link."""

PALM_OFFSET_X = 0.474
PALM_OFFSET_Y = 0.078
PALM_Z_AT_ZERO_LIFT = 0.194
FINGER_TIP_BELOW_PALM_OPEN = 0.060  # hand_motor_joint = 1.0


@dataclasses.dataclass(frozen=True)
class ObjectSpec:
    name: str
    shape: str  # "box" | "cylinder"
    size: Tuple[float, ...]  # box: (x, y, z); cylinder: (radius, length)
    mass: float
    rgba: Tuple[float, float, float, float]

    @property
    def height(self) -> float:
        return self.size[2] if self.shape == "box" else self.size[1]

    @property
    def width(self) -> float:
        return max(self.size[0], self.size[1]) if self.shape == "box" else 2 * self.size[0]

    def geometry_xml(self) -> str:
        if self.shape == "box":
            return f"<box><size>{self.size[0]} {self.size[1]} {self.size[2]}</size></box>"
        return f"<cylinder><radius>{self.size[0]}</radius><length>{self.size[1]}</length></cylinder>"

    def inertia_xml(self) -> str:
        m = self.mass
        if self.shape == "box":
            x, y, z = self.size
            ixx = m * (y * y + z * z) / 12.0
            iyy = m * (x * x + z * z) / 12.0
            izz = m * (x * x + y * y) / 12.0
        else:
            r, h = self.size
            ixx = iyy = m * (3 * r * r + h * h) / 12.0
            izz = m * r * r / 2.0
        return (
            f"<inertia><ixx>{ixx:.6e}</ixx><iyy>{iyy:.6e}</iyy><izz>{izz:.6e}</izz>"
            f"<ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia>"
        )

    def to_sdf(self, model_name: str = "obj", *, attach_to: Optional[Tuple[str, str]] = None) -> str:
        """SDF for this object.

        ``attach_to`` adds a DetachableJoint welding the object to
        ``(model, link)``. See ``GRASP_FIX_NOTE`` for why this is needed.
        """
        r, g, b, a = self.rgba
        plugin = ""
        if attach_to is not None:
            model, link = attach_to
            plugin = f"""
    <plugin filename="ignition-gazebo-detachable-joint-system"
            name="ignition::gazebo::systems::DetachableJoint">
      <parent_link>link</parent_link>
      <child_model>{model}</child_model>
      <child_link>{link}</child_link>
      <detach_topic>/{model_name}/detach</detach_topic>
    </plugin>"""
        return f"""<?xml version="1.0"?>
<sdf version="1.7">
  <model name="{model_name}">{plugin}
    <link name="link">
      <inertial><mass>{self.mass}</mass>{self.inertia_xml()}</inertial>
      <collision name="collision">
        <geometry>{self.geometry_xml()}</geometry>
        <surface>
          <friction><ode><mu>1.4</mu><mu2>1.4</mu2></ode></friction>
          <contact><ode><kp>1e6</kp><kd>50</kd></ode></contact>
        </surface>
      </collision>
      <visual name="visual">
        <geometry>{self.geometry_xml()}</geometry>
        <material>
          <ambient>{r} {g} {b} {a}</ambient>
          <diffuse>{r} {g} {b} {a}</diffuse>
          <specular>0.2 0.2 0.2 1</specular>
        </material>
      </visual>
    </link>
  </model>
</sdf>
"""


# Ten graspable objects. Widths stay <= 0.075 m so the fingers can close around
# them, heights 0.06..0.16 m so the required arm_lift stays inside 0 .. 0.69.
OBJECT_LIBRARY: List[ObjectSpec] = [
    ObjectSpec("red_can", "cylinder", (0.033, 0.120), 0.16, (0.85, 0.10, 0.10, 1.0)),
    ObjectSpec("blue_box", "box", (0.055, 0.055, 0.100), 0.15, (0.10, 0.25, 0.85, 1.0)),
    ObjectSpec("green_bottle", "cylinder", (0.028, 0.160), 0.18, (0.10, 0.70, 0.20, 1.0)),
    ObjectSpec("yellow_block", "box", (0.070, 0.045, 0.075), 0.14, (0.92, 0.82, 0.10, 1.0)),
    ObjectSpec("purple_stick", "box", (0.035, 0.035, 0.150), 0.12, (0.55, 0.15, 0.75, 1.0)),
    ObjectSpec("orange_cup", "cylinder", (0.040, 0.090), 0.13, (0.95, 0.50, 0.10, 1.0)),
    ObjectSpec("cyan_brick", "box", (0.075, 0.050, 0.060), 0.20, (0.10, 0.75, 0.80, 1.0)),
    ObjectSpec("white_tube", "cylinder", (0.025, 0.140), 0.10, (0.92, 0.92, 0.92, 1.0)),
    ObjectSpec("brown_box", "box", (0.060, 0.070, 0.110), 0.22, (0.45, 0.30, 0.15, 1.0)),
    ObjectSpec("pink_puck", "cylinder", (0.045, 0.070), 0.15, (0.95, 0.45, 0.65, 1.0)),
]

OBJECT_NAMES = [o.name for o in OBJECT_LIBRARY]


def table_sdf(*, width: float, depth: float, height: float, model_name: str = "table") -> str:
    """A static counter-style table. Solid to the floor so the base cannot drive
    underneath it, which keeps the reachable band in front of the near edge."""
    return f"""<?xml version="1.0"?>
<sdf version="1.7">
  <model name="{model_name}">
    <static>true</static>
    <link name="link">
      <collision name="collision">
        <geometry><box><size>{depth} {width} {height}</size></box></geometry>
        <surface><friction><ode><mu>1.2</mu><mu2>1.2</mu2></ode></friction></surface>
      </collision>
      <visual name="visual">
        <geometry><box><size>{depth} {width} {height}</size></box></geometry>
        <material>
          <ambient>0.65 0.60 0.52 1</ambient>
          <diffuse>0.65 0.60 0.52 1</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>
"""


class GzWorld:
    """Thin wrapper around the Ignition transport services of a running world."""

    def __init__(self, world: str = "default", timeout_ms: int = 4000, ign: str = "ign"):
        self.world = world
        self.timeout_ms = int(timeout_ms)
        self.ign = ign
        self._tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="hsr_gz_models_"))

    # -- low level ------------------------------------------------------- #
    def _call(self, service: str, reqtype: str, req: str) -> bool:
        cmd = [
            self.ign, "service",
            "-s", f"/world/{self.world}/{service}",
            "--reqtype", reqtype,
            "--reptype", "ignition.msgs.Boolean",
            "--timeout", str(self.timeout_ms),
            "--req", req,
        ]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout_ms / 1000 + 5)
        except subprocess.TimeoutExpired:
            return False
        return "data: true" in out.stdout

    @staticmethod
    def _pose_req(x: float, y: float, z: float, yaw: float) -> str:
        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)
        return (
            f"position: {{x: {x}, y: {y}, z: {z}}}, "
            f"orientation: {{x: 0, y: 0, z: {qz}, w: {qw}}}"
        )

    # -- public API ------------------------------------------------------ #
    def spawn(self, model_name: str, sdf: str, *, x: float, y: float, z: float, yaw: float = 0.0) -> bool:
        path = self._tmpdir / f"{model_name}.sdf"
        path.write_text(sdf)
        req = f'sdf_filename: "{path}", name: "{model_name}", {{POSE}}'.replace(
            "{POSE}", f"pose: {{{self._pose_req(x, y, z, yaw)}}}"
        )
        return self._call("create", "ignition.msgs.EntityFactory", req)

    def remove(self, model_name: str) -> bool:
        return self._call("remove", "ignition.msgs.Entity", f'name: "{model_name}", type: MODEL')

    def set_pose(self, model_name: str, *, x: float, y: float, z: float, yaw: float = 0.0) -> bool:
        return self._call(
            "set_pose", "ignition.msgs.Pose", f'name: "{model_name}", {self._pose_req(x, y, z, yaw)}'
        )

    def publish_empty(self, topic: str) -> bool:
        """Publish an ignition.msgs.Empty (used for the detach topic)."""
        try:
            subprocess.run(
                [self.ign, "topic", "-t", topic, "-m", "ignition.msgs.Empty", "-p", ""],
                capture_output=True, text=True, timeout=10,
            )
        except subprocess.TimeoutExpired:
            return False
        return True

    def list_models(self) -> List[str]:
        try:
            out = subprocess.run(
                [self.ign, "model", "--list"], capture_output=True, text=True, timeout=10
            ).stdout
        except subprocess.TimeoutExpired:
            return []
        names: List[str] = []
        for line in out.splitlines():
            line = line.strip()
            if line.startswith("- "):
                names.append(line[2:].strip())
        return names


# --------------------------------------------------------------------------- #
# grasp geometry helpers
# --------------------------------------------------------------------------- #
def arm_lift_for_grasp(grasp_z: float, *, finger_offset: float = FINGER_TIP_BELOW_PALM_OPEN) -> float:
    """arm_lift that puts the (open) finger tips at ``grasp_z`` in world z."""
    return grasp_z + finger_offset - PALM_Z_AT_ZERO_LIFT


def base_pose_for_grasp(obj_x: float, obj_y: float, yaw: float) -> Tuple[float, float]:
    """base_footprint pose that puts the palm above (obj_x, obj_y) at heading ``yaw``."""
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    dx = PALM_OFFSET_X * cos_y - PALM_OFFSET_Y * sin_y
    dy = PALM_OFFSET_X * sin_y + PALM_OFFSET_Y * cos_y
    return obj_x - dx, obj_y - dy


def pick_object(rng, names: Optional[Sequence[str]] = None) -> ObjectSpec:
    library = OBJECT_LIBRARY if not names else [o for o in OBJECT_LIBRARY if o.name in set(names)]
    if not library:
        raise ValueError(f"no objects match {names}")
    return library[int(rng.integers(len(library)))]


OBJECT_BY_NAME: Dict[str, ObjectSpec] = {o.name: o for o in OBJECT_LIBRARY}
