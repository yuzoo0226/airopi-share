import os
from glob import glob

from setuptools import find_packages, setup

package_name = "hsr_openpi"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        (os.path.join("share", package_name), ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="AIRoA",
    maintainer_email="yano.yuga@airoa.org",
    description="ROS 2 deployment of openpi pi0 / pi0.5 policies on the Toyota HSR.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "hsr_openpi_node = hsr_openpi.hsr_openpi_node:main",
            "reset_pose = hsr_openpi.reset_pose:main",
            "control_mode_publisher = hsr_openpi.control_mode_publisher:main",
            "random_motion = hsr_openpi.random_motion:main",
        ],
    },
)
