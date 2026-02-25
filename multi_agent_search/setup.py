from glob import glob

from setuptools import find_packages, setup

package_name = "multi_agent_search"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        (
            "share/" + package_name + "/config",
            [
                "config/config.toml",
                "config/robot_config.yaml",
                "config/amcl_params.yaml",
                "config/slam_toolbox_params.yaml",
            ],
        ),
        ("share/" + package_name + "/world", glob("world/*.world")),
        ("share/" + package_name + "/world/include", glob("world/include/*.inc")),
    ],
    package_data={"": ["py.typed"]},
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Noah Pragin",
    maintainer_email="npragin@gmail.com",
    description="A project exploring multi-agent search algorithm implementations.",
    license="Apache-2.0",
    extras_require={
        "test": [
            "pytest",
        ],
    },
    entry_points={
        "console_scripts": [
            "comms_manager = multi_agent_search.comms_manager:main",
            "brain_dead_agent = multi_agent_search.brain_dead_agent:main",
            "stage_monitor = multi_agent_search.stage_monitor:main",
        ],
    },
)
