from setuptools import find_packages, setup

setup(
    name="mlproject-yolov8",
    version="0.1",
    description="Yolov8 Object Detection - Football Player and Ball Tracker",
    license="MIT",
    packages=find_packages(include=["src", "src.*"]),
    zip_safe=False,
)
