# Drone image gps  mapper

This repository contains a Python implementation for mapping GPS coordinates onto drone camera images. It utilizes camera calibration, drone pose estimation, and projection transformation to compute the image pixel coordinates corresponding to given world coordinates.

## Features

- Conversion of world (GPS/local) coordinates to camera coordinates.
- Projection of 3D points onto the 2D image plane using camera intrinsic and extrinsic parameters.
- Example implementation using both custom functions and OpenCV's `projectPoints`.
