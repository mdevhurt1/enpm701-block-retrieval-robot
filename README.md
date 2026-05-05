# ENPM 701 Grand Challenge — Final Project

## Overview

Autonomous robot that locates colored blocks (red, green, blue) in a 7x7 ft arena,
picks them up, and delivers them to the NW collection zone. Starts from the SW launch
zone. Uses pigpio for hardware-timed encoder counting and wall-proximity localization.

## Requirements

- Raspberry Pi with pigpiod running (`sudo pigpiod`)
- Python packages: `pip install -r requirements-dev.txt`

## Running the mission

```bash
# From final_project/
sudo pigpiod          # start pigpio daemon if not running
python main.py        # autonomous run
python -m fp          # equivalent
```

## Package layout

```
final_project/
├── main.py
├── fp/
│   ├── arena.py        # arena geometry, zone bounds, coordinate math
│   ├── cli.py          # startup, hardware init, run loop
│   ├── fsm.py          # mission state machine
│   ├── hal.py          # pigpio GPIO, motors, encoders, ultrasonic, IMU, servo
│   ├── localization.py # pose tracking, wall-clearance checks
│   ├── mapping.py      # path recording and PNG visualization
│   ├── motion.py       # drive, pivot, localize, approach, grab, release
│   ├── search.py       # grid-based search planner, 360-degree scan
│   └── vision.py       # camera capture, HSV blob detection
└── calibration/
    ├── calibrate_camera.py     # compute and save camera intrinsics
    ├── blob_size.py            # display blob area readings live
    ├── encoder_test.py         # verify encoder counts per revolution
    ├── extract_hsv.py          # extract HSV range from a saved image
    ├── hsv_tuner_static.py     # tune HSV masks on a JPEG without hardware
    ├── make_timelapse.py       # stitch timelapse frames from a run
    ├── ultrasonic_test.py      # print live ultrasonic readings
    └── vision_sample.py        # capture and display a single frame
```

## Running tests

```bash
cd final_project/
python -m pytest tests/ -v
```
