# Usage 

The script can take a photo as an input or use a camera through cv2 library.
The output goes to PREFIX*.png files or shown using `cv2.imshow`.

```
usage: gaze_tracking_pipeline.py [-h]
                                 (--input-photo FILE | --take-photo | --take-video)
                                 [--output-prefix STRING] [--output-cv2]

Gaze tracking demo

optional arguments:
  -h, --help            show this help message and exit
  --input-photo FILE    Photo file to use as input (default: None)
  --take-photo          Take photo using camera 0 (default: False)
  --take-video          Take video using camera 0 (default: False)

Outputs:
  --output-prefix PREFIX
                        Optional prefix enables saving to files (default:
                        None)
  --output-cv2          Enables output using cv2.imshow() (default: False)

```

# Docker commands
*docker build -t gaze_tracking .*

*docker run --name gaze_tracking gaze_tracking*
