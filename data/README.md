# Data (contents gitignored)

Expected layout (matches what `reference/rl_vo` loaders expect):

```
data/
├── TartanAir/   <scene>/<Easy|Hard>/Pxxx/{image_left_gray/, pose_left.txt} + calibration/tartan_pinhole.yaml
├── EuRoC/       MH_01_easy/mav0/{cam0/, state_groundtruth_estimate0/} + calibration/euroc_mono.yaml
└── TUM-RGBD/    rgbd_dataset_freiburg1_desk/{rgb/, rgb.txt, groundtruth.txt} + calibration/tum.yaml
```

Sources:
- TartanAir: https://theairlab.org/tartanair-dataset/ (convert image_left to grayscale → image_left_gray)
- EuRoC: https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets
- TUM-RGBD: https://vision.in.tum.de/data/datasets/rgbd-dataset/download
- Calibration yamls: https://download.ifi.uzh.ch/rpg/ECCV24_VORL/{tartan_pinhole,euroc_mono,tum}.yaml

Locally, keep only a small subset for debugging. The full data lives on Google Drive for Colab.
