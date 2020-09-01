# TERRA data cropping
In this folder they are core codes for TERRA data cropping, in both stereo.py and thermal.py there is a function named single_image_process(), script run-pipeline call this interface to finish cropping job for stereo camera and FLIR camera.

## stereo.py: single_image_process(in_dir, out_dir, plot_dir, convt)
in_dir: input stereo raw data dirctory, which should contain left.bin, right.bin and metadata.json file from folder terra/ua-mac/raw_data/stereoTop/XXXX-XX-XX/time_stamp

out_dir: output directory which contains un-cropped RGB image

plot_dir: output directory which contains crop-to-plot RGB image

convt: an object contains plot boundary information which init by terra_common.py

## thermal.py: singe_image_process(in_dir, out_dir, plot_dir, crop_color_dir, convt)
in_dir: input FLIR raw data dirctory, which should contain ir.bin and metadata.json file from folder terra/ua-mac/raw_data/flirlrCamera/XXXX-XX-XX/time_stamp

out_dir: output directory which contains un-cropped numpy file with raw tempreture data and pseudo color image

plot_dir: output directory which contains crop-to-plot numpy file with raw tempreture data

crop_color_dir: output directory which contains crop-to-plot pseudo color image

convt: an object contains plot boundary information which init by terra_common.py
