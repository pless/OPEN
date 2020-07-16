'''
Created on Jan 17, 2020

@author: zli
'''

import sys, os.path, json, random, terra_common, math, argparse
import matplotlib.pyplot as plt
from matplotlib import cm
from glob import glob
from os.path import join
import numpy as np
from numpy.matlib import repmat
from scipy.ndimage.filters import convolve
from PIL import Image
from datetime import date, timedelta, datetime
import shutil
import cv2

#fov_alpha = 1.03 # 05-19
fov_alpha = 1.12
scan_shift = -0.03

os.environ['BETYDB_KEY'] = '9999999999999999999999999999999999999999'

class calibParam:
    def __init__(self):
        self.calibrated = False
        self.calibrationR = 0.0
        self.calibrationB = 0.0
        self.calibrationF = 0.0
        self.calibrationJ1 = 0.0
        self.calibrationJ0 = 0.0
        self.calibrationa1 = 0.0
        self.calibrationa2 = 0.0
        self.calibrationX = 0.0
        self.calibrationb1 = 0.0
        self.calibrationb2 = 0.0
        self.shuttertemperature = 0.0


# convert flir raw data into temperature C degree, for date after September 15th
# code converted from Matlab https://github.com/terraref/computing-pipeline/blob/master/scripts/FLIR/FlirRawToTemperature.m
def flirRawToTemperature(rawData, calibP):

    # Constant camera-specific parameters determined by FLIR
    # Plank constant - Flir
    R = calibP.calibrationR  # function of integration time and wavelength
    B = calibP.calibrationB  # function of wavelength
    F = calibP.calibrationF  # positive value (0 - 1)
    J0 = calibP.calibrationJ0  # global offset
    J1 = calibP.calibrationJ1 # global gain

    # Constant Atmospheric transmission parameter by Flir
    X = calibP.calibrationX
    a1 = calibP.calibrationa1
    b1 = calibP.calibrationb1
    a2 = calibP.calibrationa2
    b2 = calibP.calibrationb2

    # Constant for VPD computation (sqtrH2O)
    H2O_K1 = 1.56
    H2O_K2 = 0.0694
    H2O_K3 = -0.000278
    H2O_K4 = 0.000000685

    K0 = 273.15  # Kelvin to Celsius temperature constant

    # Environmental factors
    H = 0.1  # Relative Humidity from the gantry  (0 - 1)
    T = calibP.shuttertemperature - K0 # air temperature in degree Celsius from the gantry
    D = 2.5  # ObjectDistance - camera/canopy (m)
    E = 0.98  # bject emissivity, vegetation is around 0.98, bare soil around 0.93...

    im = rawData

    AmbTemp = T + K0
    AtmTemp = T + K0

    # Step 1: Atmospheric transmission - correction factor from air temp, relative humidity and distance sensor-object;
    # Vapour pressure deficit call here sqrtH2O => convert relative humidity and air temperature in VPD - mmHg - 1mmHg=0.133 322 39 kPa
    H2OInGperM2 = H*math.exp(H2O_K1 + H2O_K2*T + H2O_K3*math.pow(T, 2) + H2O_K4*math.pow(T, 3))
    # Atmospheric transmission correction: tao
    a1b1sqH2O = (a1+b1*math.sqrt(H2OInGperM2))
    a2b2sqH2O = (a2+b2*math.sqrt(H2OInGperM2))
    exp1 = math.exp(-math.sqrt(D/2)*a1b1sqH2O)
    exp2 = math.exp(-math.sqrt(D/2)*a2b2sqH2O)
        
    tao = X*exp1 + (1-X)*exp2  # Atmospheric transmission factor

    # Step2: Correct raw pixel values from external factors
    # General equation : Total Radiation = Object Radiation + Atmosphere Radiation + Ambient Reflection Radiation

    # Object Radiation: obj_rad
    # obj_rad = Theoretical object radiation * emissivity * atmospheric transmission
    obj_rad = im*E*tao  # For each pixel

    # Theoretical atmospheric radiation: theo_atm_rad
    theo_atm_rad = (R*J1/(math.exp(B/AtmTemp)-F)) + J0

    # Atmosphere Radiation: atm_rad
    # atm_rad = (1 - atmospheric transmission) * Theoretical atmospheric radiation
    atm_rad = repmat((1-tao)*theo_atm_rad, 480, 640)

    # Theoretical Ambient Reflection Radiation: theo_amb_refl_rad
    theo_amb_refl_rad = (R*J1/(math.exp(B/AmbTemp)-F)) + J0

    # Ambient Reflection Radiation: amb_refl_rad
    # amb_refl_rad = (1 - emissivity) * atmospheric transmission * Theoretical Ambient Reflection Radiation
    amb_refl_rad = repmat((1-E)*tao*theo_amb_refl_rad, 480, 640)

    # Total Radiation: corr_pxl_val
    corr_pxl_val = obj_rad + atm_rad + amb_refl_rad

    # Step 3:RBF equation: transformation of pixel intensity in radiometric temperature from raw values or
    # corrected values (in degree Celsius)
    pxl_temp = B/np.log(R/(corr_pxl_val-J0)*J1+F) - K0  # for each pixel
    
    return pxl_temp

def crop_thermal_imageToPlot(in_dir, out_dir, plot_dir, crop_color_dir, convt):
    
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
        
    if not os.path.isdir(plot_dir):
        os.makedirs(plot_dir)
        
    if not os.path.isdir(crop_color_dir):
        os.makedirs(crop_color_dir)
    
    list_dirs = os.listdir(in_dir)
    
    for d in list_dirs:
        in_path = os.path.join(in_dir, d)
        out_path = os.path.join(out_dir, d)
        
        if not os.path.isdir(in_path):
            continue
        
        if not os.path.isdir(out_path):
            os.mkdir(out_path)
            
        singe_image_process(in_path, out_path, plot_dir, crop_color_dir, convt)
    
    return

def singe_image_process(in_dir, out_dir, plot_dir, crop_color_dir, convt):
    # find input files
    metafile, binfile = find_input_files(in_dir)
    
    if metafile == [] or binfile == [] :
        return
    
    base_name = os.path.basename(binfile)[:-4]
    out_npy_path = os.path.join(out_dir, '{}.npy'.format(base_name))
    out_png_path = os.path.join(out_dir, '{}.png'.format(base_name))
    if os.path.isfile(out_npy_path):
       return
    
    # parse meta data
    metadata = lower_keys(load_json(os.path.join(in_dir, metafile)))
    center_position, scan_time, fov, y_ends = parse_metadata(metadata)
    
    if center_position is None:
        return

    image_shape = (640, 480)
    
    # center position/fov/imgSize to plot number, image boundaries, only dominated plot for now
    plotNum, roiBox, field_roiBox = metadata_to_imageBoundaries(center_position, fov, image_shape, convt)
    if plotNum is None:
        return
    
    plot_row, plot_col = convt.plotNum_to_fieldPartition(plotNum)
    if plotNum == 0:
        return
    
    # add scan shift to y axis
    field_roiBox = add_scan_shift_to_field_roiBox(field_roiBox, y_ends)
    
    save_dir = '{0:02d}_{1:02d}_{2:04d}'.format(plot_row, plot_col, plotNum) 
    
    # bin to image
    raw_data = load_flir_data(binfile)
    if raw_data is None:
        return
    
    create_png(raw_data, out_png_path) # create colormap png for show
    
    # raw to temperature c
    tc = rawData_to_temperature(raw_data, scan_time, metadata) # get temperature
    if tc is None:
        return
    tc_data = create_npy_tc_data(tc, out_npy_path)

    # crop image
    roi_data = tc_data[int(roiBox[0]):int(roiBox[1]), int(roiBox[2]):int(roiBox[3])]
    color_img = cv2.imread(out_png_path)
    roi_img = color_img[int(roiBox[0]):int(roiBox[1]), int(roiBox[2]):int(roiBox[3])]
    
    # save image
    s_d = os.path.join(plot_dir, save_dir)
    if not os.path.isdir(s_d):
        os.mkdir(s_d)
        
    c_d = os.path.join(crop_color_dir, save_dir)
    if not os.path.isdir(c_d):
        os.mkdir(c_d)
        
    time_stamp = os.path.basename(in_dir)
    dst_npy_path = os.path.join(s_d, '{}.npy'.format(time_stamp))
    np.save(dst_npy_path, roi_data)
    dst_png_path = os.path.join(c_d, '{}.png'.format(time_stamp))
    cv2.imwrite(dst_png_path, roi_img)
    
    # write new json file
    src_json_data = load_json(os.path.join(in_dir, metafile))
    
    add_metadata = {'gwu_added_metadata': {'xmin':field_roiBox[0], 'xmax':field_roiBox[1], 'ymin':field_roiBox[2], 'ymax':field_roiBox[3]}}
    dst_json_data = dict(src_json_data)
    dst_json_data.update(add_metadata)
    
    
    dst_json_path = os.path.join(s_d, '{}.json'.format(time_stamp))
    with open(dst_json_path, 'w') as outfile:
        json.dump(dst_json_data, outfile)
        
    dst_json_path = os.path.join(c_d, '{}.json'.format(time_stamp))
    with open(dst_json_path, 'w') as outfile:
        json.dump(dst_json_data, outfile)
    
    return

def add_scan_shift_to_field_roiBox(field_roiBox, y_ends):
    
    if y_ends == '0':     # + shift
        field_roiBox[2] = field_roiBox[2]+scan_shift
        field_roiBox[3] = field_roiBox[3]+scan_shift
    else:               # - shift   
        field_roiBox[2] = field_roiBox[2]-scan_shift
        field_roiBox[3] = field_roiBox[3]-scan_shift

    return field_roiBox

def rawData_to_temperature(rawData, scan_time, metadata):
    
    try:
        calibP = get_calibrate_param(metadata)
        tc = np.zeros((480, 640))
        
        if not calibP.calibrated:
            tc = rawData/10 - 273.15
        else:
            tc = flirRawToTemperature(rawData, calibP)
    
        return tc
    except Exception as ex:
        fail('raw to temperature fail:' + str(ex))
        return

def get_calibrate_param(metadata):
    
    try:
        sensor_fixed_meta = metadata['lemnatec_measurement_metadata']['sensor_fixed_metadata']
        sensor_variable_meta =  metadata['lemnatec_measurement_metadata']['sensor_variable_metadata']
        calibrated = sensor_fixed_meta['is calibrated']
        calibparameter = calibParam()
        if calibrated == 'True':
            return calibparameter
        if calibrated == 'False':
            calibparameter.calibrated = True
            calibparameter.calibrationR = float(sensor_fixed_meta['calibration r'])
            calibparameter.calibrationB = float(sensor_fixed_meta['calibration b'])
            calibparameter.calibrationF = float(sensor_fixed_meta['calibration f'])
            calibparameter.calibrationJ1 = float(sensor_fixed_meta['calibration j1'])
            calibparameter.calibrationJ0 = float(sensor_fixed_meta['calibration j0'])
            calibparameter.calibrationa1 = float(sensor_fixed_meta['calibration alpha1'])
            calibparameter.calibrationa2 = float(sensor_fixed_meta['calibration alpha2'])
            calibparameter.calibrationX = float(sensor_fixed_meta['calibration x'])
            calibparameter.calibrationb1 = float(sensor_fixed_meta['calibration beta1'])
            calibparameter.calibrationb2 = float(sensor_fixed_meta['calibration beta2'])
            calibparameter.shuttertemperature = float(sensor_variable_meta['shutter_temperature_[K]'])
            return calibparameter

    except KeyError as err:
        return calibParam()

def metadata_to_imageBoundaries(center_position, fov, image_shape, convt):
    
    plotNum = 0
    roiBox = []
    
    # compute field per pix
    field_dist_per_pix_1 = fov[1]/image_shape[1]
    field_dist_per_pix_0 = fov[0]/image_shape[0]
    
    
    # A: right lower point in the image
    x_a = center_position[0]-fov[0]/2
    y_a = center_position[1]-fov[1]/2
    
    # B: left upper point in the image
    x_b = center_position[0]+fov[0]/2
    y_b = center_position[1]+fov[1]/2
    
    row_a, col_a = convt.fieldPosition_to_fieldPartition(x_a, y_a)   # A range/column
    row_b, col_b = convt.fieldPosition_to_fieldPartition(x_b, y_b)   # B range/column
    
    if 0 in [row_a, row_b, col_a, col_b]:
        return None, None, None
    
    if row_a == row_b and col_a == col_b:
        plotNum = convt.fieldPartition_to_plotNum(row_a, col_a)
        roiBox = [0,image_shape[0],0,image_shape[1]]
        
    
    if row_a == row_b and col_a != col_b:
        xd_0 = (y_b-convt.np_bounds[row_a-1][col_b-1][2])/fov[1]*image_shape[1]
        #xd_1 = (convt.np_bounds[row_a-1][col_a-1][3]-y_a)/fov[1]*image_shape[1]
        
        if xd_0 > image_shape[1]/2:
            plotNum = convt.fieldPartition_to_plotNum(row_b, col_b)
            roiBox = [0, image_shape[0], 0, xd_0]
        else:
            plotNum = convt.fieldPartition_to_plotNum(row_a, col_a)
            roiBox = [0, image_shape[0],xd_0,image_shape[1]]
            
    if row_a != row_b and col_a == col_b:
        yd_0 = (x_b-convt.np_bounds[row_b-1][col_b-1][0])/fov[0]*image_shape[0]
        
        if yd_0 > image_shape[0]/2:
            plotNum = convt.fieldPartition_to_plotNum(row_b, col_b)
            roiBox = [0, yd_0, 0, image_shape[1]]
        else:
            plotNum = convt.fieldPartition_to_plotNum(row_a, col_a)
            roiBox = [yd_0, image_shape[0],0,image_shape[1]]
            
    if row_a != row_b and col_a != col_b:
        x_m = (y_b-convt.np_bounds[row_a-1][col_b-1][2])/fov[1]*image_shape[1]
        y_m = (x_b-convt.np_bounds[row_b-1][col_b-1][0])/fov[0]*image_shape[0]
        
        if x_m > image_shape[1]/2 and y_m > image_shape[0]/2:
            plotNum = convt.fieldPartition_to_plotNum(row_b, col_b)
            roiBox = [0, y_m, 0, x_m]
        elif x_m > image_shape[1]/2 and y_m < image_shape[0]/2:
            plotNum = convt.fieldPartition_to_plotNum(row_a, col_b)
            roiBox = [y_m, image_shape[0], 0, x_m]
        elif x_m <= image_shape[1]/2 and y_m <= image_shape[0]/2:
            plotNum = convt.fieldPartition_to_plotNum(row_a, col_a)
            roiBox = [y_m, image_shape[0], x_m, image_shape[1]]
        elif x_m < image_shape[1]/2 and y_m > image_shape[0]/2:
            plotNum = convt.fieldPartition_to_plotNum(row_b, col_a)
            roiBox = [0, y_m, x_m, image_shape[1]]
            
    xmin = x_a+(image_shape[0]-roiBox[1])*field_dist_per_pix_0
    xmax = x_a+(image_shape[0]-roiBox[0])*field_dist_per_pix_0
    ymin = y_a+(image_shape[1]-roiBox[3])*field_dist_per_pix_1
    ymax = y_a+(image_shape[1]-roiBox[2])*field_dist_per_pix_1
    field_roiBox = [xmin,xmax,ymin,ymax]
    
    return plotNum, roiBox, field_roiBox

def load_flir_data(file_path):
    
    try:
        im = np.fromfile(file_path, np.dtype('<u2')).reshape([480, 640])
        im = im.astype('float')
        return im
    except Exception as ex:
        fail('Error loading bin file' + str(ex))
        return

def create_png(im, outfile_path):
    
    Gmin = im.min()
    Gmax = im.max()
    At = (im-Gmin)/(Gmax - Gmin)
    
    my_cmap = cm.get_cmap('jet')
    color_array = my_cmap(At)
    
    color_array = (np.rot90(color_array, 3))
    
    plt.imsave(outfile_path, color_array)
    
    img_data = Image.open(outfile_path)
    
    return np.array(img_data)

def create_npy_tc_data(im, out_path):
    
    im_array = (np.rot90(im, 3))
    
    np.save(out_path, im_array)
    
    return im_array

def parse_metadata(metadata):
    
    try:
        gantry_meta = metadata['lemnatec_measurement_metadata']['gantry_system_variable_metadata']
        gantry_x = gantry_meta["position x [m]"]
        gantry_y = gantry_meta["position y [m]"]
        gantry_z = gantry_meta["position z [m]"]
        
        scan_time = gantry_meta["time"]
        
        cam_meta = metadata['lemnatec_measurement_metadata']['sensor_fixed_metadata']
        cam_x = cam_meta["location in camera box x [m]"]
        cam_y = cam_meta["location in camera box y [m]"]
        
        fov_x = cam_meta["field of view x [m]"]
        fov_y = cam_meta["field of view y [m]"]
        
        y_ends = gantry_meta['y end pos [m]']
        
        if "location in camera box z [m]" in cam_meta: # this may not be in older data
            cam_z = cam_meta["location in camera box z [m]"]
        else:
            cam_z = 0

    except KeyError as err:
        fail('Metadata file missing key: ' + err.args[0])
        return None, None, None, None
        
    position = [float(gantry_x), float(gantry_y), float(gantry_z)]
    center_position = [position[0]+float(cam_x), position[1]+float(cam_y), position[2]+float(cam_z)]
    fov = [float(fov_x)*fov_alpha, float(fov_y)*fov_alpha]
    
    return center_position, scan_time, fov, y_ends
    
def lower_keys(in_dict):
    if type(in_dict) is dict:
        out_dict = {}
        for key, item in in_dict.items():
            out_dict[key.lower()] = lower_keys(item)
        return out_dict
    elif type(in_dict) is list:
        return [lower_keys(obj) for obj in in_dict]
    else:
        return in_dict
    
def load_json(meta_path):
    try:
        with open(meta_path, 'r') as fin:
            return json.load(fin)
    except Exception as ex:
        fail('Corrupt metadata file, ' + str(ex))

def find_input_files(in_dir):
    json_suffix = os.path.join(in_dir, '*_metadata.json')
    jsons = glob(json_suffix)
    if len(jsons) == 0:
        fail('Could not find .json file')
        return [], []
        
        
    bin_suffix = os.path.join(in_dir, '*_ir.bin')
    bins = glob(bin_suffix)
    if len(bins) == 0:
        fail('Could not find .bin file')
        return [], []
    
    
    return jsons[0], bins[0]


def fail(reason):
    print (sys.stderr, reason)
    
def extract_roiBox_from_metadata(metadata):
    
    gwu_meta = metadata['gwu_added_metadata']
    xmin = gwu_meta["xmin"]
    xmax = gwu_meta["xmax"]
    ymin = gwu_meta["ymin"]
    ymax = gwu_meta['ymax']
    
    return xmin,xmax,ymin,ymax
    
def stitch_plot_thermal_image(in_dir, out_dir, str_date, convt):
    
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
        
    # get plot boundaries
    dir_name = os.path.basename(in_dir)
    
    plotId = dir_name.split('-')
    plot_row = int(plotId[0])
    plot_col = int(plotId[1])
    
    plot_bounds = convt.np_bounds[plot_row-1][plot_col-1]
       
    # loop image, patch it to target image
    metadata_suffix = '.json'
    metas = [os.path.basename(meta) for meta in glob(join(in_dir, '*' + metadata_suffix))]
    if len(metas) == 0:
        return
    
    # init output image
    metadata = lower_keys(load_json(os.path.join(in_dir, metas[0])))
    center_position, scan_time, fov, y_ends = parse_metadata(metadata)
    if center_position is None:
        return
    
    '''
    # original stitch codes
    field_dist_per_pix = fov[0]/640
    img_wids = int(round((plot_bounds[3]-plot_bounds[2])/field_dist_per_pix))+800
    img_hts = int(round((plot_bounds[1]-plot_bounds[0])/field_dist_per_pix))+800
    stitched_img = np.zeros((img_hts,img_wids,3),np.uint8)
    
    rect_x_min = img_wids
    rect_y_min = img_hts
    rect_x_max = 0
    rect_y_max = 0
    
    start_offset = 500
    for json_file in metas:
        json_path = os.path.join(in_dir, json_file)
        rgb_file = json_path[:-4]+'png'
        
        if not os.path.isfile(rgb_file):
            continue
        
        metadata = load_json(json_path)
        xmin,xmax,ymin,ymax = extract_roiBox_from_metadata(metadata)
        img = cv2.imread(rgb_file)
        height,width = img.shape[:2]
        x_start = int(round((plot_bounds[3]-ymax)/field_dist_per_pix))+start_offset
        y_start = int(round((plot_bounds[1]-xmax)/field_dist_per_pix))+start_offset
        if x_start < 0 or y_start < 0:
            print(1)
            continue
        
        if x_start < rect_x_min :
            rect_x_min = x_start
        if y_start < rect_y_min:
            rect_y_min = y_start
        if x_start+width > rect_x_max:
            rect_x_max = x_start+width
        if y_start+height > rect_y_max:
            rect_y_max = y_start+height
        
        stitched_img[y_start:y_start+height, x_start:x_start+width] = img
    '''
    
    # fov diff, add image weight
    image_shape = (640, 480)
    field_dist_per_pix_1 = fov[1]/image_shape[1]
    field_dist_per_pix_0 = fov[0]/image_shape[0]
    img_wids = int(round((plot_bounds[3]-plot_bounds[2])/field_dist_per_pix_1))+800
    img_hts = int(round((plot_bounds[1]-plot_bounds[0])/field_dist_per_pix_0))+800
    stitched_img = np.zeros((img_hts,img_wids,3),np.uint8)
    
    rect_x_min = img_wids
    rect_y_min = img_hts
    rect_x_max = 0
    rect_y_max = 0
    
    start_offset = 500
    for json_file in metas:
        json_path = os.path.join(in_dir, json_file)
        rgb_file = json_path[:-4]+'png'
        
        if not os.path.isfile(rgb_file):
            continue
        
        metadata = load_json(json_path)
        xmin,xmax,ymin,ymax = extract_roiBox_from_metadata(metadata)
        img = cv2.imread(rgb_file)
        height,width = img.shape[:2]
        x_start = int(round((plot_bounds[3]-ymax)/field_dist_per_pix_0))+start_offset
        y_start = int(round((plot_bounds[1]-xmax)/field_dist_per_pix_1))+start_offset
        if x_start < 0 or y_start < 0:
            print(1)
            continue
        
        if x_start < rect_x_min :
            rect_x_min = x_start
        if y_start < rect_y_min:
            rect_y_min = y_start
        if x_start+width > rect_x_max:
            rect_x_max = x_start+width
        if y_start+height > rect_y_max:
            rect_y_max = y_start+height
        
        #stitched_img[y_start:y_start+height, x_start:x_start+width] = img
        
        cv2.addWeighted(stitched_img[y_start:y_start+height, x_start:x_start+width], 0.5, img, 0.5, 0, stitched_img[y_start:y_start+height, x_start:x_start+width])
        
    save_img = stitched_img[rect_y_min:rect_y_max, rect_x_min:rect_x_max]
    # save output
    out_file_name = '{}_{}.png'.format(str_date, dir_name)
    cv2.imwrite(os.path.join(out_dir, out_file_name), save_img)
    
    return

def full_season_thermalCrop_frame(in_dir, out_dir, plot_dir, png_dir, start_date, end_date, convt):
    
    # initialize data structure
    d0 = datetime.strptime(start_date, '%Y-%m-%d').date()
    d1 = datetime.strptime(end_date, '%Y-%m-%d').date()
    deltaDay = d1 - d0
    
    print(deltaDay.days)
    
    # loop one season directories
    for i in range(deltaDay.days+1):
        str_date = str(d0+timedelta(days=i))
        print(str_date)
        
        raw_path = os.path.join(in_dir, str_date)
        
        out_path = os.path.join(out_dir, str_date)
        
        plot_path = os.path.join(plot_dir, str_date)
        
        png_path = os.path.join(png_dir, str_date)
        
        if not os.path.isdir(raw_path):
            continue
        
        if not os.path.isdir(out_path):
            os.makedirs(out_path)
            
        if not os.path.isdir(plot_path):
            os.makedirs(plot_path)
            
        if not os.path.isdir(png_path):
            os.makedirs(png_path)
        
        #crop_rgb_imageToPlot(raw_path, out_path, plot_dir, convt)
        crop_thermal_imageToPlot(raw_path, out_path, plot_path, png_path, convt)
        #full_day_gen_cc(raw_path, out_path, convt)
    
    return
    


def full_season_thermal_stitch(png_dir, stitch_dir, start_date, end_date, convt):
    
    # initialize data structure
    d0 = datetime.strptime(start_date, '%Y-%m-%d').date()
    d1 = datetime.strptime(end_date, '%Y-%m-%d').date()
    deltaDay = d1 - d0
    
    print(deltaDay.days)
    
    # loop one season directories
    for i in range(deltaDay.days+1):
        str_date = str(d0+timedelta(days=i))
        print(str_date)
        
        png_path = os.path.join(png_dir, str_date)
        
        stitch_path = os.path.join(stitch_dir, str_date)
        
        if not os.path.isdir(png_path):
            continue
        
        if not os.path.isdir(stitch_path):
            os.makedirs(stitch_path)
            
        list_dirs = os.listdir(png_path)
    
        for d in list_dirs:
            in_path = os.path.join(png_path, d)
            out_path = os.path.join(stitch_path, d)
            
            if not os.path.isdir(in_path):
                continue
            
            if not os.path.isdir(out_path):
                os.mkdir(out_path)
        
            stitch_plot_thermal_image(in_path, out_path, str_date, convt)
    
    return
    
def main():

    '''
    in_dir = os.path.join('/media/zli/Seagate Backup Plus Drive/OPEN/ua-mac/raw_data/flirlrCamera', str_date)
    out_dir = os.path.join('/media/zli/Seagate Backup Plus Drive/OPEN/ua-mac/Level_1/thermalData', str_date)
    plot_dir = os.path.join('/media/zli/Seagate Backup Plus Drive/OPEN/ua-mac/Level_2/thermalCropToPlot/', str_date)
    png_dir = os.path.join('/media/zli/Seagate Backup Plus Drive/OPEN/ua-mac/Level_2/thermalCropToPlot_png/', str_date)
    crop_thermal_imageToPlot(in_dir, out_dir, plot_dir, png_dir, convt)
    '''
    print("start...")
    
    start_date = '2019-05-23'  # S9 start date
    end_date = '2019-06-04'   # S9 end date
    
    convt = terra_common.CoordinateConverter()
    qFlag = convt.bety_query('2019-06-01') # All plot boundaries in one season should be the same, currently 2019-06-18 works best
    
    if not qFlag:
        return
    
    in_dir = os.path.join('/media/zli/Seagate Backup Plus Drive/OPEN/ua-mac/raw_data/flirlrCamera')
    out_dir = os.path.join('/media/zli/Seagate Backup Plus Drive/OPEN/ua-mac/Level_1/thermalData')
    plot_dir = os.path.join('/media/zli/Seagate Backup Plus Drive/OPEN/ua-mac/Level_2/thermalCropToPlot/')
    png_dir = os.path.join('/media/zli/Seagate Backup Plus Drive/OPEN/ua-mac/Level_2/thermalCropToPlot_png/')
    stitch_dir = os.path.join('/media/zli/Seagate Backup Plus Drive/OPEN/ua-mac/Level_2/StitchedPlotThermal')
    
    full_season_thermalCrop_frame(in_dir, out_dir, plot_dir, png_dir, start_date, end_date, convt)
    
    full_season_thermal_stitch(png_dir, stitch_dir, start_date, end_date, convt)
    
    return


def test(convt, str_date):
    
    in_dir = os.path.join('/media/zli/Seagate Backup Plus Drive/OPEN/ua-mac/Level_2/thermalCropToPlot_png/', str_date)
    out_dir = os.path.join('/media/zli/Seagate Backup Plus Drive/OPEN/ua-mac/Level_2/StitchedPlotThermal', str_date)
    
    #convt = terra_common.CoordinateConverter()
    #qFlag = convt.bety_query(str_date, False)
    
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)


    list_dirs = os.listdir(in_dir)
    
    for d in list_dirs:
        in_path = os.path.join(in_dir, d)
        out_path = os.path.join(out_dir, d)
        
        if not os.path.isdir(in_path):
            continue
        
        if not os.path.isdir(out_path):
            os.mkdir(out_path)
    
        stitch_plot_thermal_image(in_path, out_path, str_date, convt)
    
    return


if __name__ == '__main__':
    
    main()
    #test(convt, str_date)
    
    
    
    
