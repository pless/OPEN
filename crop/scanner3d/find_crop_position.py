from datetime import datetime, timedelta
from terra_common import CoordinateConverter as CC
import os, logging, traceback, time, json, warnings
from . import utils
import skimage.io as sio
import pickle
from .plyfile import PlyData
import multiprocessing
import numpy as np
import json
import gzip


def find_crop_position(raw_data_folder, ply_data_folder, output_folder, cc, log_lv=logging.INFO, compress=6):
    raw_data_folder = os.path.join(raw_data_folder, '')
    ply_data_folder = os.path.join(ply_data_folder, '')
    cpname = multiprocessing.current_process().name
    logger = logging.getLogger('ppln_' + os.path.basename(os.path.dirname(raw_data_folder)) + '_' + cpname)
    logger.setLevel(log_lv)
    formatter = logging.Formatter('%(asctime)s - %(name)s %(levelname)s:\t%(message)s')
    ch = logging.StreamHandler()
    ch.setLevel(log_lv)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.info('Start processing')
    # get file name
    for filename in os.listdir(raw_data_folder):
        if 'east' + '_0_g.png' in filename:
            east_gIm_name = filename
        if 'west' + '_0_g.png' in filename:
            west_gIm_name = filename
        if 'east' + '_0_p.png' in filename:
            east_pIm_name = filename
        if 'west' + '_0_p.png' in filename:
            west_pIm_name = filename
        if 'metadata.json' in filename:
            json_name = filename
    # read png
    try:
        east_gIm = sio.imread(os.path.join(raw_data_folder, east_gIm_name))
        east_pIm = sio.imread(os.path.join(raw_data_folder, east_pIm_name))
        west_gIm = sio.imread(os.path.join(raw_data_folder, west_gIm_name))
        west_pIm = sio.imread(os.path.join(raw_data_folder, west_pIm_name))
    except:
        logger.error('Image reading error! Skip.')
        return -1

    # check existence of ply file
    if not os.path.isdir(ply_data_folder):
        logger.error('ply folder does not exst. path{}'.format(ply_data_folder))
        return -1
    east_ply_data_path = None
    west_ply_data_path = None
    for filename in os.listdir(ply_data_folder):
        if filename.endswith('.ply'):
            if 'east' in filename:
                east_ply_data_path = os.path.expanduser(os.path.join(ply_data_folder, filename))
            if 'west' in filename:
                west_ply_data_path = os.path.expanduser(os.path.join(ply_data_folder, filename))
    if east_ply_data_path is None or west_ply_data_path is None:
        logger.error('ply file does not exist. path:{}'.format(ply_data_folder))
        return -1
    pass
    # read ply
    try:
        east_ply_data = PlyData.read(east_ply_data_path)
        west_ply_data = PlyData.read(west_ply_data_path)
    except:
        logger.error('ply file reading error! Skip. file_path:{}'.format(ply_data_folder))
        return -1
    # read json
    try:
        with open(os.path.join(raw_data_folder, json_name), 'r') as json_f:
            json_data = json.load(json_f)
        east_json_info = utils.get_json_info(json_data, sensor='east')
        west_json_info = utils.get_json_info(json_data, sensor='west')
    except:
        logger.error('Load json file unsuccessful.')
        return -4
    # offset
    logger.info('offsetting point cloud')
    east_ply_data = utils.ply_offset(east_ply_data, east_json_info)
    west_ply_data = utils.ply_offset(west_ply_data, west_json_info)
    # ply to xyz
    logger.info('ply to xyz')
    east_ply_xyz_map = utils.ply2xyz(east_ply_data, east_pIm, east_gIm, nan_init=True)
    west_ply_xyz_map = utils.ply2xyz(west_ply_data, west_pIm, west_gIm, nan_init=True)
    logger.info('find cropping position')
    # crop position
    try:
        east_crop_position_dict = utils.depth_crop_position(east_ply_xyz_map, cc)
        west_crop_position_dict = utils.depth_crop_position(west_ply_xyz_map, cc)
    except ValueError as ex:
        logger.exception(ex)
        return
    # save to file with /plot/[folder_name]_[east/west].png
    logger.info('Saving files.')
    folder_name = os.path.basename(os.path.dirname(raw_data_folder))
    folder_name_dt = datetime.strptime(folder_name[:20], '%Y-%m-%d__%H-%M-%S')
    day_split_fix_offset = timedelta(hours=12)
    folder_name_dt = folder_name_dt - day_split_fix_offset
    file_name_prefix = folder_name_dt.strftime('%Y-%m-%d__%H-%M-%S') + folder_name[20:]
    for crop_position_dict, gIm, pIm, ply_xyz_map, sensor_str in zip((east_crop_position_dict, west_crop_position_dict),(east_gIm, west_gIm), (east_pIm, west_pIm), (east_ply_xyz_map, west_ply_xyz_map), ('east', 'west')):
        for (plot_row, plot_col), (row_range, col_range) in crop_position_dict.items():
            plot_num = cc.fieldPartition_to_plotNum(plot_row, plot_col)
            plot_output_path = os.path.join(output_folder, '{}_{}_{}'.format(int(plot_row), int(plot_col), int(plot_num)))
            if not os.path.isdir(plot_output_path):
                try:
                    os.makedirs(plot_output_path)
                except FileExistsError:
                    pass
            plot_p_im = pIm[col_range[0]: col_range[1], row_range[0]: row_range[1]]
            plot_g_im = gIm[col_range[0]: col_range[1], row_range[0]: row_range[1]]
            plot_xyz_map = ply_xyz_map[col_range[0]: col_range[1], row_range[0]: row_range[1], :]
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    sio.imsave(os.path.join(plot_output_path, file_name_prefix + '_' + sensor_str + '_p' + '.png'), plot_p_im)
                    sio.imsave(os.path.join(plot_output_path, file_name_prefix + '_' + sensor_str + '_g' + '.png'), plot_g_im)
                if compress:
                    with gzip.open(os.path.join(plot_output_path, file_name_prefix + '_' + sensor_str + '_xyz' + '.npy.gz'), 'wb', compress) as npy_fp:
                        np.save(npy_fp, plot_xyz_map)
                else:
                    np.save(os.path.join(plot_output_path, file_name_prefix + '_' + sensor_str + '_xyz' + '.npy'), plot_xyz_map)
            except Exception as e:
                logger.error('File write error at {}\n {}'.format(os.path.join(plot_output_path, file_name_prefix + '_' + sensor_str), e))
                # TODO error handle (remove all the generated files)
            if plot_xyz_map[plot_xyz_map.nonzero()].size == 0:
                logger.info('map does not have nonzero element')
                continue
            cropping_metadata = dict()
            plot_x_map = plot_xyz_map[:,:, 0]
            plot_y_map = plot_xyz_map[:,:, 1]
            plot_z_map = plot_xyz_map[:,:, 2]
            plot_x_min = np.nanmin(plot_x_map[plot_x_map.nonzero()])
            plot_x_max = np.nanmax(plot_x_map[plot_x_map.nonzero()])
            plot_y_min = np.nanmin(plot_y_map[plot_y_map.nonzero()])
            plot_y_max = np.nanmax(plot_y_map[plot_y_map.nonzero()])
            plot_z_min = np.nanmin(plot_z_map[plot_z_map.nonzero()])
            plot_z_max = np.nanmax(plot_z_map[plot_z_map.nonzero()])
            cropping_metadata['origin_timestamp'] = folder_name
            cropping_metadata['crop_bbox_pixel'] = {'img_row':{'min':int(col_range[0]), 'max':int(col_range[1])}, 'img_col':{'min': int(row_range[0]), 'max': int(row_range[1])}}
            cropping_metadata['field_bbox_mm'] = {'x':{'min':plot_x_min, 'max':plot_x_max}, 'y':{'min': plot_y_min, 'max': plot_y_max}, 'z': {'min': plot_z_min, 'max': plot_z_max}}
            cropping_json_dict = {'cropping_variable_metadata': cropping_metadata, 'cropping_fix_metadata':{}}
            with open(os.path.join(plot_output_path, file_name_prefix + '_' + sensor_str + '_metadata' + '.json'), 'w') as f:
                json.dump(cropping_json_dict, f, indent=4)
    logger.info('Finished.')

    # save to file with pkl
    # folder_name = os.path.basename(os.path.dirname(raw_data_folder))
    # east_pkl_path = os.path.join('./result/', folder_name + '_' + 'east' + '.pkl')
    # west_pkl_path = os.path.join('./result/', folder_name + '_' + 'west' + '.pkl')
    # logger.info('writing to {} and {}'.format(east_pkl_path, west_pkl_path))
    # with open(east_pkl_path, 'wb') as f:
    #     pickle.dump(east_crop_position_dict, f)
    # with open(west_pkl_path, 'wb') as f:
    #     pickle.dump(west_crop_position_dict, f)

if __name__ == '__main__':
    os.environ['BETYDB_KEY'] = '9999999999999999999999999999999999999999'
    cc = CC()
    cc.bety_query('2017-05-05', useSubplot=False)
    #raw_data_folder = '/pless_nfs/home/terraref/scanner3DTop/raw_data/2018-07-25/2018-07-25__02-25-22-708/'
    #ply_data_folder = '/pless_nfs/home/terraref/scanner3DTop/Level_1/2018-07-25/2018-07-25__02-25-22-708/'
    # raw_data_folder = '/pless_nfs/home/zeyu/terraref/scanner3DTop/raw_data/2017-05-05/2017-05-05__23-58-56-845/'
    # ply_data_folder = '/pless_nfs/home/zeyu/terraref/scanner3DTop/Level_1/2017-05-05/2017-05-05__23-58-56-845/'
    raw_data_folder = '/pless_nfs/home/zeyu/terraref/scanner3DTop/raw_data/2017-06-10/2017-06-10__05-43-36-501/'
    ply_data_folder = '/pless_nfs/home/zeyu/terraref/scanner3DTop/Level_1/2017-06-10/2017-06-10__05-43-36-501/'
    find_crop_position(raw_data_folder, ply_data_folder, './test_output', cc)





