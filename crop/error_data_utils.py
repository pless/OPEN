'''
Created on Feb 18, 2020

@author: zli
'''

import sys, os, json
import cv2

def load_json(meta_path):
    try:
        with open(meta_path, 'r') as fin:
            return json.load(fin)
    except Exception as ex:
        fail('Corrupt metadata file, ' + str(ex))
        return
    
def fail(reason):
    print >> sys.stderr, reason
    
def extract_roiBox_from_metadata(metadata):
    
    gwu_meta = metadata['gwu_added_metadata']
    xmin = gwu_meta["xmin"]
    xmax = gwu_meta["xmax"]
    ymin = gwu_meta["ymin"]
    ymax = gwu_meta['ymax']
    
    return float(xmin),float(xmax),float(ymin),float(ymax)

def full_dir_scan(in_dir):
    
    plot_dirs = os.listdir(in_dir)
    
    error_list_path = os.path.join('/pless_nfs/home/zongyangli/pyWork/py3', 'error_lists_v2.txt')
    csv_handle = open(error_list_path, 'w')
    
    for d in plot_dirs:
        in_path = os.path.join(in_dir, d)
        if not os.path.isdir(in_path):
            continue
        
        plot_full_path = in_path
        '''
        plot_dirs = os.listdir(in_path)
        for plot_dir in plot_dirs:
            plot_full_path = os.path.join(in_path, plot_dir, 'stereoTop')
            if not os.path.isdir(plot_full_path):
                continue
           ''' 
        print(plot_full_path)
    
        file_lists = os.walk(plot_full_path)
        for root, dirs, files in file_lists:
            for f in files:
                if f.endswith('json'):
                    json_path = os.path.join(plot_full_path, f)
                    rgb_file = json_path[:-4]+'png'
                    if not os.path.isfile(rgb_file):
                        continue
        
                    metadata = load_json(json_path)
                    xmin,xmax,ymin,ymax = extract_roiBox_from_metadata(metadata)
                    print(xmax - xmin)
                    if xmax - xmin > 5.0:
                        # save error file path, error
                        print_line = json_path+'\n'
                        csv_handle.write(print_line)
                            #print_line = '{}\n'.format(xmax - xmin)
                            
                            # delete error png and json
                            #os.remove(json_path)
                            #os.remove(rgb_file)
                
    csv_handle.close()
    
    
    return

def bgr_2_rgb(in_dir):
    
    plot_dirs = os.listdir(in_dir)
    for d in plot_dirs:
        in_path = os.path.join(in_dir, d)
        if not os.path.isdir(in_path):
            continue
        
        plot_full_path = in_path
        file_lists = os.walk(plot_full_path)
        for root, dirs, files in file_lists:
            for f in files:
                if f.endswith('png'):
                    file_path = os.path.join(plot_full_path, f)
                    bgr_img = cv2.imread(file_path, cv2.IMREAD_COLOR)
                    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(file_path, rgb_img)
                
    return


def main():
    
    bgr_2_rgb('/data/shared/stereo_s4_result_v2/')
    #bgr_2_rgb('/media/zli/Seagate Backup Plus Drive/OPEN/ua-mac/Level_2/stereoTop')



if __name__ == '__main__':
    
    main()
