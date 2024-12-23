from tqdm import tqdm
from glob import glob
import os.path as ospth
import re

def get_file_name(file_path: str):
    return ospth.splitext(ospth.basename(file_path))[0]

def get_audios_with_folder_name (names, all_files):
    list_folders = set(names) 
    res = []
    for t in glob(all_files):
        t = t.replace('\\', '/') 
        pattern = t.split('/')[4] 
        if pattern in list_folders:
            res.append(t)
    return res

def find_files_folders (pattern, all_files): 
    return [t.replace('\\', '/') for t in glob(all_files) if pattern in t.replace('\\', '/')]

def mapper_from_flac_to_df (audio_name, base_path):
    flac_file = (base_path +  re.sub(r'.*F:/ISSAI_KSC2_unpacked/ISSAI_KSC2', '', audio_name)).replace('.flac', '.csv')
    return flac_file

