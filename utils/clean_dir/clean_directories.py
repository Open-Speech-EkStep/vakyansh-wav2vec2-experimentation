import pandas as pd
import numpy as np
import argparse
import string
import os, shutil

def remove_artefacts(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    print('Succesfully deleted contents of: ', folder)

def remove_file(file):
    try:
        os.unlink(file)
        print('File removed: ', file)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file, e))
    
def get_permission_and_dir_to_delete(folder):
    print('*'*50)
    dir_content = os.listdir(folder)
    dir_content_minus_hidden = [item for item in dir_content if item[0] != '.']
    print('Contents of ', folder, ' are the following:\n', dir_content_minus_hidden)
    folder_or_entire_dir = str(input('Enter y if you want to delete everything, else enter n if you want to delete selective items: '))

    if (folder_or_entire_dir == 'y' or folder_or_entire_dir == 'Y'):
        print('Removing all contents of:', folder)
        print('Deleting...')
        remove_artefacts(folder)

    elif (folder_or_entire_dir == 'n' or folder_or_entire_dir == 'N'):
        for file_or_dir in dir_content_minus_hidden:
            print('*'*50)
            confirmation = str(input('Do you want to delete '+file_or_dir+'? Enter y/n: '))
            if (confirmation == 'y' or confirmation == 'Y'):
                file_or_dir = os.path.join(folder, file_or_dir)
                if os.path.isfile(file_or_dir):
                    remove_file(file_or_dir)
                else:
                    # remove_artefacts(file_or_dir)
                    try:
                        shutil.rmtree(file_or_dir)
                        print('Folder removed: ', file_or_dir)
                    except Exception as e:
                        print('Failed to delete %s. Reason: %s' % (file_or_dir, e))
                    
            elif (confirmation == 'n' or confirmation == 'N'):
                print('Not deleting: ', file_or_dir)
            print('*'*50)
    return
 
def get_non_hidden_dir_items(folder):
    dir_list = os.listdir(folder)
    dir_content_minus_hidden = [item for item in dir_list if item[0] != '.']
    dir_content_minus_hidden = [os.path.join(folder, item) for item in dir_content_minus_hidden]
    return dir_content_minus_hidden

if __name__ == "__main__":
    '''
    Allows you to delete contents of results, checkpoints, data, and logs directories.
    '''
    results_dir_path = '../../results/'
    checkpoints = '../../checkpoints/'
    data = '../../data/'
    logs = '../../logs/'

    get_permission_and_dir_to_delete(results_dir_path)
    
    checkpoint_dir_content = get_non_hidden_dir_items(checkpoints)
    for item in checkpoint_dir_content:
        get_permission_and_dir_to_delete(item)
    
    data_dir_content = get_non_hidden_dir_items(data)
    for item in data_dir_content:
        get_permission_and_dir_to_delete(item)

    logs_dir_content = get_non_hidden_dir_items(logs)
    for item in logs_dir_content:
        get_permission_and_dir_to_delete(item)



