import numpy as np
import pandas as pd
import os

#split the training data into train and val sets, apart from restructuring the data organization into a format similar to fets.

wmh_folder = '/mnt/d1bdf387-8fd2-4f57-8c8a-eba9ef0baff6/Karan/WMH/dataverse_files'

client1_paths = ['/mnt/d1bdf387-8fd2-4f57-8c8a-eba9ef0baff6/Karan/WMH/dataverse_files/training/Amsterdam/GE3T', '/mnt/d1bdf387-8fd2-4f57-8c8a-eba9ef0baff6/Karan/WMH/dataverse_files/test/Amsterdam/GE3T']
client2_paths = ['/mnt/d1bdf387-8fd2-4f57-8c8a-eba9ef0baff6/Karan/WMH/dataverse_files/training/Singapore', '/mnt/d1bdf387-8fd2-4f57-8c8a-eba9ef0baff6/Karan/WMH/dataverse_files/test/Singapore']
client3_paths = ['/mnt/d1bdf387-8fd2-4f57-8c8a-eba9ef0baff6/Karan/WMH/dataverse_files/training/Utrecht', '/mnt/d1bdf387-8fd2-4f57-8c8a-eba9ef0baff6/Karan/WMH/dataverse_files/test/Utrecht']
client4_paths = ['/mnt/d1bdf387-8fd2-4f57-8c8a-eba9ef0baff6/Karan/WMH/dataverse_files/test/Amsterdam/GE1T5']
client5_paths = ['/mnt/d1bdf387-8fd2-4f57-8c8a-eba9ef0baff6/Karan/WMH/dataverse_files/test/Amsterdam/Philips_VU.PETMR_01.']

new_folder_path = '/mnt/d1bdf387-8fd2-4f57-8c8a-eba9ef0baff6/Karan/WMH/wmh_clients'

#create folders for each client
os.makedirs(new_folder_path, exist_ok=True)

client_paths = [client1_paths, client2_paths, client3_paths, client4_paths, client5_paths]

for i in range(len(client_paths)):
    os.makedirs(new_folder_path + f'/client{i+1}', exist_ok=True)
    os.makedirs(new_folder_path + f'/client{i+1}/train', exist_ok=True)
    os.makedirs(new_folder_path + f'/client{i+1}/val', exist_ok=True)
    os.makedirs(new_folder_path + f'/client{i+1}/test', exist_ok=True)

np.random.seed(42)

for i in range(len(client_paths)):
    
    subject_paths = []

    for client_path in client_paths[i]:
        #get the full path of the subjects
        subjects = os.listdir(client_path)
        for subject in subjects:
            subject_paths.append((client_path, subject))
    
    #split the subjects into train, val and test (0.6, 0.1, 0.3)
    np.random.shuffle(subject_paths)
    train_subjects = subject_paths[:int(0.6*len(subject_paths))]
    val_subjects = subject_paths[int(0.6*len(subject_paths)):int(0.7*len(subject_paths))]
    test_subjects = subject_paths[int(0.7*len(subject_paths)):]

    #copy the train, val and test data
    for subject in train_subjects:

        os.makedirs(f'{new_folder_path}/client{i+1}/train/{subject[1]}', exist_ok=True)
        os.system(f'cp {subject[0]}/{subject[1]}/pre/FLAIR.nii.gz {new_folder_path}/client{i+1}/train/{subject[1]}/flair.nii.gz')
        os.system(f'cp {subject[0]}/{subject[1]}/wmh.nii.gz {new_folder_path}/client{i+1}/train/{subject[1]}/seg.nii.gz')

    for subject in val_subjects:
            
        os.makedirs(f'{new_folder_path}/client{i+1}/val/{subject[1]}', exist_ok=True)
        os.system(f'cp {subject[0]}/{subject[1]}/pre/FLAIR.nii.gz {new_folder_path}/client{i+1}/val/{subject[1]}/flair.nii.gz')
        os.system(f'cp {subject[0]}/{subject[1]}/wmh.nii.gz {new_folder_path}/client{i+1}/val/{subject[1]}/seg.nii.gz')

    for subject in test_subjects:
                
        os.makedirs(f'{new_folder_path}/client{i+1}/test/{subject[1]}', exist_ok=True)
        os.system(f'cp {subject[0]}/{subject[1]}/pre/FLAIR.nii.gz {new_folder_path}/client{i+1}/test/{subject[1]}/flair.nii.gz')
        os.system(f'cp {subject[0]}/{subject[1]}/wmh.nii.gz {new_folder_path}/client{i+1}/test/{subject[1]}/seg.nii.gz')

    """
    for subject in train_subjects:

        os.makedirs(f'{new_folder_path}/client{i+1}/train/{subject}', exist_ok=True)
        os.system(f'cp {client_path}/{subject}/pre/FLAIR.nii.gz {new_folder_path}/client{i+1}/train/{subject}/flair.nii.gz')
        os.system(f'cp {client_path}/{subject}/wmh.nii.gz {new_folder_path}/client{i+1}/train/{subject}/seg.nii.gz')
    for subject in val_subjects:

        os.makedirs(f'{new_folder_path}/client{i+1}/val/{subject}', exist_ok=True)
        os.system(f'cp {client_path}/{subject}/pre/FLAIR.nii.gz {new_folder_path}/client{i+1}/val/{subject}/flair.nii.gz')
        os.system(f'cp {client_path}/{subject}/wmh.nii.gz {new_folder_path}/client{i+1}/val/{subject}/seg.nii.gz')

#copy the test data
for i in range(len(test_clients_paths)):

    client_path = test_clients_paths[i]
    subjects = os.listdir(client_path)

    for subject in subjects:
        os.makedirs(f'{new_folder_path}/client{i+1}/test/{subject}', exist_ok=True)
        os.system(f'cp {client_path}/{subject}/pre/FLAIR.nii.gz {new_folder_path}/client{i+1}/test/{subject}/flair.nii.gz')
        os.system(f'cp {client_path}/{subject}/wmh.nii.gz {new_folder_path}/client{i+1}/test/{subject}/seg.nii.gz')
"""


