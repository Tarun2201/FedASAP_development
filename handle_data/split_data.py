import numpy as np
import pandas as pd
import os

folder_path = '/mnt/d1bdf387-8fd2-4f57-8c8a-eba9ef0baff6/Karan/FeTS/Training_Data/MICCAI_FeTS2022_TrainingData'

partioning = pd.read_csv(folder_path + '/partitioning_1.csv')

target_clients = [4, 6, 3, 5] #3,7,11,16 are the low data clients

# Filter out rows where Partition_ID does not correspond to the target clients
partioning = partioning[partioning['Partition_ID'].isin(target_clients)]

#print the counts of the target clients
print(partioning['Partition_ID'].value_counts())

# Create a mapping between original client IDs and new IDs
client_mapping = {client_id: new_id for new_id, client_id in enumerate(target_clients, start=1)}

# Replace original client IDs with new IDs in partioning
partioning['Partition_ID'] = partioning['Partition_ID'].map(client_mapping)

train, val = 0.8, 0.2

# Create a new column 'type' in partioning
partioning['type'] = ''

np.random.seed(42)

NUM_TEST = 10

# Iterate over each client and set the 'type' column accordingly
for client_id, group in partioning.groupby('Partition_ID'):
    num_subjects = len(group)
    test_idx = NUM_TEST
    train_idx = test_idx + int(train * (num_subjects - NUM_TEST))
    indices = np.arange(num_subjects)
    np.random.shuffle(indices)

    #obtain the train, val, and test indices. But make sure that none of the indices are empty.
    test_indices = indices[:test_idx]
    train_indices = indices[test_idx:train_idx]
    val_indices = indices[train_idx:]

    if len(val_indices) == 0:
        #if only val indices are empty, then add one subject to val from test
        val_indices = [train_indices[-1]]
        train_indices = train_indices[:-1]
    
    print(f'Train: {train_indices}, Val: {val_indices}, Test: {test_indices}')
    partioning.loc[group.index, 'type'] = 'train'
    partioning.loc[group.index[val_indices], 'type'] = 'val'
    partioning.loc[group.index[test_indices], 'type'] = 'test'


print(partioning)
# Create a new folder called fets_clients
new_folder = '../fets_clients'
os.makedirs(new_folder, exist_ok=True)

# Create folders for each client
for client in client_mapping.values():
    os.makedirs(f'{new_folder}/client{client}', exist_ok=True)
    os.makedirs(f'{new_folder}/client{client}/train', exist_ok=True)
    os.makedirs(f'{new_folder}/client{client}/val', exist_ok=True)
    os.makedirs(f'{new_folder}/client{client}/test', exist_ok=True)

# Create symbolic links for each subject belonging to the top 5 clients
# Make sure that the subjects are placed in the correct train, val, and test folders
for idx, row in partioning.iterrows():
    subject_id = row['Subject_ID']
    partition_id = row['Partition_ID']
    subject_folder = os.path.join(folder_path, subject_id)
    new_folder_path = f'{new_folder}/client{partition_id}/{row["type"]}/{subject_id}'
    os.symlink(subject_folder, new_folder_path)
