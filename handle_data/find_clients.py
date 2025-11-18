import numpy as np
import pandas as pd
import os

# Paths
folder_path = '/mnt/d1bdf387-8fd2-4f57-8c8a-eba9ef0baff6/Karan/FeTS/Training_Data/MICCAI_FeTS2022_TrainingData'
partitioning = pd.read_csv(folder_path + '/partitioning_1.csv')

# Target clients list (all clients, divided into train, val, and test sets as required)
target_clients = [4, 5, 6, 13, 16, 20, 21, 3, 7, 11, 15]  # 3, 7, 11, 15 are low-data clients

# Filter the partitioning DataFrame to only include target clients
partitioning = partitioning[partitioning['Partition_ID'].isin(target_clients)]

# Print counts for sanity check
print(partitioning['Partition_ID'].value_counts())

# Create a new column 'type' to hold the partition type (train/val/test)
partitioning['type'] = ''

# Divide clients into train, val, and test sets
train_clients = target_clients[:4]  # First 4 clients
val_clients = target_clients[4:7]   # Next 3 clients
test_clients = target_clients[7:]   # Last 4 clients

# Assign the partition types
print(partitioning)
partitioning.to_csv('partitioning_original.txt', sep='\t', index=False)
partitioning.loc[partitioning['Partition_ID'].isin(train_clients), 'type'] = 'train'
partitioning.loc[partitioning['Partition_ID'].isin(val_clients), 'type'] = 'val'
partitioning.loc[partitioning['Partition_ID'].isin(test_clients), 'type'] = 'test'

# Client mapping (for convenience, assigning new IDs sequentially)
client_mapping = {client_id: new_id for new_id, client_id in enumerate(target_clients, start=1)}
print(client_mapping)
partitioning['Partition_ID'] = partitioning['Partition_ID'].map(client_mapping)

print(partitioning)
#save partitioning to a txt file
partitioning.to_csv('partitioning.txt', sep='\t', index=False)

# Create the new folder structure
new_folder = '../fets_clients'
os.makedirs(new_folder, exist_ok=True)

for client in client_mapping.values():
    os.makedirs(f'{new_folder}/client{client}/train', exist_ok=True)
    os.makedirs(f'{new_folder}/client{client}/val', exist_ok=True)
    os.makedirs(f'{new_folder}/client{client}/test', exist_ok=True)

# Create symbolic links for each subject in their respective folders
for idx, row in partitioning.iterrows():
    subject_id = row['Subject_ID']
    partition_id = row['Partition_ID']
    subject_folder = os.path.join(folder_path, subject_id)
    new_folder_path = f'{new_folder}/client{partition_id}/{row["type"]}/{subject_id}'
    os.symlink(subject_folder, new_folder_path)
