from helper_datadict import *
from skimage.measure import label, regionprops, regionprops_table
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.mixture import GaussianMixture

def safe_regionprops_table(labeled_image, properties):
    # A safe version of regionprops_table that handles invalid regions
    valid_props = []
    regions = regionprops(labeled_image)
    
    for region in regions:
        try:
            prop_values = {}
            for prop in properties:
                prop_values[prop] = getattr(region, prop)
            
            #if area < 7, skip
            if prop_values['minor_axis_length'] <= 0.001:
                continue
            valid_props.append(prop_values)
        except ValueError:
            continue
    
    return pd.DataFrame(valid_props)

def process_client_data(dataloader):
    regions = []
    for data in dataloader:
        flair = data['input'].squeeze().numpy()
        label1 = data['gt'].squeeze().numpy().astype(np.uint8)
        labeled_img = label(label1)
        
        # Using safe_regionprops_table to extract properties
        props_df = safe_regionprops_table(labeled_img, properties=['area', 'major_axis_length', 'minor_axis_length'])
        
        for _, row in props_df.iterrows():
            info = {
                'semi_major_axis_length': row['major_axis_length']/2,
                'semi_minor_axis_length': row['minor_axis_length']/2,
                'area': row['area']
            }
            regions.append(info)
    
    return pd.DataFrame(regions)

def get_client_cluster_info(client_regions, num_clusters, f=0.4):
    num_clusters = min(num_clusters, client_regions.shape[0])
    gmm = GaussianMixture(n_components=num_clusters)
    gmm.fit(client_regions[['semi_major_axis_length', 'semi_minor_axis_length']])

    #add a third feature to the GMM which has the same mean and variance and covariance with feature 1 as feature 2 but it is independent of feature 2. Now, each component should have a 3D Gaussian distribution.

    #means
    means = gmm.means_
    new_means = np.zeros((num_clusters, 3))
    new_means[:, :2] = means
    new_means[:, 2] = means[:, 1]

    #variances
    covariances = gmm.covariances_
    new_covariances = np.zeros((num_clusters, 3, 3))
    new_covariances[:, :2, :2] = covariances

    
    

    #add a column and row to each entry in variances (which contains 3 2D arrays). fill it with 0s.
    for i in range(len(covariances)):
        new_covariances[i][-1][-1] = new_covariances[i][-2][-2]
        new_covariances[i][-1][0] = new_covariances[i][-2][0]
        new_covariances[i][0][-1] = new_covariances[i][0][-2]

    def is_positive_definite(A):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False

    def ensure_positive_definite(A, epsilon=1e-8):
        while not is_positive_definite(A):
            A += np.eye(A.shape[0]) * epsilon
            epsilon *= 2  # Increase epsilon if necessary
        return A
    
    #for i in range(len(new_covariances)):
        #new_covariances[i] = ensure_positive_definite(new_covariances[i])


    clusters = gmm.predict(client_regions[['semi_major_axis_length', 'semi_minor_axis_length']])

    client_buckets = np.bincount(clusters)
    total = np.sum(client_buckets)
    num_clusters = len(client_buckets)

    x = 0
    if min(client_buckets) < f*total/num_clusters:
        x = int(np.ceil((f*total - num_clusters*min(client_buckets))/(num_clusters*(1-f))))
        client_buckets += x
    
    cluster_probs = client_buckets/(total + x*num_clusters)

    ratios = client_regions['semi_minor_axis_length']/client_regions['semi_major_axis_length']
    min_ratio = min(ratios)
    max_ratio = max(ratios)
    
    info_obj = {
        'means': new_means,
        'covariances': new_covariances,
        'cluster_probs': cluster_probs,
        'min_ratio': min_ratio,
        'max_ratio': max_ratio
    }

    return info_obj
        

    

if __name__ == "__main__":
    # Load the data
    datadict_train, _, _ = helper_federated_setup()

    np.random.seed(42)
    
    client1_data = datadict_train['client1']
    client1_dataloader = DataLoader(client1_data, batch_size=1, shuffle=False)
    client2_data = datadict_train['client2']
    client2_dataloader = DataLoader(client2_data, batch_size=1, shuffle=False)
    client3_data = datadict_train['client3']
    client3_dataloader = DataLoader(client3_data, batch_size=1, shuffle=False)
    client4_data = datadict_train['client4']
    client4_dataloader = DataLoader(client4_data, batch_size=1, shuffle=False)

    # Process each client's data separately
    print("Processing client 1 data...")
    client1_regions = process_client_data(client1_dataloader)
    print("Processing client 2 data...")
    client2_regions = process_client_data(client2_dataloader)
    print("Processing client 3 data...")
    client3_regions = process_client_data(client3_dataloader)
    print("Processing client 4 data...")
    client4_regions = process_client_data(client4_dataloader)
    
    n_clusters = 4
    info_obj_client1 = get_client_cluster_info(client1_regions, n_clusters)
    info_obj_client2 = get_client_cluster_info(client2_regions, n_clusters)
    info_obj_client3 = get_client_cluster_info(client3_regions, n_clusters)
    info_obj_client4 = get_client_cluster_info(client4_regions, n_clusters)

    print("Client 1 info:")
    print(info_obj_client1)
    print("Client 2 info:")
    print(info_obj_client2)
    print("Client 3 info:")
    print(info_obj_client3)
    print("Client 4 info:")
    print(info_obj_client4)

    #save the info objects
    np.save('client1_info.npy', info_obj_client1)
    np.save('client2_info.npy', info_obj_client2)
    np.save('client3_info.npy', info_obj_client3)
    np.save('client4_info.npy', info_obj_client4)

    print("Info objects saved successfully!")
