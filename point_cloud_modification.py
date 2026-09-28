import numpy as np

def discart_outside_cloud_in_z(cloud: PointCloud):
    
    cloud.data = cloud.data[(cloud.data[:, 2] >= -25.0) & (cloud.data[:, 2] <= 1025.0)]
