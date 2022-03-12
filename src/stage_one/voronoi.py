import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt

def voronoi_seg(centroidList, shape):
    print('Shape: ', shape)
    
    # First, fix the centroidList. Note that numpy orders their indexing
    # by row first, followed by column. Even though the indexing starts
    # from the top left in numpy, we don't need to change this, as the
    # math should account for this.
    for centroid in centroidList:
        centroid[0], centroid[1] = centroid[1], centroid[0]
    
    inputPoints = np.array(centroidList)
    vor = Voronoi(inputPoints)
    
    # Scan through each region. If the region is infinite (i.e.
    # regions[point_region_index] contains a value less than 0), then we 
    # need to make it finite, by finding the edge that the ridges should
    # touch. Then add the vertices of each ridge into a ridge list.
    # If the region is finite, then simply add the vertices into the 
    # ridge list. Note that before each add, we perform a log(n) check
    # to ensure that there are no duplicates. Overall, this task should
    # perform in O(NVlog(V)) time,
    # where N == number of regions and V == the total ridges.
    # ridgeVerticesSort1 is sorted based on the first element
    ridgeVerticesSort1 = []
    # ridgeVerticesSort2 is sorted based on the second element
    ridgeVerticesSort2 = []
    for region in vor.point_region:
        vertices = vor.regions[region]
    
        if all(v >= 0 for v in vertices):
            # This region is finite.
            for v in vertices:
                vertex = vor.ridge_vertices(v)
                # Binary search
                # Then add to the lists if unique.
            # Then, sort the lists. 
            ridgeVerticesSort1 = sorted(ridgeVerticesSort1, key = lambda v: (v[0]))
            ridgeVerticesSort2 = sorted(ridgeVerticesSort2, key = lambda v: (v[1]))
            
            
    
    print(vor.points)
    print(vor.vertices)
    print(vor.ridge_points)
    print(vor.ridge_vertices)
    print(vor.regions)
    print(vor.point_region)
    print(vor.regions[3])
    voronoi_plot_2d(vor)
    plt.show()