import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt


# Resources used:
#   https://stackoverflow.com/questions/57385472/how-to-set-a-fixed-outer-boundary-to-voronoi-tessellations
def voronoi_seg(centroidList, shape):
    # print('Shape: ', shape)
    
    # First, fix the centroidList. Note that numpy orders their indexing
    # by row first, followed by column. Even though the indexing starts
    # from the top left in numpy, we don't need to change this, as the
    # math should account for this.
    for centroid in centroidList:
        centroid[0], centroid[1] = centroid[1], centroid[0]
    
    # print(shape[0], shape[1])
    
    inputPoints = np.array(centroidList)
    vor = Voronoi(inputPoints)
    
    # I move the vertex lists out so that it is easier to update the values.
    allVertices = vor.vertices
    ridgeVertices = vor.ridge_vertices
    
    # Scan through every ridge vertex pair and ridge region points, and place them into
    # a dictionary. Then scan through each region. Match each ridge for the region to
    # the corresponding ridge in the dictionary. If the region is infinite (i.e.
    # regions[point_region_index] contains a value less than 0), then we 
    # need to make it finite, by finding the edge that the ridges should
    # touch. Then add the vertices of each ridge into a ridge list.
    # If the region is finite, then simply add the vertex pairs for each ridge on to a ridge list. 
    # Overall, this task should perform in O(NV) time,
    # where N == number of regions and V == the total ridges.
    
    # ridges contains a mapping from each point in the voronoi diagram to a tuple. This
    # tuple contains the vertices of the ridge, as well as another point that is equidistant
    # to the ridge.
    ridges = {}
    # Scan through every ridge, and record the ridges.
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, ridgeVertices):
        ridges.setdefault(p1, []).append((p2, v1, v2))
        ridges.setdefault(p2, []).append((p1, v1, v2))
    
    # print(ridges)
    
    # mappedRidgeVertices is a list that contains tuples of vertices, each representing a ridge.
    mappedRidgeVertices = []
    for p1, region in enumerate(vor.point_region):
        # Obtain the ridge vertices around the region.
        vertices = vor.regions[region]
        
        if all(v >= 0 for v in vertices):
            # This region is finite. We need to simply obtain the vertices.
            continue
        
        # We have an infinite region. First, scan through the (p2, v1, v2) of p1.
        # Once we encounter a v1 or v2 == -1, then that ridge is infinite.
        # The midpoint of p1, p2 is a point on the ridge. The gradient of the ridge
        # will be found, and the intersection point will be computed from this.
        for (p2, v1, v2) in ridges[p1]:
            if (v1 == -1 or v2 == -1):
                # Add the return from the function to allVertices and ridgeVertices
                edgeVertex = determine_edge_vertex(p1, p2, v1, v2, shape, vor)
                
    
    # print(vor.points)
    # print(vor.vertices)
    # print(vor.ridge_points)
    # print(vor.ridge_vertices)
    # print(vor.regions)
    # print(vor.point_region)
    # voronoi_plot_2d(vor)
    # plt.show()

# The function below should return a valid vertex.
def determine_edge_vertex(p1, p2, v1, v2, shape, vor):
    # Decide which edge the ridge collides with 
    # (left [0], right [defined in shape[1] - 1], top [defined in shape[0] -1], 
    # bottom [0])
    # First, find the midpoint of the two points.
    midpoint = vor.points[[p1, p2]].mean(axis=0)
    # Find the gradient in vector form.