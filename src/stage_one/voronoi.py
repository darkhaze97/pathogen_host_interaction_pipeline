import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d, cKDTree
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
    kdtree = cKDTree(inputPoints)
    
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
                # The reason why we pass max(v1, v2) in is because we want to use
                # the defined vertex as a way to find the gradient of the ridge.
                edgeVertex = determine_edge_vertex(p1, p2, max(v1, v2), shape, vor, kdtree)
                
    
    print(vor.points, centroidList)
    print(vor.vertices)
    print(vor.ridge_points)
    print(vor.ridge_vertices)
    print(vor.regions)
    print(vor.point_region)
    voronoi_plot_2d(vor)
    plt.show()

# The function below should return a valid vertex.
def determine_edge_vertex(p1, p2, vIndex, shape, vor, kdtree):
    # First, make a point set with p1 and p2. This will be used to compare if the 
    # nearest 2 neighbours are in fact p1 and p2.
    pointSet = {tuple(p) for p in vor.points[[p1, p2]]}
    # Decide which edge the ridge collides with 
    # (left [0], right [defined by shape[1] - 1], top [defined by shape[0] -1], 
    # bottom [0])
    # First, find the midpoint of the two points. The reason why we find the midpoint is because
    # we want to find a point on the ridge. p1 and p2 are equidistant from the ridge that spans
    # between them.
    midpoint = vor.points[[p1, p2]].mean(axis=0)
    # Obtain the vertex represented by vIndex. This is a known ridge vertex in the ridge,
    # and will be used to find a vector that spans this ridge.
    vertex = vor.vertices[vIndex]
    # Find the gradient vector. The gradient will be the line from the
    # ridge vertex to the midpoint.
    gradVec = midpoint - vertex
    print("For the vertex: ", vertex, ", and the midpoint: ", midpoint, ", we have gradient: ", gradVec)
    # There are 4 borders to check for intersection: x = 0, x = shape[0], y = 0, y = shape[1]
    # And we have an equation like so: a * gradVec + vertex = <x, y>, where x or y is known, as they
    # are the borders.
    # First, fix x and solve for y when x = 0
    epsilon = (-vertex[0])/gradVec[0]
    y = gradVec[1] * epsilon + vertex[1]
    # Check if y lies within our borders.
    if (y > 0 and y < shape[1]):   
        # Check if y is equidistant to p1 and p2.
        dist, nearest2Neighbours = kdtree.query([0, y], k=2)
        # If statement below checks if the computed 2 nearest neighbours
        # matches p1 and p2.
        if ({tuple(p) for p in vor.points[nearest2Neighbours]} == pointSet):
            print([0,y])
            return [0, y]
    # Next, fix x and solve for y when x = shape[0] - 1
    epsilon = (shape[0] - 1 - vertex[0])/gradVec[0]
    y = gradVec[1] * epsilon + vertex[1]
    # Check if y lies within our borders.
    
    
    # Solve for all lambdas. Whichever border the line hits that is on
    # the border of our rectangle may be the true vertex we have. We will have to test
    # if this vertex is equidistant from p1 and p2. If it isn't then, find the other
    # vertex. Check image1 of the summer research images. We will see that finding the midpoint
    # of the two outer vertices leads to a midpoint which does not lie on the ridge.
    # If we only considered the vector going from the vertex to the midpoint, then the
    # second edge vertex for this ridge will not be correct. Therefore, we need to use
    # a cKDTree. 
    # https://stackoverflow.com/questions/17857021/finding-voronoi-regions-that-contain-a-list-of-arbitrary-coordinates
    
def solve_y():
    print('Hi')