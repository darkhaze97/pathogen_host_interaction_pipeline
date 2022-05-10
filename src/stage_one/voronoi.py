import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d, cKDTree
from skimage import draw, measure
import matplotlib.pyplot as plt

# The function below is to perform a voronoi segmentation.
# ======================= TO DO WHEN FINISHED =======================
# Resources used:
#   https://stackoverflow.com/questions/57385472/how-to-set-a-fixed-outer-boundary-to-voronoi-tessellations
def voronoi_seg(centroidList, cellImg):
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
    # The kdtree will be used when finding the vertex of a voronoi region that is infinite.
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
    mappedRidgeVertices = find_finite_ridge_endpoints(vor, ridges, cellImg, kdtree)
    print(mappedRidgeVertices)
    # print(vor.points, centroidList)
    # print(vor.vertices)
    # print(vor.ridge_points)
    # print(vor.ridge_vertices)
    # print(vor.regions)
    # print(vor.point_region)
    voronoi_plot_2d(vor)
    plt.show()
    separationImg = np.ones(cellImg.shape)
    for v1, v2 in mappedRidgeVertices:
        rr, cc, val = draw.line_aa(int(v1[1]), int(v1[0]), int(v2[1]), int(v2[0]))
        separationImg[rr, cc] = 0
        cellImg[rr, cc] = -1
    separationImg = measure.label(separationImg)
    return np.where(cellImg == 0, cellImg, cellImg + separationImg)

# The function below finds the endpoints for infinite ridges. It returns the set of vertices that
# form the voronoi segmentation.
# Arguments:
#   - vor: The voronoi segmentation diagram
#   - ridges: The set of ridges surrounding a point
#   - cellImg: The cell image
#   - kdtree: A data structure used to find the nearest neighbours of a vertex.
# Returns:
#   - The list of ridge points, including the vertices that were computed from 
#     infinite ridges.
def find_finite_ridge_endpoints(vor, ridges, cellImg, kdtree):
    # mappedRidgeVertices is a list that contains tuples of vertices, each representing a ridge.
    mappedRidgeVertices = set()
    for p1 in range(len(vor.point_region)):
        # Scan through every ridge for p1
        for (p2, v1, v2) in ridges[p1]:
            if (v1 == -1 or v2 == -1):
                # If we enter here, then we have an infinite ridge.
                # The midpoint of p1, p2 is a point on the ridge. The gradient of the ridge
                # will be found, and the intersection point with the borders will be 
                # computed from this.
                # Add the return from the function to allVertices and ridgeVertices
                # The reason why we pass max(v1, v2) in is because we want to use
                # the defined vertex as a way to find the gradient of the ridge.
                edgeVertex = determine_edge_vertex(p1, p2, max(v1, v2), np.shape(cellImg), vor, kdtree)
                mappedRidgeVertices.add((tuple(vor.vertices[max(v1, v2)]), edgeVertex))
            else:
                mappedRidgeVertices.add(tuple([tuple(v) for v in vor.vertices[[v1, v2]]]))
    return mappedRidgeVertices

# This function returns a valid edge vertex, based on the voronoi segmentation.
# Arguments:
#   - p1 and p2: points which formed the voronoi ridge. Compared to the
#                nearest 2 neighbours of the computed edge vertex. This is to help
#                determine the correct edge vertex, since the edge vertex should
#                be equidistant from the two points (property of voronoi ridges).
#   - vIndex: The index of the known vertex in the voronoi ridge. This will be used to
#             compute the gradient vector of the ridge, and then to find the
#             edge vertex by finding points of intersection with the borders.
#   - shape: Used to find the maximum x and maximum y values. These values are used to
#            determine if the computed x and y values are within the borders.
#   - vor: The voronoi segmentation diagram.
#   - kdtree: A data structure to find the nearest neighbours.
# Returns:
#   - vertex [x, y]: The edge vertex. Should not return None, as the ridge vertex should always
#                    intersect one valid border. (Valid == x and y values between 0 and maxX or maxY
#                    inclusive, and is equidistant to p1 and p2, which are it's nearest neighbours).
# Sources: # https://stackoverflow.com/questions/17857021/finding-voronoi-regions-that-contain-a-list-of-arbitrary-coordinates
def determine_edge_vertex(p1, p2, vIndex, shape, vor, kdtree):
    # Extract the maxX value and the maxY value.
    maxX = shape[1] - 1
    maxY = shape[0] - 1
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
    # There are 4 borders to check for intersection: x = 0, x = shape[1], y = 0, y = shape[0]
    # And we have an equation like so: a * gradVec + vertex = <x, y>, where x or y is known, as they
    # are the borders, and a is any real number.

    # First, fix x and solve for y when x = 0
    epsilon = (-vertex[0])/gradVec[0]
    y = gradVec[1] * epsilon + vertex[1]
    # Below checks if y lies within our borders. If it does, it will determine if the nearest
    # neighbours match p1 and p2. If it does, return [0, y], else, continue.
    ret = valid_neighbours(None, y, 0, kdtree, vor, pointSet) if valid_within_borders(y, maxY) else []
    if (ret): return ret

    # Next, fix x and solve for y when x = shape[1] - 1
    epsilon = (maxX - vertex[0])/gradVec[0]
    y = gradVec[1] * epsilon + vertex[1]
    # Below checks if y lies within our borders. If it does, it will determine if the nearest
    # neighbours match p1 and p2. If it does, return [maxX, y], else, continue.
    ret = valid_neighbours(None, y, maxX, kdtree, vor, pointSet) if valid_within_borders(y, maxY) else []
    if (ret): return ret
    
    # Next, fix y and solve for x when y = 0
    epsilon = (-vertex[1])/gradVec[1]
    x = gradVec[0] * epsilon + vertex[0]
    # Below checks if x lies within our borders. If it does, it will determine if the nearest
    # neighbours match p1 and p2. If it does, return [x, 0], else, continue.
    ret = valid_neighbours(x, None, 0, kdtree, vor, pointSet) if valid_within_borders(x, maxX) else []
    if (ret): return ret

    # Next, fix y and solve for x when y = shape[0] - 1
    epsilon = (maxY - vertex[1])/gradVec[1]
    x = gradVec[0] * epsilon + vertex[0]
    # Below checks if x lies within our borders. If it does, it will determine if the nearest
    # neighbours match p1 and p2. If it does, return [x, maxY], else, continue.
    ret = valid_neighbours(x, None, maxY, kdtree, vor, pointSet) if valid_within_borders(x, maxX) else []
    if (ret): return ret
    print('How did we even end up here.')
    
# This function simply computes if value >= 0 and value <= maxBorderValue. It makes sure
# that the potential vertex is within the bounds of our image.
# Arguments:
#   - value: The x or y value to check
#   - maxBorderValue: The maximum value of x or y, defined by shape[1] or shape[0] respectively.
# Returns:
#   - True or False depending on whether the conditional holds.
def valid_within_borders(value, maxBorderValue):
    return True if (value >= 0 and value <= maxBorderValue) else False

# This function determines if the computed edge vertex's nearest neighbours are p1 and p2.
# We use the property that any point on a voronoi ridge is equidistant to the two points
# that form the ridge. 
# Arguments:
#   - x: The computed x value of the point. May be None if we are analysing the computed
#        y value instead.
#   - y: The computed y value of the point. May be None if we are analysing the computed
#        x value instead.
#   - fixedBorder: The border that will be fixed. This can either be 0, maxX (if we are
#                  fixing x value of the computed edge vertex) or maxY (if we are fixing y
#                  y value of the computed edge vertex)
#   - kdtree: A data structure used to find the nearest points.
#   - vor: The voronoi graph
#   - pointSet: {(p1), (p2)}, used to determine if the neighbours are matching p1 and p2.
# returns
#   - []: If the neighbours do not match.
#   OR vertex: the edge vertex that satisfies all conditions (is within the borders, and the
#              nearest neighbours are p1 and p2).
def valid_neighbours(x, y, fixedBorder, kdtree, vor, pointSet):
    vertex = [x, fixedBorder] if x else [fixedBorder, y]
    # Check if [x, fixedBorder] or [fixedBorder, y] is equidistant to p1 and p2.
    dist, nearest2Neighbours = kdtree.query(vertex, k=2)
    # If statement below checks if the computed 2 nearest neighbours
    # matches p1 and p2.
    if ({tuple(p) for p in vor.points[nearest2Neighbours]} == pointSet):
        print(f'Intersect with y = {fixedBorder}: ', vertex) if x else \
        print(f'Intersect with x = {fixedBorder}: ', vertex)
        return tuple(vertex)
    return []