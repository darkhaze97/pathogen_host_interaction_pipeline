import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d, cKDTree
from skimage import draw, measure
import matplotlib.pyplot as plt

# The function below is to help perform a voronoi segmentation. After the voronoi segmentation,
# it passes the voronoi image to another function to perform post-processing to find
# border vertices.
# Arguments:
#   - centroidList (list of tuples): A list of points for each centroid.
#   - cellImg (numpy array): The image of the cell.
# Returns:
#   - The newly segmented image.
# Resources used:
#   https://stackoverflow.com/questions/57385472/how-to-set-a-fixed-outer-boundary-to-voronoi-tessellations
def voronoi_seg(centroidList, cellImg):
    # First, fix the centroidList. Note that numpy orders their indexing
    # by row first, followed by column. Even though the indexing starts
    # from the top left in numpy, we don't need to change this, as the
    # math should account for this.
    for centroid in centroidList:
        centroid[0], centroid[1] = centroid[1], centroid[0]
    
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
    
    # mappedRidgeVertices is a list that contains tuples of vertices, each representing a ridge.
    mappedRidgeVertices = find_finite_ridge_endpoints(vor, ridges, cellImg, kdtree)

    meh = draw_aa_line(mappedRidgeVertices, cellImg, kdtree)
    # print(vor.points, centroidList)
    # print(vor.vertices)
    # print(vor.ridge_points)
    # print(vor.ridge_vertices)
    # print(vor.regions)
    # print(vor.point_region)
    # voronoi_plot_2d(vor)
    # plt.show()
    return meh

# The function below is similar to voronoi_seg, however, it only performs the segmentation
# for two cells. It simply forms a perpendicular line to the line between the two points.
# Then, it attempts to find border vertices by using the orthogonal line that passes through the
# midpoint of the line between the two points. This function does not use scipy.spatial.Voronoi.
# Arguments:
#   - centroidList (list of tuples): A list of points for each centroid.
#   - cellImg (numpy array): The image of the cell.
# Returns:
#   - The newly segmented image.
def voronoi_seg_alt(centroidList, cellImg):
    maxX = cellImg.shape[1] - 1
    maxY = cellImg.shape[0] - 1
    
    for centroid in centroidList:
        centroid[0], centroid[1] = centroid[1], centroid[0]
    
    kdtree = cKDTree(centroidList)
    
    midpoint = np.array(centroidList).mean(axis=0)
    gradVec = midpoint - centroidList[0]
    orthogonal = [gradVec[1], -gradVec[0]]
    # The equation that we will use is midpoint + epsilon * orthogonal = <x, y>,
    # where x or y will be defined.
    # borderVertices will store the vertices that are on the border and on the line
    # defined by midpoint + epsilon * orthogonal = <x, y>. Ultimately, len(borderVertices)
    # should be 2.
    borderVertices = []
    
    # Solve for y when x is fixed to 0.
    y = solve_y(0, midpoint, orthogonal)
    if (valid_within_borders(y, maxY)): borderVertices.append((0, y))
    
    # Solve for y when x is fixed to cellImg[1] - 1
    y = solve_y(maxX, midpoint, orthogonal)
    if (valid_within_borders(y, maxY)): borderVertices.append((maxX, y))
    
    # Solve for x when y is fixed to 0.
    x = solve_x(0, midpoint, orthogonal)
    if (valid_within_borders(x, maxX)): borderVertices.append((x, 0))
    
    # Solve for x when y is fixed to maxY
    x = solve_x(maxY, midpoint, orthogonal)
    if (valid_within_borders(x, maxX)): borderVertices.append((x, maxY))
    
    borderVertices = [tuple(borderVertices)]
    
    return draw_aa_line(borderVertices, cellImg, kdtree)

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
    maxX, maxY = np.shape(cellImg)[1] - 1, np.shape(cellImg)[0] - 1
    
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
                edgeVertex = determine_edge_vertex(p1, p2, max(v1, v2), np.shape(cellImg),
                                                   vor, kdtree, False)
                if (edgeVertex): mappedRidgeVertices.add((tuple(vor.vertices[max(v1, v2)]), edgeVertex))
            else:
                # We have a finite ridge.
                # Check if a vertex of the ridge is outside of the borders, whilst another is
                # inside the borders.
                vert1, vert2 = vor.vertices[[v1, v2]]
                newV1 = -1 if (not valid_within_borders(vert1[0], maxX) or not valid_within_borders(vert1[1], maxY)) \
                           else v1
                newV2 = -1 if (not valid_within_borders(vert2[0], maxX) or not valid_within_borders(vert2[1], maxY)) \
                           else v2
                if (newV1 >= 0 and newV2 >= 0):
                    # We enter here if both vertices are within the image dimensions.
                    mappedRidgeVertices.add(tuple([tuple(v) for v in vor.vertices[[v1, v2]]]))
                elif (max(newV1, newV2) >= 0):
                    # We enter here if only one vertex is outside of the image dimensions.
                    # Find the edge vertex, similar to when there is an infinite ridge.
                    edgeVertex = determine_edge_vertex(p1, p2, max(newV1, newV2), np.shape(cellImg),
                                                       vor, kdtree, False)
                    if (edgeVertex): mappedRidgeVertices.add((tuple(vor.vertices[max(newV1, newV2)]), edgeVertex))
                else:
                    # We enter here if both of the ridge vertices are outside of the main
                    # image dimensions.
                    edgeVertex = determine_edge_vertex(v1, v2, max(v1, v2), np.shape(cellImg),
                                                       vor, kdtree, True)
                                                   
    # print(mappedRidgeVertices)
    return mappedRidgeVertices

def determine_finite_edge_vertices(p1, p2, v1, v2, shape, vor, kdtree):
    # Extract the maxX value and the maxY value.
    maxX = shape[1] - 1
    maxY = shape[0] - 1
    # First, make a point set with p1 and p2. This will be used to compare if the 
    # nearest 2 neighbours are in fact p1 and p2.
    pointSet = {tuple(p) for p in vor.points[[p1, p2]]}
    # Decide which edges the ridge collides with.

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
def determine_edge_vertex(p1, p2, vIndex, shape, vor, kdtree, twoPt):
    
    ridgePoints = []
    
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
    # There are 4 borders to check for intersection: x = 0, x = shape[1], y = 0, y = shape[0]
    # And we have an equation like so: a * gradVec + vertex = <x, y>, where x or y is known, as they
    # are the borders, and a is any real number.

    # First, fix x and solve for y when x = 0
    y = solve_y(0, vertex, gradVec)
    # Below checks if y lies within our borders. If it does, it will determine if the nearest
    # neighbours match p1 and p2. If it does, return [0, y], else, continue.
    ret = valid_neighbours(None, y, 0, kdtree, vor, pointSet) if valid_within_borders(y, maxY) else []
    if (ret): ridgePoints.append(ret)

    # Next, fix x and solve for y when x = shape[1] - 1
    y = solve_y(maxX, vertex, gradVec)
    # Below checks if y lies within our borders. If it does, it will determine if the nearest
    # neighbours match p1 and p2. If it does, return [maxX, y], else, continue.
    ret = valid_neighbours(None, y, maxX, kdtree, vor, pointSet) if valid_within_borders(y, maxY) else []
    if (ret): ridgePoints.append(ret)
    
    # Next, fix y and solve for x when y = 0
    x = solve_x(0, vertex, gradVec)
    # Below checks if x lies within our borders. If it does, it will determine if the nearest
    # neighbours match p1 and p2. If it does, return [x, 0], else, continue.
    ret = valid_neighbours(x, None, 0, kdtree, vor, pointSet) if valid_within_borders(x, maxX) else []
    if (ret): ridgePoints.append(ret)

    # Next, fix y and solve for x when y = shape[0] - 1
    x = solve_x(maxY, vertex, gradVec)
    # Below checks if x lies within our borders. If it does, it will determine if the nearest
    # neighbours match p1 and p2. If it does, return [x, maxY], else, continue.
    ret = valid_neighbours(x, None, maxY, kdtree, vor, pointSet) if valid_within_borders(x, maxX) else []
    if (ret): ridgePoints.append(ret)

    return None if (not ridgePoints) else ridgePoints[0] if (not twoPt) else ridgePoints
    
    # print('How did we even end up here.')

# Function below is to solve: vertex + epsilon * gradVec = <x, y>, for when x is fixed.
# Used to find the border vertices of a voronoi ridge.
# Arguments:
#   - x (int): The value to set x to.
#   - vertex (tuple)
#   - gradVec (tuple)
# Returns:
#   - y, which is equal to gradVec[1] * epsilon + vertex[1]
def solve_y(x, vertex, gradVec):
    epsilon = (x - vertex[0])/gradVec[0]
    return gradVec[1] * epsilon + vertex[1]

# Function below is to solve: vertex + epsilon * gradVec = <x, y>, for when y is fixed.
# Used to find the border vertices of a voronoi ridge.
# Arguments:
#   - y (int): The value to set y to.
#   - vertex (tuple)
#   - gradVec (tuple)
# Returns:
#   - x, which is equal to gradVec[0] * epsilon + vertex[0]
def solve_x(y, vertex, gradVec):
    epsilon = (y - vertex[1])/gradVec[1]
    return gradVec[0] * epsilon + vertex[0]

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
        return tuple(vertex)
    return []

# The function below simply draws voronoi ridges on the cellImg. It then assigns the ridge
# the label of an adjacent region.
# Arguments:
#   - mappedRidgeVertices (list of tuples): The list of vertex pairs that form a voronoi ridge
#   - cellImg (numpy array): The image of the cell.
#   - kdtree: A data structure used to find the nearest points.
# Returns:
#   - The cell image that is segmented with the voronoi ridges.
def draw_aa_line(mappedRidgeVertices, cellImg, kdtree):
    # Below is to help draw lines on the cell image, to perform the voronoi segmentation.
    separationImg = np.ones(cellImg.shape)
    lineList = []
    for v1, v2 in mappedRidgeVertices:
        rr, cc, val = draw.line_aa(int(v1[1]), int(v1[0]), int(v2[1]), int(v2[0]))
        separationImg[rr, cc] = 0
        lineList.append((rr, cc))
    separationImg = measure.label(separationImg)

    # For each (rr, cc) pair in lineList, find the midpoint pixel, and obtain
    # the nearest centroid neighbour to this line. Then set the line defined by rr, cc
    # to the label of the centroid pixel.
    for rr, cc in lineList:
        # I reverse the order of the x and y, since I will be passing the midpoint into
        # a kdtree, which uses (col, row).
        midpoint = (int((cc[-1] + cc[0])/2), int((rr[-1] + rr[0])/2))
        # Find nearest neighbour with kdtree, and obtain the centroid pixel by indexing
        # with vor.points. Then assign the line to the label of the centroid pixel.
        dist, nearestNeighbour = kdtree.query(midpoint)
        # Below obtains the pixel coordinates of the centroid. It converts
        # floats into ints.
        nearestCentroid = [int(p) for p in reversed(kdtree.data[nearestNeighbour])]
        # nearestCentroid = [int(p) for p in reversed(vor.points[nearestNeighbour])]
        separationImg[rr, cc] = separationImg[nearestCentroid[0], nearestCentroid[1]]
    return np.where(cellImg == 0, cellImg, cellImg + separationImg)