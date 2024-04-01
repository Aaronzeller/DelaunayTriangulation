#Import necessary libraries
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.spatial import Delaunay 
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

#=============================================== PARAMETERS ===============================================#

#Should Delaunay triangles be drawn?
drawBlueTriag = True
drawRedTriag = True

#Should points be drawn?
drawBluePoints = True
drawRedPoints = True

#Should connecting lines for lifting be drawn?
drawConnectors = True

#Should labels for points / triangles be drawn
drawTriagLabelsBlue = False
drawPointLabelsBlue = False

drawTriagLabelsRed = False
drawPointLabelsRed = False

#Should paraboloid of lifting be drawn?
drawParaboloid = False
paraboloidStrength = 0.2

#Should the non-delaunay triangles be drawn?
drawBlueNonDelaunay = False
drawRedNonDelaunay = False

#Should the default points be chosen?
defaultPoints = False

#Number of points to be triangulated
numb_points = 50

#Should the axis be drawn?
axis = False

#Should there be debug information?
debugInfo = True

#Highlight changes -> non-delaunay triangles
highlightChanges = False

#Show the edge of the removed delaunay triangles to indicate better what happened
oldEdges = False

#How many points should be in a circle 
num_points = 100

#Highlight points inside / on circles boundary that should not be there
highlightPointsInsideCircle = True

#=============================================== INTERNAL VARIABLES ===============================================#

#Which triangles should not be drawn
list_triag = [] 

#Which triangle should be output to console
print_triag = []

#Which points should be highlighted due to being in / or circle boundary
highlight_points = []

#=============================================== COLOURS ===============================================#

# Colours
red_transparent = (1, 0, 0, 0.1)  # RGBA for red with 20% transparency
blue_transparent = (0, 0, 1, 0.1)  # RGBA for blue with 20% transparency

red = (1, 0, 0, 1)
blue = (0, 0, 1, 1)

#=============================================== METHODS ===============================================#

#Draws a triangle given the point indices - in lifted triangulation
def drawTriangleBlue(index1, index2, index3, ax):
    p1 = np.array([x[index1], y[index1], z[index1]])
    p2 = np.array([x[index2], y[index2], z[index2]])
    p3 = np.array([x[index3], y[index3], z[index3]])

    # This is how I made it work - probably there are better ways though
    triag_vertices = [list(zip([p1[0], p2[0], p3[0]], [p1[1], p2[1], p3[1]], [p1[2], p2[2], p3[2]]))]

    if highlightChanges:
        triangle = Poly3DCollection(triag_vertices, alpha=0.0, facecolor='purple', linewidths=1, edgecolors='purple')
    else:
        triangle = Poly3DCollection(triag_vertices, alpha=0.0, facecolor='blue', linewidths=1, edgecolors=blue_transparent)

    ax.add_collection3d(triangle) 

#Draws a triangle given the list of incides - in lifted triangulation
def drawTriangleBlue(triagIndices, ax):
    
    index1 = triagIndices[0]
    index2 = triagIndices[1]
    index3 = triagIndices[2]

    p1 = np.array([x[index1], y[index1], z[index1]])
    p2 = np.array([x[index2], y[index2], z[index2]])
    p3 = np.array([x[index3], y[index3], z[index3]])

    # This is how I made it work - probably there are better ways though
    triag_vertices = [list(zip([p1[0], p2[0], p3[0]], [p1[1], p2[1], p3[1]], [p1[2], p2[2], p3[2]]))]
    
    if highlightChanges:
        triangle = Poly3DCollection(triag_vertices, alpha=0.0, facecolor='purple', linewidths=1, edgecolors='purple')
    else:
        triangle = Poly3DCollection(triag_vertices, alpha=0.0, facecolor='blue', linewidths=1, edgecolors=blue_transparent)

    ax.add_collection3d(triangle)

#Draws a triangle given the point indices - in actual triangulation
def drawTriangleRed(index1, index2, index3, ax):
    p1 = np.array([x[index1], y[index1], 0])
    p2 = np.array([x[index2], y[index2], 0])
    p3 = np.array([x[index3], y[index3], 0])

    # This is how I made it work - probably there are better ways though
    triag_vertices = [list(zip([p1[0], p2[0], p3[0]], [p1[1], p2[1], p3[1]], [p1[2], p2[2], p3[2]]))]

    if highlightChanges:
        triangle = Poly3DCollection(triag_vertices, alpha=0.0, facecolor='purple', linewidths=1, edgecolors='purple')
    else:
        triangle = Poly3DCollection(triag_vertices, alpha=0.0, facecolor='red', linewidths=1, edgecolors=red_transparent)

    ax.add_collection3d(triangle) 

#Draws a triangle given the list of incides - in actual triangulation
def drawTriangleRed(triagIndices, ax):
    
    index1 = triagIndices[0]
    index2 = triagIndices[1]
    index3 = triagIndices[2]

    p1 = np.array([x[index1], y[index1], 0])
    p2 = np.array([x[index2], y[index2], 0])
    p3 = np.array([x[index3], y[index3], 0])

    # This is how I made it work - probably there are better ways though
    triag_vertices = [list(zip([p1[0], p2[0], p3[0]], [p1[1], p2[1], p3[1]], [p1[2], p2[2], p3[2]]))]
    
    if highlightChanges:
        triangle = Poly3DCollection(triag_vertices, alpha=0.0, facecolor='purple', linewidths=1, edgecolors='purple')
    else:
        triangle = Poly3DCollection(triag_vertices, alpha=0.0, facecolor='red', linewidths=1, edgecolors=red_transparent)

    ax.add_collection3d(triangle)

#Removes a triangle given the triangle index
def removeTriangle(triangleIndex):
    global list_triag
    list_triag.append(triangleIndex)

#Removes all triangles according to a list of triangle indices
def removeTriangles(triangleIndices):
    global list_triag
    for triangleIndex in triangleIndices:
        list_triag.append(triangleIndex)

#Print the triangle with corresponding index
def printTriangle(triangleIndex):
    global print_triag
    print_triag.append(triangleIndex)

#Print the triangles in the list according to their indices
def printTriangles(triangleIndices):
    global print_triag
    for triangleIndex in triangleIndices:
        print_triag.append(triangleIndex)


"""
Flip two neighbouring triangles: A B (not necessarily Lawson Flip!)

Input as indices!

0. Check if the triangles are neigbhbouring | A intersect B | == 2 
1. Remove the two triangles from being drawn -> removeTriangles
2. Determine the two points which are not shared
3. [a, b, c] [a, d, c] -> [b, d, a] [b, d, c] <- need to make sure that they are cyclic though b, d, x ensures that
"""

def flip(triangle_index1, triangle_index2):

    triangle1 = tri.simplices[triangle_index1]
    triangle2 = tri.simplices[triangle_index2]

    intersect = list(set(triangle1) & set(triangle2))

    if len(intersect) != 2: # Step 0
        print("Cannot flip non-neighbouring triangles!")
        return
    
    if debugInfo:
        print("Flipping triangles ", triangle1, " and ", triangle2)

    #Remove the triangles - not from the triangulation but they are not printed - Step 1
    removeTriangles([triangle_index1, triangle_index2]) 

    #Determine the two non-shared vertices - Step 2
    difference = list(set(triangle1) ^ set(triangle2))

    #Construct the new triangles - Step 3
    triangle_new_1 = list(difference)
    triangle_new_2 = list(difference)

    triangle_new_1.append(intersect[0])
    triangle_new_2.append(intersect[1])

    if debugInfo:
        print("New triangles are:", triangle_new_1, " and ", triangle_new_2)

    #Draw the triangles 
    if drawBlueTriag:
        drawTriangleBlue(triangle_new_1, ax)
        drawTriangleBlue(triangle_new_2, ax)

    if drawRedTriag:
        drawTriangleRed(triangle_new_1, ax)
        drawTriangleRed(triangle_new_2, ax)

#Find radius and center of circle given three points in 2D
def find_circle_center_and_radius(p1, p2, p3):

    x1 = p1[0]
    y1 = p1[1]

    x2 = p2[0]
    y2 = p2[1]

    x3 = p3[0]
    y3 = p3[1]

    # Intermediate values to simplify calculations
    ma = (y2 - y1) / (x2 - x1)
    mb = (y3 - y2) / (x3 - x2)
    
    # Center (xc, yc) of the circle
    x_center = (ma * mb * (y1 - y3) + mb * (x1 + x2) - ma * (x2 + x3)) / (2 * (mb - ma))
    y_center = -1.0 / ma * (x_center - (x1 + x2) / 2.0) + (y1 + y2) / 2.0
    
    # Radius of the circle
    radius = np.sqrt((x_center - x1)**2 + (y_center - y1)**2)
    
    return x_center, y_center, radius

#Determine points inside of a circle or on its boundary 
def points_on_or_inside_circle(points, x_center, y_center, radius):

    on_or_inside = []

    #Check the circle equation for points inside or on boundary
    for index, (x, y) in enumerate(points):
        if (x - x_center)**2 + (y - y_center)**2 <= radius**2:
            on_or_inside.append(index)
    
    return on_or_inside

#Add circles to the 2D triangulation to argue over Delaunay condition - using points
def drawCirclePoints(p1, p2, p3, x_center, y_center, radius):

    if debugInfo:
        print("Center: ", (x_center, y_center), " and Radius: ", radius)

    #We have to parametrically define the circle as we are drawing in 3D
    theta = np.linspace(0, 2 * np.pi, num_points)

    #Construct the points that make up the circle
    x = x_center + radius * np.cos(theta)
    y = y_center + radius * np.sin(theta)
    z = 0 + np.zeros_like(theta)  # Circle in the XY plane - hence z == 0

    # Plot the circle
    ax.plot(x, y, z, label='Circle', color='orange')
            


#Add circles to the 2D triangulation to argue over Delaunay condition - using point indices
def drawCircle(index1, index2, index3):

    if debugInfo:
        print("Adding circle through points: ", index1, " ", index2, " ", index3)

    triangle_1 = [x[index1], y[index1], 0]
    triangle_2 = [x[index2], y[index2], 0]
    triangle_3 = [x[index3], y[index3], 0]

    x_center, y_center, radius = find_circle_center_and_radius(triangle_1, triangle_2, triangle_3)

    drawCirclePoints(triangle_1, triangle_2, triangle_3, x_center, y_center, radius)

    #Highlights points which are inside the circle or on its boundary and not one of the three that make it
    if highlightPointsInsideCircle:
        on_or_inside = points_on_or_inside_circle(zip(x, y), x_center, y_center, radius)
        
        global highlight_points
        highlight_points = [point for point in on_or_inside if point not in [index1, index2, index3]]

        if debugInfo:
            print("Points to highlight are: ", highlight_points)

#=============================================== DELAUNAY TRIANGULATION ===============================================#

# Example 2D points
x = np.random.uniform(-1, 1, numb_points)
y = np.random.uniform(-1, 1, numb_points)

#Take the saved points
if defaultPoints:
    points_df = pd.read_csv("C:/Users/azell/Documents/Reverse-Search-Presentation/2d_points.csv")
    points = points_df.to_numpy()

    x = points[0:numb_points, 0]
    y = points[0:numb_points, 1]

# Function to lift the points to 3D: (x, y) -> (x, y, x^2 + y^2)
z = x**2 + y**2

# Performing Delaunay triangulation on the original 2D points
points2D = np.vstack([x, y]).T  # Stack the 2D points into an Nx2 array
tri = Delaunay(points2D)  # This performs the delaunay triangulation

# Creating a 3D plot (a bit larger than default)
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')


#=============================================== METHOD EXECUTIONS ===============================================#



#=============================================== PLOTTING ===============================================#

# Generating the meshgrid for the paraboloid
x_grid, y_grid = np.meshgrid(np.linspace(-1, 1, 50), np.linspace(-1, 1, 50))
z_grid = x_grid**2 + y_grid**2

# Plotting the paraboloid surface
if drawParaboloid:
    ax.plot_surface(x_grid, y_grid, z_grid, color='lightblue', alpha=paraboloidStrength) #make alpha != 0 to see the paraboloid

# Plotting the original 2D points in 3D space at z=0 (for visualization)
if drawRedPoints:
    if highlightPointsInsideCircle:
        for index, point in enumerate(zip(x, y)):
            if index in highlight_points:
                ax.scatter(*point, 0, color='orange')
            else:
                ax.scatter(*point, 0, color='red')
    else:
        ax.scatter(x, y, 0, color='red')

# Plotting the lifted points
if drawBluePoints:
    ax.scatter(x, y, z, color='blue')

# Connecting the original and lifted points to illustrate the lifting process
if drawConnectors:
    for i in range(len(x)):
        ax.plot([x[i], x[i]], [y[i], y[i]], [0, z[i]], color='gray', linestyle='--', linewidth=0.5)

# Plotting the Delaunay triangulation in 3D (projecting the 2D triangulation into 3D space)
for index, simplex in enumerate(tri.simplices):

    if index in print_triag:
        print(simplex)

    if not index in list_triag:
        # Draw the triangle for the original points at z=0 in transparent red
        if drawRedTriag:
            for i in range(3):
                start_point = points2D[simplex[i]]
                end_point = points2D[simplex[(i + 1) % 3]]
                ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 0, color=red_transparent)
        
        # Draw the triangle for the lifted points in transparent blue
        if drawBlueTriag:
            for i in range(3):
                start_point = np.append(points2D[simplex[i]], z[simplex[i]])
                end_point = np.append(points2D[simplex[(i + 1) % 3]], z[simplex[(i + 1) % 3]])
                ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], [start_point[2], end_point[2]], color=blue_transparent)
    else:
        if oldEdges and drawBlueTriag:
            for i in range(3):
                start_point = np.append(points2D[simplex[i]], z[simplex[i]])
                end_point = np.append(points2D[simplex[(i + 1) % 3]], z[simplex[(i + 1) % 3]])
                ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], [start_point[2], end_point[2]], color=blue_transparent, linestyle='--')


# Hide the axes
if not axis:
    ax.set_axis_off()

#Add the indices to so I understand which triangle is which - lifted triangulation
if drawTriagLabelsBlue:
    for i, simplex in enumerate(tri.simplices):
        if not i in list_triag:
            # Calculating the centroid of the triangle
            x_centroid = np.mean(x[simplex])
            y_centroid = np.mean(y[simplex])
            z_centroid = np.mean(z[simplex])  # Assuming z_final contains the z-values of your points
            
            # Adding the label inside the triangle
            ax.text(x_centroid, y_centroid, z_centroid, str(i), color='black')

#Add the indices to so I understand which triangle is which - actual triangulation
if drawTriagLabelsRed:
    for i, simplex in enumerate(tri.simplices):
        if not i in list_triag:
            # Calculating the centroid of the triangle
            x_centroid = np.mean(x[simplex])
            y_centroid = np.mean(y[simplex])
            z_centroid = np.mean([0, 0, 0])  # Assuming z_final contains the z-values of your points
            
            # Adding the label inside the triangle
            ax.text(x_centroid, y_centroid, z_centroid, str(i), color='black')

#Add the labels to the points - lifted triangulation
if drawPointLabelsBlue:
    for i, (x_coord, y_coord, z_coord) in enumerate(zip(x, y, z)):
        ax.text(x_coord, y_coord, z_coord, f'{i}', color='black')

#Add the labels to the points - actual triangulation
if drawPointLabelsRed:
    for i, (x_coord, y_coord, z_coord) in enumerate(zip(x, y, [0, 0, 0]*len(x))):
        ax.text(x_coord, y_coord, z_coord, f'{i}', color='black')

plt.show()


