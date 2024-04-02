#Import necessary libraries
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.spatial import Delaunay 
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import tkinter as tk
from tkinter import BooleanVar, Entry, IntVar, DoubleVar
from functools import partial


#=============================================== GLOBAL PARAMETERS ===============================================#

# Colours
red_transparent = (1, 0, 0, 0.1)  # RGBA for red with 20% transparency
blue_transparent = (0, 0, 1, 0.1)  # RGBA for blue with 20% transparency

red = (1, 0, 0, 1)
blue = (0, 0, 1, 1)

# Vertices to colour
highlight_points = []

#=============================================== DELAUNAY TRIANGULATION ===============================================#


def createDelaunay(defaultPoints, numb_points):
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

    return tri, points2D, x, y, z

#=============================================== METHODS ===============================================#

#Draws a triangle given the point indices - in lifted triangulation
def drawTriangleBlue(index1, index2, index3, ax, x, y, z, highlightChanges):

    #Make sure colour is recognised as global one
    global blue_transparent

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
def drawTriangleBlue(triagIndices, ax, x, y, z, highlightChanges):
    
    #Make sure colour is recognised as global one
    global blue_transparent

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
def drawTriangleRed(index1, index2, index3, ax, x, y, z, highlightChanges):

    #Make sure colour is recognised as global one
    global red_transparent 

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
def drawTriangleRed(triagIndices, ax, x, y, z, highlightChanges):
    
    #Make sure colour is recognised as global one
    global red_transparent

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

"""
Flip two neighbouring triangles: A B (not necessarily Lawson Flip!)

Input as indices!

0. Check if the triangles are neigbhbouring | A intersect B | == 2 
1. Remove the two triangles from being drawn -> removeTriangles
2. Determine the two points which are not shared
3. [a, b, c] [a, d, c] -> [b, d, a] [b, d, c] <- need to make sure that they are cyclic though b, d, x ensures that
"""

def flip(triangle_index1, triangle_index2, tri, debugInfo):

    triangle1 = tri.simplices[triangle_index1]
    triangle2 = tri.simplices[triangle_index2]

    intersect = list(set(triangle1) & set(triangle2))

    if len(intersect) != 2: # Step 0
        print("Cannot flip non-neighbouring triangles!")
        return
    
    if debugInfo:
        print("Flipping triangles ", triangle1, " and ", triangle2)

    #Remove the triangles - not from the triangulation but they are not printed - Step 1
    #removeTriangles([triangle_index1, triangle_index2]) 

    #Determine the two non-shared vertices - Step 2
    difference = list(set(triangle1) ^ set(triangle2))

    #Construct the new triangles - Step 3
    triangle_new_1 = list(difference)
    triangle_new_2 = list(difference)

    triangle_new_1.append(intersect[0])
    triangle_new_2.append(intersect[1])

    if debugInfo:
        print("New triangles are:", triangle_new_1, " and ", triangle_new_2)

    #Replace the exisiting triangles
    tri.simplices[triangle_index1] = triangle_new_1
    tri.simplices[triangle_index2] = triangle_new_2

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
def drawCirclePoints(p1, p2, p3, x_center, y_center, radius, ax, circle_numb_points, debugInfo):

    if debugInfo:
        print("Center: ", (x_center, y_center), " and Radius: ", radius)

    #We have to parametrically define the circle as we are drawing in 3D
    theta = np.linspace(0, 2 * np.pi, circle_numb_points)

    #Construct the points that make up the circle
    x = x_center + radius * np.cos(theta)
    y = y_center + radius * np.sin(theta)
    z = 0 + np.zeros_like(theta)  # Circle in the plane - hence z == 0

    # Plot the circle
    ax.plot(x, y, z, label='Circle', color='orange')
            


#Add circles to the 2D triangulation to argue over Delaunay condition - using point indices
def drawCircle(index1, index2, index3, x, y, z, ax, highlightPointsInsideCircle ,debugInfo, circle_numb_points):
    #Make sure that highlight_points is recognised as the global list
    global highlight_points

    if debugInfo:
        print("Adding circle through points: ", index1, " ", index2, " ", index3)

    triangle_1 = [x[index1], y[index1], 0]
    triangle_2 = [x[index2], y[index2], 0]
    triangle_3 = [x[index3], y[index3], 0]

    x_center, y_center, radius = find_circle_center_and_radius(triangle_1, triangle_2, triangle_3)

    drawCirclePoints(triangle_1, triangle_2, triangle_3, x_center, y_center, radius, ax, circle_numb_points, debugInfo)

    #Highlights points which are inside the circle or on its boundary and not one of the three that make it
    if highlightPointsInsideCircle:
        on_or_inside = points_on_or_inside_circle(zip(x, y), x_center, y_center, radius)
        
        highlight_points = [point for point in on_or_inside if point not in [index1, index2, index3]]

        if debugInfo:
            print("Points to highlight are: ", highlight_points)
    

#=============================================== PLOTTING ===============================================#

def drawDelaunay(tri, ax, points2D, x, y, z, drawBlueTriag, drawRedTriag, drawBluePoints, drawRedPoints, drawConnectors, drawTriagLabelsBlue, drawPointLabelsBlue, drawTriagLabelsRed, drawPointLabelsRed, drawParaboloid, paraboloidStrength, defaultPoints, numb_points, axis, debugInfo, circle_numb_points, highlightPointsInsideCircle):
    
    #Make sure colours are recognised to be the global ones
    global red_transparent 
    global blue_transparent
    global red 
    global blue

    #Make sure that the list of points to be highlighted is the global variable
    global highlight_points
      
    # Generating the meshgrid for the paraboloid
    x_grid, y_grid = np.meshgrid(np.linspace(-1, 1, 50), np.linspace(-1, 1, 50))
    z_grid = x_grid**2 + y_grid**2

    # Plotting the paraboloid surface
    if drawParaboloid:
        ax.plot_surface(x_grid, y_grid, z_grid, color='lightblue', alpha=paraboloidStrength) #make alpha != 0 to see the paraboloid

    # Plotting the original 2D points in 3D space at z=0 (for visualization)
    if drawRedPoints:
        if highlightPointsInsideCircle:
            if debugInfo:
                print("Highlighting Enabled!")

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

    # Hide the axes
    if not axis:
        ax.set_axis_off()

    #Add the indices to so I understand which triangle is which - lifted triangulation
    if drawTriagLabelsBlue:
        for i, simplex in enumerate(tri.simplices):
            # Calculating the centroid of the triangle
            x_centroid = np.mean(x[simplex])
            y_centroid = np.mean(y[simplex])
            z_centroid = np.mean(z[simplex])  # Assuming z_final contains the z-values of your points
            
            # Adding the label inside the triangle
            ax.text(x_centroid, y_centroid, z_centroid, str(i), color='black')

    #Add the indices to so I understand which triangle is which - actual triangulation
    if drawTriagLabelsRed:
        for i, simplex in enumerate(tri.simplices):
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

def visualize_delaunay():

    #=============================================== GUI STUFF ===============================================#

    # Create the main window
    root = tk.Tk()
    root.title("Delaunay Triangulation Visualisation")

    #Create boolean variables
    drawBlueTriag_ = tk.BooleanVar()
    drawRedTriag_ = tk.BooleanVar(value=True)
    drawBluePoints_ = tk.BooleanVar()
    drawRedPoints_ = tk.BooleanVar(value=True)
    drawConnectors_ = tk.BooleanVar()
    drawTriagLabelsBlue_ = tk.BooleanVar()
    drawPointLabelsBlue_ = tk.BooleanVar()
    drawTriagLabelsRed = tk.BooleanVar()
    drawTriagLabelsRed_ = tk.BooleanVar()
    drawPointLabelsRed_ = tk.BooleanVar()
    drawParaboloid_ = tk.BooleanVar()
    defaultPoints_ = tk.BooleanVar()
    axis_ = tk.BooleanVar()
    debugInfo_ = tk.BooleanVar()
    highlightPointsInsideCircle_ = tk.BooleanVar()

    # Create checkboxes for each parameter
    tk.Checkbutton(root, text="Draw Blue Triag", variable=drawBlueTriag_).pack()
    tk.Checkbutton(root, text="Draw Blue Points", variable=drawBluePoints_).pack()
    tk.Checkbutton(root, text="Draw Red Triag", variable=drawRedTriag_).pack()
    tk.Checkbutton(root, text="Draw Red Points", variable=drawRedPoints_).pack()
    tk.Checkbutton(root, text="Draw Connectors", variable=drawConnectors_).pack()
    tk.Checkbutton(root, text="Draw Blue Triangle Labels", variable=drawTriagLabelsBlue_).pack()
    tk.Checkbutton(root, text="Draw Blue Point Labels", variable=drawPointLabelsBlue_).pack()
    tk.Checkbutton(root, text="Draw Red Triangle Labels", variable=drawTriagLabelsRed_).pack()
    tk.Checkbutton(root, text="Draw Red Point Labels", variable=drawPointLabelsRed_).pack()
    tk.Checkbutton(root, text="Draw Paraboloid", variable=drawParaboloid_).pack()
    tk.Checkbutton(root, text="Use Default Points", variable=defaultPoints_).pack()
    tk.Checkbutton(root, text="Draw Axis", variable=axis_).pack()
    tk.Checkbutton(root, text="Print Debug Information", variable=debugInfo_).pack()
    tk.Checkbutton(root, text="Display Points Inside Circle", variable=highlightPointsInsideCircle_).pack()

    # Add entry fields for paraboloid strength
    paraboloidStrength_ = DoubleVar(value=0.2)
    tk.Label(root, text="Paraboloid Strength").pack()
    Entry(root, textvariable=paraboloidStrength_).pack()

    # Add read-only field for number of points
    numb_points_ = IntVar(value=50)
    tk.Label(root, text="Number of Points").pack()
    readonly_entry_numb_points = tk.Entry(root, textvariable=numb_points_, state='readonly')
    readonly_entry_numb_points.pack()

    # This is only used internally
    circle_numb_points_ = IntVar(value=50)

    # Add entry fields for the triangles to be flipped
    triangle_index1_ = IntVar(value=0)
    tk.Label(root, text="First Triangle").pack()
    Entry(root, textvariable=triangle_index1_).pack()

    triangle_index2_ = IntVar(value=0)
    tk.Label(root, text="Second Triangle").pack()
    Entry(root, textvariable=triangle_index2_).pack()

    # Add entry fields for the points that define circle
    circle_index1_ = IntVar(value=0)
    tk.Label(root, text="First Point").pack()
    Entry(root, textvariable=circle_index1_).pack()

    circle_index2_ = IntVar(value=0)
    tk.Label(root, text="Second Point").pack()
    Entry(root, textvariable=circle_index2_).pack()

    circle_index3_ = IntVar(value=0)
    tk.Label(root, text=">Third Point").pack()
    Entry(root, textvariable=circle_index3_).pack()

    #Initialise the Delaunay Triangulation
    tri_, points2D_, x_, y_, z_ = createDelaunay(defaultPoints=defaultPoints_.get(), numb_points=numb_points_.get())
    
    # Creating a 3D plot (a bit larger than default)
    fig_ = plt.figure(figsize=(20, 20))

    ax_ = fig_.add_subplot(111, projection='3d')

    # Function to call the visualization function with current GUI parameters
    def update_visualization(optional_function=None):

        #Save axis orientation before resetting
        axis = fig_.get_axes()[0]
        elev_, azim_ = axis.elev, axis.azim

        #Clear figure and add axis when updating
        fig_.clf()
        ax_ = fig_.add_subplot(111, projection='3d')

        #Restore orientation after resetting
        ax_.view_init(elev=elev_, azim=azim_)

        #Reset the points to highlight
        global highlight_points 
        highlight_points = []

        #Execute any optional functions here
        if optional_function is not None:
            optional_function(ax_)

        #This draws the delaunay triangulation according to the parameters
        """          Parameters                                                           Description                                              Category                              """                                                    
        drawDelaunay(tri = tri_,                                                          # triangulation object of plot                           TRIANGULATION
                     ax = ax_,                                                            # axis object of plot                                    AXIS
                     points2D = points2D_,                                                # Stacked x and y into Nx2 array                         POINTS
                     x = x_,                                                              #x coordinates of points                                 POINTS
                     y = y_,                                                              #y coordinates of points                                 POINTS
                     z = z_,                                                              #z coordinates of points z = x^2 + y^2                   POINTS
                     drawBlueTriag = drawBlueTriag_.get(),                                #Should Delaunay triangles be drawn?                     LIFTED TRIANGULATION
                     drawRedTriag = drawRedTriag_.get(),                                  #Should Delaunay triangles be drawn?                     NON LIFTED TRIANGULATION
                     drawBluePoints = drawBluePoints_.get(),                              #Should points be drawn?                                 LIFTED TRIANGULATION
                     drawRedPoints = drawRedPoints_.get(),                                #Should points be drawn?                                 NON LIFTED TRIANGULATION
                     drawConnectors = drawConnectors_.get(),                              #Should connecting lines for lifting be drawn?           LIFTING MAPPING
                     drawTriagLabelsBlue = drawTriagLabelsBlue_.get(),                    #Should labels for triangles be drawn                    LIFTED TRIANGULATION
                     drawPointLabelsBlue = drawPointLabelsBlue_.get(),                    #Should labels for points be drawn                       LIFTED TRIANGULATION
                     drawTriagLabelsRed = drawTriagLabelsRed_.get(),                      #Should labels for triangles be drawn                    NON LIFTED TRIANGULATION
                     drawPointLabelsRed = drawPointLabelsRed_.get(),                      #Should labels for points be drawn                       NON LIFTED TRIANGULATION
                     drawParaboloid = drawParaboloid_.get(),                              #Should paraboloid of lifting be drawn?                  LIFTING MAPPING
                     paraboloidStrength = paraboloidStrength_.get(),                      #Transparency of paraboloid 0=transparent, 1.0=opaque    LIFTING MAPPING
                     defaultPoints = defaultPoints_.get(),                                #Should the default points be chosen?                    POINTS
                     numb_points = numb_points_.get(),                                    #Number of points to be triangulated                     POINTS
                     axis = axis_.get(),                                                  #Should the axis be drawn?                               ACTUAL AXIS
                     debugInfo = debugInfo_.get(),                                        #Should there be debug information?                      DEBUGGING
                     circle_numb_points = circle_numb_points_.get(),                      #How many points should be in a "Delaunay" circle        NON-DELAUNAY TRIANGLES
                     highlightPointsInsideCircle = highlightPointsInsideCircle_.get())    #Show points inside "Delaunay" circle                    NON-DELAUNAY TRIANGLES


    def flipTriangles():
        #Flip the triangles internally
        flip(triangle_index1 = triangle_index1_.get(), 
             triangle_index2 = triangle_index2_.get(), 
             tri = tri_, 
             debugInfo = debugInfo_.get())

        #Partial function that makes sure orientation remains the same

        #Update figure to show changes
        update_visualization()

    def drawCircle2D():
        #Partially execute function and pass it to the update process
        circle_draw = lambda ax_param : drawCircle(index1 = circle_index1_.get(), 
                                                   index2 = circle_index2_.get(), 
                                                   index3 = circle_index3_.get(), 
                                                   circle_numb_points= circle_numb_points_.get(),
                                                   x = x_, 
                                                   y = y_, 
                                                   z = z_, 
                                                   ax = ax_param, # Will be added later
                                                   highlightPointsInsideCircle = highlightPointsInsideCircle_.get(),
                                                   debugInfo = debugInfo_.get())
         
        #Note how ax was not set as we will want to do this inside of the visualisation function
        update_visualization(circle_draw)

    # Button to update the visualization
    tk.Button(root, text="Update Visualization", command=update_visualization).pack()

    # Button to flip triangles
    tk.Button(root, text="Flip Triangles", command=flipTriangles).pack()

    # Button to draw circle
    tk.Button(root, text="Draw Circle", command=drawCircle2D).pack()

    # Start the GUI event loop
    root.mainloop()


#Start the program
visualize_delaunay()


    


