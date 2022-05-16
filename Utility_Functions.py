


def getOrientation(pts, img):
    
    # check in case there is only one point for the contour
    # if so do not process that contour 
        
    Major_Axis = 0
    Minor_Axis = 0 
    
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i,0] = pts[i,0,0]
        data_pts[i,1] = pts[i,0,1]
    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
    # The center of the object
    cntr = (int(mean[0,0]), int(mean[0,1]))
    
    # For illustrative purposes, create colored lines to indicate the 
    # major and minor principal components 
    #cv2.circle(img, cntr, 3, (255, 0, 255), 2)
    # Major axis
    p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
    # Minor axis 
    p2 = (cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0])
    angle = math.atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
    #self.drawAxis(img, cntr, p1, (0, 255, 0), 100)
    #self.drawAxis(img, cntr, p2, (255, 255, 0), 50)
    # Now create a mask of the object contour in order to calculate 
    # the major and minor feret diameters
    object_mask = np.zeros(shape=img.shape, dtype=np.uint8)
    cv2.drawContours(object_mask, pts, -1, (255, 255, 255), 1)
    P1 = (cntr[0] + 10 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 10 * eigenvectors[0,1] * eigenvalues[0,0])
    P2 = (cntr[0] - 10 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - 10 * eigenvectors[1,1] * eigenvalues[1,0])
    P3 = (cntr[0] - 10 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] - 10 * eigenvectors[0,1] * eigenvalues[0,0])
    P4 = (cntr[0] + 10 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] + 10 * eigenvectors[1,1] * eigenvalues[1,0])
    line_mask = np.zeros(shape=img.shape, dtype=np.uint8)
    drawAxis(line_mask, cntr, P2, (255, 255, 255), 1000)
    matches = np.logical_and( object_mask, line_mask )
    a = np.where(matches == True)
    a = np.asarray(a)
    a = a.transpose()
    a = np.asarray(a)
    line_mask = np.zeros(shape=img.shape, dtype=np.uint8)
    drawAxis(line_mask, cntr, P4, (255, 255, 255), 1000)
    matches = np.logical_and( object_mask, line_mask )
    b = np.where(matches == True)
    b = np.asarray(b)
    b = b.transpose()
    b = np.asarray(b)
    
    if len(a) > 0 and len(b) > 0:
        Minor_Axis = np.min(dist.cdist(a,b)) 
    
    line_mask = np.zeros(shape=img.shape, dtype=np.uint8)
    drawAxis(line_mask, cntr, P1, (255, 255, 255), 1000)
    matches = np.logical_and( object_mask, line_mask )
    a = np.where(matches == True)
    a = np.asarray(a)
    a = a.transpose()
    a = np.asarray(a)
        
    line_mask = np.zeros(shape=img.shape, dtype=np.uint8)
    drawAxis(line_mask, cntr, P3, (255, 255, 255), 1000)
    matches = np.logical_and( object_mask, line_mask )
    b = np.where(matches == True)
    b = np.asarray(b)
    b = b.transpose()
    b = np.asarray(b)
    
    if len(a) > 0 and len(b) > 0:
        Major_Axis = np.min(dist.cdist(a,b)) 
    
        
    #drawAxis(object_mask, cntr, P1, (255, 255, 255), 1000)
    #drawAxis(object_mask, cntr, P2, (255, 255, 255), 1000)
    #drawAxis(object_mask, cntr, P3, (255, 255, 255), 1000)
    #drawAxis(object_mask, cntr, P4, (255, 255, 255), 1000)

    #cv2.circle(object_mask, [x[0,0],x[0,1]], 3, (255, 0, 255), 2)

    #drawAxis(object_mask, cntr, p1, (255, 255, 255), 1000)
    #object_mask = Image.fromarray(object_mask)
    #line_mask = Image.fromarray(line_mask)
    
    #object_mask.paste(line_mask, (0, 0), line_mask)

 #   cv2.imshow("Image", object_mask)

#    cv2.waitKey(0)

    return cntr, angle, Major_Axis, Minor_Axis


def drawAxis(img, p_, q_, colour, scale):
    p = list(p_)
    q = list(q_)
    
    angle = math.atan2(p[1] - q[1], p[0] - q[0]) # angle in radians
    hypotenuse = np.sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
    
    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * math.cos(angle)
    q[1] = p[1] - scale * hypotenuse * math.sin(angle)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    
    # create the arrow hooks
    p[0] = q[0] + 9 * math.cos(angle + math.pi / 4)
    p[1] = q[1] + 9 * math.sin(angle + math.pi / 4)
    
    #cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    p[0] = q[0] + 9 * math.cos(angle - math.pi / 4)
    p[1] = q[1] + 9 * math.sin(angle - math.pi / 4)
      
    
    

def segmented_images_percents(image): 
    
    df = pd.DataFrame(columns = 
                      ['Class Number',
                      'Percent']  )   
    class_number_list = list( np.unique(image) )
    
    try: 
       
        # If the input is a file path rather than array, load the file 
        if os.path.isfile(image):
            image = np.load(image)
            
        # Check to see if either the loaded file or input array is a greyscale image 
        if type(image)  == np.ndarray and image.ndim != 2:
            exit()
            
        # Create an empty dataframe to hold the output 
        #column_names = ['Laplacian', ]
        
        for i, class_number in enumerate(class_number_list):
            # Python does not allow for two sided thresholding, therefore we 
            # have to create two one-sided images corresponding to the upper and lower 
            # thresholds and then create a final image of the difference
            # As a note, OpenCV bins to the right (upper) values when thresholding 
            #image = np.array(image, dtype=np.uint8)
            df.at[i, 'Class Number'] = class_number
            df.at[i, 'Percent'] = len(image[image == class_number].flat) / len(image.flat)
            
    except:
        print('Error encoutered. Measurments may be incomplete')
        return df
    else:
        print('Completed measurments')
        return df
    
    
    
    
def segmented_images_measurements(image, class_number_list, class_name_list):

    try: 

        # If the input is a file path rather than array, load the file 
        if os.path.isfile(image):
            image = np.load(image)
            
        # Check to see if either the loaded file or input array is a greyscale image 
        if type(image)  == np.ndarray and image.ndim != 2:
            exit()
            
        if (image.dtype != np.int8) or (image.dtype != np.uint8): 
            image = np.array(image, dtype = np.uint8)
            
        # Create an empty dataframe to hold the output 
        df = pd.DataFrame(columns = 
                      ['Intra-Image Object Index', 
                      'Class Number',
                      'Class Name',
                      'Area (px)', 
                      'X Centroid (px)', 
                      'Y Centroid (px)', 
                      'Orientation Angle (degree)', 
                      'Bounding Box A (px)', 
                      'Bounding Box B (px)', 
                      'Aspect Ratio'] )   
    
        for i, class_number in enumerate(class_number_list):
            print("Class: " + str(class_number) )
            # Python does not allow for two sided thresholding, therefore we 
            # have to create two one-sided images corresponding to the upper and lower 
            # thresholds and then create a final image of the difference
            # As a note, OpenCV bins to the right (upper) values when thresholding 
            #image = np.array(image, dtype=np.uint8)
            _, mask_lower = cv2.threshold(image, class_number, 255, type=cv2.THRESH_BINARY)
            _, mask_upper = cv2.threshold(image, class_number + 1, 255, type=cv2.THRESH_BINARY)
            greyscale = np.array(mask_lower - mask_upper, dtype=np.uint8)
            # Get the contours of each object 
            contours, hierarchy = cv2.findContours(greyscale, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) >=1:
                hierarchy = hierarchy[0]
                for z, contour in enumerate(contours):
                    if (contour.shape[0] >= 2) and (cv2.contourArea(contour) > 3 ): 
                        print("Object: " + str(z) )
                        # Append an empty row for each contour 
                        df.append(pd.Series(name= z ) )
                        # Calculate the centroids, orientation angle, diamters 
                        #cntr, angle, major, minor = getOrientation(contour, greyscale.copy() )
                        
                        """
                        # Calculate the skeletal distance 
                        blank = np.zeros(greyscale.shape, dtype = np.uint8)
                        blank = cv2.drawContours(blank, contours, z, (255,255,255), -1)
                        skeleton = cv2.ximgproc.thinning(blank, blank) # skeletonization 
                        skeletal_distance = len(skeleton[skeleton != 0 ])
                        
                        fig, ax = plt.subplots(figsize = (size, size)) 
                        plt.imshow(blank, cmap = 'Greys')
                        plt.savefig("Skeleton.png") 
                        plt.show()
                        """

                        # Begin appending data to the dataframe 
                        df.at[z, 'Intra-Image Object Index'] = z
                        df.at[z, 'Class Number'] = class_number
                        if bool(class_name_list[i] != None):
                            df.at[z, 'Class Name'] = class_name_list[i]
                        else:
                            df.at[z, 'Class Name'] = None
                        df.at[z, 'Area (px)'] = cv2.contourArea(contour) # Calculate the area of each contour
                        rect = cv2.minAreaRect(contour)
                        centroid = rect[0]
                        major, minor = rect[1]
                        angle = rect[2]
                        df.at[z, 'X Centroid (px)'] = centroid[0]
                        df.at[z, 'Y Centroid (px)'] = centroid[1]
                        df.at[z, 'Orientation Angle (degree)'] = angle
                        df.at[z, 'Bounding Box A (px)'] = major
                        df.at[z, 'Bounding Box B (px)'] = minor
                        df.at[z, 'Aspect Ratio'] = max(major, minor) / min(major, minor)
                        
                        #df.at[z, 'Skeletal Distance (px)'] = skeletal_distance
                        # Future work
                        # bool indicator of whether the object / contour touches the edge of the image 
                        #if (hierarchy[i][2] >= 0):
                        #    contour_closed = True
                        #else: 
                        #    contour_closed = False
    except:
        print('Error encoutered. Measurments may be incomplete')
        return df
    else:
        print('Completed measurments')
        return df


