import cv2 
import numpy as np
import glob
import os

#def calib(path="/Users/kyo/Documents/projects/CVision/SfM/calibration", showPix=True):
def calib(folder_path, showPix=True):
    # corners of the square blocks (vertical and horizontal)
    Ch_Dim = (8, 6)
    Sq_size = 24  #milimeters
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) 

    #  add the 3D and 2D points
    # World points 
    obj_3D = np.zeros((Ch_Dim[0] * Ch_Dim[1], 3), np.float32)
    obj_3D[:, :2] = np.mgrid[0:Ch_Dim[0], 0:Ch_Dim[1]].T.reshape(-1, 2)
    
    #print(obj_3D)
    obj_points_3D = []  # 3d point in real world space
    img_points_2D = []  # 2d points in image plane


    #image_files = glob.glob(os.path.join(path, '/*.JPG'))
    image_files = glob.glob(f"{folder_path}/*.JPG")
    print("Images trouvées :", len(image_files))
    img_size = None
    
    cnt=0
    for image in image_files:
        #print(image)
        
        imgBGR = cv2.imread(image)
        if imgBGR is None:
            print(f"Erreur lecture: {imgBGR}")
            continue
        
        imgGRAY = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2GRAY)

        #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(imgGRAY, Ch_Dim, None)
        
        if ret == True:
            obj_points_3D.append(obj_3D)
            corners2 = cv2.cornerSubPix(imgGRAY, corners, (11, 11), (-1, -1), criteria)
            img_points_2D.append(corners2)

            if showPix:    
                #img_size = imgGRAY.shape[::-1] #sauvegarde la taille
                img_corners = cv2.drawChessboardCorners(imgBGR, Ch_Dim, corners2, ret)
                #cv2.imwrite(f"calibration_corners/chessboard_{cnt}.jpg", img_corners)
                cnt+=1
                #cv2.imshow('Chessboard', img_corners)
                #cv2.waitKey(50)
            #cv2.destroyAllWindows()
    
    if len(obj_points_3D) == 0:
        raise ValueError("Aucun damier détecté. Vérifie tes images.")
            
    # Calibration
    ret, mtx, dist_coeff, R_vecs, T_vecs = cv2.calibrateCamera(obj_points_3D, img_points_2D, imgGRAY.shape[::-1], None, None)
    print("calibrated")
    
    #Save calibration parameters
    param_path = "/Users/kyo/Documents/projects/CVision/SfM/calibrated_data/calibrated_data.npz"
    np.savez(
        param_path,
        repError=ret,
        camMatrix=mtx,
        distCoeff=dist_coeff,
        R_vecs=R_vecs,
        T_vecs=T_vecs
    )

    return ret, mtx, dist_coeff #, R_vecs, T_vecs

#def run_calibration():
#calib()
    
#run_calibration()

'''
def calib(folder_path, chessboard_dim=(8,6), square_size=24):
    """
    Return :
        K
        dist
        reprojection error
    """

    obj_points = []
    img_points = []
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) 

    objp = np.zeros((chessboard_dim[0]*chessboard_dim[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_dim[0], 0:chessboard_dim[1]].T.reshape(-1,2)
    objp *= square_size

    images = glob.glob(folder_path + "/*.JPG")

    gray = None

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, chessboard_dim, None)

        if ret:
            obj_points.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            img_points.append(corners)
            
            #Draw corners
            img_size = gray.shape[::-1] #save size
            img_corners = cv2.drawChessboardCorners(img, chessboard_dim, corners2, ret)
            cv2.imshow("Corners", img_corners)
            #cv2.imshow('Chessboard', imgBGR)
            cv2.waitKey(50)
        cv2.destroyAllWindows()
        

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points,
        img_points,
        gray.shape[::-1],
        None,
        None
    )

    return ret, K, dist'''