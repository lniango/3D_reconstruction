import cv2 
import numpy as np
import glob

def calib(showPix=True):
    # corners of the square blocks (vertical and horizontal)
    Ch_Dim = (8, 6)
    Sq_size = 24  #milimeters
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) 

    #  add the 3D and 2D points
    # World points 
    obj_3D = np.zeros((Ch_Dim[0] * Ch_Dim[1], 3), np.float32)
    obj_3D[:, :2] = np.mgrid[0:Ch_Dim[0], 0:Ch_Dim[1]].T.reshape(-1, 2)
    '''index = 0
    for i in range(Ch_Dim[0]):
        for j in range(Ch_Dim[1]):
            obj_3D[index][0] = j * Sq_size
            obj_3D[index][1] = i * Sq_size
            index += 1'''
    #print(obj_3D)
    obj_points_3D = []  # 3d point in real world space
    img_points_2D = []  # 2d points in image plane


    image_files = glob.glob("/Users/kyo/Documents/projects/CVision/SfM/calibration/*.JPG")
    print("Images trouvées :", len(image_files))
    img_size = None
    
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
                img_size = imgGRAY.shape[::-1] #sauvegarde la taille
                cv2.drawChessboardCorners(imgBGR, Ch_Dim, corners2, ret)
                #cv2.imwrite("projects/CVision/SfM/calibrated_data/chessboard.jpg", img)
                #cv2.imshow('Chessboard', imgBGR)
                #cv2.waitKey(50)
   # cv2.destroyAllWindows()
    
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

    return ret, mtx, dist_coeff, R_vecs, T_vecs

def run_calibration():
    calib()
    
run_calibration()