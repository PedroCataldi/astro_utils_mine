import numpy as np


############## Get Rotation Matrix ##############
    
def rotador_mio(x, y, z, vx, vy, vz):

    Jx_total = vx
    Jy_total = vy
    Jz_total = vz
    
    Jmod = np.sqrt(np.square(Jx_total)+np.square(Jy_total)+np.square(Jz_total))
    Jmodxy = np.sqrt(np.square(Jx_total)+np.square(Jy_total))
    Jmodxz = np.sqrt(np.square(Jx_total)+np.square(Jz_total))
    Jmodyz = np.sqrt(np.square(Jy_total)+np.square(Jz_total))
    
    # Ángulos de Euler
    sentita=Jmodxy/Jmod
    costita=Jz_total/Jmod
    senfi=Jx_total/Jmodxy
    cosfi=-Jy_total/Jmodxy
    sensi=0
    cossi=1
     
    # Matriz de rotación
    a11=cossi*cosfi-costita*sensi*senfi
    a12=cossi*senfi+costita*cosfi*sensi
    a13=sensi*sentita
    a21=-sensi*cosfi-costita*senfi*cossi
    a22=-sensi*senfi+costita*cosfi*cossi
    a23=cossi*sentita
    a31=sentita*senfi
    a32=-sentita*cosfi
    a33=costita
     
    return a11, a12, a13, a21, a22, a23, a31, a32, a33
    
############## Vel and Post Cartesian to Spherical ##############

def cartesian_to_spherical( x, y, z, v_x, v_y, v_z):

    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    theta = np.arctan2(hxy, z)
    phi = np.arctan2(y, x)
    n1 = x ** 2 + y ** 2
    n2 = n1 + z ** 2
    v_r = (x * v_x + y * v_y + z * v_z) / np.sqrt(n2)
    v_th = (z * (x * v_x + y * v_y) - n1 * v_z) / (n2 * np.sqrt(n1))
    v_p = -1 * (v_x * y - x * v_y) / n1

    return r, theta, phi, v_r, v_th, v_p

############## Positions from Cartesian to Spherical ##############

def appendSpherical_np(xyz):
    ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
    xy = xyz[:,0]**2 + xyz[:,1]**2
    ptsnew[:,3] = np.sqrt(xy + xyz[:,2]**2)
    #ptsnew[:,4] = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
    ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    ptsnew[:,5] = np.arctan2(xyz[:,1], xyz[:,0])
    return ptsnew
    
############## Convert angle to range form [-pi,pi] to [0,2pi] ##############

def convert_angle_to_0_2pi_interval(angle):
    new_angle = np.arctan2(np.sin(angle), np.cos(angle))
    if (new_angle < 0):
        new_angle = abs(new_angle) + 2 * (np.pi - abs(new_angle))
    return new_angle
    
############## Get Rotation Matrix ##############

def rotation_matrix_from_vectors(vec1, vec2):
    
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix
############## Rotate given the Rotation Matrix ##############

def rotation_matrix(axis, theta):
    costheta = np.cos(theta)
    sintheta = np.sin(theta)

    ux = axis[0]
    uy = axis[1]
    uz = axis[2]

    a = costheta + ux ** 2 * (1 - costheta)
    b = ux * uy * (1 - costheta) - uz * sintheta
    c = ux * uz * (1 - costheta) + uy * sintheta

    d = uy * ux * (1 - costheta) + uz * sintheta
    e = costheta + uy ** 2 * (1 - costheta)
    f = uy * uz * (1 - costheta) - ux * sintheta

    g = uz * ux * (1 - costheta) - uy * sintheta
    h = uz * uy * (1 - costheta) + ux * sintheta
    i = costheta + uz ** 2 * (1 - costheta)

    arr = np.array([[a, b, c], [d, e, f], [g, h, i]])

    return arr

def get_rotation_matrix(L):

    L_dir = L / np.linalg.norm(L)

    axZ = np.array([0, 0, 1])

    ort = np.cross(L_dir, axZ)
    ort = ort / np.linalg.norm(ort)

    costheta = np.dot(L_dir, axZ)
    theta = np.arccos(costheta)

    R = rotation_matrix(ort, theta)

    return R

