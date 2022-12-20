import argparse
from pathlib import Path
import numpy as np
import h5py
from scipy.io import loadmat
import torch
from tqdm import tqdm
import pickle
import cv2
import pycolmap

import struct
from os.path import join
import math
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation

from . import logger
from .utils.parsers import parse_retrieval, names_to_pair
from .utils.geometry import pose_matrix_from_qvec_tvec


def interpolate_scan(scan, kp):
    h, w, c = scan.shape
    kp = kp / np.array([[w-1, h-1]]) * 2 - 1
    assert np.all(kp > -1) and np.all(kp < 1)
    scan = torch.from_numpy(scan).permute(2, 0, 1)[None]
    kp = torch.from_numpy(kp)[None, None]
    grid_sample = torch.nn.functional.grid_sample

    # To maximize the number of points that have depth:
    # do bilinear interpolation first and then nearest for the remaining points
    interp_lin = grid_sample(
        scan, kp, align_corners=True, mode='bilinear')[0, :, 0]
    interp_nn = torch.nn.functional.grid_sample(
        scan, kp, align_corners=True, mode='nearest')[0, :, 0]
    interp = torch.where(torch.isnan(interp_lin), interp_nn, interp_lin)
    valid = ~torch.any(torch.isnan(interp), 0)

    kp3d = interp.T.numpy()
    valid = valid.numpy()
    return kp3d, valid


def get_scan_pose(dataset_dir, rpath):
    split_image_rpath = rpath.split('/')
    floor_name = split_image_rpath[-3]
    scan_id = split_image_rpath[-2]
    image_name = split_image_rpath[-1]
    building_name = image_name[:3]

    path = Path(
        dataset_dir, 'database/alignments', floor_name,
        f'transformations/{building_name}_trans_{scan_id}.txt')
    with open(path) as f:
        raw_lines = f.readlines()

    P_after_GICP = np.array([
        np.fromstring(raw_lines[7], sep=' '),
        np.fromstring(raw_lines[8], sep=' '),
        np.fromstring(raw_lines[9], sep=' '),
        np.fromstring(raw_lines[10], sep=' ')
    ])

    return P_after_GICP

def get_all_camera_poses(images_path):

    camera_poses = {}

    for pose_path in (images_path / 'poses/').iterdir():
        id = int(str(pose_path.name).split("_")[0])
        camera_poses[id] = {}

        with open(pose_path) as file:
            for line in file:
                #print(line)
                if line.__contains__("worldPose"):
                    t = line.rstrip()
                    t = t.replace(" ", "").split("worldPose")[1].replace("]", "").replace("[", "")
                    t = [float(tp) for tp in t.split(",")]
                    worldPose = np.array(t).reshape((4,4)).T
                    #worldPose = np.flip(worldPose.T, 1).astype(np.float64)
                    camera_poses[id]["worldPose"] = worldPose

                if line.__contains__("translation "):
                    t = line.rstrip()
                    t = t.replace(" ", "").split("translation")[1].replace("]", "").replace("[", "")
                    #print(t)
                    t = [float(tp) for tp in t.split(",")]
                    #print(t)
                    T = np.array(t)
                    T = T
                    camera_poses[id]["translation"] = T

                if line.__contains__("intrinsics"):
                    t = line.rstrip()
                    #print(t)
                    t = t.replace(" ", "").split("intrinsics")[1].replace("]", "").replace("[", "")
                    t = [float(tp) for tp in t.split(",")]
                    intrinsics = np.array(t).reshape((3,3))
                    camera_poses[id]["intrinsics"] = intrinsics
                    
                if line.__contains__("quartenion2"):
                    #print("Line contains quaternion2")
                    t = line.rstrip()
                    t = t.replace(" ", "").split("quartenion2")[1].replace("]", "").replace("[", "")
                    t = [float(tp) for tp in t.split(",")]
                    qvec = np.array(t)
                    #print(qvec)
                    camera_poses[id]["qvec"] = qvec
                    
                    
                if line.__contains__("translation2"):
                    t = line.rstrip()
                    t = t.replace(" ", "").split("translation2")[1].replace("]", "").replace("[", "")
                    t = [float(tp) for tp in t.split(",")]
                    tvec = np.array(t)
                    camera_poses[id]["tvec"] = tvec

    return camera_poses


def quaternion_rotation_matrix(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.

    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3)

    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix.
             This rotation matrix converts a point in the local reference
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]

    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)

    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)

    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1

    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])

    return rot_matrix


def pose_from_cluster(dataset_dir, q, retrieved, feature_file, match_file,
                      skip=None):
    print("FEATURE KEYS")
    print(q, feature_file[q].keys())
    images_path = dataset_dir / "images/"
    q_img = cv2.imread(str(images_path / q))
    height, width = q_img.shape[:2]
    cx = .5 * width
    cy = .5 * height
    #fov = 82 # oneplus
    fov = 82.1 # dji mini 3 pro
    focal_length = (width * 0.5) / np.tan(fov * 0.5 * np.pi/180) #Drone: (6.72 / 9.6) * width #4032. * 28. / 36. # 1432.0432 iphone
    #print(focal_length)
    #print(focal_length)

    all_mkpq = []
    all_mkpr = []
    all_mkp3d = []
    all_indices = []
    kpq = feature_file[q]['keypoints'].__array__()
    num_matches = 0

    for i, r in enumerate(retrieved):
        print("FEATURE KEYS")
        print(r, feature_file[r]["image_size"].__array__())
        kpr = feature_file[r]['keypoints'].__array__()
        pair = names_to_pair(q, r)
        print("MATCH KEYS")
        print(pair, match_file[pair]["matches0"].shape, match_file[pair]["matching_scores0"].shape)
        m = match_file[pair]['matches0'].__array__()
        print(m)
        v = (m > -1)

        if skip and (np.count_nonzero(v) < skip):
            print("skipping... ", r)
            continue

        mkpq, mkpr = kpq[v], kpr[m[v]]
        #print(len(mkpq), len(mkpr))
        #print(mkpr[:5])
        num_matches += len(mkpq)

        # Load depthmap as an image
        img_id = str(r)[:-5]
        img_id = int(img_id.split("_")[-1])

        # Get the pose from file
        #Tr = get_scan_pose(dataset_dir, r)
        pose_path = ""

        for p in (images_path / 'poses/').iterdir():
            id = int(str(p.name).split("_")[0])

            if id == img_id:
                pose_path = p
                break

        with open(pose_path) as file:
            for line in file:
                if line.__contains__("worldPose"):
                    t = line.rstrip()
                    t = t.replace(" ", "").split("worldPose")[1].replace("]", "").replace("[", "")
                    t = [float(tp) for tp in t.split(",")]
                    worldPose = np.array(t).reshape((4,4)).T
                    print(worldPose)

                if line.__contains__("translation "):
                    t = line.rstrip()
                    t = t.replace(" ", "").split("translation")[1].replace("]", "").replace("[", "")
                    t = [float(tp) for tp in t.split(",")]
                    T = np.array(t)
                    T = T

                if line.__contains__("intrinsics"):
                    t = line.rstrip()
                    t = t.replace(" ", "").split("intrinsics")[1].replace("]", "").replace("[", "")
                    t = [float(tp) for tp in t.split(",")]
                    intrinsics = np.array(t).reshape((3,3)).T
                    
                if line.__contains__("quaternion2"):
                    t = line.rstrip()
                    t = t.replace(" ", "").split("quaternion2")[1].replace("]", "").replace("[", "")
                    t = [float(tp) for tp in t.split(",")]
                    quaternion = np.array(t)
        #print(intrinsics, worldPose, T)


        #print(str(r), img_id)
        depthmap_path = ""
        for p in (images_path / 'depth/').iterdir():
            d_id = int(str(p)[:-4].split("_")[-1])
            if d_id is img_id:
                depthmap_path = p
                break

        confmap_path = ""
        for p in (images_path / 'confidence/').iterdir():
            c_id = int(str(p)[:-4].split("_")[-1])
            if c_id is img_id:
                confmap_path = p
                break
            #print(p)
        #print(depthmap_path, d_id)
        #scan_r = cv2.imread(str(depthmap_path))
        def load_depth(depth_name):
            with open(depth_name, mode='rb') as file:
                file_content = file.read()
            #header = file_content[:1024] # 1024 bit header
            #file_content = file_content[1024:]
            file_content = struct.unpack('f'* ((len(file_content)) // 4), file_content)
            depth = np.reshape(file_content, (192,256))
            depth = np.flip(depth.T, 1).astype(np.float64)
            return depth

        def load_conf(conf_name):
            with open(conf_name, mode='rb') as file:
                file_content = file.read()
            file_content = struct.unpack('B'* ((len(file_content))), file_content)
            conf = np.reshape(file_content, (192,256))
            conf = np.flip(conf.T, 1).astype(np.uint8)
            return conf

        scan_r = load_depth(depthmap_path)
        confmap = load_conf(confmap_path)
        print(depthmap_path, confmap_path)
        print(confmap.shape)

        #fig, axs = plt.subplots(1,2)
        #axs[0].imshow(scan_r)
        #print(scan_r.shape)
        #print(height, width)
        scan_h, scan_w = scan_r.shape

        intrinsics[0,0] = intrinsics[0,0] / (width/scan_w)
        intrinsics[1,1] = intrinsics[1,1] / (height/scan_h)

        intrinsics[0,2] = intrinsics[0,2] / (height/scan_h)
        intrinsics[1,2] = intrinsics[1,2] / (width/scan_w)

        flip_YZ = np.eye(4)
        flip_YZ[1,1] = -1
        flip_YZ[2,2] = -1

        worldPose = worldPose @ flip_YZ

        # Reshape rgb to the same size as depthmap
        #resized_q_img = cv2.resize(q_img, (scan_w, scan_h), interpolation=cv2.INTER_AREA)
        #newX = (currentX / currentWidth) * newWidth
        #newY = (currentY / currentHeight) * newHeight
        resized_kpr = np.array([np.array([int((kp[1] / height) * scan_h), int((kp[0] / width) * scan_w)]) for kp in mkpr])
        #axs[0].scatter(resized_kpr[:,0], resized_kpr[:,1], c='r', s=5)

        #axs[1].imshow(q_img)
        #axs[1].scatter(mkpq[:,0], mkpq[:,1], c='r', s=5)


        #plt.show()
        #kp = kp / np.array([[w-1, h-1]]) * 2 - 1

        #print(resized_scan_r.shape)
        #scan_r = loadmat(Path(dataset_dir, r + '.mat'))["XYZcut"]

        # Sample depth from the depthmap
        #mkp3d, valid = interpolate_scan(scan_r, mkpr)

        mkp3d=[]

        #print(resized_kpr)
        #print("MINMAX")
        #print(np.max(resized_kpr[:,0]), np.min(resized_kpr[:,0]))
        #print(np.max(resized_kpr[:,1]), np.min(resized_kpr[:,1]))

        valid = []

        for idx, kp in enumerate(resized_kpr):
            #print()
            #x_original, y_original = kp[0], kp[1]
            #print(x_original, y_original)
            #print(resized_kpr[i][0], resized_kpr[i][1])
            r_y, r_x = kp
            #r_y = scan_h - r_y - 1 # Convert from Bottom Left to Top Left
            #print(r_y, r_x)
            depth = -1*scan_r[r_y, r_x]
            confidence = confmap[r_y, r_x]
            confInt = confidence*255.0

            if np.abs(depth) < 0.05 or np.abs(depth) > 2.5 or confInt < 2:
                print("CONFIDENCE IS NOT MET")
                valid.append(False)
            else:
                valid.append(True)
            #print(depth)
            #depth = resized_scan_r[x_original,y_original,0]/255

            #print(r_y, r_x, -depth)

            #n_ry = (r_y - 0) / (256 - 0)
            #n_rx = (r_x - 0) / (192 - 0)
            #print(n_ry, n_rx)

            inv_int = np.linalg.inv(intrinsics)
            localpoint = (inv_int @ np.array([r_x, r_y, 1.0]).T) * (-1*depth)
            #print(localpoint)

            worldPoint_h = worldPose @ np.array([localpoint[0], localpoint[1], localpoint[2], 1.0]).T
            #print(worldPoint_h)
            worldPoint = np.array([worldPoint_h[0] / worldPoint_h[3], worldPoint_h[1] / worldPoint_h[3], worldPoint_h[2] / worldPoint_h[3]])
            #print(worldPoint)
            mkp3d.append(worldPoint)

            #print(depth)
        mkp3d = np.array(mkp3d)
        print(mkp3d.shape)

        #print(R + T)
        #print(T)
        # Project to world coordinates
        #print(mkp3d.shape, R.shape, T.shape)

        #if len(mkp3d) == 0:
        #    continue
        #mkp3d = ((R @ mkp3d.T).T + T)
        #print(mkp3d.shape)

        #all_mkpq.append(mkpq[valid])
        #all_mkpr.append(mkpr[valid])
        #all_mkp3d.append(mkp3d[valid])
        #all_indices.append(np.full(np.count_nonzero(valid), i))
        print("VALID")
        print(valid)
        all_mkpq.append(mkpq[valid])
        all_mkpr.append(mkpr[valid])
        all_mkp3d.append(mkp3d[valid])
        all_indices.append(np.full(np.count_nonzero(valid), i))

    if len(all_mkpq) < 1 or len(all_mkpr) < 0:
        return None, None, None, None, None, None
    all_mkpq = np.concatenate(all_mkpq, 0)
    all_mkpr = np.concatenate(all_mkpr, 0)
    all_mkp3d = np.concatenate(all_mkp3d, 0)
    all_indices = np.concatenate(all_indices, 0)

    print(len(all_mkpq), len(all_mkpr), len(all_mkp3d), len(all_indices))

    cfg = {
        'model': 'SIMPLE_RADIAL',
        'width': width,
        'height': height,
        'params': [focal_length, cx, cy, 0.0]
    }
    ret = pycolmap.absolute_pose_estimation(
        all_mkpq, all_mkp3d, cfg, 48.00)

    #print("\n\nret:", ret)
    #print(ret)

    print(q)
    rvar = [float(ret["qvec"][0]), float(ret["qvec"][1]), float(ret["qvec"][2]), float(ret["qvec"][3])]

    # rvar = [float(lvar[0]), float(lvar[1]), float(lvar[2]), float(lvar[2])]
    # Rotation = transformations.quaternion_matrix(rvar)
    # Translation = np.eye(4)
    # Translation[0][3] = float(lvar[4])
    # Translation[1][3] = float(lvar[5])
    # Translation[2][3] = float(lvar[6])
    # Tvar = np.matmul(-Rotation.transpose(),Translation)

    #print(pycolmap.quaternion_matrix(rvar))

    R = np.eye(4)
    R[0:3,0:3] = pycolmap.qvec_to_rotmat(rvar)
    t = np.eye(4)
    t[0][3] = float(ret["tvec"][0])
    t[1][3] = float(ret["tvec"][1])
    t[2][3] = float(ret["tvec"][2])
    #t.
    Rt = np.matmul(-R.transpose(), t) #-R.transpose()  t
    #Rt = pose_matrix_from_qvec_tvec(ret["qvec"], ret["tvec"]) #-R.transpose() @ t
    #print(f'x->to right, y->bottom, z->in front')
    #print(f"x: {Tvar[0]} y: {Tvar[1]} z: {Tvar[2]}")
    #Rt = np.zeros((4,4))
    #Rt = np.c_[R, Tvar.T]

    print(Rt)

    ret["Rt"] = Rt

    #print(quaternion_rotation_matrix(ret["qvec"]))
    ret['cfg'] = cfg
    return ret, all_mkpq, all_mkpr, all_mkp3d, all_indices, num_matches


def main(dataset_dir, retrieval, features, matches, results,
         skip_matches=None):

    assert retrieval.exists(), retrieval
    assert features.exists(), features
    assert matches.exists(), matches

    retrieval_dict = parse_retrieval(retrieval)
    queries = list(retrieval_dict.keys())
    print(queries)

    feature_file = h5py.File(features, 'r', libver='latest')
    match_file = h5py.File(matches, 'r', libver='latest')

    poses = {}
    logs = {
        'features': features,
        'matches': matches,
        'retrieval': retrieval,
        'loc': {},
    }
    logger.info('Starting localization...')

    found_poses=[]

    for q in tqdm(queries):
        db = retrieval_dict[q]
        ret, mkpq, mkpr, mkp3d, indices, num_matches = pose_from_cluster(
            dataset_dir, q, db, feature_file, match_file, skip_matches)

        if ret == None:
            continue
        found_poses.append(q)
        poses[q] = (ret['qvec'], ret['tvec'], ret["Rt"])
        logs['loc'][q] = {
            'db': db,
            'PnP_ret': ret,
            'keypoints_query': mkpq,
            'keypoints_db': mkpr,
            '3d_points': mkp3d,
            'indices_db': indices,
            'num_matches': num_matches,
        }
        print(logs['loc'][q]['indices_db'])

    logger.info(f'Writing poses to {results}...')
    with open(results, 'w') as f:
        for q in found_poses:
            qvec, tvec, Rt = poses[q]
            qvec = ' '.join(map(str, qvec))
            tvec = ' '.join(map(str, tvec))
            Rt = ' '.join(map(str, Rt.flatten()))
            name = q.split("/")[-1]
            f.write(f'{name} {qvec} {tvec}\n{Rt}\n')

    logs_path = f'{results}_logs.pkl'
    logger.info(f'Writing logs to {logs_path}...')
    with open(logs_path, 'wb') as f:
        pickle.dump(logs, f)
    logger.info('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=Path, required=True)
    parser.add_argument('--retrieval', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--matches', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--skip_matches', type=int)
    args = parser.parse_args()
    main(**args.__dict__)
