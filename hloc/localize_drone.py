import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import axes_grid
from pathlib import Path
import h5py
import argparse

import sys
sys.path.append('../')

from hloc import extract_features, match_features, visualization, pairs_from_exhaustive, localize_dma
from hloc.utils.parsers import parse_retrieval, names_to_pair
#from .utils.parsers import parse_retrieval, names_to_pair
#from .utils.geometry import pose_matrix_from_qvec_tvec


def calculate_px_in_meters(resolution, altitude, intrinsics):
    height, width = resolution
    corner_px = np.array([
        [0,0],
        [height, 0],
        [0, width],
        [height, width]
    ])

    point3d = np.ones((4,3))
    point3d[:, :2] = corner_px

    inv_I = np.linalg.inv(intrinsics)
    
    
    for i, p in enumerate(point3d):
    	point3d[i, :] = (inv_I @ p.T) * -altitude
    img_in_m = ((point3d[0,0] - point3d[1,0]), (point3d[0, 1] - point3d[2, 1]))
    px_ratio = (height / img_in_m[0], width / img_in_m[1])

    return px_ratio, img_in_m


def parse_altitude_from_name(img_path):
    return int(img_path.split("=")[-1].split("_")[0].split(".")[0])
    
def parse_img_index_from_name(img_path):
    img_id = str(img_path)[:-5]
    img_id = int(img_id.split("_")[-1])
    return img_id
    
def get_data_paths_from_name(images_dir, img_path):
    img_id = parse_img_index_from_name(img_path)
    
    pose_path=""
    depth_path=""
    conf_path=""
    
    for p in (images_dir / 'poses/').iterdir():
        pose_id = int(str(p.stem).split("_")[0])
        if pose_id == img_id:
            pose_path = p
            break
            
    for p in (images_dir / 'depth/').iterdir():
        d_id = int(str(p.stem).split("_")[-1])
        if d_id is img_id:
            depth_path = p
            break
            
    for p in (images_dir / 'confidence/').iterdir():
        c_id = int(str(p.stem).split("_")[-1])
        if c_id is img_id:
            conf_path = p
            break

    return pose_path, depth_path, conf_path
    
def calculate_sampling_resolution(img_in_m, query_px_ratio):
    return (int(img_in_m[0] * query_px_ratio[0]), int(img_in_m[1] * query_px_ratio[1]))

def downsample_image(img, resolution):
    resized_img = cv2.resize(img, (resolution[1], resolution[0]), interpolation = cv2.INTER_AREA)

    return resized_img

def save_images_to_path(imgs, img_names, path, query=False):
    new_img_names = []
    if query:
        pathName = Path(img_names[0])
        ext = pathName.suffix
        for i, img in enumerate(imgs):
            save_name = "tile_" + str(i) + ext
            new_img_names.append(save_name)
            cv2.imwrite(str(path / save_name), img)
    else:
        for i, img in enumerate(imgs):
            cv2.imwrite(str(path / img_names[i]), img)
            new_img_names.append(img_names[i])
    return new_img_names
    
    
def get_pose_from_file(pose_path):

    t = np.zeros((1,3))
    Rt = np.zeros((4,4))
    I = np.zeros((3,3))
    
    with open(pose_path) as file:
        for line in file:
            if line.__contains__("worldPose"):
                Rt = np.eye(4)
                rline = line.rstrip()
                rline = rline.replace(" ", "").split("worldPose")[1].replace("]", "").replace("[", "")
                rline = [float(tp) for tp in rline.split(",")]
                Rt[:3, :3] = np.array(rline).reshape((3,3)).T

            if line.__contains__("translation "):
                rline = line.rstrip()
                rline = rline.replace(" ", "").split("translation")[1].replace("]", "").replace("[", "")
                rline = [float(tp) for tp in rline.split(",")]
                t = np.array(rline)
                Rt[-1, :3] = t

            if line.__contains__("intrinsics"):
                rline = line.rstrip()
                rline = rline.replace(" ", "").split("intrinsics")[1].replace("]", "").replace("[", "")
                rline = [float(tp) for tp in rline.split(",")]
                I = np.array(rline).reshape((3,3))
                
    return t, Rt, I
    
def split_img_into_tiles(img, window_size, overlap):

    height, width, _ = list(img.shape)
    hTile, wTile = window_size 
    hTile = int(hTile)
    wTile = int(wTile)
    
    nTilesH = np.uint8(np.ceil(height / hTile))
    nTilesW = np.uint8(np.ceil(width / wTile))
    
    # Total remainders
    overlapW = int(wTile*overlap[1]) #nTilesW * wTile - width
    overlapH = int(hTile*overlap[0]) #nTilesH * hTile - height
    
    # Initialize array of tile boxes
    tiles = []#np.zeros((nTilesW * nTilesH, 4), np.uint16)

    # Determine proper tile boxes
    k = 0
    y = 0
    
    ncols=1
    nrows=1
    
    runCols=True
    runRows=True
    
    y=0
    while(runRows):
        x=0
        while(runCols):
            k+=1
            tileImg = img[int(y):int(y)+int(hTile), int(x):int(x)+int(wTile), :]
            tiles.append(tileImg)
            
            if x + wTile - overlapW + wTile < img.shape[1]:
                x = x + wTile - overlapW
                ncols+=1
            else:
                runCols = False
                break

        if y + hTile - overlapH + hTile < img.shape[0]:
            nrows+=1
            y = y + hTile - overlapH
            runCols = True
        else:
            runRows=False
            runCols=False
            break
    
    return tiles, nrows, int(np.ceil(ncols/nrows))
    
def visualize_tiles_and_original(original_img, tiles, nrows, ncols):
    f = plt.figure(figsize=(10, 3))
    ag = axes_grid.Grid(f, (1, 2, 1), (1, 1), axes_pad=(0.0,0.0), aspect=True)
    ag[0].imshow(original_img)

    ag = axes_grid.Grid(f, (1, 2, 2), (nrows, ncols), axes_pad=(0.0,0.0), aspect=True)

    for k in range(len(tiles)):
        tile = tiles[k]
        ag[k].imshow(tile/255)
        ag[k].set_xticks([])
        ag[k].set_yticks([])
    plt.subplots_adjust(wspace=0.1, hspace=0.0)
    plt.show(block=True)
    
    pass
    
    
def visualize_tiles_and_references(references, tiles, tnrows, tncols):
    init_ncols = 6
    
    if len(references) / init_ncols < 1:
        rncols=len(references) % init_ncols
        rnrows=1
    else:
        rncols=init_ncols
        rnrows=int(np.floor(len(references)/rncols)) + 1

    f = plt.figure(figsize=(10, 3))
    ag = axes_grid.Grid(f, (1, 2, 1), (rnrows, rncols), axes_pad=(0.0,0.0), aspect=True)
    extent = [0, references[0].shape[1], 0, references[0].shape[0]]
    for i, rimg in enumerate(references):
        ag[i].imshow(rimg)
        ag[i].set_xticks([])
        ag[i].set_yticks([])

    ag = axes_grid.Grid(f, (1, 2, 2), (tnrows, tncols), axes_pad=(0.0,0.0), aspect=True)

    for k in range(len(tiles)):
        tile = tiles[k]
        ag[k].imshow(tile/255)
        ag[k].set_xticks([])
        ag[k].set_yticks([])
    plt.subplots_adjust(wspace=0.1, hspace=0.0)
    plt.show(block=True)
    
def main(dataset_dir, query_dir, db_dir, images_dir, vis_input=False):
    
    # Define feature configuration
    feature_conf = extract_features.confs['superpoint_inloc']
    matcher_conf = match_features.confs['superglue']

    queries = [p.relative_to(query_dir).as_posix() for p in (query_dir).iterdir() if p.is_file()]
    references = [p.relative_to(db_dir).as_posix() for p in (db_dir).iterdir() if p.is_file()]
    
    # Hard code drone intrinsics, in the future, read from the file
    qI = np.eye(3)
    qI[0,0], qI[1,1], qI[0,2], qI[1,2] = 2204.82, 2204.82, 1080, 1920
    
    for q in queries[:5]:
        print(q)
        altitude = parse_altitude_from_name(q)
        
        qimg = cv2.imread(str(query_dir / q))
        qheigth, qwidth, _ = qimg.shape
        q_resolution = (qheigth, qwidth)
        
        q_px_ratio, q_size_in_m = calculate_px_in_meters(resolution=q_resolution,
                                                         altitude=altitude,
                                                         intrinsics=qI)
                                                         
        
        # Read pose, depth, conf paths of refernece images
        # Calculate new size in meters
        # Downsample the image into the same relative resolution as drone image                            
        resized_references = []
                                    
        for r in references:
            rimg = cv2.imread(str(db_dir / r))
            rheigth, rwidth, _ = rimg.shape
            r_resolution = (rheigth, rwidth)
            
            pose_path, depth_path, conf_path = get_data_paths_from_name(images_dir, r)
            
            rt, rRt, rI = get_pose_from_file(pose_path)
            
            
            _, r_size_in_m = calculate_px_in_meters(resolution=r_resolution,
                                                    altitude=1,
                                                    intrinsics=rI)
                                                    
            new_resolution = calculate_sampling_resolution(img_in_m=r_size_in_m, query_px_ratio=q_px_ratio)
            resized_rimg = downsample_image(img=rimg, resolution=new_resolution)
            resized_references.append(resized_rimg)
        
        # Split the drone image into overlapping tiles using a mean resolution of reference images
        mean_resolution = np.mean([ r.shape[:2] for r in resized_references], axis=0)
        splitted_qimg, nrows, ncols = split_img_into_tiles(img=qimg, window_size=mean_resolution, overlap=(0.1,0.1))
        
        # Save the splitted images and reference images as new dataset
        tile_db_dir=dataset_dir / "images2" / Path(q).stem / "mapping"
        tile_query_dir=dataset_dir / "images2" / Path(q).stem / "query"
        
        tile_db_dir.mkdir(parents=True, exist_ok=True)
        tile_query_dir.mkdir(parents=True, exist_ok=True)
        
        # Save new references
        new_references = save_images_to_path(imgs=resized_references, img_names=references, path=tile_db_dir)
        new_query = save_images_to_path(imgs=splitted_qimg, img_names=[q], path=tile_query_dir, query=True)
        
        
        print(new_references)
        print(new_query)
        
        # Calculate the feature points of new references
        output_path = dataset_dir / "images2" / Path(q).stem / "outputs"
        feature_path=output_path / 'features.h5'
        output_path.mkdir(parents=True, exist_ok=True)
        features = extract_features.main(feature_conf, tile_db_dir, 
                      image_list=new_references, 
                      feature_path=feature_path)
        
        # Calcualte the feature points of new query
        features = extract_features.main(feature_conf, tile_query_dir, 
                     image_list=new_query, 
                     feature_path=feature_path)
                     
        # Calculate pairs
        loc_pairs_path = output_path / 'pairs-query.txt'
        pairs_from_exhaustive.main(loc_pairs_path, image_list=new_query, ref_list=new_references)
        
        
        # Match features
        matches_path = output_path / 'matches.h5'
        matches = match_features.main(matcher_conf, loc_pairs_path, features=features, matches=matches_path)
        
        # visualize tiles and downsampled reference images
        if vis_input:
            visualize_tiles_and_references(references=resized_references, tiles=splitted_qimg, tnrows=nrows, tncols=ncols)
            visualize_tiles_and_original(original_img=qimg, tiles=splitted_qimg, nrows=nrows, ncols=ncols)
        
        # Perform localization
        results_path = output_path / 'results.txt'
        localize_dma.main(dataset_dir=dataset_dir, retrieval=loc_pairs_path, images_path=dataset_dir / "images2" / Path(q).stem, features=features, matches=matches, results=results_path, skip_matches=20)
        
        visualization.visualize_loc(results=results_path, image_dir=dataset_dir / "images2" / Path(q).stem / 'query', db_image_dir=dataset_dir / "images2" / Path(q).stem / 'mapping', n=len(splitted_qimg), top_k_db=1, seed=2)
        
        #match_file = h5py.File(str(matches_path), 'r', libver='latest')
        #pair = names_to_pair(new_query[0], new_references[0])
        #matches = match_file[pair]['matches0'].__array__()
        
        #print(matches)
        
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=Path, required=True)
    parser.add_argument('--query_dir', type=Path, required=True)
    parser.add_argument('--db_dir', type=Path, required=True)
    parser.add_argument('--images_dir', type=Path, required=True)
    parser.add_argument('--vis_input', type=bool, required=False, default=False)
    args = parser.parse_args()
    main(**args.__dict__)
        
        
    
    


