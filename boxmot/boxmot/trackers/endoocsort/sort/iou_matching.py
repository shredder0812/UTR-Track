# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

"""
IOU matching for EndoOcSort tracker.
"""

from __future__ import absolute_import
import numpy as np
from . import linear_assignment


def iou_xyxy(bbox, candidates):
    """
    Compute intersection over union in XYXY format.
    
    Args:
        bbox: Single bbox [x1, y1, x2, y2]
        candidates: Array of bboxes [[x1, y1, x2, y2], ...]
        
    Returns:
        Array of IOU scores
    """
    bbox = np.asarray(bbox).flatten()
    candidates = np.asarray(candidates)
    
    if len(candidates) == 0:
        return np.array([])
    
    # Ensure 2D
    if candidates.ndim == 1:
        candidates = candidates.reshape(1, -1)
    
    # Intersection
    tl_x = np.maximum(bbox[0], candidates[:, 0])
    tl_y = np.maximum(bbox[1], candidates[:, 1])
    br_x = np.minimum(bbox[2], candidates[:, 2])
    br_y = np.minimum(bbox[3], candidates[:, 3])
    
    intersection_w = np.maximum(0.0, br_x - tl_x)
    intersection_h = np.maximum(0.0, br_y - tl_y)
    area_intersection = intersection_w * intersection_h
    
    # Union
    bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    candidates_area = (candidates[:, 2] - candidates[:, 0]) * (candidates[:, 3] - candidates[:, 1])
    area_union = bbox_area + candidates_area - area_intersection
    
    iou = area_intersection / (area_union + 1e-6)
    return iou


def iou_cost(tracks, detections, track_indices=None, detection_indices=None):
    """
    Compute IOU cost matrix between tracks and detections.
    
    Args:
        tracks: List of Track objects
        detections: List of Detection objects
        track_indices: Indices of tracks to match
        detection_indices: Indices of detections to match
        
    Returns:
        Cost matrix of shape (len(track_indices), len(detection_indices))
        where entry (i, j) is 1 - IOU
    """
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))
    
    cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
    
    for row, track_idx in enumerate(track_indices):
        # Skip tracks that haven't been updated recently
        if tracks[track_idx].time_since_update > 1:
            cost_matrix[row, :] = linear_assignment.INFTY_COST
            continue
        
        # Get track bbox in XYXY
        track_bbox = tracks[track_idx].to_xyxy()
        
        # Get detection bboxes
        det_bboxes = np.array([detections[i].xyxy for i in detection_indices])
        
        # Compute IOU and convert to cost
        ious = iou_xyxy(track_bbox, det_bboxes)
        cost_matrix[row, :] = 1.0 - ious
    
    return cost_matrix


def asso_with_velocity(tracks, detections, velocities, k_observations, 
                       inertia=0.2, w=640, h=480):
    """
    OcSort-style association incorporating velocity.
    
    Args:
        tracks: Predicted track positions (N x 4)
        detections: Detection positions (M x 4)
        velocities: Track velocities (N x 2)
        k_observations: Previous observations (N x 4)
        inertia: Weight for velocity component
        w, h: Frame dimensions
        
    Returns:
        Cost matrix with velocity compensation
    """
    N = len(tracks)
    M = len(detections)
    
    if N == 0 or M == 0:
        return np.zeros((N, M))
    
    # Compute basic IOU cost
    cost_matrix = np.zeros((N, M))
    for i in range(N):
        ious = iou_xyxy(tracks[i], detections)
        cost_matrix[i, :] = 1.0 - ious
    
    # Add velocity component
    if velocities is not None and len(velocities) > 0:
        for i in range(N):
            if k_observations[i].sum() < 0:  # No previous observation
                continue
                
            vel = velocities[i]
            if np.linalg.norm(vel) < 0.01:  # Negligible velocity
                continue
            
            # Predict next position based on velocity
            for j in range(M):
                det_center = np.array([
                    (detections[j][0] + detections[j][2]) / 2,
                    (detections[j][1] + detections[j][3]) / 2
                ])
                track_center = np.array([
                    (tracks[i][0] + tracks[i][2]) / 2,
                    (tracks[i][1] + tracks[i][3]) / 2
                ])
                
                # Compute velocity-based distance
                diff = det_center - track_center
                diff_norm = diff / (np.linalg.norm(diff) + 1e-6)
                
                # Velocity agreement (cosine similarity)
                vel_agree = np.dot(vel, diff_norm)
                
                # Add to cost (negative agreement increases cost)
                cost_matrix[i, j] += inertia * (1.0 - vel_agree)
    
    return cost_matrix
