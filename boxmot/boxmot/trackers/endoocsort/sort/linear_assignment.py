# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

"""
Linear assignment using Hungarian algorithm for EndoOcSort.
"""

from __future__ import absolute_import
import numpy as np
from scipy.optimize import linear_sum_assignment


# Large cost for gating/invalid associations
INFTY_COST = 1e5


def min_cost_matching(distance_metric, max_distance, tracks, detections,
                     track_indices=None, detection_indices=None):
    """
    Solve linear assignment problem using Hungarian algorithm.
    
    Args:
        distance_metric: Callable that returns cost matrix
        max_distance: Maximum allowed distance
        tracks: List of Track objects
        detections: List of Detection objects
        track_indices: Indices of tracks to match
        detection_indices: Indices of detections to match
        
    Returns:
        (matches, unmatched_tracks, unmatched_detections)
    """
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))
    
    if len(detection_indices) == 0 or len(track_indices) == 0:
        return [], track_indices, detection_indices
    
    # Compute cost matrix
    cost_matrix = distance_metric(
        tracks, detections, track_indices, detection_indices
    )
    
    # Gate based on max_distance
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5
    
    # Solve assignment
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    
    # Extract matches
    matches, unmatched_tracks, unmatched_detections = [], [], []
    
    for col, detection_idx in enumerate(detection_indices):
        if col not in col_indices:
            unmatched_detections.append(detection_idx)
    
    for row, track_idx in enumerate(track_indices):
        if row not in row_indices:
            unmatched_tracks.append(track_idx)
    
    for row, col in zip(row_indices, col_indices):
        track_idx = track_indices[row]
        detection_idx = detection_indices[col]
        if cost_matrix[row, col] > max_distance:
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
        else:
            matches.append((track_idx, detection_idx))
    
    return matches, unmatched_tracks, unmatched_detections


def matching_cascade(distance_metric, max_distance, cascade_depth, tracks,
                    detections, track_indices=None, detection_indices=None):
    """
    Run matching cascade over different track ages.
    
    Args:
        distance_metric: Callable returning cost matrix
        max_distance: Maximum allowed distance
        cascade_depth: Maximum age to consider
        tracks: List of Track objects
        detections: List of Detection objects
        track_indices: Track indices
        detection_indices: Detection indices
        
    Returns:
        (matches, unmatched_tracks, unmatched_detections)
    """
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))
    
    unmatched_detections = detection_indices
    matches = []
    
    for level in range(cascade_depth):
        if len(unmatched_detections) == 0:
            break
        
        # Select tracks at this cascade level
        track_indices_l = [
            k for k in track_indices
            if tracks[k].time_since_update == 1 + level
        ]
        
        if len(track_indices_l) == 0:
            continue
        
        # Match at this level
        matches_l, _, unmatched_detections = min_cost_matching(
            distance_metric, max_distance, tracks, detections,
            track_indices_l, unmatched_detections
        )
        
        matches += matches_l
    
    # Remaining tracks are unmatched
    unmatched_tracks = list(set(track_indices) - set(k for k, _ in matches))
    
    return matches, unmatched_tracks, unmatched_detections


def gate_cost_matrix(kf, cost_matrix, tracks, detections, track_indices,
                    detection_indices, gated_cost=INFTY_COST,
                    only_position=False):
    """
    Invalidate cost matrix entries using Mahalanobis gating.
    
    Args:
        kf: Kalman filter instance
        cost_matrix: Cost matrix to gate
        tracks: List of Track objects
        detections: List of Detection objects
        track_indices: Track indices
        detection_indices: Detection indices
        gated_cost: Cost to assign for gated entries
        only_position: If True, only gate on position
        
    Returns:
        Gated cost matrix
    """
    gating_dim = 2 if only_position else 4
    gating_threshold = chi2inv95[gating_dim]
    
    measurements = np.asarray([detections[i].xysr for i in detection_indices])
    
    for row, track_idx in enumerate(track_indices):
        track = tracks[track_idx]
        gating_distance = track.gating_distance(
            measurements, only_position=only_position
        )
        cost_matrix[row, gating_distance > gating_threshold] = gated_cost
    
    return cost_matrix


# Chi-squared threshold for Mahalanobis gating
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919
}
