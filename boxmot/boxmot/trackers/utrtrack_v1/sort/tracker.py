# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license
"""Multi-target association and track management of UTR-Track-v1."""

from __future__ import annotations

import numpy as np

from boxmot.trackers.utrtrack.sort import iou_matching, linear_assignment
from boxmot.trackers.utrtrack_v1.sort.track import TrackV1


class TrackerV1:
    """Association loop of UTR-Track-v1.

    Per frame (after :meth:`predict`):

    1. **Three-tier association** (:meth:`_match`)
       A. appearance cascade with a chi^2 Mahalanobis gate in XYSR space,
          confirmed tracks vs high-confidence detections (``conf >= byte_low_conf_thresh``);
       B. IoU matching of tentative / just-missed tracks vs remaining high-confidence detections;
       C. IoU matching of still-unmatched confirmed tracks vs low-confidence detections
          (BYTE-style; low-confidence detections never start a track).
    2. **Transfer learning** for every unmatched track (Source -> Primary, see :class:`TrackV1`).
    3. **Re-acquisition** of lost confirmed tracks (Hungarian on an appearance/IoU cost,
       independent of the Mahalanobis gate, which keeps growing while a track is lost).
    4. **Identity-preserving birth control** for every remaining high-confidence detection
       (:meth:`_initiate_track`): a detection never replaces a live track.
    5. **Lost-track absorption** (:meth:`_absorb_lost_tracks`).
    6. Deleted tracks are dropped and the appearance gallery is refreshed.

    No step caps the number of tracks: two separated lesions keep two identities.
    """

    def __init__(
        self,
        metric,
        max_iou_dist: float = 0.9,
        max_age: int = 500,
        n_init: int = 3,
        ema_alpha: float = 0.95,
        mc_lambda: float = 0.98,
        reacquire_min_iou: float = 0.01,
        reacquire_max_cost: float = 0.75,
        reacquire_max_age: int = 500,
        transfer_stale_gap: int = 4,
        byte_low_conf_thresh: float = 0.45,
        byte_max_iou_dist: float | None = None,
        q_calibration_scale: float | None = 1.0,
        absorption_iou: float = 0.3,
        absorption_app_sim: float = 0.7,
        birth_iou: float = 0.3,
        birth_contain: float | None = 0.8,
    ):
        self.metric = metric
        self.max_iou_dist = max_iou_dist
        self.max_age = max_age
        self.n_init = n_init
        self.ema_alpha = ema_alpha
        self.mc_lambda = mc_lambda
        self.tracks: list[TrackV1] = []
        self._next_id = 1

        self.reacquire_min_iou = reacquire_min_iou
        self.reacquire_max_cost = reacquire_max_cost
        self.reacquire_max_age = reacquire_max_age
        self.transfer_stale_gap = transfer_stale_gap
        self.byte_low_conf_thresh = byte_low_conf_thresh
        self.byte_max_iou_dist = byte_max_iou_dist if byte_max_iou_dist is not None else max_iou_dist
        self.q_calibration_scale = q_calibration_scale
        self.absorption_iou = absorption_iou
        self.absorption_app_sim = absorption_app_sim
        self.birth_iou = birth_iou
        self.birth_contain = birth_contain  # None = overlap (IoU) criterion only
        # Counters of the birth-control step (diagnostics only)
        self.birth_gating_events = {"dropped_duplicate": 0, "reacquired_lost": 0}

    # ----------------------------------------------------------------- frame loop
    def predict(self) -> None:
        """Propagate every track one frame forward."""
        for track in self.tracks:
            track.predict()

    def update(self, detections, frame_id=None, img_width=None, img_height=None) -> None:
        """Associate ``detections`` with the current tracks and manage the track set."""
        matches, unmatched_tracks, unmatched_detections = self._match(detections)

        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(detections[detection_idx], frame_id=frame_id)

        for track_idx in unmatched_tracks:
            self.tracks[track_idx].apply_transfer_learning(
                frame_id=frame_id, img_width=img_width, img_height=img_height,
            )

        if unmatched_tracks and unmatched_detections:
            recovered_matches, unmatched_tracks, unmatched_detections = self._recover_lost_tracks(
                detections, unmatched_tracks, unmatched_detections,
            )
            for track_idx, detection_idx in recovered_matches:
                self.tracks[track_idx].update(detections[detection_idx], frame_id=frame_id)

        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx], frame_id=frame_id)

        self._absorb_lost_tracks()

        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Appearance gallery of confirmed tracks (real and transfer-learning features)
        active_targets = [t.id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.id for _ in track.features]
        if len(features) > 0:
            self.metric.partial_fit(np.asarray(features), np.asarray(targets), active_targets)

    # ----------------------------------------------------------------- association
    def _match(self, detections):
        """Three-tier association; returns ``(matches, unmatched_tracks, unmatched_detections)``.

        Only unmatched *high-confidence* detections are returned: low-confidence leftovers
        are dropped so that they can never create a new identity.
        """
        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feat for i in detection_indices])
            targets = np.array([tracks[i].id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            return linear_assignment.gate_cost_matrix(
                cost_matrix,
                tracks,
                dets,
                track_indices,
                detection_indices,
                self.mc_lambda,
                measurement_fn=lambda d: d.to_xysr(),
            )

        high_det_indices = [i for i, d in enumerate(detections) if d.conf >= self.byte_low_conf_thresh]
        low_det_indices = [i for i, d in enumerate(detections) if d.conf < self.byte_low_conf_thresh]

        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Tier A: appearance cascade, confirmed tracks vs high-confidence detections
        matches_a, unmatched_tracks_a, unmatched_high = linear_assignment.matching_cascade(
            gated_metric,
            self.metric.matching_threshold,
            self.max_age,
            self.tracks,
            detections,
            confirmed_tracks,
            high_det_indices,
        )

        # Tier B: IoU, tentative + just-missed tracks vs remaining high-confidence detections
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if self.tracks[k].time_since_update == 1
        ]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if self.tracks[k].time_since_update != 1
        ]
        matches_b, unmatched_tracks_b, unmatched_high = linear_assignment.min_cost_matching(
            iou_matching.iou_cost,
            self.max_iou_dist,
            self.tracks,
            detections,
            iou_track_candidates,
            unmatched_high,
        )

        still_unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))

        # Tier C: still-unmatched confirmed tracks vs low-confidence detections (pure IoU)
        byte_track_candidates = [k for k in still_unmatched_tracks if self.tracks[k].is_confirmed()]
        matches_c = []
        if byte_track_candidates and low_det_indices:
            matches_c, _, _ = linear_assignment.min_cost_matching(
                self._byte_iou_cost,
                self.byte_max_iou_dist,
                self.tracks,
                detections,
                byte_track_candidates,
                low_det_indices,
            )

        matched_in_c = {t for t, _ in matches_c}
        unmatched_tracks = [k for k in still_unmatched_tracks if k not in matched_in_c]
        matches = matches_a + matches_b + matches_c
        return matches, unmatched_tracks, unmatched_high

    @staticmethod
    def _byte_iou_cost(tracks, detections, track_indices=None, detection_indices=None):
        """``1 - IoU`` cost without the ``time_since_update > 1`` cut-off of ``iou_cost``,
        so tracks missed for several frames can still take a low-confidence detection."""
        if track_indices is None:
            track_indices = np.arange(len(tracks))
        if detection_indices is None:
            detection_indices = np.arange(len(detections))
        cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
        for row, track_idx in enumerate(track_indices):
            bbox = tracks[track_idx].to_tlwh()
            candidates = np.asarray([detections[i].tlwh for i in detection_indices])
            cost_matrix[row, :] = 1.0 - iou_matching.iou(bbox, candidates)
        return cost_matrix

    # ----------------------------------------------------------------- re-acquisition
    def _recover_lost_tracks_cost(self, tracks, detections, track_indices, detection_indices):
        """Cost of re-attaching a lost track to a detection.

        A pair is admissible if ``IoU >= reacquire_min_iou`` or the appearance distance is
        ``<= reacquire_max_cost``; its cost is ``app - 0.5 * IoU + 0.03 * time_since_update``.
        The Mahalanobis gate is deliberately not used here.
        """
        cost_matrix = np.full((len(track_indices), len(detection_indices)), linear_assignment.INFTY_COST)
        for row, t_idx in enumerate(track_indices):
            track = tracks[t_idx]
            t_box = track.to_tlbr()
            for col, d_idx in enumerate(detection_indices):
                det = detections[d_idx]
                d_box = self._det_tlbr(det)
                iou = self._bbox_iou(t_box, d_box)
                app_cost = self._appearance_cost(track, det)

                passes_iou = iou >= self.reacquire_min_iou
                passes_app = app_cost is not None and app_cost <= self.reacquire_max_cost
                if not (passes_iou or passes_app):
                    continue
                effective_app = app_cost if app_cost is not None else self.reacquire_max_cost
                cost_matrix[row, col] = effective_app - 0.5 * iou + 0.03 * track.time_since_update
        return cost_matrix

    def _recover_lost_tracks(self, detections, unmatched_tracks, unmatched_detections):
        """Hungarian re-acquisition of lost confirmed tracks (several per frame allowed)."""
        candidate_tracks = [
            t_idx
            for t_idx in unmatched_tracks
            if self.tracks[t_idx].is_confirmed() and self.tracks[t_idx].time_since_update <= self.reacquire_max_age
        ]
        if not candidate_tracks or not unmatched_detections:
            return [], unmatched_tracks, unmatched_detections

        recovered, rem_tracks, rem_dets = linear_assignment.min_cost_matching(
            self._recover_lost_tracks_cost,
            linear_assignment.INFTY_COST - 1,  # reject only pairs gated to INFTY_COST
            self.tracks,
            detections,
            candidate_tracks,
            unmatched_detections,
        )
        ineligible_tracks = [t for t in unmatched_tracks if t not in candidate_tracks]
        return recovered, rem_tracks + ineligible_tracks, rem_dets

    # ----------------------------------------------------------------- track birth
    def _initiate_track(self, detection, frame_id=None) -> None:
        """Identity-preserving birth control, then (if allowed) start a new track.

        For an unmatched high-confidence detection ``d``:

        1. if ``d`` overlaps a track matched in this frame (``IoU > birth_iou``) or lies
           inside it (``IoMin > birth_contain``), it is a duplicate box of an already
           tracked object: ignore it;
        2. otherwise, if ``d`` overlaps a lost track (``IoU > birth_iou``), update the lost
           track with ``d`` (its identity is kept);
        3. otherwise start a new tentative track.

        The containment test is applied to matched tracks only: assigning a small box to
        a *lost* track because it lies inside its prediction would shrink the track abruptly.
        """
        x, y, w, h = detection.tlwh
        new_bbox = [x, y, x + w, y + h]

        best_lost, best_lost_iou = None, -1.0
        for track in self.tracks:
            if track.is_deleted():
                continue
            tb = track.to_tlbr()
            ov = self._bbox_iou(new_bbox, tb)
            if track.time_since_update == 0:
                dup = ov > self.birth_iou or (
                    self.birth_contain is not None
                    and self._bbox_iomin(new_bbox, tb) > self.birth_contain)
                if dup:
                    self.birth_gating_events["dropped_duplicate"] += 1
                    return
            elif ov > self.birth_iou and ov > best_lost_iou:
                best_lost, best_lost_iou = track, ov
        if best_lost is not None:
            best_lost.update(detection, frame_id=frame_id)
            self.birth_gating_events["reacquired_lost"] += 1
            return

        # Inherited merge rule. After the checks above every live track with IoU > 0.3 has
        # already been handled, so it can only remove tracks that exceeded max_age in this
        # frame (their appearance features are handed over to the new track).
        tracks_to_remove = []
        merged_features = []
        for i, track in enumerate(self.tracks):
            track_bbox = track.to_tlbr()
            x1 = max(new_bbox[0], track_bbox[0])
            y1 = max(new_bbox[1], track_bbox[1])
            x2 = min(new_bbox[2], track_bbox[2])
            y2 = min(new_bbox[3], track_bbox[3])
            if x2 > x1 and y2 > y1:
                intersection = (x2 - x1) * (y2 - y1)
                bbox1_area = (new_bbox[2] - new_bbox[0]) * (new_bbox[3] - new_bbox[1])
                bbox2_area = (track_bbox[2] - track_bbox[0]) * (track_bbox[3] - track_bbox[1])
                union = bbox1_area + bbox2_area - intersection
                iou = intersection / union if union > 0 else 0
                if iou > 0.3:
                    tracks_to_remove.append(i)
                    if track.features:
                        merged_features.extend(track.features[-5:])

        new_track = TrackV1(
            detection,
            self._next_id,
            self.n_init,
            self.max_age,
            self.ema_alpha,
            transfer_stale_gap=self.transfer_stale_gap,
            q_calibration_scale=self.q_calibration_scale,
        )
        if merged_features and detection.feat is not None:
            for old_feat in merged_features:
                if len(new_track.features) < 10:
                    new_track.features.append(old_feat)
        for i in sorted(tracks_to_remove, reverse=True):
            del self.tracks[i]
        self.tracks.append(new_track)
        self._next_id += 1

    # ----------------------------------------------------------------- absorption
    def _absorb_lost_tracks(self) -> list[tuple[int, int]]:
        """Merge a lost track into a matched track that is following the same lesion.

        A confirmed track matched in this frame (``time_since_update == 0``) absorbs at most
        one confirmed *lost* track with an older (smaller) id when
        ``IoU >= absorption_iou`` or appearance similarity ``>= absorption_app_sim``
        (the lost box is the Primary prediction). The matched track takes the older id,
        ``max(hits)`` and up to five gallery features; its filter state is kept, and the
        lost track is deleted.

        Requiring one matched and one lost track separates "two tracks on one lesion"
        (one object yields one detection, so only one of them can be matched) from "two
        lesions crossing" (both are matched). Known failure: under full occlusion the
        occluder can take the identity of the occluded object.

        Returns:
            List of ``(kept_id, absorbed_id)`` pairs, for diagnostics.
        """
        if len(self.tracks) < 2:
            return []
        matched = [i for i, t in enumerate(self.tracks)
                   if t.is_confirmed() and t.time_since_update == 0]
        lost = [i for i, t in enumerate(self.tracks)
                if t.is_confirmed() and t.time_since_update > 0]
        if not matched or not lost:
            return []

        absorbed, removed = [], set()
        for mi in matched:
            if mi in removed:
                continue
            tm = self.tracks[mi]
            best_idx, best_score = None, -1.0
            for li in lost:
                if li in removed or li == mi:
                    continue
                tl = self.tracks[li]
                if tl.id >= tm.id:
                    continue  # only an older identity can be inherited
                iou = self._bbox_iou(tm.to_tlbr(), tl.to_tlbr())
                app = -1.0
                if tm.features and tl.features:
                    fa = np.asarray(tm.features[-1], dtype=np.float64)
                    fb = np.asarray(tl.features[-1], dtype=np.float64)
                    na, nb = np.linalg.norm(fa), np.linalg.norm(fb)
                    if na > 0 and nb > 0:
                        app = float(np.dot(fa, fb) / (na * nb))
                if iou >= self.absorption_iou or app >= self.absorption_app_sim:
                    score = max(iou, app)
                    if score > best_score:
                        best_idx, best_score = li, score
            if best_idx is not None:
                tl = self.tracks[best_idx]
                old_id, new_id = tm.id, tl.id
                tm.id = tl.id
                tm.hits = max(tm.hits, tl.hits)
                for f in tl.features[-5:]:
                    if len(tm.features) < 10:
                        tm.features.append(f)
                removed.add(best_idx)
                absorbed.append((new_id, old_id))

        for idx in sorted(removed, reverse=True):
            del self.tracks[idx]
        return absorbed

    # ----------------------------------------------------------------- geometry helpers
    @staticmethod
    def _det_tlbr(detection) -> np.ndarray:
        x, y, w, h = detection.tlwh
        return np.array([x, y, x + w, y + h], dtype=np.float32)

    @staticmethod
    def _bbox_iou(box1, box2) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        a1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
        a2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
        union = a1 + a2 - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _bbox_iomin(box1, box2) -> float:
        """Intersection over the smaller box (1 when one box lies inside the other)."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        a1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
        a2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
        m = min(a1, a2)
        return inter / m if m > 0 else 0.0

    def _appearance_cost(self, track, detection) -> float | None:
        """Cosine distance between a detection and the gallery of ``track`` (None if unavailable)."""
        if detection.feat is None:
            return None
        if track.id not in self.metric.samples or len(self.metric.samples[track.id]) == 0:
            return None
        det_feat = np.asarray(detection.feat).reshape(1, -1)
        costs = self.metric.distance(det_feat, np.array([track.id]))
        return float(costs[0, 0])
