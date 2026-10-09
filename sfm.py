"""Incremental Structure from Motion.

1. Undistort keypoints, match every image pair and keep matches consistent with an
   essential matrix (geometric verification).
2. Pick an initial pair with many inliers, a non-planar/non-rotational motion and a large
   triangulation angle, triangulate it and bundle-adjust it.
3. Repeatedly register the image that sees the most reconstructed points (PnP + RANSAC),
   extend existing tracks to it, triangulate new tracks, and periodically run a global
   bundle adjustment followed by outlier filtering.
"""
from dataclasses import dataclass, field
from itertools import combinations
from time import perf_counter

import cv2 as cv
import numpy as np

from bundle_adjustment import bundle_adjustment, params_to_pose, pose_to_params
from matching import match_descriptors
from pose import (camera_center, essential_inliers, estimate_pose, homography_inlier_count,
                  undistort_keypoints)
from triangulation import (depths, reprojection_errors, triangulate_points,
                           valid_triangulations)


@dataclass
class SfMOptions:
    ratio: float = 0.8                 # Lowe ratio test
    essential_threshold: float = 2.0   # px, RANSAC threshold for the essential matrix
    min_pair_inliers: int = 30         # verified matches needed to link two images
    init_max_homography_ratio: float = 0.8
    init_min_median_angle: float = 3.0  # degrees
    init_candidates: int = 30
    min_triangulation_angle: float = 1.5  # degrees
    max_reprojection_error: float = 4.0   # px
    pnp_threshold: float = 4.0            # px
    min_pnp_inliers: int = 30
    ba_every: int = 3                  # global bundle adjustment every N registered images
    ba_max_nfev: int = 50
    final_ba_max_nfev: int = 200
    verbose: bool = True


@dataclass
class Track:
    xyz: np.ndarray
    color: np.ndarray
    obs: dict = field(default_factory=dict)  # view index -> keypoint index


class Reconstruction:
    def __init__(self, views, K, dist=None, options=None):
        self.views = views
        self.K = np.asarray(K, dtype=np.float64)
        self.dist = None if dist is None else np.asarray(dist, dtype=np.float64).ravel()
        self.opt = options or SfMOptions()
        self.points2d = [undistort_keypoints(v.keypoints, self.K, self.dist) for v in views]
        self.poses = {}  # view index -> (R, t)
        self.tracks = {}
        self.kp_to_track = [-np.ones(len(v.keypoints), dtype=int) for v in views]
        self.pair_matches = {}  # (i, j) with i < j -> (M, 2) keypoint indices
        self.initial_pair = None
        self._next_track_id = 0

    def log(self, msg):
        if self.opt.verbose:
            print(msg)

    # ------------------------------------------------------------------ matching
    def match_all_pairs(self):
        start = perf_counter()
        for i, j in combinations(range(len(self.views)), 2):
            m = match_descriptors(self.views[i].descriptors, self.views[j].descriptors,
                                  ratio=self.opt.ratio)
            if len(m) < self.opt.min_pair_inliers:
                continue
            inl = essential_inliers(self.points2d[i][m[:, 0]], self.points2d[j][m[:, 1]], self.K,
                                    threshold=self.opt.essential_threshold)
            if inl.sum() >= self.opt.min_pair_inliers:
                self.pair_matches[(i, j)] = m[inl]
        self.log(f"Matched {len(self.views)} images: {len(self.pair_matches)} verified pairs "
                 f"in {perf_counter() - start:.1f}s")

    def matches_between(self, a, b):
        """Verified matches oriented as (keypoint in a, keypoint in b)."""
        if (a, b) in self.pair_matches:
            return self.pair_matches[(a, b)]
        if (b, a) in self.pair_matches:
            return self.pair_matches[(b, a)][:, ::-1]
        return None

    # ------------------------------------------------------------------ tracks
    def add_track(self, xyz, obs):
        tid = self._next_track_id
        self._next_track_id += 1
        v0, k0 = next(iter(obs.items()))
        self.tracks[tid] = Track(xyz=np.asarray(xyz, dtype=np.float64), color=self.views[v0].colors[k0],
                                 obs=dict(obs))
        for v, k in obs.items():
            self.kp_to_track[v][k] = tid
        return tid

    def add_observation(self, tid, view, kp):
        self.tracks[tid].obs[view] = kp
        self.kp_to_track[view][kp] = tid

    def remove_observation(self, tid, view):
        kp = self.tracks[tid].obs.pop(view)
        self.kp_to_track[view][kp] = -1
        if len(self.tracks[tid].obs) < 2:
            for v, k in self.tracks[tid].obs.items():
                self.kp_to_track[v][k] = -1
            del self.tracks[tid]

    # ------------------------------------------------------------------ initialization
    def _evaluate_pair(self, i, j):
        m = self.pair_matches[(i, j)]
        p1, p2 = self.points2d[i][m[:, 0]], self.points2d[j][m[:, 1]]
        h_ratio = homography_inlier_count(p1, p2) / len(m)
        if h_ratio > self.opt.init_max_homography_ratio:
            return None
        R, t, good = estimate_pose(p1, p2, self.K, threshold=self.opt.essential_threshold)
        if R is None or good.sum() < self.opt.min_pair_inliers:
            return None
        R0, t0 = np.eye(3), np.zeros((3, 1))
        X = triangulate_points(p1[good], p2[good], self.K, R0, t0, R, t)
        ok = valid_triangulations(X, p1[good], p2[good], self.K, R0, t0, R, t,
                                  self.opt.max_reprojection_error, self.opt.min_triangulation_angle)
        if ok.sum() < self.opt.min_pair_inliers:
            return None
        c2 = camera_center(R, t)
        rays1 = X[ok] / np.linalg.norm(X[ok], axis=1, keepdims=True)
        rays2 = (X[ok] - c2) / np.linalg.norm(X[ok] - c2, axis=1, keepdims=True)
        median_angle = np.degrees(np.median(np.arccos(np.clip(np.sum(rays1 * rays2, 1), -1, 1))))
        return dict(pair=(i, j), R=R, t=t, matches=m[good][ok], X=X[ok], n=int(ok.sum()),
                    median_angle=median_angle, h_ratio=h_ratio)

    def initialize(self):
        candidates = sorted(self.pair_matches, key=lambda p: -len(self.pair_matches[p]))
        results = [r for r in (self._evaluate_pair(i, j) for i, j in candidates[:self.opt.init_candidates])
                   if r is not None]
        if not results:
            raise RuntimeError("No image pair suitable for initialization (not enough parallax or "
                               "verified matches)")
        wide = [r for r in results if r["median_angle"] >= self.opt.init_min_median_angle]
        if wide:
            best = max(wide, key=lambda r: r["n"])
        else:
            best = max(results, key=lambda r: r["median_angle"])
            self.log(f"Warning: no pair has a median triangulation angle >= "
                     f"{self.opt.init_min_median_angle} deg, using the widest one")

        i, j = best["pair"]
        self.initial_pair = (i, j)
        self.poses[i] = (np.eye(3), np.zeros((3, 1)))
        self.poses[j] = (best["R"], best["t"])
        for (ki, kj), xyz in zip(best["matches"], best["X"]):
            self.add_track(xyz, {i: ki, j: kj})
        self.log(f"Initial pair {self.views[i].name} / {self.views[j].name}: {best['n']} points, "
                 f"median angle {best['median_angle']:.1f} deg, homography ratio {best['h_ratio']:.2f}")
        self.global_bundle_adjustment(self.opt.ba_max_nfev)

    # ------------------------------------------------------------------ registration
    def correspondences_2d3d(self, view):
        """Unregistered keypoints of `view` matched to an existing track: (keypoints, track ids)."""
        kvs, tids = [], []
        for reg in self.poses:
            m = self.matches_between(view, reg)
            if m is None:
                continue
            tid = self.kp_to_track[reg][m[:, 1]]
            sel = (tid >= 0) & (self.kp_to_track[view][m[:, 0]] < 0)
            kvs.append(m[sel, 0])
            tids.append(tid[sel])
        if not kvs:
            return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
        kv, first = np.unique(np.concatenate(kvs), return_index=True)
        return kv, np.concatenate(tids)[first]

    def register_view(self, view):
        kv, tids = self.correspondences_2d3d(view)
        if len(kv) < self.opt.min_pnp_inliers:
            return False
        obj = np.array([self.tracks[t].xyz for t in tids])
        img = self.points2d[view][kv]
        ok, rvec, tvec, inliers = cv.solvePnPRansac(
            obj, img, self.K, None, iterationsCount=2000, reprojectionError=self.opt.pnp_threshold,
            confidence=0.9999, flags=cv.SOLVEPNP_EPNP)
        if not ok or inliers is None or len(inliers) < self.opt.min_pnp_inliers:
            return False
        inliers = inliers.ravel()
        rvec, tvec = cv.solvePnPRefineLM(obj[inliers], img[inliers], self.K, None, rvec, tvec)
        R, _ = cv.Rodrigues(rvec)
        t = tvec.reshape(3, 1)

        err = reprojection_errors(obj, img, self.K, R, t)
        good = (err < self.opt.max_reprojection_error) & (depths(obj, R, t) > 0)
        if good.sum() < self.opt.min_pnp_inliers:
            return False
        self.poses[view] = (R, t)
        for k, tid in zip(kv[good], tids[good]):
            if view not in self.tracks[tid].obs:
                self.add_observation(tid, view, k)
        self.log(f"Registered {self.views[view].name}: {int(good.sum())}/{len(kv)} 2D-3D inliers")
        return True

    def _extend_tracks(self, tids, view, kps):
        """Add keypoints `kps` of registered `view` to tracks `tids` when they reproject well."""
        if len(tids) == 0:
            return
        R, t = self.poses[view]
        X = np.array([self.tracks[tid].xyz for tid in tids])
        err = reprojection_errors(X, self.points2d[view][kps], self.K, R, t)
        ok = (err < self.opt.max_reprojection_error) & (depths(X, R, t) > 0)
        for tid, kp in zip(tids[ok], kps[ok]):
            if view not in self.tracks[tid].obs and self.kp_to_track[view][kp] < 0:
                self.add_observation(tid, view, kp)

    def triangulate_view(self, view):
        """Extend tracks seen by `view` to its neighbours (and vice versa), then triangulate
        matches that belong to no track yet."""
        n_new = 0
        Rv, tv = self.poses[view]
        for reg in list(self.poses):
            if reg == view:
                continue
            m = self.matches_between(view, reg)
            if m is None:
                continue
            tid_v, tid_r = self.kp_to_track[view][m[:, 0]], self.kp_to_track[reg][m[:, 1]]
            sel = (tid_v < 0) & (tid_r >= 0)
            self._extend_tracks(tid_r[sel], view, m[sel, 0])
            sel = (tid_v >= 0) & (tid_r < 0)
            self._extend_tracks(tid_v[sel], reg, m[sel, 1])

            free = (self.kp_to_track[view][m[:, 0]] < 0) & (self.kp_to_track[reg][m[:, 1]] < 0)
            if not free.any():
                continue
            m = m[free]
            p1, p2 = self.points2d[view][m[:, 0]], self.points2d[reg][m[:, 1]]
            Rr, tr = self.poses[reg]
            X = triangulate_points(p1, p2, self.K, Rv, tv, Rr, tr)
            ok = valid_triangulations(X, p1, p2, self.K, Rv, tv, Rr, tr,
                                      self.opt.max_reprojection_error, self.opt.min_triangulation_angle)
            for (kv, kr), xyz in zip(m[ok], X[ok]):
                self.add_track(xyz, {view: kv, reg: kr})
            n_new += int(ok.sum())
        return n_new

    # ------------------------------------------------------------------ refinement
    def _observation_arrays(self):
        tids = list(self.tracks)
        cam_idx, pt_idx, uv = [], [], []
        for p, tid in enumerate(tids):
            for v, k in self.tracks[tid].obs.items():
                cam_idx.append(v)
                pt_idx.append(p)
                uv.append(self.points2d[v][k])
        return tids, np.array(cam_idx), np.array(pt_idx), np.array(uv).reshape(-1, 2)

    def global_bundle_adjustment(self, max_nfev):
        if not self.tracks:
            return
        start = perf_counter()
        views = sorted(self.poses)
        view_pos = {v: c for c, v in enumerate(views)}
        tids, obs_view, pt_idx, uv = self._observation_arrays()
        cam_params = np.array([pose_to_params(*self.poses[v]) for v in views])
        points = np.array([self.tracks[t].xyz for t in tids])
        i, j = self.initial_pair
        cams, pts, info = bundle_adjustment(
            cam_params, points, np.array([view_pos[v] for v in obs_view]), pt_idx, uv, self.K,
            fixed_cams=(view_pos[i],), scale_pair=(view_pos[i], view_pos[j]), max_nfev=max_nfev)
        for v, params in zip(views, cams):
            self.poses[v] = params_to_pose(params)
        for tid, xyz in zip(tids, pts):
            self.tracks[tid].xyz = xyz
        self.log(f"Bundle adjustment ({len(views)} cams, {len(tids)} points, {len(uv)} obs): "
                 f"RMS {info['initial_rms']:.2f} -> {info['final_rms']:.2f} px "
                 f"in {perf_counter() - start:.1f}s")
        self.filter_tracks()

    def filter_tracks(self):
        """Drop observations with a large reprojection error or behind the camera, then tracks
        observed fewer than twice or without enough parallax."""
        removed_obs = 0
        tids, obs_view, pt_idx, uv = self._observation_arrays()
        if not tids:
            return
        X = np.array([self.tracks[t].xyz for t in tids])
        bad = np.zeros(len(uv), dtype=bool)
        for v in np.unique(obs_view):
            sel = obs_view == v
            R, t = self.poses[v]
            err = reprojection_errors(X[pt_idx[sel]], uv[sel], self.K, R, t)
            bad[sel] = (err > self.opt.max_reprojection_error) | (depths(X[pt_idx[sel]], R, t) <= 0)
        for v, p in zip(obs_view[bad], pt_idx[bad]):
            if tids[p] in self.tracks and v in self.tracks[tids[p]].obs:
                self.remove_observation(tids[p], v)
                removed_obs += 1

        centers = {v: camera_center(*self.poses[v]) for v in self.poses}
        removed_tracks = 0
        min_cos = np.cos(np.radians(self.opt.min_triangulation_angle))
        for tid in list(self.tracks):
            track = self.tracks[tid]
            rays = np.array([track.xyz - centers[v] for v in track.obs])
            rays /= np.linalg.norm(rays, axis=1, keepdims=True)
            if np.min(rays @ rays.T) > min_cos:
                for v, k in track.obs.items():
                    self.kp_to_track[v][k] = -1
                del self.tracks[tid]
                removed_tracks += 1
        if removed_obs or removed_tracks:
            self.log(f"  filtered {removed_obs} observations, {removed_tracks} low-parallax tracks")

    # ------------------------------------------------------------------ driver
    def run(self):
        self.match_all_pairs()
        self.initialize()
        failed = set()
        since_ba = 0
        while True:
            candidates = [v for v in range(len(self.views)) if v not in self.poses and v not in failed]
            if not candidates:
                break
            scored = sorted(((len(self.correspondences_2d3d(v)[0]), v) for v in candidates), reverse=True)
            registered = False
            for n_corr, v in scored:
                if n_corr < self.opt.min_pnp_inliers:
                    break
                if self.register_view(v):
                    registered = True
                    n_new = self.triangulate_view(v)
                    self.log(f"  {n_new} new points, {len(self.tracks)} total")
                    since_ba += 1
                    if since_ba >= self.opt.ba_every:
                        self.global_bundle_adjustment(self.opt.ba_max_nfev)
                        since_ba = 0
                    failed.clear()
                    break
                failed.add(v)
            if not registered:
                break

        self.global_bundle_adjustment(self.opt.final_ba_max_nfev)
        unregistered = [self.views[v].name for v in range(len(self.views)) if v not in self.poses]
        self.log(f"Reconstruction: {len(self.poses)}/{len(self.views)} images registered, "
                 f"{len(self.tracks)} points, mean track length {self.mean_track_length():.2f}")
        if unregistered:
            self.log(f"Not registered: {', '.join(unregistered)}")
        return self

    # ------------------------------------------------------------------ results
    def points_and_colors(self):
        if not self.tracks:
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8)
        return (np.array([t.xyz for t in self.tracks.values()]),
                np.array([t.color for t in self.tracks.values()], dtype=np.uint8))

    def mean_track_length(self):
        return float(np.mean([len(t.obs) for t in self.tracks.values()])) if self.tracks else 0.0

    def reprojection_rms(self):
        tids, obs_view, pt_idx, uv = self._observation_arrays()
        if not tids:
            return float("nan")
        X = np.array([self.tracks[t].xyz for t in tids])
        sq = []
        for v in np.unique(obs_view):
            sel = obs_view == v
            sq.append(reprojection_errors(X[pt_idx[sel]], uv[sel], self.K, *self.poses[v]) ** 2)
        return float(np.sqrt(np.mean(np.concatenate(sq))))


def run_sfm(views, K, dist=None, options=None):
    return Reconstruction(views, K, dist, options).run()
