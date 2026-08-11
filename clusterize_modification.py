import numpy as np
import sys
from dataclasses import dataclass

from spyral.core.point_cloud import PointCloud
from spyral.core.clusterize import LabeledCloud, Direction


sys.path.append('/home/danilo-marcato/Documents/TriplClust')

sys.path.insert(0, "/home/danilo-marcato/Documents/ransac_cpp")

sys.path.append('/home/danilo-marcato/Documents/Python_binding_ransac_STD17_lines')

import pyTrackinglines
from ransac_circle_fit import ransac_circle_fit, ransac_multi_circle_fit
import pyTriplClust

@dataclass
class RevaluationParameters:
    threshold_circle: float = 25.0
    threshold_line: float = 25.0
    min_inliers: int = 20
    n_iterations: int = 2000
    origin_weight: float = 2.0
    max_revaluation: int = 5
    
    zone_1: float = 10.0
    zone_2: float = 25.0
    dm: float = 10.0
    
    rng_seed: int = 42   # NOTE: never passed into ransac_circle_fit / ransac_multi_circle_fit /
    # RANSAC_Line from this file. If those functions don't read a seed
    # internally, results aren't actually reproducible via this field.


def Select_Event_Cloud(event_number, cloud_group):
    """Pull a single event's point cloud out of an HDF5-like group."""
    event_name = f"cloud_{event_number}"
    if event_name in cloud_group:
        event_data = cloud_group[event_name]
        cloud = PointCloud(event_number, event_data[:].copy())
        return cloud
    else:
        return None


def Create_Cluster(cloud, labels):
    """Turn a flat (cloud, labels) pair into a list of LabeledCloud objects, one per unique label."""
    clusters: list[LabeledCloud] = []
    for label in np.unique(labels):
        mask = (labels == label)
        clusters.append(
            LabeledCloud(
                label,
                Direction.NONE,
                PointCloud(cloud.event_number, cloud.data[mask]),
                np.flatnonzero(mask),
            )
        )
    return clusters


def Relabel_Cluster(revaluation_labels, current_label, noise_label=-1):
    """
    Remap an arbitrary label set into a contiguous block starting at
    `current_label`, keeping `noise_label` entries as noise (-1).
    Returns (relabeled_array, count_of_new_labels_assigned).
    """
    new_cluster_labels = np.full(
        len(revaluation_labels), noise_label, dtype=int)
    i = 0
    for label in np.unique(revaluation_labels):
        if label == noise_label:
            continue
        mask = revaluation_labels == label
        new_cluster_labels[mask] = int(i + current_label)
        i += 1
    return new_cluster_labels, i


def TriplClust(cloud):
    """Run the TriplClust algorithm on a point cloud and wrap the result into clusters."""
    obj = pyTriplClust.Run_clustering()

    Z_SCALE = 584.0/1000.0  # scale between x&y/z dimations of the detector
    initial = obj.Init(
        cloud.data[:, 0], cloud.data[:, 1], cloud.data[:, 2] * Z_SCALE, True)
    solve = obj.Solve()
    triplClust_labels = np.array(obj.GetIDs())
    triplClust_clusters = Create_Cluster(cloud, triplClust_labels)
    return triplClust_clusters, triplClust_labels


def RANSAC_Line(x, y, revaluation_parms=RevaluationParameters()):
    """Fit line(s) via LMedS RANSAC over (x, y) pairs (here: z vs unrolled arc length)."""
    q = np.ones(len(x))
    z = np.zeros(len(x))
    obj = pyTrackinglines.LMedS()
    inicia = obj.Init(x, y, z, q)
    resuelve = obj.Solve(revaluation_parms.threshold_line,
                         revaluation_parms.min_inliers, revaluation_parms.n_iterations)
    clusters = obj.GetClusters()

    return clusters


def Revaluation_Step(cluster_data, idx_remaining, revaluation_parms=RevaluationParameters(), use_radius=True):
    """
    Fit a single circle to the remaining (not-yet-assigned) hits in a cluster,
    then, among the hits near that circle, unroll the arc length and fit a
    line in (z, arc) space to pick out one helical track's worth of hits.
    Returns the *global* indices (into cluster_data) of the accepted hits,
    or None if no good track was found this round.
    """
    best_circle = ransac_circle_fit(
        cluster_data[idx_remaining, 0:3],
        min_inliers=revaluation_parms.min_inliers,
        threshold_circle=revaluation_parms.threshold_circle,
        n_iterations=revaluation_parms.n_iterations,
        origin_weight=revaluation_parms.origin_weight,
    )

    if best_circle is None:
        return None
    # if best_circle.inlier_fraction > 0.9:
    #     return best_circle.inlier_indices

    x_c = best_circle.center_x
    y_c = best_circle.center_y
    radius = best_circle.radius

    x = cluster_data[idx_remaining, 0]
    y = cluster_data[idx_remaining, 1]
    dist2 = (x - x_c)**2 + (y - y_c)**2

    inside_mask = dist2 <= (radius + revaluation_parms.threshold_circle)**2

    z = cluster_data[idx_remaining[inside_mask], 2]

    x_trans = x - x_c
    y_trans = y - y_c

    theta0 = np.arctan2(y_trans[0], x_trans[0])

    phi = -np.pi - theta0

    x_rot = x_trans * np.cos(phi) - y_trans * np.sin(phi)
    y_rot = x_trans * np.sin(phi) + y_trans * np.cos(phi)

    theta = np.arctan2(y_rot[inside_mask], x_rot[inside_mask])

    if use_radius:
        arc = theta * radius
    else:
        rho = np.linalg.norm(np.column_stack(
            (x_rot[inside_mask], y_rot[inside_mask])), axis=1)
        arc = theta * rho

    lines = RANSAC_Line(z, arc, revaluation_parms)
    if lines is None or len(lines) == 0:
        return None

    idx = []
    back = False
    main_angle_degrees = None
    for i, l in enumerate(lines):
        m, _ = l.ClusterFitP1
        angle_degrees = np.degrees(np.arctan(m))

        if i == 0:

            main_angle_degrees = angle_degrees
            if main_angle_degrees < 0:
                back = True
            idx.append(l.ClusterIndex)
            continue

        # NOTE: asymmetric acceptance windows depending on `back` — presumably
        # intentional (handling tracks curving the other way), but worth a
        # one-line comment explaining the physical reasoning so it's not
        # mistaken for a typo (7.5/2.5 swapped) by a future reader.
        if back:
            if main_angle_degrees - 7.5 < angle_degrees < main_angle_degrees + 2.5:
                idx.append(l.ClusterIndex)
                continue
        else:
            if main_angle_degrees - 2.5 < angle_degrees < main_angle_degrees + 7.5:
                idx.append(l.ClusterIndex)
                continue

    if len(idx) == 0:
        return None

    l_mask = np.concatenate(idx)
    return idx_remaining[inside_mask][l_mask]


def TriplClust_Revaluation(cluster_data, revaluation_parms=RevaluationParameters()):
    """
    Iteratively peel tracks out of a single TriplClust cluster: repeatedly
    fit one circle+line "track" via Revaluation_Step, label its hits, remove
    them from the pool, and repeat up to max_revaluation times or until
    too few hits remain.
    """
    idx_remaining = np.arange(len(cluster_data))
    track_label = np.full(len(idx_remaining), -1, dtype=int)
    track_number = 0

    for _ in range(revaluation_parms.max_revaluation):
        if len(idx_remaining) < revaluation_parms.min_inliers:
            break

        inliers_global = Revaluation_Step(
            cluster_data, idx_remaining, revaluation_parms)
        if inliers_global is None or len(inliers_global) == 0:
            break

        track_label[inliers_global] = track_number
        track_number += 1

        # remove this round's inliers from the remaining pool
        # NOTE: np.setdiff1d returns a SORTED array, which is what feeds
        # theta0 = arctan2(y_trans[0], x_trans[0]) on the next iteration
        # (see the note in Revaluation_Step above).
        idx_remaining = np.setdiff1d(
            idx_remaining, inliers_global, assume_unique=False)

    return track_label


def Clusters_Revaluation(clusters, labels, revaluation_parms=RevaluationParameters()):
    """
    For each cluster produced by TriplClust, try fitting multiple circles.
    If more than one circle fits, the cluster is likely multiple overlapping
    tracks and gets split via TriplClust_Revaluation + relabeling; otherwise
    it's kept as a single cluster.
    """
    new_labels = np.full(len(labels), -1, dtype=int)
    current_label = 0

    for cluster in clusters:
        if cluster.label == -1:
            continue
        mask = labels == cluster.label

        cluster_data = cluster.point_cloud.data

        fitted_circles = ransac_multi_circle_fit(
            cluster_data[:, 0:3],
            max_circles=revaluation_parms.max_revaluation,
            threshold_circle=revaluation_parms.threshold_circle,
            n_iterations=revaluation_parms.n_iterations,
            min_inliers_absolute=revaluation_parms.min_inliers,
            verbose=False)

        if len(fitted_circles) > 1:
            revaluation_labels = TriplClust_Revaluation(
                cluster_data, revaluation_parms)
            revaluation_labels, number_labels = Relabel_Cluster(
                revaluation_labels, current_label)
            current_label += number_labels
            new_labels[mask] = revaluation_labels
        else:
            new_labels[mask] = current_label
            current_label += 1
            
    return new_labels

def Clusters_Metrics(clusters, labels, revaluation_parms=RevaluationParameters()):
    """
    Compute one summary-metrics row per (non-noise) cluster: fitted circle
    center + track angle, plus entry/exit-point averages, for use by
    Join_Clusters to decide which clusters likely belong to the same track.
 
    Each row is:
        [label, center_x, center_y, angle_degrees,
         entry_x, entry_y, entry_z,
         exit_x, exit_y, exit_z,
         fit_ok]
 
    `fit_ok` is 1 if both the circle fit and line fit succeeded, else 0.
    `label` is always the real cluster label (even on fit failure) so
    Join_Clusters can look clusters up reliably instead of relying on row
    position lining up with `clusters`.
    """
    cluster_metrics = []
    for cluster in clusters:
        if cluster.label == -1:
            continue
 
        cluster_data = cluster.point_cloud.data
        x = cluster_data[:, 0]
        y = cluster_data[:, 1]
        z = cluster_data[:, 2]
 
        # entry/exit point averages over the raw cluster (first/last 3 hits),
        # fixed to average across points (rows) per-axis, not across axes
        # within a single point.
        entry_x, entry_y, entry_z = (
            np.mean(cluster_data[:3, 0]), np.mean(cluster_data[:3, 1]), np.mean(cluster_data[:3, 2]))
        exit_x, exit_y, exit_z = (
            np.mean(cluster_data[-3:, 0]), np.mean(cluster_data[-3:, 1]), np.mean(cluster_data[-3:, 2]))
 
        best_circle = ransac_circle_fit(
            cluster_data[:, 0:3],
            min_inliers=revaluation_parms.min_inliers,
            threshold_circle=revaluation_parms.threshold_circle,
            n_iterations=revaluation_parms.n_iterations,
            origin_weight=0.0,
        )
 
        if best_circle is None:
            cluster_metrics.append([cluster.label, 0.0, 0.0, -1.0,
                                     entry_x, entry_y, entry_z,
                                     exit_x, exit_y, exit_z, 0])
            continue
 
        # Restrict the arc/line fit to the circle's own inliers instead of
        # running it over every point in the cluster (previously `inliers`
        # was computed but never used to filter x/y/z).
        inliers = best_circle.inlier_indices
        x_c = best_circle.center_x
        y_c = best_circle.center_y
        radius = best_circle.radius
 
        x_in = x[inliers]
        y_in = y[inliers]
        z_in = z[inliers]
 
        x_trans = x_in - x_c
        y_trans = y_in - y_c
        theta0 = np.arctan2(y_trans[0], x_trans[0])
        phi = -np.pi - theta0
        x_rot = x_trans * np.cos(phi) - y_trans * np.sin(phi)
        y_rot = x_trans * np.sin(phi) + y_trans * np.cos(phi)
        theta = np.arctan2(y_rot, x_rot)
        arc = theta * radius
 
        lines = RANSAC_Line(z_in, arc, revaluation_parms)
 
        if lines is None or len(lines) == 0:
            cluster_metrics.append([cluster.label, x_c, y_c, -1.0,
                                     entry_x, entry_y, entry_z,
                                     exit_x, exit_y, exit_z, 0])
            continue
 
        best_line = lines[0]
        m, b = best_line.ClusterFitP1
        angle_degrees = np.degrees(np.arctan(m))
        cluster_metrics.append([cluster.label, x_c, y_c, angle_degrees,
                                 entry_x, entry_y, entry_z,
                                 exit_x, exit_y, exit_z, 1])
 
    return np.array(cluster_metrics)
 
 
def Join_Clusters(cluster_metrics, clusters, labels, cloud, revaluation_parms=RevaluationParameters()):
    """
    Merge clusters whose fitted circle centers (and, loosely, track angle)
    are close enough to plausibly be the same track split across multiple
    TriplClust clusters.
    """
    # Look clusters up by label rather than by position: Clusters_Metrics
    # skips noise clusters, so cluster_metrics rows are NOT guaranteed to be
    # in the same order/position as `clusters`.
    cluster_by_label = {c.label: c for c in clusters if c.label != -1}
 
    groups = []
    used = set()
    n = len(cluster_metrics)
 
    for i in range(n):
        if i in used:
            continue
        used.add(i)
        group = [i]
 
        # A failed circle/line fit has only placeholder metrics (0.0, 0.0,
        # -1.0) -- it can't be meaningfully compared to other clusters, so
        # keep it as its own singleton group instead of letting it merge
        # with (or absorb) real clusters based on those placeholders.
        if cluster_metrics[i, 10] == 0:
            groups.append(group)
            continue
 
        for j in range(i + 1, n):
            if j in used:
                continue
            if cluster_metrics[j, 10] == 0:
                continue  # skip failed-fit rows -- their metrics aren't meaningful
 
            dx = cluster_metrics[i, 1] - cluster_metrics[j, 1]
            dy = cluster_metrics[i, 2] - cluster_metrics[j, 2]
            dm = cluster_metrics[i, 3] - cluster_metrics[j, 3]
 
            if np.hypot(dx, dy) < revaluation_parms.zone_1:
                group.append(j)
                used.add(j)
            elif np.hypot(dx, dy) < revaluation_parms.zone_2 and abs(dm) < 10.0:
                group.append(j)
                used.add(j)
 
        groups.append(group)
 
    newLabels = np.full(len(labels), -1)
    for groupNumber, group in enumerate(groups):
        for row in group:
            cluster_label = cluster_metrics[row, 0]
            cluster_obj = cluster_by_label.get(cluster_label)
            if cluster_obj is None:
                continue
            indx = np.where(labels == cluster_obj.label)[0]
            newLabels[indx] = groupNumber
 
    newClusters = Create_Cluster(cloud, newLabels)
    return newClusters, newLabels
 
 
def New_Clustering_Method(cloud, revaluation_parms=RevaluationParameters()):
    """
    Full pipeline: TriplClust -> per-cluster circle-fit based revaluation
    (splitting overlapping tracks) -> per-cluster metrics -> merge clusters
    that look like the same track (Join_Clusters).
    """
    clusters_TriplClust, labels_TriplClust = TriplClust(cloud)
 
    new_labels = Clusters_Revaluation(
        clusters_TriplClust, labels_TriplClust, revaluation_parms)
    new_clusters = Create_Cluster(cloud, new_labels)
 
    cluster_metrics = Clusters_Metrics(new_clusters, new_labels, revaluation_parms)
    joined_clusters, joined_labels = Join_Clusters(
        cluster_metrics, new_clusters, new_labels, cloud, revaluation_parms)
 
    return joined_clusters, joined_labels

