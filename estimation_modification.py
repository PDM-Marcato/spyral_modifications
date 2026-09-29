from pathlib import Path
import numpy as np
import math
import h5py as h5
from dataclasses import dataclass
from scipy.stats import linregress
from spyral.core.cluster import Cluster, Direction
import sys

sys.path.insert(0, "/home/danilo-marcato/Documents/ransac_cpp")

from ransac_circle_fit import ransac_circle_fit, ransac_multi_circle_fit
@dataclass
class NewEstimateResult:

    vertex_x: float
    vertex_y: float
    vertex_z: float
    center_x: float
    center_y: float
    center_z: float
    polar: float
    azimuthal: float
    brho: float
    dEdx: float
    sqrt_dEdx: float
    dE: float
    arclength: float
    direction: int

@dataclass
class Circle:
    center_x: float
    center_y: float
    radius: float
    
def Closest_Origin(center_x, center_y, radius):
    # Calculate distance from origin to center
    d = np.sqrt(center_x**2 + center_y**2)
    
    x = center_x - (center_x / d) * radius
    y = center_y - (center_y / d) * radius
    
    return (x, y)
    
def Track_Classification(cluster):
    
    x = cluster.data[:, 0]
    y = cluster.data[:, 1]
    z = cluster.data[:, 2]
    
    x_m = np.mean(x)
    y_m = np.mean(y)
    radius_cut = 30.0
    
    dist_squared = (x - x_m) ** 2 + (y - y_m) ** 2
    inside_count = np.sum(dist_squared <= radius_cut**2)
    if (inside_count/ len(cluster.data) > 0.75):
        return False
    else:
        return True
        
def Define_First_Arc(cluster, best_circle):
    x = cluster.data[:, 0]
    y = cluster.data[:, 1]
    z = cluster.data[:, 2]
    
    # Best-fit circle
    x_c = best_circle.center_x
    y_c = best_circle.center_y
    radius = best_circle.radius

    vertex_x, vertex_y = Closest_Origin(x_c, y_c, radius)

    x_trans = x - x_c
    y_trans = y - y_c

    theta0 = np.arctan2(vertex_y - y_c, vertex_x - x_c)

    phi = -np.pi - theta0

    x_rot = (x_trans * np.cos(phi)- y_trans * np.sin(phi))

    y_rot = (x_trans * np.sin(phi) + y_trans * np.cos(phi))

    theta = np.arctan2(y_rot, x_rot)

    # Arc length
    arc = theta * radius

    maximum = np.argmax(arc)
    
    return cluster_data[: (maximum + 1)]
    
    
def get_first_segment(z, arc, wrap_threshold= np.pi):
    """
    Given z and arc (theta, range [-pi, pi]) arrays, return a boolean mask
    selecting only the first contiguous segment (first revolution).

    wrap_threshold: how large a negative jump in arc must be to count as
                     a wrap-around to a new segment. Since arc in [-pi, pi],
                     a real wrap jump is close to -2*pi; 1.5*pi is a safe cut.
    """
    order = np.argsort(z)
    arc_sorted = arc[order]

    darc = np.diff(arc_sorted)

    # A wrap is a large NEGATIVE jump (from near +pi down to near -pi)
    wrap_idx = np.where(darc < -wrap_threshold)[0]

    if len(wrap_idx) == 0:
        first_segment_sorted_mask = np.ones_like(arc_sorted, dtype=bool)
    else:
        end = wrap_idx[0] + 1  # first wrap -> end of first segment
        first_segment_sorted_mask = np.zeros_like(arc_sorted, dtype=bool)
        first_segment_sorted_mask[:end] = True

    mask = np.zeros_like(z, dtype=bool)
    mask[order[first_segment_sorted_mask]] = True
    return mask

def helix_segment_length(first_arc, best_circle):
    """
    Arc length of a constant-pitch helix segment, using closed form.
    
    A helix with constant pitch, parametrized by theta, traces a straight
    line in the "unrolled" (r*theta, z) plane -> length = hypot(r*d_theta, d_z).
    
    Parameters
    ----------
    radius : float
        Helix radius (mm).
    theta0, theta1 : float
        Start/end angles in radians. Should already be unwrapped
        (i.e. NOT wrapped into [-pi, pi]) so d_theta reflects the true
        angular sweep, not the shortest-path wraparound.
    z0, z1 : float
        Start/end Z positions (mm).

    Returns
    -------
    length : float
        Arc length of the helix segment (mm).
    """
    x_c, y_c, radius = best_circle.center_x, best_circle.center_y, best_circle.radius
    
    z0 = first_arc[0,2]
    z1 = first_arc[-1,2]
    
    theta = np.arctan2(first_arc[:,1] - y_c, first_arc[:,0] - x_c)
    theta = np.unwrap(theta)
    theta0 = theta[0]
    theta1 = np.mean(theta[-3:])

    d_theta = theta1 - theta0
    d_z = z1 - z0
    return np.hypot(radius * d_theta, d_z)

def numerical_segment_length(first_arc, best_circle):
    x = first_arc[: , 0]
    y = first_arc[:, 1]
    z = first_arc[:, 2]
    x_c, y_c, radius = best_circle.center_x, best_circle.center_y, best_circle.radius
    x_trans = x - x_c
    y_trans = y - y_c
    theta0 = np.arctan2(y_trans[0], x_trans[0])
    phi = - theta0
    x_rot = x_trans*np.cos(phi) - y_trans*np.sin(phi)
    y_rot = x_trans*np.sin(phi) + y_trans*np.cos(phi)
    theta = np.arctan2(y_rot, x_rot)
    
    arc =np.unwrap(theta)*best_circle.radius
    z_fix = z-z[0]
    length = np.sqrt(arc**2 + z_fix**2)
    return length
    
def least_squares_circle_through_point(
    x: np.ndarray, y: np.ndarray, vertex: tuple[float, float]
) -> tuple[float, float, float, float]:
    px, py = vertex

    # Reduced coordinates, relative to the vertex instead of the mean
    u = x - px
    v = y - py

    # A circle through the origin has no constant term:
    # u^2 + v^2 = 2*uc*u + 2*vc*v
    # Minimizing the algebraic error gives the linear system A * c = B
    Suv = np.sum(u * v)
    Suu = np.sum(u**2.0)
    Svv = np.sum(v**2.0)
    Suuv = np.sum(u**2.0 * v)
    Suvv = np.sum(u * v**2.0)
    Suuu = np.sum(u**3.0)
    Svvv = np.sum(v**3.0)

    matrix_a = np.array([[Suu, Suv], [Suv, Svv]])
    matrix_b = np.array([(Suuu + Suvv) * 0.5, (Suuv + Svvv) * 0.5])
    try:
        c = np.linalg.solve(matrix_a, matrix_b)
    except Exception:
        return (np.nan, np.nan, np.nan, np.nan)

    xc = c[0] + px
    yc = c[1] + py

    # The radius is fixed by the constraint: distance from center to vertex
    radius = np.hypot(c[0], c[1])

    radii = np.sqrt((x - xc) ** 2.0 + (y - yc) ** 2.0)
    residual = np.sum((radii - radius) ** 2.0)
    return (xc, yc, radius, residual)

def Curved_Tracks_Estimation(cluster, det_params):
    direction = cluster.direction # We already know the direction from the clustering phase

    reaction_vertex = np.array([0.0, 0.0, 0.0])  # reaction vertex
    spyral_center = np.array([0.0, 0.0, 0.0])  # spiral center
    cluster_data = cluster.data.copy()

    if direction == Direction.BACKWARD:
        cluster_data = np.flip(cluster_data, axis=0)
    
    best_circle = ransac_circle_fit(cluster_data, threshold_circle=15.0, n_iterations=2000,
                                min_inliers=10, origin_weight=0.0, rng_seed=123)

    if best_circle is None:
        return None
        
    x = cluster_data[:, 0]
    y = cluster_data[:, 1]
    z = cluster_data[:, 2]
    if direction == Direction.BACKWARD:
        z = -z

    x_c, y_c, radius = best_circle.center_x, best_circle.center_y, best_circle.radius
    vertex_x, vertex_y = Closest_Origin(x_c, y_c, radius)


    x_trans, y_trans = x - x_c, y - y_c
    theta0 = np.arctan2(vertex_y - y_c, vertex_x - x_c)
    phi = -theta0
    x_rot = x_trans * np.cos(phi) - y_trans * np.sin(phi)
    y_rot = x_trans * np.sin(phi) + y_trans * np.cos(phi)
    phi = np.arctan2(y_rot, x_rot)

    mask_first = get_first_segment(z, phi)
    first_arc = cluster_data[mask_first,:]


    best_circle = ransac_circle_fit(first_arc, threshold_circle=15.0, n_iterations=2000,
                            min_inliers=10, origin_weight=0.0, rng_seed=123)

    if best_circle is None:
        return None

    spyral_center[:2] = best_circle.center_x, best_circle.center_y
    reaction_vertex[:2] = Closest_Origin(best_circle.center_x, best_circle.center_y, best_circle.radius)

    vertex_rho = np.linalg.norm(reaction_vertex[:2])

    if vertex_rho > det_params.beam_region_radius:
        return None
    

    rho_to_vertex = np.linalg.norm((first_arc[:, :2] - reaction_vertex[:2]), axis=1)

    test_index = max(10, int(len(first_arc) * 0.5))
    fit = linregress(cluster_data[:test_index, 2], rho_to_vertex[:test_index])

    if fit.slope == 0.0:  # type: ignore
        return None
    
    reaction_vertex[2] = -1.0 * fit.intercept / fit.slope  # type: ignore
    spyral_center[2] = reaction_vertex[2]

    polar = math.atan(fit.slope)
    if (polar > 0.0 and direction == Direction.BACKWARD) or (polar < 0.0 and direction == Direction.FORWARD):
        return None
    elif direction is Direction.BACKWARD:
        polar += math.pi

    # From the trigonometry of the system to the center
    azimuthal = math.atan2(reaction_vertex[1] - spyral_center[1], reaction_vertex[0] - spyral_center[0])
    if azimuthal < 0:
        azimuthal += 2.0 * math.pi
    azimuthal += math.pi * 0.5
    if azimuthal > math.pi * 2.0:
        azimuthal -= 2.0 * math.pi

    brho = (
            det_params.magnetic_field * best_circle.radius * 0.001 / np.abs(math.sin(polar))
        )  # Sometimes our angle is in the wrong quadrant
    if np.isnan(brho):
        brho = 0.0


    charge_deposited = 0
    arclength = None
    track_measurement = numerical_segment_length(first_arc, best_circle)
    for idx in range(len(first_arc)):
        if first_arc[idx, 4] == 0.5:
            charge_deposited += first_arc[idx, 3]
        else:
            charge_deposited += first_arc[idx, 3] / 1.6
    
        if track_measurement[idx] > 150.0:
            arclength = track_measurement[idx]
            break
    
    if arclength is None:
        # Track never reached 200mm — use its full measured length instead
        arclength = track_measurement[-1]
    
    if arclength == 0.0:
        return None
    #arclength = helix_segment_length(first_arc, best_circle)
    dEdx = charge_deposited/arclength
    
    return NewEstimateResult(
            vertex_x=reaction_vertex[0],
            vertex_y=reaction_vertex[1],
            vertex_z=reaction_vertex[2],
            center_x=spyral_center[0],
            center_y=spyral_center[1],
            center_z=spyral_center[2],
            polar=polar,
            azimuthal=azimuthal,
            brho=brho,
            dEdx=dEdx,
            sqrt_dEdx=np.sqrt(np.fabs(dEdx)),
            dE=charge_deposited,
            arclength=arclength,
            direction=direction.value,
        )

def Straight_Tracks_Estimation(cluster, ref_est, det_params):
    direction = cluster.direction # We already know the direction from the clustering phase

    reaction_vertex = np.array([0.0, 0.0, 0.0])  # reaction vertex
    spyral_center = np.array([0.0, 0.0, 0.0])  # spiral center
    cluster_data = cluster.data.copy()

    if direction == Direction.BACKWARD:
        cluster_data = np.flip(cluster_data, axis=0)
        
    x = cluster_data[:, 0]
    y = cluster_data[:, 1]
    z = cluster_data[:, 2]
    if direction == Direction.BACKWARD:
        z = -z

    vertex_x, vertex_y = ref_est.vertex_x, ref_est.vertex_y
    reaction_vertex[:2] = vertex_x, vertex_y 
    
    x_c, y_c, radius, _ = least_squares_circle_through_point_geometric(x, y, (vertex_x, vertex_y))
    #x_c, y_c, radius, _ = least_squares_circle_through_point(x, y, (vertex_x, vertex_y))
    best_circle = Circle(center_x =  x_c,
                         center_y = y_c,
                         radius = radius)


    x_trans, y_trans = x - x_c, y - y_c
    theta0 = np.arctan2(vertex_y - y_c, vertex_x - x_c)
    phi = -theta0
    x_rot = x_trans * np.cos(phi) - y_trans * np.sin(phi)
    y_rot = x_trans * np.sin(phi) + y_trans * np.cos(phi)
    phi = np.arctan2(y_rot, x_rot)

    mask_first = get_first_segment(z, phi)
    first_arc = cluster_data[mask_first,:]
    

    spyral_center[:2] = x_c, y_c
    reaction_vertex[:2] = vertex_x, vertex_y

    vertex_rho = np.linalg.norm(reaction_vertex[:2])

    if vertex_rho > det_params.beam_region_radius:
        return None
    

    rho_to_vertex = np.linalg.norm((first_arc[:, :2] - reaction_vertex[:2]), axis=1)

    test_index = min(len(first_arc), max(10, int(len(first_arc) * 0.5)))
    if test_index < 3:  # linregress needs at least 2 points; 3+ gives a meaningful fit
        return None
    #test_index = max(10, int(len(first_arc) * 0.5))
    fit = linregress(cluster_data[:test_index, 2], rho_to_vertex[:test_index])

    if fit.slope == 0.0:  # type: ignore
        return None
    
    reaction_vertex[2] = -1.0 * fit.intercept / fit.slope  # type: ignore
    spyral_center[2] = reaction_vertex[2]

    polar = math.atan(fit.slope)
    if (polar > 0.0 and direction == Direction.BACKWARD) or (polar < 0.0 and direction == Direction.FORWARD):
        return None
    elif direction is Direction.BACKWARD:
        polar += math.pi

    # From the trigonometry of the system to the center
    azimuthal = math.atan2(reaction_vertex[1] - spyral_center[1], reaction_vertex[0] - spyral_center[0])
    if azimuthal < 0:
        azimuthal += 2.0 * math.pi
    azimuthal += math.pi * 0.5
    if azimuthal > math.pi * 2.0:
        azimuthal -= 2.0 * math.pi

    brho = (
            det_params.magnetic_field * best_circle.radius * 0.001 / np.abs(math.sin(polar))
        )  # Sometimes our angle is in the wrong quadrant
    if np.isnan(brho):
        brho = 0.0


    charge_deposited = 0
    arclength = None
    track_measurement = numerical_segment_length(first_arc, best_circle)
    for idx in range(len(first_arc)):
        if first_arc[idx, 4] == 0.5:
            charge_deposited += first_arc[idx, 3]
        else:
            charge_deposited += first_arc[idx, 3] / 1.6
    
        if track_measurement[idx] > 150.0:
            arclength = track_measurement[idx]
            break
    
    if arclength is None:
        # Track never reached 200mm — use its full measured length instead
        arclength = track_measurement[-1]
    
    if arclength == 0.0:
        return None
    #arclength = helix_segment_length(first_arc, best_circle)
    dEdx = charge_deposited/arclength
    
    return NewEstimateResult(
            vertex_x=reaction_vertex[0],
            vertex_y=reaction_vertex[1],
            vertex_z=reaction_vertex[2],
            center_x=spyral_center[0],
            center_y=spyral_center[1],
            center_z=spyral_center[2],
            polar=polar,
            azimuthal=azimuthal,
            brho=brho,
            dEdx=dEdx,
            sqrt_dEdx=np.sqrt(np.fabs(dEdx)),
            dE=charge_deposited,
            arclength=arclength,
            direction=direction.value,
        )


def my_estimate_physics(cluster, ref_est, det_params):
	
    if Track_Classification(cluster):
        result = Curved_Tracks_Estimation(cluster, det_params)
        if result is not None and ref_est is None:
            ref_est = result  # first valid curved track is the reference
    elif ref_est is None:
        result = Curved_Tracks_Estimation(cluster, det_params)
    else:
	    result = Straight_Tracks_Estimation(cluster, ref_est, det_params)
    return result
