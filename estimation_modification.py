import numpy as np
import sys
sys.path.append('/home/danilo-marcato/Documents/RANSAC')
import pyTracking

def RANSAC_Fit(data, dist = 20.0, min_points = 10, interaction = 1500):
    #data = cluster.point_cloud.data
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    q = data[:, 3]

    obj = pyTracking.Ransac_circle()
    obj.Init(x, y, z, q)
    obj.Solve(dist, min_points, interaction)
    fitted_circle = obj.GetClusters()
    return fitted_circle

def Closest_Origin(center_x, center_y, radius):
    # Calculate distance from origin to center
    d = np.sqrt(center_x**2 + center_y**2)
    
    x = center_x - (center_x / d) * radius
    y = center_y - (center_y / d) * radius
    
    return (x, y)

def First_Arc(cluster_data):
    reaction_vertex = np.array([0.0, 0.0, 0.0])  # reaction vertex
    spyral_center = np.array([0.0, 0.0, 0.0, 0.0])  # spiral center
    
    rho_to_vertex = np.linalg.norm(cluster_data[1:, :2] - reaction_vertex[:2], axis=1)#
    maximum = np.argmax(np.linalg.norm(cluster_data[:, :2], axis=1) > 155.0)
    
    if(maximum==0):
        maximum = len(cluster_data)
    
    first_arc = cluster_data[: (maximum + 1)]
        
    fitted_circle = RANSAC_Fit(first_arc, min_points = max(10, int(maximum * 0.3)))
    
    if len(fitted_circle) == 0:
        return None, None, None

    else:
        best_circle = fitted_circle[0]
        x_c, y_c, radius = best_circle.ClusterFitP1
        spyral_center[:2] = x_c, y_c
        spyral_center[3] = radius
        reaction_vertex[:2] = Closest_Origin(x_c, y_c, radius)
        first_arc = first_arc[best_circle.ClusterIndex, :]
    
    return reaction_vertex, spyral_center, first_arc

def Angles_Estimation(first_arc, reaction_vertex, spyral_center, det_params):
    
    rho_to_vertex = np.linalg.norm(first_arc[:, :2] - reaction_vertex[:2], axis=1)#
    maximum = np.argmax(rho_to_vertex)
    
    if len(first_arc[:maximum, 2]) >= 2:
        fit = linregress(first_arc[:maximum, 2],rho_to_vertex[:maximum])
    else:
        return None, None, None, None
    
    # fitted values
    y_fit = fit.slope * first_arc[:maximum, 2] + fit.intercept

    if fit.slope == 0.0:  # type: ignore
        return None, None, None, None

    reaction_vertex[2] = -1.0 * fit.intercept / fit.slope  # type: ignore
    spyral_center[2] = reaction_vertex[2]

    vertex_rho = np.linalg.norm(reaction_vertex[:2])
    if vertex_rho > det_params.beam_region_radius:
        return None, None, None, None

    polar = math.atan(fit.slope)  # type: ignore

    # From the trigonometry of the system to the center
    azimuthal = math.atan2(reaction_vertex[1] - spyral_center[1], reaction_vertex[0] - spyral_center[0])
    
    if azimuthal < 0:
        azimuthal += 2.0 * math.pi
    azimuthal += math.pi * 0.5
    if azimuthal > math.pi * 2.0:
        azimuthal -= 2.0 * math.pi

    return reaction_vertex, spyral_center, polar, azimuthal

def Brho_Estimation(spyral_center, polar, det_params):
    brho = (
            det_params.magnetic_field * spyral_center[3] * 0.001 / np.abs(math.sin(polar))
        )  # Sometimes our angle is in the wrong quadrant
    if np.isnan(brho):
        brho = 0.0
    return brho

def dE_Estimation(first_arc, spyral_center):
    theta = np.arctan2(first_arc[:,1] - spyral_center[1], first_arc[:,0] - spyral_center[0])
    
    dtheta = theta[-1] - theta[0]
    dz = first_arc[-1,2] - first_arc[0,2]
    
    dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))
    
    arclength = np.sqrt(spyral_center[3]**2 * dtheta**2+ dz**2)

    charge_deposited = first_arc[0, 3]
    small_pad_cutoff = -1  # Where do we cross from big pads to small pads
    for idx in range(len(first_arc) - 1):
        # Stop integrating if we leave the small pad region
        if np.linalg.norm(first_arc[idx + 1, :2]) > 152.0:
            small_pad_cutoff = idx + 1
            break
        # arclength += np.linalg.norm(first_arc[idx + 1, :3] - first_arc[idx, :3])
        charge_deposited += first_arc[idx + 1, 3]
    if charge_deposited == first_arc[0, 3]:
        return None, None, None

    dEdx = charge_deposited / arclength

    return charge_deposited, arclength, dEdx
    
def My_Estimation(
    cluster_index: int,
    cluster: Cluster,
    ic_amplitude: float,
    ic_centroid: float,
    ic_integral: float,
    ic_multiplicity: float,
    orig_run: int,
    orig_event: int,
    det_params: DetectorParameters,
) -> EstimateResult:

    cluster_data = cluster.data.copy()
    direction = Direction.FORWARD
    reaction_vertex, spyral_center, first_arc = First_Arc(cluster_data)
    if reaction_vertex is None:
        return None
        
    reaction_vertex, spyral_center, polar, azimuthal = Angles_Estimation(first_arc, reaction_vertex, spyral_center, det_params)
    if reaction_vertex is None:
        return None
    
    if(polar < 0):
        cluster_data = np.flip(cluster_data, axis=0)
        direction = Direction.BACKWARD
        reaction_vertex, spyral_center, first_arc = First_Arc(cluster_data)
        if reaction_vertex is None:
            return None
        reaction_vertex, spyral_center, polar, azimuthal = Angles_Estimation(first_arc, reaction_vertex, spyral_center, det_params)
        if reaction_vertex is None:
            return None
        polar += math.pi
    
    brho = Brho_Estimation(spyral_center, polar, det_params)
    
    charge_deposited, arclength, dEdx = dE_Estimation(first_arc, spyral_center)
    if charge_deposited is None:
        return None

    return EstimateResult(
            event=cluster.event,
            cluster_index=cluster_index,
            cluster_label=cluster.label,
            orig_run=orig_run,
            orig_event=orig_event,
            ic_amplitude=ic_amplitude,
            ic_centroid=ic_centroid,
            ic_integral=ic_integral,
            ic_multiplicity=ic_multiplicity,
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
