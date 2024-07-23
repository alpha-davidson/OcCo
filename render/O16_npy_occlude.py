import numpy as np, open3d

INPUT_PATH = 'data/O16/split/train.npy'
COMPLETE_PATH = 'data/O16/occluded/train_complete.npy'
OCCLUDED_PATH = 'data/O16/occluded/train_occluded.npy'

REQUIRED_CHARGE = 80 # Sensitivity to noise; higher value means more noise removal (and vice versa)
NUM_OF_POINTS = 512
OCCLUSION_PERSISTANCE = 0.75 # Percentage of points that should be left after occlusion
SNAPSHOTS_PER_CLOUD = 10
CAMERA_BOUNDARIES = [[-300, 300], [-200, 1200], [-300, 300]] # Camera can appear anywhere on the surface of this rectangular prism

#np.random.seed(12079522)

def downsamplePointCloud(points, new_num_of_points):
    indices = np.random.choice(points.shape[0], size=new_num_of_points, replace=False)
    new_points = points[indices]
    return new_points

def randomCameraPosition(borders):
    random_x = np.random.randint(borders[0][0], borders[0][1])
    random_z = np.random.randint(borders[1][0], borders[1][1])
    random_y = np.random.randint(borders[2][0], borders[2][1])
    camera = np.asarray([random_x, random_z, random_y])
    x_z_y = np.random.randint(0, 3)
    left_right = np.random.randint(0, 2)
    camera[x_z_y] = borders[x_z_y][left_right]
    return camera

def occludePointCloud(points, new_num_of_points, camera):
    START_RADIUS = 1000 # Hold-over from previous occlusion method; doesn't impact much
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points)
    radius = START_RADIUS
    new_points = np.asarray([])
    while len(new_points) < new_num_of_points:
        _, visible_points = pcd.hidden_point_removal(camera, radius)
        new_pcd = pcd.select_by_index(visible_points)
        new_points = np.asarray(new_pcd.points)
        radius *= new_num_of_points / len(new_points) # Insanely fast and insanely accurate
    if len(new_points) > new_num_of_points:
        new_points = downsamplePointCloud(new_points, new_num_of_points)
    return new_points

def normalizePointCloud(points):
    # -1 to 1
    points[:,0] = points[:,0] / 250
    points[:,1] = points[:,1] / 250
    points[:,2] = points[:,2] / 500 - 1
    return points

data = np.load(INPUT_PATH, mmap_mode='r')
num_of_events = data.shape[0]
print('Number of events: ' + str(num_of_events))

complete_clouds = []
occluded_clouds = []
starting_num_of_points = int(NUM_OF_POINTS / OCCLUSION_PERSISTANCE)

for event in data:
    event = event[np.any(event, axis=1)] # Remove all-zero rows
    event = event[np.where(event[:,4] >= REQUIRED_CHARGE)] # Remove noise
    points = event[:,:3] # Take only x,y,z
    num_of_points = points.shape[0]
    if num_of_points < starting_num_of_points:
        continue
    points = downsamplePointCloud(points, starting_num_of_points)
    occluded_clouds_i = []
    for j in range(SNAPSHOTS_PER_CLOUD):
        camera = randomCameraPosition(CAMERA_BOUNDARIES)
        new_points = occludePointCloud(points, NUM_OF_POINTS, camera)
        new_points = normalizePointCloud(new_points)
        occluded_clouds_i.append(new_points)
    points = downsamplePointCloud(points, NUM_OF_POINTS) # Downsample
    points = normalizePointCloud(points)
    complete_clouds.append(points)
    occluded_clouds.append(occluded_clouds_i)

complete_clouds = np.asarray(complete_clouds)
occluded_clouds = np.asarray(occluded_clouds)

print('Now saving ' + str(complete_clouds.shape[0]) + ' complete point clouds...')
np.save(COMPLETE_PATH, complete_clouds)
np.save(OCCLUDED_PATH, occluded_clouds)