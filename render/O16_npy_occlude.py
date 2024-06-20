import numpy as np, open3d

INPUT_PATH = 'temp/train.npy'
COMPLETE_PATH = 'temp/train_complete.npy'
OCCLUDED_PATH = 'temp/train_occluded.npy'

REQUIRED_CHARGE = 80 # Sensitivity to noise; higher value means more noise removal (and vice versa)
REQUIRED_POINTS = 500 # Required number of points, after noise removal, to be considered for occlusion
OCCLUSION_PERSISTANCE = 0.7 # Percentage of points that should be left after occlusion
SNAPSHOTS_PER_CLOUD = 10
CAMERA_BOUNDARIES = [[-300, 300], [-200, 1200], [-300, 300]] # Camera can appear anywhere on the surface of this rectangular prism

def randomCameraPosition(borders):
    random_x = np.random.randint(borders[0][0], borders[0][1])
    random_z = np.random.randint(borders[1][0], borders[1][1])
    random_y = np.random.randint(borders[2][0], borders[2][1])
    camera = np.asarray([random_x, random_z, random_y])
    x_z_y = np.random.randint(0, 3)
    left_right = np.random.randint(0, 2)
    camera[x_z_y] = borders[x_z_y][left_right]
    return camera

def smartOccludePointCloud(points, camera, percentage):
    START_RADIUS = 1000
    RADIUS_GROWTH = 1.05
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points)
    new_points = np.asarray([])
    radius = START_RADIUS
    while len(new_points) / len(points) < percentage:
        radius *= RADIUS_GROWTH
        _, visible_points = pcd.hidden_point_removal(camera, radius)
        new_pcd = pcd.select_by_index(visible_points)
        new_points = np.asarray(new_pcd.points)
    return new_points

data = np.load(INPUT_PATH, mmap_mode='r')
num_of_events = data.shape[0]
print('Number of events: ' + str(num_of_events))

complete_clouds = []
occluded_clouds = []

for index in range(int(num_of_events / 100)):
    if (index % 100 == 0):
        print('Now on: Event #' + str(index))
    event = data[index]
    event = event[np.any(event, axis=1)] # Remove all-zero rows
    event = event[np.where(event[:,4] >= REQUIRED_CHARGE)] # Remove noise
    points = event[:,:3] # Take only x,y,z
    num_of_points = points.shape[0]
    if num_of_points < REQUIRED_POINTS:
        continue
    points = points[np.random.choice(num_of_points, size=REQUIRED_POINTS, replace=False)] # Downsample
    new_num_of_points = int(REQUIRED_POINTS * OCCLUSION_PERSISTANCE)
    complete_clouds.append(points)
    occluded_clouds_i = []
    for j in range(SNAPSHOTS_PER_CLOUD):
        camera = randomCameraPosition(CAMERA_BOUNDARIES)
        new_points = smartOccludePointCloud(points, camera, OCCLUSION_PERSISTANCE)
        print(new_points.shape[0])
        new_points = new_points[np.random.choice(new_points.shape[0], size=new_num_of_points, replace=False)] # Downsample
        occluded_clouds_i.append(new_points)
    occluded_clouds.append(occluded_clouds_i)

complete_clouds = np.asarray(complete_clouds)
occluded_clouds = np.asarray(occluded_clouds)
print('Now saving ' + str(complete_clouds.shape[0]) + ' complete point clouds...')
np.save(COMPLETE_PATH, complete_clouds)
np.save(OCCLUDED_PATH, occluded_clouds)