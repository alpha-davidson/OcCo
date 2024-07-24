import argparse, numpy as np, open3d

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, default='data/O16/split/train.npy')
parser.add_argument('--complete_path', type=str, default='data/O16/occluded/train_complete.npy')
parser.add_argument('--occluded_path', type=str, default='data/O16/occluded/train_occluded.npy')
parser.add_argument('--required_charge', type=float, default=80) # Sensitivity to noise; higher value means more noise removal (and vice versa)
parser.add_argument('--num_of_points', type=int, default=512)
parser.add_argument('--occlusion_persistance', type=float, default=0.75) # Percentage of points that should be left after occlusion
parser.add_argument('--snapshots_per_cloud', type=int, default=10)
parser.add_argument('--camera_boundaries', type=float, nargs=6, default=[[-300, 300], [-200, 1200], [-300, 300]]) # Camera can appear anywhere on the surface of this rectangular prism
args = parser.parse_args()

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

data = np.load(args.input_path, mmap_mode='r')
num_of_events = data.shape[0]
print('Number of events: ' + str(num_of_events))

complete_clouds = []
occluded_clouds = []
starting_num_of_points = int(args.num_of_points / args.occlusion_persistance)

for event in data:
    event = event[np.any(event, axis=1)] # Remove all-zero rows
    event = event[np.where(event[:,4] >= args.required_charge)] # Remove noise
    points = event[:,:3] # Take only x,y,z
    num_of_points = points.shape[0]
    if num_of_points < starting_num_of_points:
        continue
    points = downsamplePointCloud(points, starting_num_of_points)
    occluded_clouds_i = []
    for j in range(args.snapshots_per_cloud):
        camera = randomCameraPosition(args.camera_boundaries)
        new_points = occludePointCloud(points, args.num_of_points, camera)
        new_points = normalizePointCloud(new_points)
        occluded_clouds_i.append(new_points)
    points = downsamplePointCloud(points, args.num_of_points) # Downsample
    points = normalizePointCloud(points)
    complete_clouds.append(points)
    occluded_clouds.append(occluded_clouds_i)

complete_clouds = np.asarray(complete_clouds)
occluded_clouds = np.asarray(occluded_clouds)

print('Now saving ' + str(complete_clouds.shape[0]) + ' complete point clouds...')
np.save(args.complete_path, complete_clouds)
np.save(args.occluded_path, occluded_clouds)