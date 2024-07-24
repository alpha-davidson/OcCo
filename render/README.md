This directory contains code that generates occluded point clouds that can be fed into the model. Currently, it only operates on O16 `.npy` files, but the overall process can likely be extended to other experiments or file formats.

Necessary packages:
- `matplotlib` (optional)
- `numpy`
- `open3d`
- `scikit-learn`
- `tensorpack`

The default name of the environment used throughout the process is `occo-data`.

Steps for generating valid data:
1. Given a raw O16 `.npy` file (i.e. `O16_w_event_keys.npy`), run `O16_npy_split.sh` to split it into training and testing sets of data. Then, run it again to split the training set further into training and validation sets (you may want to run that second split again occasionally to prevent overfitting).
2. Given any O16 `.npy` file, run `O16_npy_occlude.sh` to generate occluded point clouds. This script will output one `.npy` file representing complete point clouds and one `.npy` file representing occluded point clouds.
3. OPTIONAL: If you wish to visualize the generated occlusions, you can run `O16_npy_visualize.ipynb` to view graphs of complete clouds compared with their respective occlusions. The function that generates the graphs is `comparePointClouds()`; feel free to call this in whatever way you wish.
4. Given a complete clouds data file and its respective occluded clouds data file, run `O16_npy_serialize.sh` to combine the two files into one `.lmdb` file that can be fed into the model!

For each step, double-check the paths and parameters in the shell script before running it.