## Occlusion Completion: TensorFlow Implementation

This is the `TensorFlow` implementation for Occlusion Completion (OcCo). It is the primary framework for Occlusion Completion; the `PyTorch` implementation has not yet been tested.

### Setup
It is unknown if `Requirements_TF.txt` works in its current form for a `pipenv` environment. Instead, use `conda`:
1. `conda create -n occo-tf python=3.7`
2. `conda install python-lmdb>=0.9 numpy=1.19.5 pyarrow>=0.10.0 matplotlib>=2.1.0 tensorflow-gpu=1.15.0`
3. `conda install -c open3d-admin open3d>=0.9.0.0`
4. `conda install -c nvidia cuda-nvcc cuda-cudart-dev`
5. `pip install msgpack==0.5.6 tensorpack==0.8.9 open3d-python==0.7.0.0`

As of June 28 2024, this resulted in the following versions:
- `cuda-cudart-dev==12.4.127`
- `cuda-nvcc==12.4.131`
- `matplotlib==3.5.3`
- `msgpack==0.5.6`
- `numpy==1.19.5`
- `open3d==0.11.2`
- `open3d-python==0.7.0.0`
- `pyarrow==8.0.0`
- `python-lmdb==1.4.0`
- `tensorflow-gpu==1.15.0`
- `tensorpack==0.8.9`

The model may work using newer versions of some of these packages, but no gurantees.

Next, to actually be able to train the model, run the following two commands:
1. `cd` into `$HOME/.conda/envs/occo-tf/lib/python3.7/site-packages/tensorflow_core/` and run `ln -s libtensorflow_framework.so.1 libtensorflow_framework.so`.
2. `cd` into `$HOME/OcCo/OcCo_TF/pc_distance/` and run `make`.

### Training

The model is currently set to train for 30 epochs using the `train_completion.py` script. Submit it as a SLURM job using `train_completion.sh`.

Note: the `--num_input_points` and `--num_gt_points` flags are for clarifying to the model how many points are in each occluded and complete point cloud, respectively, **only in the datasets provided using the** `--lmdb_train` **and** `--lmdb_valid` **flags. They do not influence how many points the model predicts when completing a point cloud.** To change the actual completions of the model, edit the architecture itself.

### Architecture

The encoder-decoder architectures used in training the model can be found in `completion_models/`. There are 3 models already implemented for you: Dynamic Graph Convolutional Neural Network (DGCNN), Point Completion Network (PCN), and PointNet. Each can be used with either Chamfer Distance (CD) or Earth Mover Distance (EMD) as the loss function.

The default architecture is a PCN with Chamfer Distance, the details of which can be found in `completion_models/pcn_cd.py`. To specify a different architecture, set the `--model_type` flag accordingly.