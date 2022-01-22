from mlagents.envs.environment import UnityEnvironment
from PIL import Image
from skimage import io, img_as_ubyte
import numpy as np
import random
import os
import sys
import argparse
from utils import ancestral_sampler_fix_sigma, get_random_color


parser = argparse.ArgumentParser(
    description="Generate images of a given type_id or car_id or color_id")
parser.add_argument("--env_path", type=str,
                    default="./Build-linux/VehicleX.x86_64")
parser.add_argument("--out_dir", type=str, required=True,
                    help="where to save generated images, REQUIRED")
parser.add_argument("--n_images", type=int, required=True,
                    help="how many images to generate, REQUIRED")
parser.add_argument("--car_id", type=int,
                    help="Vehicle model id, integer in range of [1, 1362]")
parser.add_argument("--type_id", type=int, help="""
Vehicle type id from the following values:
0 sedan
1 suv
2 van
3 hatchback
4 mpv
5 pickup
6 bus
7 truck
8 estate
9 sportscar
10 RV
""")

parser.add_argument("--color_id", type=int, help="""
Vehicle color id from the following values:
0 yellow
1 orange
2 green
3 gray
4 red
5 blue
6 white
7 golden
8 brown
9 black
10 purple
11 pink
""")

opt = parser.parse_args()

if not os.path.isdir(opt.out_dir):
    os.mkdir(opt.out_dir)

if (opt.color_id is not None and opt.color_id not in range(0, 12)) or \
   (opt.car_id is not None and opt.car_id not in range(1, 1363)) or \
   (opt.type_id is not None and opt.type_id not in range(0, 11)):
    print("Error: color_id, car_id, or type_id in wrong interval, check usage info.")
    sys.exit(1)

if opt.n_images < 1:
    print("Error: n_images < 1")
    sys.exit(1)


train_mode = False
env = UnityEnvironment(file_name=opt.env_path, timeout_wait=180)
default_brain = env.brain_names[0]

N_IMAGES = opt.n_images
n_generated = 0
CNT_INIT = len(os.listdir(opt.out_dir))

env_info = env.reset(train_mode=False)[default_brain]
intensity_range = (0.0, 3.0)
light_direction_range = (45., 135.)
camera_height_range = (4., 13.)
camera_distance_y_range = (-2., 13.)
camera_distance_x_range = (-5., 5.)
scene_id_range = (1, 59)

N_BATCH = 100
q = N_BATCH
while True:
    if q >= N_BATCH:
        q = 0
        # angle = np.random.permutation(
        #    ancestral_sampler_fix_sigma(mu=[0] * 6, size=N_IMAGES))
        angle = np.random.uniform(0.0, 360.0, size=N_BATCH)

        temp_intensity_list = np.random.normal(
            loc=0, scale=np.sqrt(0.4), size=N_BATCH)
        temp_light_direction_x_list = np.random.normal(
            loc=45, scale=np.sqrt(50), size=N_BATCH)
        Cam_height_list = np.random.normal(loc=4, scale=2, size=N_BATCH)
        Cam_distance_y_list = np.random.normal(loc=-2, scale=3, size=N_BATCH)

    angle[q] = angle[q] % 360
    temp_intensity_list[q] = min(
        max(intensity_range[0], temp_intensity_list[q]), intensity_range[1])
    temp_light_direction_x_list[q] = min(max(light_direction_range[0], temp_light_direction_x_list[q]),
                                         light_direction_range[1])
    Cam_height_list[q] = min(
        max(camera_height_range[0], Cam_height_list[q]), camera_height_range[1])
    Cam_distance_y_list[q] = min(max(
        camera_distance_y_range[0], Cam_distance_y_list[q]), camera_distance_y_range[1])
    Cam_distance_x = random.uniform(
        camera_distance_x_range[0], camera_distance_x_range[1])
    scene_id = random.randint(scene_id_range[0], scene_id_range[1])

    color_R, color_G, color_B = get_random_color()

    env_info = env.step([[angle[q], temp_intensity_list[q], temp_light_direction_x_list[q],
                          Cam_distance_y_list[q], Cam_distance_x, Cam_height_list[q],
                          scene_id, train_mode]])[default_brain]

    done = env_info.local_done[0]
    q += 1
    if done:
        env_info = env.reset(train_mode=False)[default_brain]
        #       env_info = env.reset(train_mode=True)[default_brain]
        continue

    car_id = int(env_info.vector_observations[0][4])
    color_id = int(env_info.vector_observations[0][5])
    type_id = int(env_info.vector_observations[0][6])

    print(f"car_id:{car_id}, type_id:{type_id}, color_id:{color_id}")

    if (opt.car_id is not None and car_id != opt.car_id) or \
       (opt.color_id is not None and color_id != opt.color_id) or \
       (opt.type_id is not None and type_id != opt.type_id):
        continue

    observation_gray = np.array(env_info.visual_observations[1])
    x, y = (observation_gray[0, :, :, 0] > 0).nonzero()
    observation = np.array(env_info.visual_observations[0])

    if observation.shape[3] == 3 and len(y) > 0 and min(y) > 10 and min(x) > 10:
        ori_img = observation[0, min(
            x) - 10:max(x) + 10, min(y) - 10:max(y) + 10, :]

        filename = "{}_{}_{}_{}.jpg".format(
            color_id, type_id, car_id, CNT_INIT + n_generated)
        io.imsave(os.path.join(opt.out_dir, filename), img_as_ubyte(ori_img))

        n_generated += 1
        print("----> Good images: {}".format(n_generated))
        if n_generated >= N_IMAGES:
            break

        #img = Image.fromarray(ori_img).convert("RGB")
