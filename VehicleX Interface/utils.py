import numpy as np
import json
import random
import argparse
import time


def write_json(json_path, cam_info, attribute_list, task_info, best_score):
    idx = 0
    for attribute in cam_info['attributes'].items():
        attribute_name = attribute[0]
        attribute_content = attribute[1]
        if attribute_content[0] == 'Gaussian Mixture':
            attribute_content[2] = [
                float(i) for i in attribute_list[idx: idx + len(attribute_content[2])]]
            idx += len(attribute_content[2])
        if attribute_content[0] == 'Gaussian':
            attribute_content[2] = float(attribute_list[idx])
            idx += 1
    cam_info["FD distance"] = best_score
    with open(json_path, 'w') as outfile:
        json.dump(task_info, outfile, indent=4)


def sample_values(attribute_list, variance_list, order_list, dataset_size):
    cnt = 0
    for attribute_name in order_list:
        if attribute_name == "orientation":
            angle = np.random.permutation(ancestral_sampler(
                mu=attribute_list[cnt:cnt + 6], sigma=variance_list[:6], size=dataset_size * 3))
            cnt = cnt + 6
        if attribute_name == "light intensity":
            temp_intensity_list = np.random.normal(
                loc=attribute_list[cnt], scale=variance_list[cnt], size=dataset_size + 100)
            cnt = cnt + 1
        if attribute_name == "light direction":
            temp_light_direction_x_list = np.random.normal(
                loc=attribute_list[cnt], scale=variance_list[cnt], size=dataset_size + 100)
            cnt = cnt + 1
        if attribute_name == "camera height":
            Cam_height_list = np.random.normal(
                loc=attribute_list[cnt], scale=variance_list[cnt], size=dataset_size + 100)
            cnt = cnt + 1
        if attribute_name == "camera distance":
            Cam_distance_y_list = np.random.normal(
                loc=attribute_list[9], scale=variance_list[9], size=dataset_size + 100)
            cnt = cnt + 1
    return angle, temp_intensity_list, temp_light_direction_x_list, Cam_height_list, Cam_distance_y_list


def get_color_distribution(id_num, model_num):
    vehicle_info = {}
    vehicle_info['model'] = []
    vehicle_info['R'] = []
    vehicle_info['G'] = []
    vehicle_info['B'] = []
    for i in range(id_num):
        vehicle_info['model'].append(i % model_num)
        vehicle_info['R'].append(float(random.randint(0, 255)) / 255)
        vehicle_info['G'].append(float(random.randint(0, 255)) / 255)
        vehicle_info['B'].append(float(random.randint(0, 255)) / 255)
    return vehicle_info


def get_random_color():
    return tuple([float(random.randint(0, 255)) / 255 for i in range(3)])
    
# 0 yellow
# 1 orange
# 2 green
# 3 gray
# 4 red
# 5 blue
# 6 white
# 7 golden
# 8 brown
# 9 black
# 10 purple
# 11 pink

color_to_rgb = {
    0: [(255, 201, 31), (255, 207, 32), (251, 226, 18), (224, 225, 61)],
    1: [(247, 134, 22), (249, 164, 88), (246, 174, 32), (247, 134, 22)],
    2: [(21, 92, 45), (102, 184, 31), (29, 56, 62), (69, 89, 75), (131, 197, 102),
        (78, 100, 67), (176, 238, 110)],
    3: [(151, 154, 151), (69, 75, 79), (60, 63, 71), (38, 40, 42), (51, 58, 60),
        (54, 58, 63), (160, 161, 153)],
    4: [(192, 14, 26), (218, 25, 24), (123, 26, 34), (73, 17, 29), (182, 15, 37),
        (115, 32, 33), (222, 15, 24), (169, 71, 68)],
    5: [(175, 214, 228), (27, 103, 112), (34, 46, 70), (48, 76, 126), (35, 49, 85),
        (99, 123, 167), (57, 71, 98), (11, 156, 241), (17, 37, 82)],
    6: [(255, 255, 246), (234, 234, 234), (223, 221, 208), (252, 249, 241)],
    7: [(218,165,32), (184,134,11), (255,215,0), (194, 148, 79)],
    8: [(101, 63, 35), (34, 27, 25), (119, 92, 62), (64, 46, 43), (58, 42, 27),
        (181, 160, 121), (69, 56, 49)],
    9: [(10, 12, 23), (13, 17, 22), (10, 10, 10), (28, 29, 33)],
    10:[(98, 18, 118), (107, 31, 123)],
    11:[(242, 31, 153), (223, 88, 145), (253, 214, 205)]
}

def get_color_by_id(color_id):
    if color_id not in color_to_rgb:
        raise ValueError("Bad color_id")
    rgb_uint = random.sample(color_to_rgb[color_id], 1)[0]
    return tuple(map(lambda x: float(x) / 255, rgb_uint))


model_to_type = {'102': '0', '104': '0', '109': '0', '114': '0', '116': '0', '120': '0', '127': '0', '133': '0', '137': '0', '139': '0', '13': '0', '140': '0', '143': '0', '156': '0', '15': '0', '160': '0', '161': '0', '166': '0', '167': '0', '168': '0', '172': '0', '17': '0', '40': '0', '43': '0', '45': '0', '46': '0', '47': '0', '48': '0', '51': '0', '54': '0', '64': '0', '66': '0', '74': '0', '77': '0', '82': '0', '90': '0', '94': '0', '96': '0', '98': '0', '9': '0', '107': '10', '138': '10', '147': '10', '152': '10', '1': '10', '21': '10', '29': '10', '37': '10', '3': '10', '6': '10', '105': '1', '124': '1', '130': '1', '136': '1', '164': '1', '171': '1', '36': '1', '39': '1', '50': '1', '56': '1', '61': '1', '63': '1', '71': '1', '76': '1', '7': '1', '84': '1', '92': '1', '99': '1', '121': '2', '129': '2', '148': '2', '165': '2', '16': '2', '2': '2', '44': '2', '78': '2', '101': '3', '103': '3', '106': '3', '108': '3', '10': '3', '110': '3', '111': '3', '11': '3', '122': '3', '126': '3', '132': '3', '134': '3', '144': '3', '146': '3', '154': '3', '155': '3', '158': '3', '163': '3', '169': '3', '173': '3', '177': '3', '18': '3', '24': '3', '25': '3', '26': '3', '31': '3', '32': '3', '33': '3', '55': '3', '58': '3', '60': '3', '67': '3', '68': '3', '70': '3', '73': '3', '75': '3', '79': '3', '85': '3', '93': '3', '5': '4', '86': '4', '145': '5', '162': '5', '170': '5', '87': '5', '91': '5', '123': '6', '128': '6', '153': '6', '4': '6', '57': '6', '97': '6', '117': '7', '142': '7', '149': '7', '174': '7', '30': '7', '34': '7', '38': '7', '49': '7', '80': '7', '81': '7', '88': '7', '159': '8', '52': '8', '89': '8', '8': '8', '100': '9', '112': '9', '113': '9', '115': '9', '118': '9', '119': '9', '125': '9', '12': '9', '131': '9', '135': '9', '141': '9', '14': '9', '150': '9', '151': '9', '157': '9', '175': '9', '176': '9', '19': '9', '20': '9', '22': '9', '23': '9', '27': '9', '28': '9', '35': '9', '41': '9', '42': '9', '53': '9', '59': '9', '62': '9', '65': '9', '69': '9', '72': '9', '83': '9', '95': '9'}

type_to_model = {}
for mod, typ in model_to_type.items():
    mods = type_to_model.setdefault(int(typ), [])
    mods.append(int(mod))


def get_random_car_by_type(type_id):
    if type_id not in type_to_model:
        raise ValueError("Bad type_id")
    model_id = random.sample(type_to_model[type_id], 1)[0]
    return model_id
        


def get_cam_attr(cam_info):
    control_list = []
    attribute_list = []
    variance_list = []
    order_list = []
    for attribute in cam_info['attributes'].items():
        attribute_name = attribute[0]
        order_list.append(attribute_name)
        attribute_content = attribute[1]
        if attribute_content[0] == 'Gaussian Mixture':
            range_info = attribute_content[1]
            mean_list = attribute_content[2]
            var_list = attribute_content[3]
            control_list.extend([np.arange(
                range_info[0], range_info[1], range_info[2]) for i in range(len(mean_list))])
            attribute_list.extend(mean_list)
            variance_list.extend(var_list)
        if attribute_content[0] == 'Gaussian':
            range_info = attribute_content[1]
            mean_list = attribute_content[2]
            var_list = attribute_content[3]
            control_list.append(
                np.arange(range_info[0], range_info[1], range_info[2]))
            attribute_list.append(mean_list)
            variance_list.append(var_list)
    return control_list, attribute_list, variance_list


def ancestral_sampler(mu, sigma, size=1):
    """ Draw samples uniformly from normal distributions of mu=mu[i], std=sqrt(sigma[i]) """
    pi = [1 / 6 for i in range(6)]
    sample = []
    z_list = np.random.uniform(size=size)
    low = 0  # low bound of a pi interval
    high = 0  # high bound of a pi interval
    for index in range(len(pi)):
        if index > 0:
            low += pi[index - 1]
        if index == (len(pi) - 1):
            high = 1.1
        else:
            high += pi[index]
        s = len([z for z in z_list if low <= z < high])
        sample.extend(np.random.normal(
            loc=mu[index], scale=np.sqrt(sigma[index]), size=s))
    return sample


def ancestral_sampler_fix_sigma(mu, size=1):
    """ Draw samples uniformly from normal distributions of mu=mu[i], std=sqrt(20) """
    sigma = [20 for i in range(6)]
    pi = [1 / 6 for i in range(6)]
    sample = []
    z_list = np.random.uniform(size=size)
    low = 0  # low bound of a pi interval
    high = 0  # high bound of a pi interval
    for index in range(len(pi)):
        if index > 0:
            low += pi[index - 1]
        if index == (len(pi) - 1):
            high = 1.1
        else:
            high += pi[index]
        s = len([z for z in z_list if low <= z < high])
        sample.extend(np.random.normal(
            loc=mu[index], scale=np.sqrt(sigma[index]), size=s))
    return sample


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
