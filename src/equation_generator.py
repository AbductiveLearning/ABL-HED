import os
import itertools
import random
import numpy as np
from PIL import Image
import pickle

def get_sign_path_list(data_dir, sign_names):
    sign_num = len(sign_names)
    index_dict = dict(zip(sign_names, list(range(sign_num))))
    ret = [[] for _ in range(sign_num)]
    for path in os.listdir(data_dir):
        if (path in sign_names):
            index = index_dict[path]
            sign_path = os.path.join(data_dir, path)
            for p in os.listdir(sign_path):
                ret[index].append(os.path.join(sign_path, p))
    return ret

def split_pool_by_rate(pools, rate, seed = None):
    if seed is not None:
        random.seed(seed)
    ret1 = []
    ret2 = []
    for pool in pools:
        random.shuffle(pool)
        num = int(len(pool) * rate)
        ret1.append(pool[:num])
        ret2.append(pool[num:])
    return ret1, ret2

def int_to_system_form(num, system_num):
    if num is 0:
        return "0"
    ret = ""
    while (num > 0):
        ret += str(num % system_num)
        num //= system_num
    return ret[::-1]

def generator_equations(left_opt_len, right_opt_len, res_opt_len, system_num, label, generate_type):
    expr_len = left_opt_len + right_opt_len
    num_list = "".join([str(i) for i in range(system_num)])
    ret = []
    if generate_type == "all":
        candidates = itertools.product(num_list, repeat = expr_len)
    else:
        candidates = [''.join(random.sample(['0', '1'] * expr_len, expr_len))]
        random.shuffle(candidates)
    for nums in candidates:
        left_num = "".join(nums[:left_opt_len])
        right_num = "".join(nums[left_opt_len:])
        left_value = int(left_num, system_num)
        right_value = int(right_num, system_num)
        result_value = left_value + right_value
        if (label == 'negative'):
            result_value += random.randint(-result_value, result_value)
            if (left_value + right_value == result_value):
                continue
        result_num = int_to_system_form(result_value, system_num)
        #leading zeros
        if (res_opt_len != len(result_num)):
            continue
        if ((left_opt_len > 1 and left_num[0] == '0') or (right_opt_len > 1 and right_num[0] == '0')):
            continue

        #add leading zeros
        if (res_opt_len < len(result_num)):
            continue
        while (len(result_num) < res_opt_len):
            result_num = '0' + result_num
            #continue
        ret.append(left_num + '+' + right_num + '=' + result_num) # current only consider '+' and '='
        #print(ret[-1])
    return ret

def generator_equation_by_len(equation_len, system_num = 2, label = 0, require_num = 1):
    generate_type = "one"
    ret = []
    equation_sign_num = 2 # '+' and '='
    while len(ret) < require_num:
        left_opt_len = random.randint(1, equation_len - 1 - equation_sign_num)
        right_opt_len = random.randint(1, equation_len - left_opt_len - equation_sign_num)
        res_opt_len = equation_len - left_opt_len - right_opt_len - equation_sign_num
        ret.extend(generator_equations(left_opt_len, right_opt_len, res_opt_len, system_num, label, generate_type))
    return ret

def generator_equations_by_len(equation_len, system_num = 2, label = 0, repeat_times = 1, keep = 1, generate_type = "all"):
    ret = []
    equation_sign_num = 2 # '+' and '='
    for left_opt_len in range(1, equation_len - (2 + equation_sign_num) + 1):
        for right_opt_len in range(1, equation_len - left_opt_len - (1 + equation_sign_num) + 1):
            res_opt_len = equation_len - left_opt_len - right_opt_len - equation_sign_num
            for i in range(repeat_times): #generate more equations
                if random.random() > keep ** (equation_len):
                    continue
                ret.extend(generator_equations(left_opt_len, right_opt_len, res_opt_len, system_num, label, generate_type))
    return ret

def generator_equations_by_max_len(max_equation_len, system_num = 2, label = 0, repeat_times = 1, keep = 1, generate_type = "all", num_per_len = None):
    ret = []
    equation_sign_num = 2 # '+' and '='
    for equation_len in range(3 + equation_sign_num, max_equation_len + 1):
        if (num_per_len is None):
            ret.extend(generator_equations_by_len(equation_len, system_num, label, repeat_times, keep, generate_type))
        else:
            ret.extend(generator_equation_by_len(equation_len, system_num, label, require_num = num_per_len))
    return ret

def generator_equation_images(image_pools, equations, signs, shape, seed, is_color):
    if (seed is not None):
        random.seed(seed)
    ret = []
    sign_num = len(signs)
    sign_index_dict = dict(zip(signs, list(range(sign_num))))
    for equation in equations:
        data = []
        for sign in equation:
            index = sign_index_dict[sign]
            pick = random.randint(0, len(image_pools[index]) - 1)
            if is_color:
                image = Image.open(image_pools[index][pick]).convert('RGB').resize(shape)
            else:
                image = Image.open(image_pools[index][pick]).convert('I').resize(shape)
            image_array = np.array(image)
            image_array = (image_array-127)*(1./128)
            data.append(image_array)
        ret.append(np.array(data))
    return ret

def get_equation_std_data(data_dir, sign_dir_lists, sign_output_lists, shape = (28, 28), train_max_equation_len = 10, test_max_equation_len = 10, system_num = 2, tmp_file_prev =
None, seed = None, train_num_per_len = 10, test_num_per_len = 10, is_color = False):
    tmp_file = ""
    if (tmp_file_prev is not None):
        tmp_file = "%s_train_len_%d_test_len_%d_sys_%d_.pk" % (tmp_file_prev, train_max_equation_len, test_max_equation_len, system_num)
    if (os.path.exists(tmp_file)):
        return pickle.load(open(tmp_file, "rb"))

    image_pools = get_sign_path_list(data_dir, sign_dir_lists)
    train_pool, test_pool = split_pool_by_rate(image_pools, 0.8, seed)

    ret = {}
    for label in ["positive", "negative"]:
        print("Generating equations.")
        train_equations = generator_equations_by_max_len(train_max_equation_len, system_num, label, num_per_len = train_num_per_len)
        test_equations = generator_equations_by_max_len(test_max_equation_len, system_num, label, num_per_len = test_num_per_len)
        print(train_equations)
        print(test_equations)
        print("Generated equations.")
        print("Generating equation image data.")
        ret["train:%s" % (label)] = generator_equation_images(train_pool, train_equations, sign_output_lists, shape, seed, is_color)
        ret["test:%s" % (label)] = generator_equation_images(test_pool, test_equations, sign_output_lists, shape, seed, is_color)
        print("Generated equation image data.")

    if (tmp_file_prev is not None):
        pickle.dump(ret, open(tmp_file, "wb"))
    return ret

if __name__ == "__main__":
    data_dirs = ["../dataset/mnist_images", "../dataset/random_images"] #, "../dataset/cifar10_images"]
    tmp_file_prevs = ["mnist_equation_data", "random_equation_data"] #, "cifar10_equation_data"]
    for data_dir, tmp_file_prev in zip(data_dirs, tmp_file_prevs):
        data = get_equation_std_data(data_dir = data_dir,\
                                     sign_dir_lists = ['0', '1', '10', '11'],\
                                     sign_output_lists = ['0', '1', '+', '='],\
                                     shape = (28, 28),\
                                     train_max_equation_len = 26, \
                                     test_max_equation_len = 26, \
                                     system_num = 2, \
                                     tmp_file_prev = tmp_file_prev, \
                                     train_num_per_len = 300, \
                                     test_num_per_len = 300, \
                                     is_color = False)
