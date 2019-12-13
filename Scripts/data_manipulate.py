import numpy as np, cv2, os, tensorflow as tf
from Scripts.boundary_target import get_boundary_target

def get_batch(index, dir_name, batchsize, data_dir):
    img_index_range = range(index * batchsize, (index + 1) * batchsize)
    input_list, label_list, boundary_target_list = [], [], []
    for i in img_index_range:
        input_bgr = cv2.imread(os.path.join(data_dir, dir_name, '{:05d}.png'.format(i + 1)))
        label_gray = cv2.imread(
            os.path.join(data_dir, dir_name, '{:05d}_matte.png'.format(i + 1)),
            cv2.IMREAD_GRAYSCALE)
        temp_input = preprocess(input_bgr)
        temp_label = preprocess(label_gray, gray=True)
        temp_boundary_target = np.expand_dims(get_boundary_target(label_gray), -1)
        input_list.append(temp_input)
        label_list.append(temp_label)
        boundary_target_list.append(temp_boundary_target)
    return np.array(input_list), np.array(label_list), np.array(boundary_target_list)

def preprocess(img, gray=False, single=False):
    '''
    :param img:     rgb portrait image or single channel matte image
    :param gray:    bool value to identify param img
    :param single:  if model runs in single image test mode, set to True
    :return:
    '''
    if gray:
        img = img / 255.
        img = np.expand_dims(img, -1)
    else:
        img = np.expand_dims(img, 0)/255. if single else img / 255.
    return img

def postprocess(net_out, single=False):
    '''
    :param net_out: model output image
    :param single:  if model runs in single image test mode, set to True
    :return:        the matte image file that can be directly saved
    '''
    net_out = np.squeeze(net_out, -1)
    net_out = np.clip(net_out, 0.0, 1.0)
    net_out = net_out * 255.
    out = net_out.astype(np.uint8)
    if single:
        out = np.squeeze(out, 0)
    return out

def sigmoid_with_T(x, t=1):
    '''
    sigmoid with np
    :param x:   input image array, shape [height, width], value range 0~255
    :param t:   "a temperature which produces a softer probability distribution.",
                according to the paper
    :return:    softened boundary attention map
    '''
    z = 1 / (1 + np.exp(-x/t))
    return z

def save_imgs(index, pred, sematic_out, epoch_dir, batchsize, result_dir):
    img_index_range = range(index * batchsize, (index + 1) * batchsize)
    cnt = 0
    for i in img_index_range:
        img_to_save = postprocess(pred[cnt])
        sematic_to_save = postprocess(sematic_out[cnt])
        cv2.imwrite(os.path.join(result_dir,
                                 '{:03d}'.format(epoch_dir),
                                 '{:05d}_pred.png'.format(i + 1)),
                    img_to_save)
        cv2.imwrite(os.path.join(result_dir,
                                 '{:03d}'.format(epoch_dir),
                                 '{:05d}_sematic.png'.format(i + 1)),
                    sematic_to_save)
        cnt += 1

def soften(boundary_attention_map, t=1):
    z = 1 / (1 + tf.exp(-boundary_attention_map/t))
    return z
