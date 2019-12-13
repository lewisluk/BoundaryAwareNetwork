import tensorflow as tf

def iou(pred, label):
    '''
    calculate intersection over union between predict image batch and label image batch
    :param pred:    model output image batch, shape [batch, height, width, 1]
    :param label:   labgel image batch, shape [batch, height, width, 1]
    :return:        mean iou
    '''
    pred = tf.clip_by_value(pred, 0.0, 1.0)
    pred_sum = tf.reduce_sum(pred)
    label_sum = tf.reduce_sum(label)
    intersection = tf.reduce_sum(pred * label)
    union = pred_sum + label_sum - intersection
    return tf.reduce_mean(intersection / union)

# def iou(pred, label):
#     '''
#     calculate intersection over union between predict image batch and label image batch
#     :param pred:    model output image batch, shape [batch, height, width, 1]
#     :param label:   labgel image batch, shape [batch, height, width, 1]
#     :return:        mean iou
#     '''
#     return tf.metrics.mean_iou(predictions=tf.sigmoid(pred),
#                                labels=label,
#                                num_classes=2)

# def compute_mean_iou(pred, label):
#     unique_labels = np.unique(label)
#     num_unique_labels = len(unique_labels)
#
#     I = np.zeros(num_unique_labels)
#     U = np.zeros(num_unique_labels)
#
#     for index, val in enumerate(unique_labels):
#         pred_i = pred == val
#         label_i = label == val
#
#         I[index] = float(np.sum(np.logical_and(label_i, pred_i)))
#         U[index] = float(np.sum(np.logical_or(label_i, pred_i)))
#
#
#     mean_iou = np.mean(I / U)
#     return mean_iou

