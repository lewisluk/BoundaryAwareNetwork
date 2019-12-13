import tensorflow as tf, cv2, numpy as np
from Scripts.boundary_target import get_boundary_target

def refine_loss(logits, labels, boundary_target):
    gamma1 = 0.5
    gamma2 = 1-gamma1
    factor_lambda= 1.5

    dy_logits, dx_logits = tf.image.image_gradients(logits)
    dy_labels, dx_labels = tf.image.image_gradients(labels)

    # magnitudes of logits and labels gradients
    Mpred = tf.sqrt(tf.square(dy_logits)+tf.square(dx_logits))
    Mimg = tf.sqrt(tf.square(dy_labels)+tf.square(dx_labels))

    # define cos loss and mag loss
    cosL = (1-tf.abs(dx_labels*dx_logits+dy_labels*dy_logits))*Mpred
    magL = tf.maximum(factor_lambda*Mimg-Mpred,0)

    # define mask
    M_bound = boundary_target/255.

    # define total refine loss
    refineLoss = (gamma1*cosL + gamma2*magL)*M_bound
    return tf.reduce_mean(refineLoss)
    # return  tf.reduce_mean(refineLoss), Mpred, Mimg

def load_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)/255.
    mask = np.expand_dims(np.expand_dims(mask, 0), -1)
    return mask

if __name__=='__main__':
    pred_placeholder = tf.placeholder(dtype=tf.float32,
                                      shape=[None, None, None, 1])
    label_placeholder = tf.placeholder(dtype=tf.float32,
                                      shape=[None, None, None, 1])
    boundary_placeholder = tf.placeholder(dtype=tf.float32,
                                      shape=[None, None, None, 1])
    pred_mask_path = '/media/lewisluk/新加卷/GeekVision/data_boundaryAwareNet/result (1)/100/00020_pred.png'
    label_mask_path = '/media/lewisluk/新加卷1/dataset/portrai_seg/dataset/testing/00020_matte.png'
    pred = load_mask(pred_mask_path)
    label = load_mask(label_mask_path)
    boundary = get_boundary_target(cv2.imread(label_mask_path,
                                              cv2.IMREAD_GRAYSCALE))
    boundary = np.expand_dims(np.expand_dims(boundary, 0), -1)

    refineL, Mpred, Mimg = refine_loss(pred_placeholder, label_placeholder, boundary_placeholder)
    with tf.Session() as sess:
        rl, mp, mi = sess.run([refineL,
                        Mpred,
                        Mimg],
                 feed_dict={
                     pred_placeholder: pred,
                     label_placeholder: label,
                     boundary_placeholder: boundary
                 })
    print(rl)
    print(mp.shape, np.max(mp), np.min(mp), np.std(mp))
    print(mi.shape, np.max(mi), np.min(mi), np.std(mi))

    mp = np.clip(np.squeeze(np.squeeze(mp*255., 0), -1), 0., 255.).astype(np.uint8)
    mi = np.clip(np.squeeze(np.squeeze(mi*255., 0), -1), 0., 255.).astype(np.uint8)
    cv2.imshow('mpred', mp)
    cv2.imshow('mimg', mi)
    cv2.waitKey(0)