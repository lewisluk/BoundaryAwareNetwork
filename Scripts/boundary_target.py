import cv2, numpy as np

def get_boundary_target(img, hysteresis_thresh1=5, hysteresis_thresh2=5, morph_type=0):
    '''
    :param img:                 np array of mask, shape [height, width], value range 0~255
    :param hysteresis_thresh1:  first threshold for the hysteresis procedure in canny detector
    :param hysteresis_thresh2:  second threshold for the hysteresis procedure in canny detector
    :param morph_type:          a param to select morph structuring element type,
                                0 for RECT, 1 for CROSS, 2 for ELLIPSE
    :return:                    softened boundary attention map after sigmoid,
                                the same shape with img, value range 0~1
    '''

    # an empirical value representing the canonical width boundary,
    # set to 50 in paper, but we use 25 here which makes the output
    # looks more like the one in paper
    W = 20

    # morph structuring element type
    morph_dict = {0:cv2.MORPH_RECT, 1:cv2.MORPH_CROSS, 2:cv2.MORPH_ELLIPSE}

    edge = cv2.Canny(img,
                     hysteresis_thresh1,
                     hysteresis_thresh2)

    portrait_ratio = get_portrait_ratio(img)
    edge_radius = int(portrait_ratio*W)
    edge_kernel = cv2.getStructuringElement(morph_dict[morph_type],
                                            (edge_radius, edge_radius))
    dilated_edge = cv2.dilate(edge, edge_kernel)

    return dilated_edge

def get_portrait_ratio(img):
    '''
    a function to get potrait/image_size ratio
    :param img: np array of mask, shape [height, width], value range 0~255
    :return:    potrait/image_size ratio
    '''
    total_area = img.size
    portrait_area = np.count_nonzero(img)
    return  portrait_area/total_area

if __name__=='__main__':
    img = cv2.imread('/media/lewisluk/新加卷/dataset/portrai_seg/dataset/testing/00001_matte.png',
                     cv2.IMREAD_GRAYSCALE)
    edge = get_boundary_target(img)
    cv2.imshow('raw mask', img)
    cv2.imshow('result edge', edge)
    cv2.waitKey(0)
    cv2.imwrite('result.png', edge)