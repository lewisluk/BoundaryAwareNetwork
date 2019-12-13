# Author: LU Yifu
# Original paper: https://arxiv.org/abs/1901.03814
from Scripts.metrics import *
from Scripts.network_ops import ops
from Scripts.losses import *
from Scripts.data_manipulate import *

class FCN_4s():
    def __init__(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.input = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='x_input')
        self.output = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='y_label')
        # boundary_attention_map has value range 0~255
        self.boundary_attention_map = tf.placeholder(tf.float32,
                                                     shape=[None, None, None, 1],
                                                     name='BAM')
        self.training = tf.placeholder(tf.bool, shape=None, name='istraining')
        self.data_dir = '/data/portrait_seg_data'
        self.result_dir = 'result'
        self.train_dir_name = 'training'
        self.test_dir_name = 'testing'
        self.noisy_label = tf.placeholder(tf.float32, shape=None)
        self.learningrate = 1e-4
        self.epochs = 100
        self.batchsize = 10 # divisible from 100
        self.segmentation_loss_alpha = 0.6
        self.boundary_attention_loss_beta = 0.3
        self.refine_loss_gamma = 0.1

    def restore(self, ckpt_path=None):
        if ckpt_path:
            self.saver.restore(self.sess, ckpt_path)
        else:
            self.saver.restore(self.sess, self.result_dir + '/model.ckpt')

    def build(self, inputs, BAmap):
        '''
        :param inputs:  input of the network, shape should be [batchsize, None, None, 3]
        :return:
        '''
        net_ops = ops(self.training)

        x = net_ops.first_layer(inputs, scope='first_layer')    #1/4

        x1 = net_ops.bottleneck_convocation(x, scope='block1')
        x1 = net_ops.max_pooling(x1)                            #1/8

        x2 = net_ops.bottleneck_convocation(x1, scope='block2')
        x2 = net_ops.max_pooling(x2)                            #1/16

        x3 = net_ops.bottleneck_convocation(x2, scope='block3')
        x3 = net_ops.max_pooling(x3)                            #1/32

        x4 = net_ops.upsample_and_add(x3, x2)                   #1/16

        x5 = net_ops.upsample_and_add(x4, x1)                   #1/8

        x6 = net_ops.upsample_and_add(x5, x)                    #1/4

        sematic_output = net_ops.layer_before_FFM(x6,
                                                  interpolate_target_shape=tf.shape(inputs)[1:3],
                                                  scope='layer_before_FFM')

        boundary_feature = net_ops.boundary_mining(inputs, BAmap)

        FFM_output = net_ops.feature_fusion_module(semantic_input=sematic_output,
                                                   boundary_feature=boundary_feature)
        return sematic_output, FFM_output


    def init_network(self):
        # loss
        self.sematic_output, self.FFM_output = self.build(self.input,
                                                          self.boundary_attention_map)

        # "to improve numerical stability"
        softened_boundary_attention_map = soften(self.boundary_attention_map, t=1)

        self.boundary_attention_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.sematic_output, labels=softened_boundary_attention_map)
        self.segmentation_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.FFM_output, labels=self.output)
        self.refine_loss = refine_loss(logits=self.FFM_output,
                                       labels=self.output,
                                       boundary_target=self.boundary_attention_map)

        self.loss = self.segmentation_loss_alpha*self.segmentation_loss\
                    + self.boundary_attention_loss_beta*self.boundary_attention_loss\
                    + self.refine_loss_gamma*self.refine_loss

        # self.IOU, self.iou_op = iou(label=self.output, pred=self.FFM_output)
        self.IOU = iou(label=self.output, pred=self.FFM_output)
        # self.IOU = compute_mean_iou(label=self.output,
        #                             pred=self.FFM_output)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.opt = tf.train.AdamOptimizer(learning_rate=self.learningrate).minimize(self.loss)
        self.saver = tf.train.Saver(max_to_keep=self.epochs)
        self.sess.run(tf.global_variables_initializer())
        # self.sess.run(tf.local_variables_initializer())

    def trainer(self):
        # create dir for saving models if haven't been created
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        # start training loop
        train_size, val_size = 1700, 300
        for epoch in range(self.epochs):
            cnt, tr_loss, tr_iou, tr_bal, tr_sl, tr_rl = \
                0, [], [], [], [], []
            for index in range(int(train_size/self.batchsize)):

                input_batch, label_batch, boundary_target_batch = get_batch(index,
                                                                            self.train_dir_name,
                                                                            self.batchsize,
                                                                            self.data_dir)
                _, loss, iou, pred, sl, bal, rl = self.sess.run([
                        self.opt,
                        self.loss,
                        self.IOU,
                        self.FFM_output,
                        self.segmentation_loss,
                        self.boundary_attention_loss,
                        self.refine_loss
                    ],
                        feed_dict={self.input: input_batch,
                                   self.output: label_batch,
                                   self.boundary_attention_map: boundary_target_batch,
                                   self.training: True})

                tr_loss.append(loss)
                tr_iou.append(iou)
                tr_bal.append(bal)
                tr_sl.append(sl)
                tr_rl.append(rl)
                cnt += 1

            print(
                'Ep:{}, Iter:{}, Loss:{:.4f}, IOU:{:.4f}, segL:{:.4f}, bouL:{:.4f}, refL:{:.4f}'.format(
                    epoch + 1,
                    cnt,
                    np.mean(tr_loss),
                    np.mean(tr_iou),
                    np.mean(tr_sl),
                    np.mean(tr_bal),
                    np.mean(tr_rl)
                ))

            if not os.path.exists("%s/%03d" % (self.result_dir, epoch + 1)):
                os.makedirs("%s/%03d" % (self.result_dir, epoch + 1))

            val_loss, val_iou = [], []
            for index in range(int(val_size/self.batchsize)):
                val_input, val_label, boundary_target_batch = get_batch(index,
                                                                        self.test_dir_name,
                                                                        self.batchsize,
                                                                        self.data_dir)

                sematic_out, pred, temp_loss, temp_iou = self.sess.run([
                    self.sematic_output,
                    self.FFM_output,
                    self.loss,
                    self.IOU
                ], feed_dict={self.input: val_input,
                              self.output: val_label,
                              self.boundary_attention_map: boundary_target_batch,
                              self.training: False})

                val_loss.append(temp_loss)
                val_iou.append(temp_iou)
                save_imgs(index, pred, sematic_out, epoch+1, self.batchsize, self.result_dir)
            print('Validation for Epoch:{:d}, Loss:{:.4f}, IOU:{:.4f}'.format(
                epoch + 1,
                np.mean(val_loss),
                np.mean(val_iou)
            ))
            self.saver.save(self.sess, self.result_dir + '/{:03d}/model.ckpt'.format(epoch + 1))
