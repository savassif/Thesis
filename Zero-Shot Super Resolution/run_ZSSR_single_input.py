import sys
import os
import configs
import ZSSR
import tensorflow as tf

def main(input_img, ground_truth, kernels, gpu, conf_str, results_path):
    # Choose the wanted GPU
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = '%s' % gpu
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    # 0 input for ground-truth or kernels means None
    ground_truth = None if ground_truth == '0' else ground_truth
    print '*****', kernels
    kernels = None if kernels == '0' else kernels.split(';')[:-1]

    # Setup configuration and results directory
    conf = configs.Config()
    if conf_str is not None:
        exec ('conf = configs.%s' % conf_str)
    conf.result_path = results_path

    # Run ZSSR on the image
    net = ZSSR.ZSSR(input_img, conf, ground_truth, kernels)
    net.run()


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
