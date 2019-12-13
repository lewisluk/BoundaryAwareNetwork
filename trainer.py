from Scripts.model import FCN_4s

if __name__ == '__main__':
    model = FCN_4s()
    model.init_network()
    model.restore()
    model.trainer()