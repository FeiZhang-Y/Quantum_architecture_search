import pickle


def load_pkl(input_file):  # 加载pkl文件
    f = open(input_file, 'rb')
    output_file = pickle.load(f)
    f.close()
    return output_file


def save_pkl(data, loc):
    f = open(loc, 'wb')
    pickle.dump(data, file=f)
    f.close()
    return 0
