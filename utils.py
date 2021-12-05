import datetime
import dgl
import errno
import numpy as np
import os
import pickle
import random
import torch

from dgl.data.utils import download, get_download_dir, _get_dgl_url
from pprint import pprint
from scipy import sparse
from scipy import io as sio
import pandas as pd
from sklearn import preprocessing

from NodeFeature import *

from sklearn.preprocessing import OneHotEncoder

def set_random_seed(seed=0):
    """Set random seed.
    Parameters
    ----------
    seed : int
        Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def mkdir_p(path, log=True):
    """Create a directory for the specified path.
    Parameters
    ----------
    path : str
        Path name
    log : bool
        Whether to print result for directory creation
    """
    try:
        os.makedirs(path)
        if log:
            print('Created directory {}'.format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path) and log:
            print('Directory {} already exists.'.format(path))
        else:
            raise

def get_date_postfix():
    """Get a date based postfix for directory name.
    Returns
    -------
    post_fix : str
    """
    dt = datetime.datetime.now()
    post_fix = '{}_{:02d}-{:02d}-{:02d}'.format(
        dt.date(), dt.hour, dt.minute, dt.second)

    return post_fix

def setup_log_dir(args, sampling=False):
    """Name and create directory for logging.
    Parameters
    ----------
    args : dict
        Configuration
    Returns
    -------
    log_dir : str
        Path for logging directory
    sampling : bool
        Whether we are using sampling based training
    """
    date_postfix = get_date_postfix()
    log_dir = os.path.join(
        args['log_dir'],
        '{}_{}'.format(args['dataset'], date_postfix))

    if sampling:
        log_dir = log_dir + '_sampling'

    mkdir_p(log_dir)
    return log_dir

# The configuration below is from the paper.
default_configure = {
    'lr': 0.005,             # Learning rate
    'num_heads': [8],        # Number of attention heads for node-level attention
    'hidden_units': 8,
    'dropout': 0.6,
    'weight_decay': 0.001,   # 正则https://zhuanlan.zhihu.com/p/62393636
    'num_epochs': 200,
    'patience': 100
}

sampling_configure = {
    'batch_size': 20
}

def setup(args):
    args.update(default_configure)
    set_random_seed(args['seed'])
    args['dataset'] = 'ACMRaw' if args['hetero'] else 'ACM'
    args['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    args['log_dir'] = setup_log_dir(args)
    return args

def setup_for_sampling(args):
    args.update(default_configure)
    args.update(sampling_configure)
    set_random_seed()
    args['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    args['log_dir'] = setup_log_dir(args, sampling=True)
    return args

def get_binary_mask(total_size, indices):
    mask = torch.zeros(total_size)
    mask[indices] = 1
    return mask.byte()

# python main.py 用处理过的数据
def load_acm(remove_self_loop):

    data_path = get_download_dir() + '/ACM3025.pkl'
    # ACM3025.pkl-->包含feature, label, PAP, PLP, PTP(似乎没用)，test_idx, train_idx, val_idx
    print("data_path =", data_path)
    # url = 'dataset/ACM3025.pkl'
    # download(_get_dgl_url(url), path=data_path)

    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    print("data =\n", data)
    # np, show()
    labels, features = torch.from_numpy(data['label'].todense()).long(), \
                       torch.from_numpy(data['feature'].todense()).float()
    num_classes = labels.shape[1]
    labels = labels.nonzero()[:, 1]

    if remove_self_loop:
        num_nodes = data['label'].shape[0]
        data['PAP'] = sparse.csr_matrix(data['PAP'] - np.eye(num_nodes))
        data['PLP'] = sparse.csr_matrix(data['PLP'] - np.eye(num_nodes))
        # data['PTP'] = sparse.csr_matrix(data['PTP'] - np.eye(num_nodes))


    # Adjacency matrices for meta path based neighbors
    # (Mufei): I verified both of them are binary adjacency matrices with self loops
    author_g = dgl.from_scipy(data['PAP'])
    subject_g = dgl.from_scipy(data['PLP'])
    gs = [author_g, subject_g]

    train_idx = torch.from_numpy(data['train_idx']).long().squeeze(0)
    val_idx = torch.from_numpy(data['val_idx']).long().squeeze(0)
    test_idx = torch.from_numpy(data['test_idx']).long().squeeze(0)

    num_nodes = author_g.number_of_nodes()
    train_mask = get_binary_mask(num_nodes, train_idx)
    val_mask = get_binary_mask(num_nodes, val_idx)
    test_mask = get_binary_mask(num_nodes, test_idx)

    print('dataset loaded')
    pprint({
        'dataset': 'ACM',
        'train': train_mask.sum().item() / num_nodes,
        'val': val_mask.sum().item() / num_nodes,
        'test': test_mask.sum().item() / num_nodes
    })

    return gs, features, labels, num_classes, train_idx, val_idx, test_idx, \
           train_mask, val_mask, test_mask

# 这个对应的是未处理的数据
# python main.py --hetero
def load_acm_raw(remove_self_loop): ####################################
    assert not remove_self_loop
    data_path = get_download_dir() + '/ACM.mat'
    # 如果已经下载数据就不用这两个句子了
    #url = 'dataset/ACM.mat'
    #download(_get_dgl_url(url), path=data_path)
    data = sio.loadmat(data_path)
    # print("data =\n", data)
    # np, show()
    p_vs_l = data['PvsL']       # paper-field?
    p_vs_a = data['PvsA']       # paper-author
    p_vs_t = data['PvsT']       # paper-term, bag of words
    p_vs_c = data['PvsC']       # paper-conference, labels come from that

    print("p_vs_c =", p_vs_c)
    print(type(p_vs_c))
    print("len(p_vs_c) =", p_vs_c.shape)

    # We assign
    # (1) KDD papers as class 0 (data mining),
    # (2) SIGMOD and VLDB papers as class 1 (database),
    # (3) SIGCOMM and MOBICOMM papers as class 2 (communication)
    conf_ids = [0, 1, 9, 10, 13]
    label_ids = [0, 1, 2, 2, 1]

    p_vs_c_filter = p_vs_c[:, conf_ids]
    p_selected = (p_vs_c_filter.sum(1) != 0).A1.nonzero()[0]
    p_vs_l = p_vs_l[p_selected]
    p_vs_a = p_vs_a[p_selected]
    p_vs_t = p_vs_t[p_selected]
    p_vs_c = p_vs_c[p_selected]

    hg = dgl.heterograph({
        ('paper', 'pa', 'author'): p_vs_a.nonzero(),
        ('author', 'ap', 'paper'): p_vs_a.transpose().nonzero(),
        ('paper', 'pf', 'field'): p_vs_l.nonzero(),
        ('field', 'fp', 'paper'): p_vs_l.transpose().nonzero()
    })

    features = torch.FloatTensor(p_vs_t.toarray())

    pc_p, pc_c = p_vs_c.nonzero()
    labels = np.zeros(len(p_selected), dtype=np.int64)
    for conf_id, label_id in zip(conf_ids, label_ids):
        labels[pc_p[pc_c == conf_id]] = label_id
    labels = torch.LongTensor(labels)
    # 分类数
    num_classes = 3

    float_mask = np.zeros(len(pc_p))
    for conf_id in conf_ids:
        pc_c_mask = (pc_c == conf_id)
        float_mask[pc_c_mask] = np.random.permutation(np.linspace(0, 1, pc_c_mask.sum()))
    print("float_mask =", float_mask)

    train_idx = np.where(float_mask <= 0.2)[0]
    val_idx = np.where((float_mask > 0.2) & (float_mask <= 0.3))[0]
    test_idx = np.where(float_mask > 0.3)[0]
    print(train_idx)
    print(val_idx)
    print(test_idx)

    num_nodes = hg.number_of_nodes('paper')
    train_mask = get_binary_mask(num_nodes, train_idx)
    val_mask = get_binary_mask(num_nodes, val_idx)
    test_mask = get_binary_mask(num_nodes, test_idx)
    print("hg =\n", hg)
    print("features =\n", features)
    print("labels =\n", labels)
    print("num_classes =\n", num_classes)
    print("train_idx =\n", train_idx)
    print("val_idx =\n", val_idx)
    print("test_idx =\n", test_idx)
    print("train_mask =\n", train_mask)
    print("val_mask =\n", val_mask)
    print("test_mask =\n", test_mask)
    return hg, features, labels, num_classes, train_idx, val_idx, test_idx, \
            train_mask, val_mask, test_mask

# 这个对应的是未处理的数据--->处理自己的数据
# python main.py --hetero
def load_mydata_raw(remove_self_loop): ####################################
    assert not remove_self_loop
    data_path = get_download_dir() + '/mydataset.mat'
    # 如果已经下载数据就不用这两个句子了
    # url = 'dataset/ACM.mat'
    # download(_get_dgl_url(url), path=data_path)
    data = sio.loadmat(data_path)
    print("小伙子什么回事！！！")
    # print("data =\n", data)
    # 该交通方式去过哪？od对？-->交通方式对应的OD对
    p_vs_l = data['PvsL']       # paper-field? 文章领域
    # 把每个赋值完才赋值下一个数
    # paperID  领域
    # 12345      0
    # 23412      0
    # 89731      0
    # 按领域区分
    # 交通方式对应的用户------->交通方式对应的用户
    p_vs_a = data['PvsA']       # paper-author 文章作者
    # paperID  作者
    # 148      0
    # 3852     0
    # 3852     0
    # 特性的是特征，交通模式的特征？-->选择当前路径时，产生路径(最大距离，最小距离，平均距离，中数距离)，
    # 时间（最大，最小，平均，中数时间），价钱(最大，最小，平均，中数价钱-->可以获取的)，
    # 天气等稍后获取，相当于一种交通方式的偏好
    #               -------->交通方式对应的特征，五种特征，价格，速度，使用次数，能源使用，距离
    p_vs_t = data['PvsT']       # paper-term, bag of words 文章术语-->文章中有哪些重要的词
    # PaperID  重要词
    # 0         0
    # 32        0
    # 41        0
    # 如果怕麻烦，可以把交通方式对应标签放上去
    #              --------->交通方式对应的标签
    p_vs_c = data['PvsC']       # paper-conference, labels come from that
    # PaperID 会议
    # 0       0
    # 1       0
    # 2       0
    # print(type(p_vs_a))
    # 取出前多少组数据-->批次处理，想想是不是从这里出发？？

    # 数据集的长度
    num = p_vs_l.shape[0]
    print("len(p_vs_a) =", p_vs_a.shape[0])
    num = 308507
    # num = 50000
    # num = 5000
    p_vs_l = p_vs_l[:num]
    p_vs_a = p_vs_a[:num]
    p_vs_t = p_vs_t[:num]
    p_vs_c = p_vs_c[:num]
    # print("len(p_vs_a) =", p_vs_a.shape[0])
    # print("p_vs_a =\n", p_vs_a)
    # np, show()
    # We assign
    # (1) KDD papers as class 0 (data mining),
    # (2) SIGMOD and VLDB papers as class 1 (database),
    # (3) SIGCOMM and MOBICOMM papers as class 2 (communication)
    # conf_ids = [0, 1, 9, 10, 13]
    conf_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # label_ids = [0, 1, 2, 2, 1]
    label_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    p_vs_c_filter = p_vs_c[:, conf_ids]
    p_selected = (p_vs_c_filter.sum(1) != 0).A1.nonzero()[0]
    # od
    p_vs_l = p_vs_l[p_selected]
    # 用户
    p_vs_a = p_vs_a[p_selected]
    # 对应项(session id or SID)的特征or属性
    p_vs_t = p_vs_t[p_selected] # -->在之后会改变，之前使用label赋予属性的思路不对
    # print("len(p_vs_t) =", p_vs_t.shape)

    # 所属分类
    p_vs_c = p_vs_c[p_selected]
    # print("len(p_vs_a) =", p_vs_a.shape[0])
    # print("p_vs_a =\n", p_vs_a)
    # np, show()
    # 自己构图
    hg = dgl.heterograph({
        # 交通方式-->用户？
        ('paper', 'pa', 'author'): p_vs_a.nonzero(),
        ('author', 'ap', 'paper'): p_vs_a.transpose().nonzero(),
        # 交通方式-->od对？
        ('paper', 'pf', 'field'): p_vs_l.nonzero(),
        ('field', 'fp', 'paper'): p_vs_l.transpose().nonzero()
    })

    print("hg =", hg)

    # 用户特征-->官方给的属性文件，第一个参数为用户个数，第2个参数为特征个数
    # torch.FloatTensor()-->>该函数能将矩阵转换为tensor
    #hg.nodes['author'].data['feature'] = torch.ones(815, 66)
    # od对特征-->距离
    #hg.nodes['field'].data['feature'] = torch.ones(986, 1)

    #print("hg.nodes['paper'] =", hg.nodes['author'].data['feature'])
    # np, show()
    # -->融合用户的特征
    # od_encode = pd.read_csv("../p38dglproject/dataset/output/beijing/train_click_od.csv", usecols=['od_encode'])
    od_distance = pd.read_csv("../p38dglproject/dataset/output/beijing/train_click_od.csv", usecols=['manhattan'])
    # 空间属性
    o_d_lng_lat = pd.read_csv("../p38dglproject/dataset/output/beijing/beijing_nonoise.csv", usecols=['o_lng','o_lat', 'd_lng', 'd_lat'])
    # 时间属性
    timedata = pd.read_csv("../p38dglproject/dataset/output/beijing/time_feature.csv", usecols=['req_time_hour','req_time_weekday', 'elapsed_time', 'minute', 'month', 'day'])
    # 气象特征属性
    max_min_temp = pd.read_csv("../p38dglproject/dataset/output/beijing/time_feature.csv", usecols=['max_temp', 'min_temp', 'wind'])
    weatherdata = pd.read_csv("../p38dglproject/dataset/output/beijing/time_feature.csv",
                              usecols=['weather'])
    winddata = pd.read_csv("../p38dglproject/dataset/output/beijing/time_feature.csv",
                              usecols=['wind'])
    # plan中的特征
    # plandata = pd.read_csv("../p38dglproject/dataset/output/beijing/plan_feature.csv")

    # myfeature = pd.read_csv("../p38dglproject/dataset/output/beijing/norm_context.csv")
    # myfeature_new = myfeature.values

    # 原始属性--》主要是用户属性p0-p65
    origin_feature = pd.read_csv("../p38dglproject/dataset/output/beijing/beijing_nonoise.csv")
    # 这个属性，不包含sid、req_time、o、d、plan_time、plans、click_time、click_mode、city_flag_o
    del origin_feature['sid']
    del origin_feature['pid']
    del origin_feature['req_time']
    del origin_feature['o']
    del origin_feature['d']
    del origin_feature['plan_time']
    del origin_feature['plans']
    del origin_feature['click_time']
    # 这几项按道理是需要的
    del origin_feature['o_lng']
    del origin_feature['o_lat']
    del origin_feature['d_lng']
    del origin_feature['d_lat']
    del origin_feature['click_mode']
    # origin_feature['click_mode'] = origin_feature['click_mode'] - 1
    del origin_feature['city_flag_o']

    # print("origin_feature =", origin_feature)
    # plan属性
    # -->实验证明，plan_mode_fea很重要
    plan_mode_fea = pd.read_csv("../p38dglproject/dataset/output/beijing/plan_feature.csv", usecols=['mode_feas_1', 'mode_feas_2', 'mode_feas_3','mode_feas_4', 'mode_feas_5', 'mode_feas_6', 'mode_feas_7', 'mode_feas_8', 'mode_feas_9','mode_feas_10', 'mode_feas_11'])
    plan_firstmode_fea = pd.read_csv("../p38dglproject/dataset/output/beijing/plan_feature.csv", usecols=['first_mode'])
    plan_speed_fea = pd.read_csv("../p38dglproject/dataset/output/beijing/plan_feature.csv", usecols=['speed_feas_1', 'speed_feas_2', 'speed_feas_3','speed_feas_4', 'speed_feas_5', 'speed_feas_6', 'speed_feas_7', 'speed_feas_8', 'speed_feas_9','speed_feas_10', 'speed_feas_11'])

    plan_price_fea = pd.read_csv("../p38dglproject/dataset/output/beijing/plan_feature.csv", usecols=['price_feas_1', 'price_feas_2', 'price_feas_3','price_feas_4', 'price_feas_5', 'price_feas_6', 'price_feas_7', 'price_feas_8', 'price_feas_9','price_feas_10', 'price_feas_11'])

    # plan_distance_fea = pd.read_csv("../p38dglproject/dataset/output/beijing/plan_feature.csv", usecols=['distance_feas_1', 'distance_feas_2', 'distance_feas_3','distance_feas_4', 'distance_feas_5', 'distance_feas_6', 'distance_feas_7', 'distance_feas_8', 'distance_feas_9','distance_feas_10', 'distance_feas_11'])

    # plan_eta_fea = pd.read_csv("../p38dglproject/dataset/output/beijing/plan_feature.csv", usecols=['eta_feas_1', 'eta_feas_2', 'eta_feas_3','eta_feas_4', 'eta_feas_5', 'eta_feas_6', 'eta_feas_7', 'eta_feas_8', 'eta_feas_9','eta_feas_10', 'eta_feas_11'])

    plan_energy_fea = pd.read_csv("../p38dglproject/dataset/output/beijing/plan_feature.csv", usecols=['energy_feas_1', 'energy_feas_2', 'energy_feas_3','energy_feas_4', 'energy_feas_5', 'energy_feas_6', 'energy_feas_7', 'energy_feas_8', 'energy_feas_9','energy_feas_10', 'energy_feas_11'])

    # plan_mode_NLP = pd.read_csv("../p38dglproject/dataset/output/beijing/plan_feature.csv", usecols=['svd_mode_0', 'svd_mode_1', 'svd_mode_2','svd_mode_3', 'svd_mode_4', 'svd_mode_5', 'svd_mode_6', 'svd_mode_7', 'svd_mode_8','svd_mode_9'])

    # od相关数据处理
    od_feat = pd.read_csv(path + who + '/train_click_od.csv')
    del od_feat['Unnamed: 0']
    # -->od特征<----结合了pid的特征
    od_data_feat = pd.read_csv(path + who + '/beijing_od_unique_feature.csv')
    # 想od特征数据中添加索引od
    od_data_feat['od'] = od_data_feat['od_unique']
    # 删除其中的无用特征
    del od_data_feat['od_unique']
    # 数据拼接
    od_feat = od_feat.merge(od_data_feat, 'left', ['od'])
    del od_feat['Unnamed: 0']
    # 删除无关特征
    # sid,o,d,click_mode,od,od_encode,manhattan,sid_ID,od_encode_ID,od_ID,
    del od_feat['sid']
    del od_feat['o']
    del od_feat['d']
    del od_feat['click_mode']
    del od_feat['od']
    del od_feat['od_encode']
    del od_feat['manhattan']
    del od_feat['sid_ID']
    del od_feat['od_encode_ID']
    del od_feat['od_ID']

    # print("od_feat =", od_feat)

    # -->pid数据处理
    pid_feat = pd.read_csv(path + who + '/train_click_pid.csv')
    # print("pid_data =", pid_feat, pid_feat.shape)
    del pid_feat['Unnamed: 0']
    # pid特征<-----融合了od的特征
    pid_data_feat = pd.read_csv(path + who + '/beijing_pid_unique_feature.csv')
    del pid_data_feat['Unnamed: 0']
    # print("pid_data_feat =", pid_data_feat, pid_data_feat.shape)
    # 想pid特征数据中添加索引pid
    pid_data_feat['pid'] = pid_data_feat['pid_unique']
    # 删除其中的无用特征
    del pid_data_feat['pid_unique']
    # np,show()
    # 数据拼接
    pid_feat = pid_feat.merge(pid_data_feat, 'left', ['pid'])
    # sid,pid,click_mode,sid_ID,pid_ID
    # print("pid_feat =", pid_feat, pid_feat.shape)
    del pid_feat['sid']
    del pid_feat['pid']
    del pid_feat['click_mode']
    del pid_feat['sid_ID']
    del pid_feat['pid_ID']
    # print("pid_feat ", pid_feat, pid_feat.shape)
    # np,show()

    #------->转化为数组<-------#
    # 通过图注意力网络获得od及pid的新特征！！！
    pid_feat_feature = pid_feat.values
    od_feat_feature = od_feat.values

    # dataframe转化为array:https://blog.csdn.net/weixin_39223665/article/details/79935467
    # pidfeaturearray = pidfeaturenew.values
    # od_feature = od_encode.values
    # 浮点数的onehot编码太长了，机器很难存下，尽量换成整形
    od_distance_feature = od_distance.values.astype(np.int)
    # 空间特征
    '''
    # o部分相关数据导入<--与d数量有关，与pid数量有关，与时间time有关
    o_time = pd.read_csv(path + who + '/o_time_embedding.csv')
    o_unique = pd.read_csv(path + who + '/o_unique.csv')
    o_pid_unique = pd.read_csv(path + who + '/o_pid_unique.csv')

    d_time = pd.read_csv(path + who + '/d_time_embedding.csv')
    d_unique = pd.read_csv(path + who + '/d_unique.csv')
    d_pid_unique = pd.read_csv(path + who + '/d_pid_unique.csv')

    od_pid_unique = pd.read_csv(path + who + '/od_pid_unique.csv')
    # o和d-->对d,o, pid, time的特征
    # o部分
    o_d_lng_lat = o_d_lng_lat.merge(o_time, 'left', ['o'])
    o_d_lng_lat = o_d_lng_lat.merge(o_unique, 'left', ['o'])
    o_d_lng_lat = o_d_lng_lat.merge(o_pid_unique, 'left', ['o'])
    print("o_d_lng_lat =", o_d_lng_lat)
    # d部分
    o_d_lng_lat = o_d_lng_lat.merge(d_time, 'left', ['d'])
    o_d_lng_lat = o_d_lng_lat.merge(d_unique, 'left', ['d'])
    o_d_lng_lat = o_d_lng_lat.merge(d_pid_unique, 'left', ['d'])
    # od部分
    o_d_lng_lat['od'] = o_d_lng_lat['o']+o_d_lng_lat['d']
    o_d_lng_lat = o_d_lng_lat.merge(od_pid_unique, 'left', ['od'])
    del o_d_lng_lat['o']
    del o_d_lng_lat['d']
    del o_d_lng_lat['od']
    '''
    # print("o_d_lng_lat =", o_d_lng_lat)
    o_d_lng_lat_feature = o_d_lng_lat.values
    # np,show()
    # print("type(o_lng_feature) =", type(o_d_lng_lat_feature))
    # 时间特征
    timefeature = timedata.values

    # 气象特征
    max_min_temparrany = max_min_temp.values
    weatherdataarrany = weatherdata.values
    winddataarrany = winddata.values

    # plan特征
    # 实验证明它很重要
    plan_mode_feaarrany = plan_mode_fea.values

    # 首推模式onehot编码,尽量减1，.mat中的模式都减1
    plan_firstmode_feaarrany = plan_firstmode_fea.values - 1
    # 速度编码
    plan_speed_feaarrany = plan_speed_fea.values.astype(np.int)
    plan_price_feaarrany = (plan_price_fea.values/100).astype(np.int)
    # 这三个属性不好评价
    # plan_distance_feaarrany = plan_distance_fea.values
    # plan_eta_feaarrany = plan_eta_fea.values
    # plan_mode_NLParrany = plan_mode_NLP.values
    # 交通方式使用能源的类型
    plan_energy_feaarrany = plan_energy_fea.values
    # 本身就是onehot编码
    origin_feature_new = origin_feature.values

    # 标准化--->对于分类可能有用
    scaler = preprocessing.StandardScaler()

    # 最大最小归一化更合适
    # plan_speed_feaarrany = scaler.fit_transform(plan_speed_feaarrany)

    # 可考虑使用最大最小归一化方法，将所有值隐射到[0,1]之间

    min_max_scaler = preprocessing.MinMaxScaler()
    # o_d_lng_lat_feature = min_max_scaler.fit_transform(o_d_lng_lat_feature)
    # print("o_d_lng_lat_feature =", o_d_lng_lat_feature, o_d_lng_lat_feature.shape)
    # timefeature = min_max_scaler.fit_transform(timefeature)

    # ------>之前这里用最大最小归一化的<----------
    # 速度的值用归一化方式的结果似乎比onehot编码的效果要好
    plan_speed_feaarrany = min_max_scaler.fit_transform(plan_speed_feaarrany)

    # plan_firstmode_feaarrany = min_max_scaler.fit_transform(plan_firstmode_feaarrany)
    # plan_price_feaarrany = min_max_scaler.fit_transform(plan_price_feaarrany)
    # plan_energy_feaarrany = min_max_scaler.fit_transform(plan_energy_feaarrany)
    # od_distance_feature = min_max_scaler.fit_transform(od_distance_feature)
    # 用最大最小归一化效果达到，0.7422
    max_min_temparrany = min_max_scaler.fit_transform(max_min_temparrany)
    winddataarrany = min_max_scaler.fit_transform(winddataarrany)

    # max_min_temparrany = scaler.fit_transform(max_min_temparrany)
    # winddataarrany = scaler.fit_transform(winddataarrany)

    # 前两个用normalize，后一个用max_min
    # plan_distance_feaarrany = min_max_scaler.fit_transform(plan_distance_feaarrany)
    # plan_eta_feaarrany = min_max_scaler.fit_transform(plan_eta_feaarrany)
    # plan_mode_NLParrany = min_max_scaler.fit_transform(plan_mode_NLParrany)

    # 通过图注意力获得的od,pid 新特征
    # od_feat_feature = min_max_scaler.fit_transform(od_feat_feature)
    # pid_feat_feature = min_max_scaler.fit_transform(pid_feat_feature)
    # 效果超级差
    # od_feature = min_max_scaler.fit_transform(od_feature)

    # 正则归一化？--->似乎对于分类效果明显的增强了
    # o_d_lng_lat_feature = preprocessing.normalize(o_d_lng_lat_feature, norm='l2')
    # print("o_d_lng_lat_feature =", o_d_lng_lat_feature, o_d_lng_lat_feature.shape)
    # timefeature = preprocessing.normalize(timefeature, norm='l2')
    # 速度的值用归一化方式的结果似乎比onehot编码的效果要好<----已放到最大最小归一化处
    # plan_firstmode_feaarrany = preprocessing.normalize(plan_firstmode_feaarrany, norm='l2')
    # plan_price_feaarrany = preprocessing.normalize(plan_price_feaarrany, norm='l2')
    # plan_energy_feaarrany = preprocessing.normalize(plan_energy_feaarrany, norm='l2')
    # od_distance_feature = preprocessing.normalize(od_distance_feature, norm='l2')
    # 字符不能转换
    # weatherdataarrany = preprocessing.normalize(weatherdataarrany, norm='l2')
    # 最大最小归一化似乎结果更好
    # max_min_temparrany = preprocessing.normalize(max_min_temparrany, norm='l2')
    # winddataarrany = preprocessing.normalize(winddataarrany, norm='l2')

    # 前两个用normalize，后一个用max_min
    # plan_distance_feaarrany = preprocessing.normalize(plan_distance_feaarrany, norm='l2')
    # plan_eta_feaarrany = preprocessing.normalize(plan_eta_feaarrany, norm='l2')
    # plan_mode_NLParrany = preprocessing.normalize(plan_mode_NLParrany, norm='l2')

    # 通过图注意力获得的od,pid 新特征
    # od_feat_feature = preprocessing.normalize(od_feat_feature, norm='l2')
    # pid_feat_feature = preprocessing.normalize(pid_feat_feature, norm='l2')

    # od编码特征
    # od_feature = preprocessing.normalize(od_feature, norm='l2')

    # ---->使用onehot编码总体效果提升，应该是欧式距离可加减<----- # !!! 占用空间太多
    # ------>如果内存够，全用onehot编码都行<------#
    # 对od的特征进行onehot编码处理
    enc = OneHotEncoder(sparse=False)
    # od_feature = enc.fit_transform(od_feature)
    # 利用onehot编码可以增强效果
    o_d_lng_lat_feature = enc.fit_transform(o_d_lng_lat_feature)
    # print("o_d_lng_lat_feature =", o_d_lng_lat_feature, o_d_lng_lat_feature.shape)
    # 时间
    timefeature = enc.fit_transform(timefeature)

    # 速度这项值用归一化效果更好<---已放到最大最小归一化处
    plan_firstmode_feaarrany = enc.fit_transform(plan_firstmode_feaarrany)
    plan_price_feaarrany = enc.fit_transform(plan_price_feaarrany)
    plan_energy_feaarrany = enc.fit_transform(plan_energy_feaarrany)
    od_distance_feature = enc.fit_transform(od_distance_feature)

    # o_d_lng_lat_feature = scaler.fit_transform(o_d_lng_lat_feature)
    # timefeature = scaler.fit_transform(timefeature)

    # plan_firstmode_feaarrany = scaler.fit_transform(plan_firstmode_feaarrany)
    # plan_price_feaarrany = scaler.fit_transform(plan_price_feaarrany)
    # plan_energy_feaarrany = scaler.fit_transform(plan_energy_feaarrany)
    # od_distance_feature = scaler.fit_transform(od_distance_feature)

    # 前两个用normal，后一个用max_min
    # plan_distance_feaarrany = enc.fit_transform(plan_distance_feaarrany)
    # plan_eta_feaarrany = enc.fit_transform(plan_eta_feaarrany)
    # plan_mode_NLParrany = enc.fit_transform(plan_mode_NLParrany)

    # 气象
    # 天气情况--->对于字符而言，还是得用onehot编码
    # print("weatherdataarrany =", weatherdataarrany, weatherdataarrany.shape)
    weatherdataarrany = enc.fit_transform(weatherdataarrany)
    # print("weatherdataarrany =", weatherdataarrany, weatherdataarrany.shape)
    # 气温-->最大最小均值效果似乎更好
    # max_min_temparrany = enc.fit_transform(max_min_temparrany)
    # 风力强度 -->用原数据效果好一点-->感觉最大最小归一化不错呢？
    # winddataarrany = enc.fit_transform(winddataarrany)

    # 对应行拼接：https://blog.csdn.net/zyl1042635242/article/details/43162031
    # 只有交通方式的特征
    # 自定义特征
    # termfeature = p_vs_t.toarray()  # -->这个特征来自于label，不是很适合使用

    # 交通方式之间的关联特征

    # -------------------->重要特征<------------------------#

    # -->用户属性，时间，空间(位置，od对)，plan中的属性：mode，speed，firstmode(首推模式)，price
    # pid_feat_feature
    # od_feat_feature
    # 时间属性
    termfeature = timefeature

    # plan中的属性，分别是：mode，speed，firstmode(首推模式)，price
    termfeature = np.concatenate((termfeature, plan_mode_feaarrany), axis=1)
    termfeature = np.concatenate((termfeature, plan_speed_feaarrany), axis=1)
    termfeature = np.concatenate((termfeature, plan_firstmode_feaarrany), axis=1)
    termfeature = np.concatenate((termfeature, plan_price_feaarrany), axis=1)

    # termfeature = np.concatenate((termfeature, plan_distance_feaarrany), axis=1)
    # termfeature = np.concatenate((termfeature, plan_eta_feaarrany), axis=1)
    # termfeature = np.concatenate((termfeature, plan_mode_NLParrany), axis=1)

    # 交通方式使用的能量是不同的--->使用能量种类不同
    termfeature = np.concatenate((termfeature, plan_energy_feaarrany), axis=1)
    # -->距离对交通方式的选择具有一定的影响，比如小于10km时，更希望步行或乘坐自行车
    # termfeature = np.concatenate((termfeature, od_distance_feature), axis=1)
    # 气象特征-->全部加入，性能就提升了，不要单独放入(单独放入，性能似乎下降！！！)
    # -->天气状况
    # termfeature = np.concatenate((termfeature, weatherdataarrany), axis=1)
    # -->最高，最低温度
    # termfeature = np.concatenate((termfeature, max_min_temparrany), axis=1)
    # -->风力强度-->0.7058
    # termfeature = np.concatenate((termfeature, winddataarrany), axis=1)

    # od_feature---没什么用处啊，别瞎加
    # termfeature = np.concatenate((termfeature, od_feature),axis=1)

    # print("termfeature =", termfeature, termfeature.shape)

    #--------->保存当前生成的原始特征<--------#
    '''
    # 保存sid的此时的特征，再用图注意力学习每个节点的特征
    print("准备保存sid之前的属性（特征）")
    sid_feat = pd.DataFrame(termfeature)
    sid_feat.to_csv("../p38dglproject/dataset/output/beijing/beijing_sid_feat.csv")
    np,show()
    '''

    # -------->尝试如何将生成图注意力生成的sid特征直接用于训练？<---------#不保存，？但相当于每次都得重新训练，可能更久？
    # sid_unique_pid_feature = extract_sid_pid_feat()
    # sid_unique_od_feature = extract_sid_od_feat()
    # sid_unique_pid_without_od_feature = extract_sid_pid_without_od_feat()
    # sid<---pid信息融入sid

    sid_unique_pid_feature = pd.read_csv("../p38dglproject/dataset/output/beijing/beijing_sid_unique_pid_feature.csv")
    del sid_unique_pid_feature['Unnamed: 0']
    del sid_unique_pid_feature['sid_unique_pid']
    # print("sid_pid_feature =", sid_unique_pid_feature, sid_unique_pid_feature.shape)
    # 数值类型发生变化
    sid_unique_pid_featurearray = sid_unique_pid_feature.values
    # 归一化后，就正常了：https://ssjcoding.github.io/2019/03/27/normalization-and-standardization/
    # sid_unique_pid_featurearray = preprocessing.normalize(sid_unique_pid_featurearray, norm='l2')
    # sid_unique_pid_featurearray = min_max_scaler.fit_transform(sid_unique_pid_featurearray)
    # print("sid_unique_pid_featurearray =", sid_unique_pid_featurearray, sid_unique_pid_featurearray.shape)

    # sid<---od信息融入sid
    sid_unique_od_feature = pd.read_csv("../p38dglproject/dataset/output/beijing/beijing_sid_unique_od_feature.csv")
    del sid_unique_od_feature['Unnamed: 0']
    del sid_unique_od_feature['sid_unique_od']
    # print("sid_od_feature =", sid_unique_od_feature, sid_unique_od_feature.shape)
    # 数值类型发生变化---->这里的值比较大，必须归一化
    sid_unique_od_featurearray = sid_unique_od_feature.values
    # 归一化后，就正常了：https://ssjcoding.github.io/2019/03/27/normalization-and-standardization/
    sid_unique_od_featurearray = preprocessing.normalize(sid_unique_od_featurearray, norm='l2')
    # sid_unique_od_featurearray = min_max_scaler.fit_transform(sid_unique_od_featurearray)
    # print("sid_unique_od_featurearray =", sid_unique_od_featurearray, sid_unique_od_feature.shape)


    # 不对od对融合信息pid对对sid的影响---->说实话，效果真不错0.7474
    sid_unique_pid_without_od_feature = pd.read_csv("../p38dglproject/dataset/output/beijing/beijing_sid_unique_pid_without_od_feature.csv")
    # print("sid_unique_pid_without_od_feature =", sid_unique_pid_without_od_feature, sid_unique_pid_without_od_feature.shape)
    del sid_unique_pid_without_od_feature['Unnamed: 0']
    del sid_unique_pid_without_od_feature['sid_unique_pid_without_od']
    # print("sid_unique_pid_without_od_feature =", sid_unique_pid_without_od_feature, sid_unique_pid_without_od_feature.shape)
    # 数值类型发生变化
    sid_unique_pid_without_od_featurearray = sid_unique_pid_without_od_feature.values
    # 归一化后，就正常了：https://ssjcoding.github.io/2019/03/27/normalization-and-standardization/
    # 效果达到0.7416
    # sid_unique_pid_without_od_featurearray = preprocessing.normalize(sid_unique_pid_without_od_featurearray, norm='l2')
    # 效果只有0.7368
    # sid_unique_pid_without_od_featurearray = min_max_scaler.fit_transform(sid_unique_pid_without_od_featurearray)
    # print("sid_unique_pid_without_od_feature =", sid_unique_pid_without_od_featurearray, sid_unique_pid_without_od_featurearray.shape)

    
    # 不对od对融合信息pid对对sid的影响
    sid_unique_od_without_pid_feature = pd.read_csv(
        "../p38dglproject/dataset/output/beijing/beijing_sid_unique_od_without_pid_feature.csv")
    # print("sid_unique_od_without_pid_feature =", sid_unique_od_without_pid_feature, sid_unique_od_without_pid_feature.shape)
    # 删除无关列
    del sid_unique_od_without_pid_feature['Unnamed: 0']
    del sid_unique_od_without_pid_feature['sid_unique_od_without_pid']
    # print("sid_unique_od_without_pid_feature =", sid_unique_od_without_pid_feature,
    #      sid_unique_od_without_pid_feature.shape)
    # 数值类型发生变化-->该值不需要归一化就很不错，原因是在次之前也算是做了相应的归一化操作
    sid_unique_od_without_pid_featurearray = sid_unique_od_without_pid_feature.values
    # print("sid_unique_od_without_pid_feature =", sid_unique_od_without_pid_featurearray,
    #      sid_unique_od_without_pid_featurearray.shape)

    # GAT layer的结果
    # 下面的不动！！！！！！！！！！！！！！！~~~~~~~~~~~~~~
    # 融合该结果后，似乎效果也变差了？等等看看什么回事<----------------按道理会更好！！-->果然是od特征提取出现问题
    # 按设计，用以下两项特征就有最好的效果
    # 单独使用它达到了的效果--->0.7206-->测试pid对结果影响时，不应该用它，而适合用下面的选择
    # termfeature = np.concatenate((termfeature, sid_unique_pid_featurearray), axis=1)
    # 用了它，效果反而变差了？-->测试od对结果影响时，不应该用它，而适合用下面的选择
    # termfeature = np.concatenate((termfeature, sid_unique_od_featurearray), axis=1)

    # 单独用无od影响的pid对sid的特征，效果达到了--->0.6991
    termfeature = np.concatenate((termfeature, sid_unique_pid_without_od_featurearray), axis=1)

    # 单独用无pid影响的od对sid的特征，效果达到了-->0.7365
    # termfeature = np.concatenate((termfeature, sid_unique_od_without_pid_featurearray), axis=1)

    # 其实这两个特征更像sid中的额外特征(OD,PID中特地取出来的)
    # -->利用图注意力机制生成的新特征-->这个值使结果提升到了0.7341？-->用户特征
    # termfeature = np.concatenate((termfeature, pid_feat_feature), axis=1)
    # -->这个值似乎使得效果变差？可能是od坐标特征暂时无用？
    # termfeature = np.concatenate((termfeature, od_feat_feature), axis=1)

    # 用户属性
    termfeature = np.concatenate((termfeature, origin_feature_new), axis=1)
    # od对，空间属性
    # termfeature = np.concatenate((termfeature, o_d_lng_lat_feature), axis=1)

    # 相当于提取交通方式的特征
    features = torch.FloatTensor(termfeature[:num])
    print("features =", features,  features.shape)
    print("len(features) =", len(features))

    pc_p, pc_c = p_vs_c.nonzero()
    labels = np.zeros(len(p_selected), dtype=np.int64)
    for conf_id, label_id in zip(conf_ids, label_ids):
        labels[pc_p[pc_c == conf_id]] = label_id
    # print(labels, labels.shape, min(labels), max(labels))
    # np,show()
    labels = torch.LongTensor(labels)
    # 赋予标签
    print("labels =", labels)
    # 分类数
    num_classes = 11
    # 掩码？-->用于拆分训练集、验证集和测试集
    float_mask = np.zeros(len(pc_p))
    for conf_id in conf_ids:
        pc_c_mask = (pc_c == conf_id)
        float_mask[pc_c_mask] = np.random.permutation(np.linspace(0, 1, pc_c_mask.sum()))
    print("float_mask =", float_mask)

    train_idx = np.where(float_mask <= 0.7)[0]
    val_idx = np.where((float_mask > 0.7) & (float_mask <= 0.8))[0]
    test_idx = np.where(float_mask > 0.8)[0]

    print("len(train_idx) =", len(train_idx))
    print("len(val_idx) =", len(val_idx))
    print("len(test_idx) =", len(test_idx))
    # 这里就挺明确了，以SID为主
    num_nodes = hg.number_of_nodes('paper')
    train_mask = get_binary_mask(num_nodes, train_idx)
    val_mask = get_binary_mask(num_nodes, val_idx)
    test_mask = get_binary_mask(num_nodes, test_idx)

    print("hg =\n", hg)
    return hg, features, labels, num_classes, train_idx, val_idx, test_idx, \
            train_mask, val_mask, test_mask


def load_data(dataset, remove_self_loop=False):
    if dataset == 'ACM':
        return load_acm(remove_self_loop)
    elif dataset == 'ACMRaw':
        return load_acm_raw(remove_self_loop)
    elif dataset == "Mydataset":
        return load_mydata_raw(remove_self_loop)
    else:
        return NotImplementedError('Unsupported dataset {}'.format(dataset))

class EarlyStopping(object):
    def __init__(self, patience=10):
        dt = datetime.datetime.now()
        # self.filename = 'early_stop_{}_{:02d}-{:02d}-{:02d}.pth'.format(
        #    dt.date(), dt.hour, dt.minute, dt.second)
        self.filename = 'early_stop.pth'
        self.patience = patience
        self.counter = 0
        self.best_acc = None
        self.best_loss = None
        self.early_stop = False

    def step(self, loss, acc, model):
        if self.best_loss is None:
            self.best_acc = acc
            self.best_loss = loss
            # 上一次的模型接着导入训练
            # self.load_checkpoint(model)
            self.save_checkpoint(model)
        elif (loss > self.best_loss) and (acc < self.best_acc):
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if (loss <= self.best_loss) and (acc >= self.best_acc):
                self.save_checkpoint(model)
            self.best_loss = np.min((loss, self.best_loss))
            self.best_acc = np.max((acc, self.best_acc))
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        """Saves model when validation loss decreases."""
        torch.save(model.state_dict(), self.filename)

    def load_checkpoint(self, model):
        """Load the latest checkpoint."""
        model.load_state_dict(torch.load(self.filename))
