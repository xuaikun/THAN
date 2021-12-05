# encoding: utf-8
# 利用图注意力网络提取目标节点(可变维度的)特征！！！！
import dgl
import pandas as pd
import numpy as np
import torch as th
from dgl.nn import GATConv
import torch.nn as nn
import dgl.function as fn
from dgl.nn.pytorch import GraphConv
from joblib.numpy_pickle_utils import xrange
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
import torch.nn.functional as F

# 最大最小归一化
min_max_scaler = preprocessing.MinMaxScaler()
'''
u = [0, 1, 0, 0, 1]
v = [0, 1, 2, 3, 2]
# g = dgl.bipartite((u, v))
g = dgl.heterograph({('o_type', 'od_type', 'd_type'): (u, v)})
u_feat = th.tensor(np.random.rand(2, 5).astype(np.float32))
v_feat = th.tensor(np.random.rand(4, 10).astype(np.float32))
gatconv = GATConv((5,10), 2, 3)
res = gatconv(g, (u_feat, v_feat))
print(res)
np,show()
'''

path = '../p38dglproject/dataset/output/'
who = 'beijing'

# 经纬度拆分函数
def split_od(data):
    data['o_lng'] = data['o'].apply(lambda x: float(x.split(',')[0])).astype(np.float)
    data['o_lat'] = data['o'].apply(lambda x: float(x.split(',')[1])).astype(np.float)
    data['d_lng'] = data['d'].apply(lambda x: float(x.split(',')[0])).astype(np.float)
    data['d_lat'] = data['d'].apply(lambda x: float(x.split(',')[1])).astype(np.float)
    return data


# 对用户特征进行降维
def generate_profile_features(data):
    # 人的属性
    profile_data = pd.read_csv('../p38dglproject/dataset/data_set_phase2/profiles.csv')
    # print("profile_data =", profile_data)
    # x等于不包括PID的值
    x = profile_data.drop(['pid'], axis=1).values
    # print("x =", x)
    # 奇异值降维-20维
    svd = TruncatedSVD(n_components=20, n_iter=20, random_state=42)
    # print("svd =", svd)
    svd_x = svd.fit_transform(x)
    svd_feas = pd.DataFrame(svd_x)
    # print("svd_feas =", svd_feas)
    svd_feas.columns = ['svd_attribute_{}'.format(i) for i in range(20)]
    # svd_feas加入pid列
    svd_feas['pid'] = profile_data['pid'].values
    # print("svd_feas =", svd_feas)
    # data['pid'] = data['pid'].fillna(-1)
    # print("data =", data)
    data = data.merge(svd_feas, on='pid', how='left')
    return data

# 计算O或D对时间的偏好

def get_embedding(col):
    """
    Get the functional embedding of OD
    :param col: O or D
    :return: embedding
    """
    queries_data = pd.read_csv(path + who + '/beijing_nonoise.csv', parse_dates=['req_time'])
    queries_data["time_bins"] = queries_data["req_time"].dt.hour // 2
    data = queries_data
    o_time_nums = data.groupby([col, "time_bins"], as_index=False)[col].agg({"{}_nums".format(col): "count"})
    rows = o_time_nums[col].nunique()
    o_time_features = np.zeros((rows, 12))
    o_time_features = pd.DataFrame(o_time_features)
    o_time_features[col] = o_time_nums[col].unique().tolist()

    for i, nums in enumerate(o_time_nums.values):
        o_time_features.loc[o_time_features[col] == nums[0], nums[1]] = nums[2]

    cols = o_time_features.columns.tolist()
    cols.remove(col)
    sum_temp = o_time_features[cols].sum(axis=1)
    o_time_features[cols] = o_time_features[cols].div(sum_temp, axis="rows")
    cols = o_time_features.columns.tolist()
    cols.remove(col)
    print("Topic model starting...")
    svd_enc = LatentDirichletAllocation(n_components=5, max_iter=50)
    mode_svd = svd_enc.fit_transform(o_time_features[cols].values)
    mode_svd = pd.DataFrame(mode_svd)
    mode_svd.columns = ['{}_time_{}'.format(col, i) for i in range(5)]
    result = pd.concat([o_time_features[col], mode_svd], axis=1)
    print("{} embedding done.".format(col))
    return result

# unique--->单独出现的情况
# count-->所有出现情况，一对数出现的次数都统计，而unique只统计一次
# 统计对于grouped_col, target_col出现的频率，data,'d','pid'表示有多少'pid'使用了'd'
def get_unique(data, grouped_col, target_col):
    unique = data.groupby(grouped_col)[target_col].nunique().reset_index().rename(
        columns={target_col: "unique_{}".format(target_col)})
    result_sum = data.groupby(grouped_col, as_index=False)[target_col].agg({"{}_count".format(target_col): "count"})
    unique = unique.merge(result_sum, on=grouped_col, how="left")
    unique["{}_ratio".format(target_col)] = unique["unique_{}".format(target_col)].div(
        unique["{}_count".format(target_col)])
    return unique

# 是否提取od的额外特征：o对d,time,pid的特征，d对o，time，pid的特征以及od对pid的特征
od_flag = True

# 提取od对的特征
def extract_od_list():
    od_data = pd.read_csv(path + who + '/train_click_od.csv')
    print(od_data, od_data.shape)
    '''
    pid_data = pd.read_csv(path + who + '/train_click_pid.csv')
    # o或d对时间的偏好
    o_time = get_embedding("o")
    o_time.to_csv(path + who + '/o_time_embedding.csv', index=False)
    d_time = get_embedding("d")
    d_time.to_csv(path + who + '/d_time_embedding.csv', index=False)

    queries = od_data
    # 相应o和d出现的次数
    o_unique = get_unique(queries, "o", "d")
    o_unique.to_csv(path + who + '/o_unique.csv', index=False)
    d_unique = get_unique(queries, "d", "o")
    d_unique.to_csv(path + who + '/d_unique.csv', index=False)
    queries["pid"] = pid_data["pid"].fillna(-1)
    o_pid_unique = get_unique(queries, "o", "pid")
    o_pid_unique.to_csv(path + who + '/o_pid_unique.csv', index=False)
    d_pid_unique = get_unique(queries, "d", "pid")
    d_pid_unique.to_csv(path + who + '/d_pid_unique.csv', index=False)
    queries["od"] = queries["o"] + queries["d"]
    od_pid_unique = get_unique(queries, "od", "pid")
    od_pid_unique.to_csv(path + who + '/od_pid_unique.csv', index=False)
    '''

    # --->将od边上的频率统计出来
    od_count = od_data.groupby(['od'], as_index=False)['od'].agg({'od_count': 'count'})
    # 将od出现频率拼接到数据中
    od_data = od_data.merge(od_count, 'left', ['od'])
    # 向数据->添加o_ID
    # train_click_od = pd.read_csv(filepath + who + '/train_click_od.csv')
    print(od_data, od_data.shape)

    # od_count = od_data['od_count']
    # od_count = pd.DataFrame(od_count)
    # print("od_count =", od_count)
    # print("od_count.values =", od_count.values)
    # 对使用频率进行最大最小归一化-->出现0值，这是不好的现象哦？
    # od_count = min_max_scaler.fit_transform(od_count.values)
    # od_count = preprocessing.normalize(od_count.values, norm='l2')
    # od_data['od_count'] = od_count
    # print(od_data, od_data.shape)
    # np,show()
    # 该语句可查看sid是否唯一
    # --> 体现了节点的唯一性
    o_value_unique = od_data['o'].unique()

    # 找出节点o的唯一特征
    tempdata = pd.DataFrame()
    o_value_unique_new = pd.DataFrame({'o_unique': o_value_unique})
    tempdata['o_unique'] = o_value_unique_new['o_unique']
    print(o_value_unique)
    tempdata['o_lng'] = o_value_unique_new['o_unique'].apply(lambda x: float(x.split(',')[0])).astype(np.float)
    tempdata['o_lat'] = o_value_unique_new['o_unique'].apply(lambda x: float(x.split(',')[1])).astype(np.float)
    print(tempdata)

    o_value_unique = list(o_value_unique)
    print(od_data['o'].unique())
    print(len(od_data['o'].unique()))
    # 节点o的长度-->节点o的数量
    len_o = len(od_data['o'].unique())

    # 给o节点添加编号
    od_ID = []
    for i in range(len(o_value_unique)):
        od_ID.append(i)
    odvalue = od_data['o']
    pid_ID_list = []
    for i in odvalue:
        pid_ID_list.append(od_ID[o_value_unique.index(i)])
    od_data['o_ID'] = pid_ID_list

    # 向数据->添加d_ID
    # 该语句可查看sid是否唯一
    d_value_unique = od_data['d'].unique()
    d_value_unique_new = pd.DataFrame({'d_unique': d_value_unique})
    tempdata['d_unique'] = d_value_unique_new['d_unique']
    tempdata['d_lng'] = d_value_unique_new['d_unique'].apply(lambda x: float(x.split(',')[0])).astype(np.float)
    tempdata['d_lat'] = d_value_unique_new['d_unique'].apply(lambda x: float(x.split(',')[1])).astype(np.float)
    print(tempdata)

    d_value_unique = list(d_value_unique)
    print(od_data['d'].unique())
    print(len(od_data['d'].unique()))

    # 节点d的长度-->节点o的数量
    len_d = len(od_data['d'].unique())
    od_ID = []
    # 给d节点添加编号
    for i in range(len(d_value_unique)):
        od_ID.append(i)
    odvalue = od_data['d']
    pid_ID_list = []
    for i in odvalue:
        pid_ID_list.append(od_ID[d_value_unique.index(i)])
    od_data['d_ID'] = pid_ID_list

    # -->节点的唯一特征
    tempdata.to_csv(path + who + '/tempdata.csv')

    # 提取od的经纬度
    o_lng_lat_raw = pd.read_csv(path + who + '/tempdata.csv', usecols=['o_unique'])
    d_lng_lat_raw = pd.read_csv(path + who + '/tempdata.csv', usecols=['d_unique'])
    o_lng_lat = pd.read_csv(path + who + '/tempdata.csv', usecols=['o_unique', 'o_lng', 'o_lat'])
    d_lng_lat = pd.read_csv(path + who + '/tempdata.csv', usecols=['d_unique', 'd_lng', 'd_lat'])

    # 删除任何含nan的行-->去除噪声,用户信息缺失(pid = nan)或未点击选项(mode = 0)
    o_lng_lat = o_lng_lat.dropna(axis=0, how='any')
    d_lng_lat = d_lng_lat.dropna(axis=0, how='any')

    # 补充一列o或d用做索引
    o_lng_lat['o'] = o_lng_lat['o_unique']
    d_lng_lat['d'] = d_lng_lat['d_unique']

    print("o_lng_lat =", o_lng_lat)
    print("d_lng_lat =", d_lng_lat)

    if od_flag == True:
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
        o_lng_lat = o_lng_lat.merge(o_time, 'left', ['o'])
        o_lng_lat = o_lng_lat.merge(o_unique, 'left', ['o'])
        o_lng_lat = o_lng_lat.merge(o_pid_unique, 'left', ['o'])
        print("o_lng_lat =", o_lng_lat)
        # d部分
        d_lng_lat = d_lng_lat.merge(d_time, 'left', ['d'])
        d_lng_lat = d_lng_lat.merge(d_unique, 'left', ['d'])
        d_lng_lat = d_lng_lat.merge(d_pid_unique, 'left', ['d'])
        print("d_lng_lat =", d_lng_lat)

        # 删除无关项
        del o_lng_lat['o']
        del d_lng_lat['d']

    del o_lng_lat['o_unique']
    del d_lng_lat['d_unique']
    print("o_lng_lat =", o_lng_lat)
    # 空间特征
    o_lng_lat_feature = o_lng_lat.values
    d_lng_lat_feature = d_lng_lat.values

    # 利用onehot编码增强效果
    # enc = OneHotEncoder(sparse=False)
    # o_lng_lat_feature = enc.fit_transform(o_lng_lat_feature)
    # d_lng_lat_feature = enc.fit_transform(d_lng_lat_feature)
    # 维度少是少，表达的信息不见得少
    # o_d_lng_lat_feature = min_max_scaler.fit_transform(o_d_lng_lat_feature)
    o_lng_lat_feature = min_max_scaler.fit_transform(o_lng_lat_feature)
    d_lng_lat_feature = min_max_scaler.fit_transform(d_lng_lat_feature)

    # 节点特征维度
    o_lng_lat_feature_v = o_lng_lat_feature.shape[1]
    d_lng_lat_feature_v = d_lng_lat_feature.shape[1]

    print("o_lng_lat_feature =", o_lng_lat_feature, o_lng_lat_feature.shape)
    print("d_lng_lat_feature =", d_lng_lat_feature, d_lng_lat_feature.shape)

    # 节点特征，特征转化为tensor类型
    o_features = th.FloatTensor(o_lng_lat_feature)
    d_features = th.FloatTensor(d_lng_lat_feature)
    # np,show()

    # -------->可修改部分<---------#

    # 节点数量

    # 节点o的数量
    o_node_number = len_o
    # 节点d的数量
    d_node_number = len_d

    # 节点编号

    # 源节点的标号
    o = od_data['o_ID']
    # 目标节点的标号
    d = od_data['d_ID']

    # 初始特征维度

    # 节点o的特征维度
    o_feat_D = o_lng_lat_feature_v
    # 节点d的特征维度
    d_feat_D = d_lng_lat_feature_v

    # 源节点特征,()括号中的数分别为节点数量和特征维度
    # o_feat = th.tensor(np.random.rand(node_number_o, feature_o).astype(np.float32))
    o_feat = o_features
    # 目标节点特征
    # d_feat = th.tensor(np.random.rand(node_number_d, feature_d).astype(np.float32))
    d_feat = d_features
    '''
    # 更新目标节点的特征维度-->与d节点相同维度
    # -->节点d更新后的特征
    feat_update = d_feat_D
    myattention(o, o_node_number, o_feat, o_feat_D, d, d_node_number, d_feat, d_feat_D, feat_update, 'd_unique',
                d_lng_lat_raw, od_data['od_count'], edge_flag=True)
    '''
    return o, d



# 提取od对的特征
def extract_od_feat():
    od_data = pd.read_csv(path + who + '/train_click_od.csv')
    print(od_data, od_data.shape)
    '''
    pid_data = pd.read_csv(path + who + '/train_click_pid.csv')
    # o或d对时间的偏好
    o_time = get_embedding("o")
    o_time.to_csv(path + who + '/o_time_embedding.csv', index=False)
    d_time = get_embedding("d")
    d_time.to_csv(path + who + '/d_time_embedding.csv', index=False)

    queries = od_data
    # 相应o和d出现的次数
    o_unique = get_unique(queries, "o", "d")
    o_unique.to_csv(path + who + '/o_unique.csv', index=False)
    d_unique = get_unique(queries, "d", "o")
    d_unique.to_csv(path + who + '/d_unique.csv', index=False)
    queries["pid"] = pid_data["pid"].fillna(-1)
    o_pid_unique = get_unique(queries, "o", "pid")
    o_pid_unique.to_csv(path + who + '/o_pid_unique.csv', index=False)
    d_pid_unique = get_unique(queries, "d", "pid")
    d_pid_unique.to_csv(path + who + '/d_pid_unique.csv', index=False)
    queries["od"] = queries["o"] + queries["d"]
    od_pid_unique = get_unique(queries, "od", "pid")
    od_pid_unique.to_csv(path + who + '/od_pid_unique.csv', index=False)
    '''

    # --->将od边上的频率统计出来
    od_count = od_data.groupby(['od'], as_index=False)['od'].agg({'od_count': 'count'})
    # 将od出现频率拼接到数据中
    od_data = od_data.merge(od_count, 'left', ['od'])
    # 向数据->添加o_ID
    # train_click_od = pd.read_csv(filepath + who + '/train_click_od.csv')
    print(od_data, od_data.shape)
    o_d_count = pd.DataFrame()
    o_d_count['o_d_count'] = od_data['od_count']
    print(o_d_count, o_d_count.shape)
    o_d_count.to_csv(path + who + '/o_d_count.csv')
    np, show()
    # print("od_count.values =", od_count.values)
    # 对使用频率进行最大最小归一化-->出现0值，这是不好的现象哦？
    # od_count = min_max_scaler.fit_transform(od_count.values)
    # od_count = preprocessing.normalize(od_count.values, norm='l2')
    # od_data['od_count'] = od_count
    # print(od_data, od_data.shape)
    # np,show()
    # 该语句可查看sid是否唯一
    # --> 体现了节点的唯一性
    o_value_unique = od_data['o'].unique()

    # 找出节点o的唯一特征
    tempdata = pd.DataFrame()
    o_value_unique_new = pd.DataFrame({'o_unique': o_value_unique})
    tempdata['o_unique'] = o_value_unique_new['o_unique']
    print(o_value_unique)
    tempdata['o_lng'] = o_value_unique_new['o_unique'].apply(lambda x: float(x.split(',')[0])).astype(np.float)
    tempdata['o_lat'] = o_value_unique_new['o_unique'].apply(lambda x: float(x.split(',')[1])).astype(np.float)
    print(tempdata)

    o_value_unique = list(o_value_unique)
    print(od_data['o'].unique())
    print(len(od_data['o'].unique()))
    # 节点o的长度-->节点o的数量
    len_o = len(od_data['o'].unique())

    # 给o节点添加编号
    od_ID = []
    for i in range(len(o_value_unique)):
        od_ID.append(i)
    odvalue = od_data['o']
    pid_ID_list = []
    for i in odvalue:
        pid_ID_list.append(od_ID[o_value_unique.index(i)])
    od_data['o_ID'] = pid_ID_list

    # 向数据->添加d_ID
    # 该语句可查看sid是否唯一
    d_value_unique = od_data['d'].unique()
    d_value_unique_new = pd.DataFrame({'d_unique': d_value_unique})
    tempdata['d_unique'] = d_value_unique_new['d_unique']
    tempdata['d_lng'] = d_value_unique_new['d_unique'].apply(lambda x: float(x.split(',')[0])).astype(np.float)
    tempdata['d_lat'] = d_value_unique_new['d_unique'].apply(lambda x: float(x.split(',')[1])).astype(np.float)
    print(tempdata)

    d_value_unique = list(d_value_unique)
    print(od_data['d'].unique())
    print(len(od_data['d'].unique()))

    # 节点d的长度-->节点o的数量
    len_d = len(od_data['d'].unique())
    od_ID = []
    # 给d节点添加编号
    for i in range(len(d_value_unique)):
        od_ID.append(i)
    odvalue = od_data['d']
    pid_ID_list = []
    for i in odvalue:
        pid_ID_list.append(od_ID[d_value_unique.index(i)])
    od_data['d_ID'] = pid_ID_list
    o_d_od_ID_data = pd.DataFrame()
    o_d_od_ID_data['od_ID'] = od_data['od_ID']
    o_d_od_ID_data['o_ID'] = od_data['o_ID']
    o_d_od_ID_data['d_ID'] = od_data['d_ID']
    o_d_od_ID_data.to_csv(path + who + '/o_d_od_ID_data.csv')
    # np,show()
    # -->节点的唯一特征
    tempdata.to_csv(path + who + '/tempdata.csv')

    # 提取od的经纬度
    o_lng_lat_raw = pd.read_csv(path + who + '/tempdata.csv', usecols=['o_unique'])
    d_lng_lat_raw = pd.read_csv(path + who + '/tempdata.csv', usecols=['d_unique'])
    o_lng_lat = pd.read_csv(path + who + '/tempdata.csv', usecols=['o_unique', 'o_lng', 'o_lat'])
    d_lng_lat = pd.read_csv(path + who + '/tempdata.csv', usecols=['d_unique', 'd_lng', 'd_lat'])

    # 删除任何含nan的行-->去除噪声,用户信息缺失(pid = nan)或未点击选项(mode = 0)
    o_lng_lat = o_lng_lat.dropna(axis=0, how='any')
    d_lng_lat = d_lng_lat.dropna(axis=0, how='any')

    # 补充一列o或d用做索引
    o_lng_lat['o'] = o_lng_lat['o_unique']
    d_lng_lat['d'] = d_lng_lat['d_unique']

    print("o_lng_lat =", o_lng_lat)
    print("d_lng_lat =", d_lng_lat)

    if od_flag == True:
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
        o_lng_lat = o_lng_lat.merge(o_time, 'left', ['o'])
        o_lng_lat = o_lng_lat.merge(o_unique, 'left', ['o'])
        o_lng_lat = o_lng_lat.merge(o_pid_unique, 'left', ['o'])
        print("o_lng_lat =", o_lng_lat)
        # d部分
        d_lng_lat = d_lng_lat.merge(d_time, 'left', ['d'])
        d_lng_lat = d_lng_lat.merge(d_unique, 'left', ['d'])
        d_lng_lat = d_lng_lat.merge(d_pid_unique, 'left', ['d'])
        print("d_lng_lat =", d_lng_lat)

        # 删除无关项
        del o_lng_lat['o']
        del d_lng_lat['d']

    del o_lng_lat['o_unique']
    del d_lng_lat['d_unique']
    print("o_lng_lat =", o_lng_lat)
    # 空间特征
    o_lng_lat_feature = o_lng_lat.values
    d_lng_lat_feature = d_lng_lat.values

    # 利用onehot编码增强效果
    # enc = OneHotEncoder(sparse=False)
    # o_lng_lat_feature = enc.fit_transform(o_lng_lat_feature)
    # d_lng_lat_feature = enc.fit_transform(d_lng_lat_feature)
    # 维度少是少，表达的信息不见得少
    # o_d_lng_lat_feature = min_max_scaler.fit_transform(o_d_lng_lat_feature)
    o_lng_lat_feature = min_max_scaler.fit_transform(o_lng_lat_feature)
    d_lng_lat_feature = min_max_scaler.fit_transform(d_lng_lat_feature)

    # 节点特征维度
    o_lng_lat_feature_v = o_lng_lat_feature.shape[1]
    d_lng_lat_feature_v = d_lng_lat_feature.shape[1]

    print("o_lng_lat_feature =", o_lng_lat_feature, o_lng_lat_feature.shape)
    print("d_lng_lat_feature =", d_lng_lat_feature, d_lng_lat_feature.shape)

    # 节点特征，特征转化为tensor类型
    o_features = th.FloatTensor(o_lng_lat_feature)
    d_features = th.FloatTensor(d_lng_lat_feature)
    # np,show()

    # -------->可修改部分<---------#

    # 节点数量

    # 节点o的数量
    o_node_number = len_o
    # 节点d的数量
    d_node_number = len_d

    # 节点编号

    # 源节点的标号
    o = od_data['o_ID']
    # 目标节点的标号
    d = od_data['d_ID']

    # 初始特征维度

    # 节点o的特征维度
    o_feat_D = o_lng_lat_feature_v
    # 节点d的特征维度
    d_feat_D = d_lng_lat_feature_v

    # 源节点特征,()括号中的数分别为节点数量和特征维度
    # o_feat = th.tensor(np.random.rand(node_number_o, feature_o).astype(np.float32))
    o_feat = o_features
    # 目标节点特征
    # d_feat = th.tensor(np.random.rand(node_number_d, feature_d).astype(np.float32))
    d_feat = d_features

    # 更新目标节点的特征维度-->与d节点相同维度
    # -->节点d更新后的特征
    feat_update = d_feat_D
    myattention(o, o_node_number, o_feat, o_feat_D, d, d_node_number, d_feat, d_feat_D, feat_update, 'd_unique',
                d_lng_lat_raw, od_data['od_count'], edge_flag=True)
    # 更新目标节点的特征维度-->与o节点相同维度
    # -->节点o更新后的特征
    feat_update = o_feat_D
    myattention(d, d_node_number, d_feat, d_feat_D, o, o_node_number, o_feat, o_feat_D, feat_update, 'o_unique',
                o_lng_lat_raw, od_data['od_count'], edge_flag=True)

    # 拼接o和d更新后的特征为od更新部分特征：od = od(原)||o(更新)||d(更新)-->更新为od(原)
    # --->od的特征
    # d的特征 = d特征 + o特征
    d_feat_new = pd.read_csv(path + who + '/beijing_d_unique_feature.csv')
    d_feat_new['d'] = d_feat_new['d_unique']
    # 删除无关列
    d_feat_new = d_feat_new.drop(['Unnamed: 0'], axis=1)
    d_feat_new = d_feat_new.drop('d_unique', axis=1)
    # o的特征 = d特征+ o特征
    o_feat_new = pd.read_csv(path + who + '/beijing_o_unique_feature.csv')
    o_feat_new['o'] = o_feat_new['o_unique']
    # 删除无关列
    o_feat_new = o_feat_new.drop(['Unnamed: 0'], axis=1)
    o_feat_new = o_feat_new.drop(['o_unique'], axis=1)

    od_data = pd.read_csv(path + who + '/train_click_od.csv')
    # od特征拼接
    od_data = od_data.merge(d_feat_new, 'left', ['d'])
    od_data = od_data.merge(o_feat_new, 'left', ['o'])
    if od_flag == True:
        od_data = od_data.merge(od_pid_unique, 'left', ['od'])
    od_data = od_data.drop(['Unnamed: 0'], axis=1)
    od_data.to_csv(path + who + '/train_click_od_new.csv')
    print(od_data)

# 提取user and od对的特征
def extract_user_od_feat():
    print("extract_user_od_feat")
    # -->od特征
    od_data = pd.read_csv(path + who + '/train_click_od_new.csv')
    del od_data['Unnamed: 0']
    # 向数据->添加o_ID
    # train_click_od = pd.read_csv(filepath + who + '/train_click_od.csv')
    print(od_data, od_data.shape)
    # 该语句可查看sid是否唯一
    # --> 体现了节点的唯一性-->od_ID已经包含在里面了的
    # 找出节点o的唯一特征
    # 按'od'列删除重复的项，并保留第一次出现的项，https://www.cnblogs.com/zlc364624/p/12293666.html
    # 只包含od唯一的数据，是为了获取od的特征呢
    temp_od_data_feat = od_data.drop_duplicates(subset=['od'], keep='first', inplace=False)
    # 拆分od-->为了获得单独的特征
    # temp_od_data_feat = split_od(temp_od_data_feat)
    o_d_lng_lat = pd.DataFrame()
    o_d_lng_lat['o_lng'] = temp_od_data_feat['o'].apply(lambda x: float(x.split(',')[0])).astype(np.float)
    o_d_lng_lat['o_lat'] = temp_od_data_feat['o'].apply(lambda x: float(x.split(',')[1])).astype(np.float)
    o_d_lng_lat['d_lng'] = temp_od_data_feat['d'].apply(lambda x: float(x.split(',')[0])).astype(np.float)
    o_d_lng_lat['d_lat'] = temp_od_data_feat['d'].apply(lambda x: float(x.split(',')[1])).astype(np.float)
    # 相当于是目前唯一od对于的数据，作为保存是的索引
    od_data_raw = pd.DataFrame()
    od_data_raw['od_unique'] = temp_od_data_feat['od']
    print("temp_od_data_feat =", temp_od_data_feat, temp_od_data_feat.shape)

    # 节点od的长度-->节点od的数量
    len_o = len(od_data['od'].unique())
    # od特征？合并原始特征!!!
    od_data_feat = temp_od_data_feat
    # 利用onehot编码增强效果-->od原始特征
    enc = OneHotEncoder(sparse=False)
    # o_lng_lat_feature = enc.fit_transform(o_d_lng_lat[['o_lng', 'o_lat']])
    # d_lng_lat_feature = enc.fit_transform(o_d_lng_lat[['d_lng', 'd_lat']])
    # 利用最大最小归一化不见得差?maybe?
    o_lng_lat_feature = min_max_scaler.fit_transform(o_d_lng_lat[['o_lng', 'o_lat']])
    d_lng_lat_feature = min_max_scaler.fit_transform(o_d_lng_lat[['d_lng', 'd_lat']])
    print("o_lng_lat_feature =", o_lng_lat_feature, o_lng_lat_feature.shape)
    print("d_lng_lat_feature =", d_lng_lat_feature, d_lng_lat_feature.shape)

    '''
    od_data_feat = temp_od_data_feat.loc[:,
                   ['o_unique0', 'o_unique1', 'o_unique2', 'o_unique3', 'o_unique4', 'o_unique5', 'o_unique6',
                    'o_unique7', 'o_unique8', 'o_unique9',
                    'd_unique0', 'd_unique1', 'd_unique2', 'd_unique3', 'd_unique4', 'd_unique5', 'd_unique6',
                    'd_unique7', 'd_unique8', 'd_unique9']]
    '''
    # 删除无关特征
    # sid,o,d,click_mode,od,od_encode,manhattan,sid_ID,od_encode_ID,od_ID,
    del od_data_feat['sid']
    # od_data_feat = od_data_feat.drop(['sid'], axis=1)
    del od_data_feat['o']
    del od_data_feat['d']
    del od_data_feat['click_mode']
    del od_data_feat['od']
    del od_data_feat['od_encode']
    del od_data_feat['manhattan']
    del od_data_feat['sid_ID']
    del od_data_feat['od_encode_ID']
    del od_data_feat['od_ID']
    # np,show()
    # 合并原始数据？
    # od_data_feat = np.concatenate((od_data_feat, o_lng_lat_feature), axis=1)
    # od_data_feat = np.concatenate((od_data_feat, d_lng_lat_feature), axis=1)
    print("od_data_feat =", od_data_feat, od_data_feat.shape)

    # 节点特征数值类型转换
    o_lng_lat_feature = od_data_feat.values  # --->本身就是数组，不用变了
    print("o_lng_lat_feature =", o_lng_lat_feature)
    # 节点特征维度
    o_lng_lat_feature_v = o_lng_lat_feature.shape[1]
    print("o_lng_lat_feature =", o_lng_lat_feature, o_lng_lat_feature.shape)


    #------------->以下为PID的特征<--------------#
    pid_data = pd.read_csv(path + who + '/train_click_pid.csv')
    pid_feat_data = pd.read_csv(path + who + '/beijing_nonoise.csv')
    # 向数据->添加pid_ID
    # 该语句可查看sid是否唯一
    # 节点pid的长度-->节点pid的数量--->到此为止，用户信息仍然为42342
    print(len(pid_data['pid'].unique()))
    print(len(pid_feat_data['pid'].unique()))
    len_d = len(pid_data['pid'].unique())

    # np,show()
    # 按pid删除重复行
    temp_pid_data_feat = pid_feat_data.drop_duplicates(subset=['pid'], keep='first', inplace=False)
    print("temp_pid_data_feat =", temp_pid_data_feat,temp_pid_data_feat.shape)

    # -->节点的唯一特征
    # 提取pid原始值，用于后期索引使用
    pid_data_raw = pd.DataFrame()
    pid_data_raw['pid_unique'] = temp_pid_data_feat['pid']
    print(pid_data_raw, pid_data_raw.shape)

    newPid = pd.DataFrame()
    newPid['pid'] = temp_pid_data_feat['pid']
    print("newPid =", newPid)
    newPid = generate_profile_features(newPid)
    print("newPid =", newPid)
    del newPid['pid']
    print("newPid =", newPid)
    # np, show()

    # 删除无关特征
    pid_data_feat = temp_pid_data_feat
    del pid_data_feat['sid']
    del pid_data_feat['pid']
    del pid_data_feat['req_time']
    del pid_data_feat['o']
    del pid_data_feat['d']
    del pid_data_feat['plan_time']
    del pid_data_feat['plans']
    del pid_data_feat['click_time']
    # 这几项按道理是需要的
    del pid_data_feat['o_lng']
    del pid_data_feat['o_lat']
    del pid_data_feat['d_lng']
    del pid_data_feat['d_lat']
    del pid_data_feat['click_mode']
    del pid_data_feat['city_flag_o']

    print("pid_data_feat =", pid_data_feat, pid_data_feat.shape)
    # 改变特征数值类型
    # d_lng_lat_feature = newPid.values
    d_lng_lat_feature = pid_data_feat.values
    print("d_lng_lat_feature =", d_lng_lat_feature)

    # 节点特征维度
    d_lng_lat_feature_v = d_lng_lat_feature.shape[1]
    print("d_lng_lat_feature =", d_lng_lat_feature, d_lng_lat_feature.shape)

    # -------->可修改部分<---------#

    # 节点特征，特征转化为tensor类型
    o_features = th.FloatTensor(o_lng_lat_feature)
    d_features = th.FloatTensor(d_lng_lat_feature)
    # np,show()
    # 节点数量

    # 节点o的数量
    o_node_number = len_o
    # 节点d的数量
    d_node_number = len_d

    # 节点编号

    # 源节点的标号
    o = od_data['od_ID']
    # 目标节点的标号
    d = pid_data['pid_ID']

    # 初始特征维度

    # 节点o的特征维度
    o_feat_D = o_lng_lat_feature_v
    # 节点d的特征维度
    d_feat_D = d_lng_lat_feature_v

    # 源节点特征,()括号中的数分别为节点数量和特征维度
    # o_feat = th.tensor(np.random.rand(node_number_o, feature_o).astype(np.float32))
    o_feat = o_features
    # 目标节点特征
    # d_feat = th.tensor(np.random.rand(node_number_d, feature_d).astype(np.float32))
    d_feat = d_features

    print("pid_data =", pid_data, min(pid_data['pid']))
    print("od_data =", od_data)

    pid_data = pid_data.merge(od_data, 'left', ['sid'])
    pid_od_count = pid_data.groupby(by=['pid', 'od'])
    newdf = pid_od_count.size()
    pid_od_count = newdf.reset_index(name='pid_od_count')
    pid_od_count = pd.DataFrame(pid_od_count)
    pid_od_count.to_csv(path + who + '/pid_od_count_test.csv')
    print("pid_data =", pid_od_count, max(pid_od_count['pid_od_count']), min(pid_od_count['pid_od_count']))

    pid_data = pid_data.merge(pid_od_count, 'left', ['pid', 'od'])
    # 按pid删除重复行
    # pid_data = pid_data.drop_duplicates(subset=['pid'], keep='first', inplace=False)
    # print("temp_pid_data_feat =", pid_data, pid_data.shape)

    print("pid_data =", pid_data, pid_data.shape)
    pid_data = pid_data.dropna(axis=0, how='any')
    print("pid_data", pid_data, pid_data.shape, max(pid_data['pid_od_count']), min(pid_data['pid_od_count']))
    pid_od_count = pd.DataFrame()
    pid_od_count['pid_od_count'] = pid_data['pid_od_count']
    pid_od_count.to_csv(path + who + '/pid_od_count.csv')
    # np,show()
    # 更新目标节点的特征维度-->与pid节点相同维度
    # -->节点pid更新后的特征--->
    feat_update = d_feat_D
    myattention(o, o_node_number, o_feat, o_feat_D, d, d_node_number, d_feat, d_feat_D, feat_update, 'pid_unique',
                 pid_data_raw, pid_data['pid_od_count'], edge_flag=True)
    # 更新目标节点的特征维度-->与od节点相同维度
    # -->节点od更新后的特征-->相当于保留维度的一半(o和d特征维度之后)(在o和d合并阶段已经扩充过维度了，在这里就不在扩充维度)
    feat_update = o_feat_D
    myattention(d, d_node_number, d_feat, d_feat_D, o, o_node_number, o_feat, o_feat_D, feat_update, 'od_unique',
                od_data_raw, pid_data['pid_od_count'], edge_flag=True)

    # return o, o_node_number, o_feat, o_feat_D, d, d_node_number, d_feat, d_feat_D, feat_update, pid_data_raw, od_data_raw

# 提取sid对&od的特征
def extract_sid_od_feat():
    print("extract_sid_od_feat")
    # sid特征 & od特征
    sid_feat = pd.read_csv(path + who + '/beijing_sid_feat.csv')
    # 删除无关列
    del sid_feat['Unnamed: 0']
    pid_data = pd.read_csv(path + who + '/train_click_pid.csv')
    sid_feat['sid'] = pid_data['sid']
    sid_feat['sid_ID'] = pid_data['sid_ID']

    # sid节点数量
    sid_node_number = len(sid_feat['sid'].unique())
    # sid节点标号
    sid = sid_feat['sid_ID']
    # 按sid删除重复行-->不重复统计节点特征
    temp_sid_data_feat = sid_feat.drop_duplicates(subset=['sid'], keep='first', inplace=False)
    print("temp_sid_data_feat =", temp_sid_data_feat)

    # 原数据
    sid_raw = pd.DataFrame()
    sid_raw['sid_unique_od'] = temp_sid_data_feat['sid']

    # 删除标号列，无特征属性
    del temp_sid_data_feat['sid']
    del temp_sid_data_feat['sid_ID']
    # sid节点特征th.FloatTensor(o_lng_lat_feature)
    sid_feature = th.FloatTensor(temp_sid_data_feat.values)
    # sid节点特征维度
    sid_feature_D = temp_sid_data_feat.shape[1]

    # -->以上为sid节点：节点数量，节点序号，节点特征，节点特征维度

    # od相关数据处理
    od_data = pd.read_csv(path + who + '/train_click_od.csv')
    del od_data['Unnamed: 0']
    # -->od特征<----结合了pid的特征
    od_data_feat = pd.read_csv(path + who + '/beijing_od_unique_feature.csv')
    # 向od特征数据中添加索引od
    od_data_feat['od'] = od_data_feat['od_unique']
    # 删除其中的无用特征
    del od_data_feat['od_unique']
    # 数据拼接
    od_data = od_data.merge(od_data_feat, 'left', ['od'])
    # 向数据->添加o_ID
    # 节点od的长度-->节点od的数量
    od_node_number = len(od_data['od'].unique())

    # od节点的标号
    od = od_data['od_ID']

    # 按'od'列删除重复的项，并保留第一次出现的项，https://www.cnblogs.com/zlc364624/p/12293666.html
    temp_od_data_feat = od_data.drop_duplicates(subset=['od'], keep='first', inplace=False)

    # 拆分od-->为了获得单独的特征
    o_d_lng_lat = pd.DataFrame()
    o_d_lng_lat['o_lng'] = temp_od_data_feat['o'].apply(lambda x: float(x.split(',')[0])).astype(np.float)
    o_d_lng_lat['o_lat'] = temp_od_data_feat['o'].apply(lambda x: float(x.split(',')[1])).astype(np.float)
    o_d_lng_lat['d_lng'] = temp_od_data_feat['d'].apply(lambda x: float(x.split(',')[0])).astype(np.float)
    o_d_lng_lat['d_lat'] = temp_od_data_feat['d'].apply(lambda x: float(x.split(',')[1])).astype(np.float)

    # 利用onehot编码增强效果-->od原始特征
    enc = OneHotEncoder(sparse=False)
    # o_lng_lat_feature = enc.fit_transform(o_d_lng_lat[['o_lng', 'o_lat']])
    # d_lng_lat_feature = enc.fit_transform(o_d_lng_lat[['d_lng', 'd_lat']])
    # 最大最小归一化
    o_lng_lat_feature = min_max_scaler.fit_transform(o_d_lng_lat[['o_lng', 'o_lat']])
    d_lng_lat_feature = min_max_scaler.fit_transform(o_d_lng_lat[['d_lng', 'd_lat']])

    print("o_lng_lat_feature =", o_lng_lat_feature, o_lng_lat_feature.shape)
    print("d_lng_lat_feature =", d_lng_lat_feature, d_lng_lat_feature.shape)

    # 原数据
    od_raw = pd.DataFrame()
    od_raw['od_unique'] = temp_od_data_feat['od']
    od_data = temp_od_data_feat
    # od特征
    '''
    od_data_feat = temp_od_data_feat.loc[:, ['od_unique0', 'od_unique1', 'od_unique2', 'od_unique3', 'od_unique4', 'od_unique5', 'od_unique6', 'od_unique7', 'od_unique8', 'od_unique9']]
    print("od_data_feat =", od_data_feat, od_data_feat.shape)
    '''
    # 删除无关特征
    # sid,o,d,click_mode,od,od_encode,manhattan,sid_ID,od_encode_ID,od_ID,
    del od_data['sid']
    del od_data['o']
    del od_data['d']
    del od_data['click_mode']
    del od_data['od']
    del od_data['od_encode']
    del od_data['manhattan']
    del od_data['sid_ID']
    del od_data['od_encode_ID']
    del od_data['od_ID']

    # 拼接原始特征
    # od_data = np.concatenate((od_data, o_lng_lat_feature), axis=1)
    # od_data = np.concatenate((od_data, d_lng_lat_feature), axis=1)
    print("od_data_feat =", od_data, od_data.shape)

    # od节点特征
    od_feature = th.FloatTensor(od_data.values)
    print("od_feature =", od_feature)
    # np,show()
    # od特征维度
    od_feature_D = od_feature.shape[1]
    # 更新目标节点的特征维度--->维度太大，可用小一点的维度代替,比如源节点的维度
    feat_update = od_feature_D
    # feat_update = 134  # -->pid-->sid也是134维，维度低一点，产生结果快一点
    dst_data_feat = myattention(od, od_node_number, od_feature, od_feature_D, sid, sid_node_number, sid_feature, sid_feature_D, feat_update, 'sid_unique_od',
                sid_raw, None, edge_flag=False)
    return dst_data_feat

# 提取sid对&pid的特征
def extract_sid_pid_feat():
    print("extract_sid_pid_feat")
    # sid特征 & od特征
    # beijing_sid_unique_feature--->操作的文件可修改，单独效果如何？融合之后又如何？
    # --> 是utils.py合并的特征，不带sid
    sid_feat = pd.read_csv(path + who + '/beijing_sid_feat.csv')
    # 删除无关列
    del sid_feat['Unnamed: 0']
    print("sid_feat =", sid_feat, sid_feat.shape)
    # 给以上数据添加sid和sid_ID
    pid_data = pd.read_csv(path + who + '/train_click_pid.csv')
    sid_feat['sid'] = pid_data['sid']
    sid_feat['sid_ID'] = pid_data['sid_ID']

    # sid节点数量
    sid_node_number = len(sid_feat['sid'].unique())
    # sid节点标号
    sid = sid_feat['sid_ID']
    # 按sid删除重复行-->不重复统计节点特征
    temp_sid_data_feat = sid_feat.drop_duplicates(subset=['sid'], keep='first', inplace=False)
    print("temp_pid_data_feat =", temp_sid_data_feat)
    # 原数据
    sid_raw = pd.DataFrame()
    sid_raw['sid_unique_pid'] = temp_sid_data_feat['sid']

    # 删除标号列，无特征属性
    del temp_sid_data_feat['sid']
    del temp_sid_data_feat['sid_ID']
    # sid节点特征th.FloatTensor(o_lng_lat_feature)
    sid_feature = th.FloatTensor(temp_sid_data_feat.values)
    # sid节点特征维度
    sid_feature_D = temp_sid_data_feat.shape[1]

    # -->以上为sid节点：节点数量，节点序号，节点特征，节点特征维度

    # -->pid数据处理
    pid_data = pd.read_csv(path + who + '/train_click_pid.csv')
    print("pid_data =", pid_data, pid_data.shape)
    del pid_data['Unnamed: 0']
    # pid特征<-----融合了od的特征
    pid_data_feat = pd.read_csv(path + who + '/beijing_pid_unique_feature.csv')
    del pid_data_feat['Unnamed: 0']
    print("pid_data_feat =", pid_data_feat, pid_data_feat.shape)
    # 想pid特征数据中添加索引pid
    pid_data_feat['pid'] = pid_data_feat['pid_unique']
    # 删除其中的无用特征
    del pid_data_feat['pid_unique']
    # 数据拼接
    pid_data = pid_data.merge(pid_data_feat, 'left', ['pid'])
    # 节点pid的长度-->节点pid的数量
    pid_node_number = len(pid_data['pid'].unique())
    # 目标节点的标号
    pid = pid_data['pid_ID']

    # 按pid删除重复行
    temp_pid_data_feat = pid_data.drop_duplicates(subset=['pid'], keep='first', inplace=False)
    print("temp_pid_data_feat =", temp_pid_data_feat)
    # 原始数据, 可能后期索引会用到
    pid_raw = pd.DataFrame()
    pid_raw['pid_unique'] = temp_pid_data_feat['pid']
    pid_data_feat = temp_pid_data_feat
    '''
    pid_data_feat = temp_pid_data_feat.loc[:,
                   ['pid_unique0', 'pid_unique1', 'pid_unique2', 'pid_unique3', 'pid_unique4', 'pid_unique5', 'pid_unique6',
                    'pid_unique7', 'pid_unique8', 'pid_unique9']]
    '''
    # 删除无关特征
    # sid,pid,click_mode,sid_ID,pid_ID
    del pid_data_feat['sid']
    del pid_data_feat['pid']
    del pid_data_feat['click_mode']
    del pid_data_feat['sid_ID']
    del pid_data_feat['pid_ID']

    # 其中包含了pid原始特征
    pid_data_feat_origin = pd.read_csv(path + who + '/beijing_nonoise.csv')
    # 按pid删除重复行
    temp_pid_data_feat_origin = pid_data_feat_origin.drop_duplicates(subset=['pid'], keep='first', inplace=False)
    # 删除无关特征
    pid_data_feat_origin = temp_pid_data_feat_origin
    del pid_data_feat_origin['sid']
    del pid_data_feat_origin['pid']
    del pid_data_feat_origin['req_time']
    del pid_data_feat_origin['o']
    del pid_data_feat_origin['d']
    del pid_data_feat_origin['plan_time']
    del pid_data_feat_origin['plans']
    del pid_data_feat_origin['click_time']
    # 这几项按道理是需要的
    del pid_data_feat_origin['o_lng']
    del pid_data_feat_origin['o_lat']
    del pid_data_feat_origin['d_lng']
    del pid_data_feat_origin['d_lat']
    del pid_data_feat_origin['click_mode']
    del pid_data_feat_origin['city_flag_o']
    print("pid_data_feat_origin =", pid_data_feat_origin, pid_data_feat_origin.shape)

    # 合并学习到的pid特征和原始特征
    # pid_data_feat = np.concatenate((pid_data_feat, pid_data_feat_origin), axis=1)

    # pid节点特征
    pid_feature = th.FloatTensor(pid_data_feat.values)
    #  pid节点特征维度
    pid_feature_D = pid_data_feat.shape[1]

    # 更新目标节点的特征维度-->因为sid维度太大了，可以考虑用源节点特征维度
    feat_update = pid_feature_D
    dst_data_feat = myattention(pid, pid_node_number, pid_feature, pid_feature_D, sid, sid_node_number, sid_feature, sid_feature_D, feat_update, 'sid_unique_pid',
                sid_raw, None, edge_flag=False)
    return dst_data_feat

# 提取sid对&od(不受pid影响)的特征
def extract_sid_od_without_pid_feat():
    print("extract_sid_od_without_pid_feat")
    # sid特征 & od特征
    sid_feat = pd.read_csv(path + who + '/beijing_sid_feat.csv')
    # 删除无关列
    del sid_feat['Unnamed: 0']
    pid_data = pd.read_csv(path + who + '/train_click_pid.csv')
    sid_feat['sid'] = pid_data['sid']
    sid_feat['sid_ID'] = pid_data['sid_ID']

    # sid节点数量
    sid_node_number = len(sid_feat['sid'].unique())
    # sid节点标号
    sid = sid_feat['sid_ID']
    # 按sid删除重复行-->不重复统计节点特征
    temp_sid_data_feat = sid_feat.drop_duplicates(subset=['sid'], keep='first', inplace=False)
    print("temp_sid_data_feat =", temp_sid_data_feat)

    # 原数据
    sid_raw = pd.DataFrame()
    sid_raw['sid_unique_od_without_pid'] = temp_sid_data_feat['sid']

    # 删除标号列，无特征属性
    del temp_sid_data_feat['sid']
    del temp_sid_data_feat['sid_ID']
    # sid节点特征th.FloatTensor(o_lng_lat_feature)
    sid_feature = th.FloatTensor(temp_sid_data_feat.values)
    # sid节点特征维度
    sid_feature_D = temp_sid_data_feat.shape[1]

    # -->以上为sid节点：节点数量，节点序号，节点特征，节点特征维度

    # od相关数据处理
    od_data = pd.read_csv(path + who + '/train_click_od_new.csv')
    del od_data['Unnamed: 0']
    # 向数据->添加o_ID
    # 节点od的长度-->节点od的数量
    od_node_number = len(od_data['od'].unique())

    # od节点的标号
    od = od_data['od_ID']

    # 按'od'列删除重复的项，并保留第一次出现的项，https://www.cnblogs.com/zlc364624/p/12293666.html
    temp_od_data_feat = od_data.drop_duplicates(subset=['od'], keep='first', inplace=False)

    # 拆分od-->为了获得单独的特征
    o_d_lng_lat = pd.DataFrame()
    o_d_lng_lat['o_lng'] = temp_od_data_feat['o'].apply(lambda x: float(x.split(',')[0])).astype(np.float)
    o_d_lng_lat['o_lat'] = temp_od_data_feat['o'].apply(lambda x: float(x.split(',')[1])).astype(np.float)
    o_d_lng_lat['d_lng'] = temp_od_data_feat['d'].apply(lambda x: float(x.split(',')[0])).astype(np.float)
    o_d_lng_lat['d_lat'] = temp_od_data_feat['d'].apply(lambda x: float(x.split(',')[1])).astype(np.float)

    # 利用onehot编码增强效果-->od原始特征
    enc = OneHotEncoder(sparse=False)
    # o_lng_lat_feature = enc.fit_transform(o_d_lng_lat[['o_lng', 'o_lat']])
    # d_lng_lat_feature = enc.fit_transform(o_d_lng_lat[['d_lng', 'd_lat']])
    # 最大最小归一化
    o_lng_lat_feature = min_max_scaler.fit_transform(o_d_lng_lat[['o_lng', 'o_lat']])
    d_lng_lat_feature = min_max_scaler.fit_transform(o_d_lng_lat[['d_lng', 'd_lat']])

    print("o_lng_lat_feature =", o_lng_lat_feature, o_lng_lat_feature.shape)
    print("d_lng_lat_feature =", d_lng_lat_feature, d_lng_lat_feature.shape)

    # 原数据
    od_raw = pd.DataFrame()
    od_raw['od_unique'] = temp_od_data_feat['od']
    od_data = temp_od_data_feat
    # od特征
    '''
    od_data_feat = temp_od_data_feat.loc[:, ['od_unique0', 'od_unique1', 'od_unique2', 'od_unique3', 'od_unique4', 'od_unique5', 'od_unique6', 'od_unique7', 'od_unique8', 'od_unique9']]
    print("od_data_feat =", od_data_feat, od_data_feat.shape)
    '''
    # 删除无关特征
    # sid,o,d,click_mode,od,od_encode,manhattan,sid_ID,od_encode_ID,od_ID,
    del od_data['sid']
    del od_data['o']
    del od_data['d']
    del od_data['click_mode']
    del od_data['od']
    del od_data['od_encode']
    del od_data['manhattan']
    del od_data['sid_ID']
    del od_data['od_encode_ID']
    del od_data['od_ID']

    # 拼接原始特征
    od_data = np.concatenate((od_data, o_lng_lat_feature), axis=1)
    od_data = np.concatenate((od_data, d_lng_lat_feature), axis=1)
    print("od_data_feat =", od_data, od_data.shape)

    # od节点特征
    od_feature = th.FloatTensor(od_data)
    print("od_feature =", od_feature)
    # np,show()
    # od特征维度
    od_feature_D = od_feature.shape[1]
    # 更新目标节点的特征维度--->维度太大，可用小一点的维度代替,比如源节点的维度
    feat_update = od_feature_D
    # feat_update = 134  # -->pid-->sid也是134维，维度低一点，产生结果快一点
    dst_data_feat = myattention(od, od_node_number, od_feature, od_feature_D, sid, sid_node_number, sid_feature, sid_feature_D, feat_update, 'sid_unique_od_without_pid',
                sid_raw, None, edge_flag=False)
    return dst_data_feat

# 提取不带od对特征的pid对sid对的特征的影响
def extract_sid_pid_without_od_feat():
    print("extract_sid_pid_without_od_feat")
    # sid特征 & od特征
    # beijing_sid_unique_feature--->操作的文件可修改，单独效果如何？融合之后又如何？
    # --> 是utils.py合并的特征，不带sid
    sid_feat = pd.read_csv(path + who + '/beijing_sid_feat.csv')
    # 删除无关列
    del sid_feat['Unnamed: 0']
    print("sid_feat =", sid_feat, sid_feat.shape)
    # 给以上数据添加sid和sid_ID
    pid_data = pd.read_csv(path + who + '/train_click_pid.csv')
    sid_feat['sid'] = pid_data['sid']
    sid_feat['sid_ID'] = pid_data['sid_ID']

    # sid节点数量
    sid_node_number = len(sid_feat['sid'].unique())
    # sid节点标号
    sid = sid_feat['sid_ID']
    # 按sid删除重复行-->不重复统计节点特征
    temp_sid_data_feat = sid_feat.drop_duplicates(subset=['sid'], keep='first', inplace=False)
    print("temp_pid_data_feat =", temp_sid_data_feat)
    # 原数据
    sid_raw = pd.DataFrame()
    sid_raw['sid_unique_pid_without_od'] = temp_sid_data_feat['sid']

    # 删除标号列，无特征属性
    del temp_sid_data_feat['sid']
    del temp_sid_data_feat['sid_ID']
    # sid节点特征th.FloatTensor(o_lng_lat_feature)
    sid_feature = th.FloatTensor(temp_sid_data_feat.values)
    # sid节点特征维度
    sid_feature_D = temp_sid_data_feat.shape[1]

    # -->以上为sid节点：节点数量，节点序号，节点特征，节点特征维度

    # -->pid数据处理
    pid_data = pd.read_csv(path + who + '/train_click_pid.csv')
    print("pid_data =", pid_data, pid_data.shape)
    del pid_data['Unnamed: 0']
    '''
    del pid_data_feat['Unnamed: 0']
    print("pid_data_feat =", pid_data_feat, pid_data_feat.shape)
    # 想pid特征数据中添加索引pid
    pid_data_feat['pid'] = pid_data_feat['pid_unique']
    # 删除其中的无用特征
    del pid_data_feat['pid_unique']
    # 数据拼接
    pid_data = pid_data.merge(pid_data_feat, 'left', ['pid'])
    '''
    # 节点pid的长度-->节点pid的数量
    pid_node_number = len(pid_data['pid'].unique())
    # 目标节点的标号
    pid = pid_data['pid_ID']

    # 按pid删除重复行
    temp_pid_data_feat = pid_data.drop_duplicates(subset=['pid'], keep='first', inplace=False)
    print("temp_pid_data_feat =", temp_pid_data_feat)
    # 原始数据, 可能后期索引会用到
    pid_raw = pd.DataFrame()
    pid_raw['pid_unique'] = temp_pid_data_feat['pid']
    pid_data_feat = temp_pid_data_feat
    '''
    pid_data_feat = temp_pid_data_feat.loc[:,
                   ['pid_unique0', 'pid_unique1', 'pid_unique2', 'pid_unique3', 'pid_unique4', 'pid_unique5', 'pid_unique6',
                    'pid_unique7', 'pid_unique8', 'pid_unique9']]
    '''

    # 删除无关特征
    # sid,pid,click_mode,sid_ID,pid_ID
    del pid_data_feat['sid']
    del pid_data_feat['pid']
    del pid_data_feat['click_mode']
    del pid_data_feat['sid_ID']
    del pid_data_feat['pid_ID']

    # 其中包含了pid原始特征
    pid_data_feat_origin = pd.read_csv(path + who + '/beijing_nonoise.csv')
    # 按pid删除重复行
    temp_pid_data_feat_origin = pid_data_feat_origin.drop_duplicates(subset=['pid'], keep='first', inplace=False)
    # 删除无关特征
    pid_data_feat_origin = temp_pid_data_feat_origin
    del pid_data_feat_origin['sid']
    del pid_data_feat_origin['pid']
    del pid_data_feat_origin['req_time']
    del pid_data_feat_origin['o']
    del pid_data_feat_origin['d']
    del pid_data_feat_origin['plan_time']
    del pid_data_feat_origin['plans']
    del pid_data_feat_origin['click_time']
    # 这几项按道理是需要的
    del pid_data_feat_origin['o_lng']
    del pid_data_feat_origin['o_lat']
    del pid_data_feat_origin['d_lng']
    del pid_data_feat_origin['d_lat']
    del pid_data_feat_origin['click_mode']
    del pid_data_feat_origin['city_flag_o']
    print("pid_data_feat_origin =", pid_data_feat_origin, pid_data_feat_origin.shape)

    # 合并学习到的pid特征和原始特征
    pid_data_feat = np.concatenate((pid_data_feat, pid_data_feat_origin), axis=1)

    # pid节点特征
    pid_feature = th.FloatTensor(pid_data_feat)
    #  pid节点特征维度
    pid_feature_D = pid_data_feat.shape[1]

    # 更新目标节点的特征维度-->因为sid维度太大了，可以考虑用源节点特征维度
    feat_update = pid_feature_D
    dst_data_feat = myattention(pid, pid_node_number, pid_feature, pid_feature_D, sid, sid_node_number, sid_feature, sid_feature_D, feat_update, 'sid_unique_pid_without_od',
                sid_raw, None, edge_flag=False)
    return dst_data_feat

# log归一化
def nonlinearity_normalization_lg(data_value_after_lg, data_col_max_values_after_lg):
    """ Data normalization using lg
    Args:
        data_value_after_lg: The data to be normalized
        data_col_max_values_after_lg: The maximum value of data's columns
    """
    data_shape = data_value_after_lg.shape
    data_rows = data_shape[0]
    data_cols = data_shape[1]

    for i in xrange(0, data_rows, 1):
        for j in xrange(0, data_cols, 1):
            data_value_after_lg[i][j] = \
                data_value_after_lg[i][j] / data_col_max_values_after_lg[j]

    return data_value_after_lg

# 反切函数归一化
def atan_normalization(data):
    import math
    data = [math.atan(x)*2/math.pi for x in data]
    data = np.array(data)
    return data

# 线性变换-->降维
# 输入：需要降维的tensor，需要输出的维度
# 输出：返回修改维度的tensor
def LinearChange(res, dst_feat_D):
    # print("res =", res, res.shape, dst_feat_D)
    fc = nn.Linear(res.shape[1], dst_feat_D, bias=True)
    Ba = nn.BatchNorm1d(dst_feat_D)
    Dr = nn.Dropout(0.2)

    dst_feat = fc(res)
    # print("dst_feat =", dst_feat, dst_feat.shape)
    dst_feat = Ba(dst_feat)
    # 比relu强多了
    dst_feat = F.leaky_relu(dst_feat)
    # dst_feat = F.softmax(dst_feat, dim=1)
    dst_feat = Dr(dst_feat)
    '''
    dst_feat = F.relu(fc(dst_feat))
    dst_feat = Ba(dst_feat)
    dst_feat = Dr(dst_feat)

    dst_feat = F.relu(fc(dst_feat))
    dst_feat = Ba(dst_feat)
    dst_feat = Dr(dst_feat)
    '''
    return dst_feat

# ----------->修改部分<----------#
# str:源节点，srt_num:源节点数量，srt_feat:源节点特征，srt_feat_D:源节点特征维度
# dst:目标节点，dst_num:目标节点数量，dst_feat:源节点特征，dst_feat_D:源节点特征维度
# dst_feat_update:目标节点更新维度
# dst_name:目标节点名称
def myattention(srt, srt_num, srt_feat, srt_feat_D, dst, dst_num, dst_feat, dst_feat_D, dst_feat_update, dst_name, dst_data_raw, od_count, edge_flag):
    # 需要计算融入边的信息
    print("dst_feat_update =", dst_feat_update)
    print("srt_feat_D =", srt_feat_D)
    print("dst_feat_D =", dst_feat_D)
    print("srt_feat =", srt_feat, srt_feat.shape)
    print("dst_feat =", dst_feat, dst_feat.shape)
    if edge_flag == True:
        print("edge_flag = True")
        # 边出现频率
        # print("od_count =", od_count, od_count.shape)
        od_count = od_count.values
        # print("od_count =", od_count, od_count.shape)
        # 从二维降为一维
        od_count = od_count.reshape(-1, 1)
        # print("2D od_count =", od_count, od_count.shape)

        # data_col_max_values_after_lg = np.log10(od_count.max(axis=0))
        # od_count = nonlinearity_normalization_lg(od_count,
        #                               data_col_max_values_after_lg)
        # od_count = atan_normalization(od_count)
        # 反切函数归一化
        # od_count = atan_normalization(od_count)
        # print("atan od_count =", od_count, max(od_count))
        # 对使用频率进行最大最小归一化-->之后看看谁更好？
        # od_count = min_max_scaler.fit_transform(od_count)
        # od_count = preprocessing.normalize(od_count, norm='l2')
        # --->是否可以考虑在此之前把边的特征融入到各节点上？
        # ---需要为tensor--->这样才能正常运行

        # 旨在更新d的特征
        g_o_d = dgl.heterograph({('srt_type', 'srt_dst_type', 'dst_type'): (srt, dst)})
        # 源节点的特征
        # 源节点特征定义为'ft'
        g_o_d.srcdata.update({'ft': srt_feat})
        # 向'srt_dst_type'中输入权重信息(使用频率)
        g_o_d.edata['srt_dst_type'] = th.FloatTensor(od_count)
        # print("g.edges['srt_dst_type'] =", g_o_d.edges['srt_dst_type'])
        # 将源节点特征与边特征相乘可得目标节点更新后的特征
        g_o_d.update_all(fn.u_mul_e('ft', 'srt_dst_type', 'm'),
                        fn.sum('m', 'ft'))
        # 利用边权重处理过的目的节点特征<----更新部分
        # -->这步操作，其实只给出了目标节点的更新部分，而源节点未发生变化
        res_d = g_o_d.dstdata['ft']
        # 线性变换-->维持之前的维度不变-->降维
        res_d = LinearChange(res_d, dst_feat_D)
        # print("res1 =", res_d, res_d.shape)
        # tensor转为numpy
        res_d_a = res_d.detach().numpy()
        # print("resd_a =", res_d_a, res_d_a.shape)
        # res_d_a = min_max_scaler.fit_transform(res_d_a)
        res_d_a = preprocessing.normalize(res_d_a, norm='l2')
        # res_d_a = atan_normalization(res_d_a)
        print("res_d_a =", res_d_a, res_d_a.shape, min(res_d_a[0]), min(res_d_a[1]), max(res_d_a[0]), max(res_d_a[1]))

        res_d = res_d_a
        # 拼接目标节点原始特征--->但似乎效果变差了？-->只能用更新变量做注意力？
        # res_d = np.concatenate((res_d, dst_feat), axis=1)
        print("dst_feat =", dst_feat, dst_feat.shape)
        # np,show()
        # 节点特征维度更新-->进来是什么维度，出去什么维度
        dst_feat_D_new = res_d.shape[1]
        # 输出维度特征更新
        # dst_feat_update_new = dst_feat_D_new
        # 更新d节点的特征-->归一化后，重新转换为tensor类型
        dst_feat_new = th.FloatTensor(res_d)
        # print("dst_feat =", dst_feat, dst_feat.shape)

        # 旨在更新o的特征
        g_d_o = dgl.heterograph({('srt_type', 'srt_dst_type', 'dst_type'): (dst, srt)})
        # 源节点特征定义为'ft'--->这次的源节点是d节点
        g_d_o.srcdata.update({'ft': dst_feat})
        # 向'srt_dst_type'中输入权重信息(使用频率)
        g_d_o.edata['srt_dst_type'] = th.FloatTensor(od_count)
        # print("g.edges['srt_dst_type'] =", g_d_o.edges['srt_dst_type'])
        # 将源节点特征与边特征相乘可得目标节点更新后的特征
        g_d_o.update_all(fn.u_mul_e('ft', 'srt_dst_type', 'm'),
                         fn.sum('m', 'ft'))
        # 利用边权重处理过的目的节点特征<----更新部分
        # -->这步操作，其实只给出了目标节点的更新部分，而源节点未发生变化
        res_o = g_d_o.dstdata['ft']
        # 线性变换-->维持之前的维度不变-->降维
        res_o = LinearChange(res_o, srt_feat_D)
        # print("res1 =", res_o, res_o.shape)
        # tensor转为numpy
        res_o_a = res_o.detach().numpy()
        # print("res1_a =", res_o_a, res_o_a.shape)
        # res_o_a = min_max_scaler.fit_transform(res_o_a)
        res_o_a = preprocessing.normalize(res_o_a, norm='l2')
        # res_o_a = atan_normalization(res_o_a)
        # print("res_o_a =", res_o_a, res_o_a.shape, min(res_o_a[0]), min(res_o_a[1]), max(res_o_a[0]), max(res_o_a[1]))
        
        res_o = res_o_a
        # 拼接源节点原始特征，
        # res_o = np.concatenate((res_o, srt_feat), axis=1)

        # 节点维度特征更新
        srt_feat_D_new = res_o.shape[1]
        
        # 更新o节点的特征-->归一化后，重新转换为tensor类型
        srt_feat_new = th.FloatTensor(res_o)
        # print("srt_feat =", srt_feat, srt_feat.shape)

        # ----->中途不要随意改变原始参数，要放到最后赋值<------#
        dst_feat = dst_feat_new
        srt_feat = srt_feat_new
        dst_feat_D = dst_feat_D_new
        # 个人觉得：进来什么维度，出去什么维度
        # dst_feat_update = dst_feat_update_new
        srt_feat_D = srt_feat_D_new

    print("dst_feat_update =", dst_feat_update)
    print("srt_feat_D =", srt_feat_D)
    print("dst_feat_D =", dst_feat_D)
    print("srt_feat =", srt_feat, srt_feat.shape)
    print("dst_feat =", dst_feat, dst_feat.shape)

    # 构建二部图-->重新利用图注意力学习节点特征
    g = dgl.heterograph({('srt_type', 'srt_dst_type', 'dst_type'): (srt, dst)})
    print("g =", g)
    g0 = dgl.heterograph({('srt_type', 'srt_dst_type', 'dst_type'): (dst, srt)})

    # 调用图注意力网络-->(a, b)括号中第一个数应该为源节点特征维度，第二个数应该为目标节点特征维度,紧接着为输出特征的维度，5和1表示输出1x5的特征
    # 输出特征维度可变, 其中的1表示每个节点更新后的特征只有1行，如果使用多头注意力机制可用：num_heads=3
    # srt_feat_D-->源节点特征维度，dst_feat_D-->目标节点特征维度，dst_feat_update-->目标节点更新后的维度-->1头注意力
    # 1表示1头注意力机制，可以考虑使用多头注意力机制，--->多头注意力机制对性能起到均衡作用
    gatconv = GATConv((srt_feat_D, dst_feat_D), dst_feat_update, 8, 0.6, 0.6, activation=F.elu)  # -->构建网络
    gatconv0 = GATConv((dst_feat_D, srt_feat_D), srt_feat_D, 8, 0.6, 0.6, activation=F.elu)  # -->构建网络
    # gatconv = GraphConv(srt_feat_D, dst_feat_D, norm='both', weight=True, bias=True)
    # res = conv(g, (u_fea, v_fea))
    # g为二部图，(u_feat, v_feat)分别为源节点特征和目标节点特征，res为目标节点特征更新值
    # -->相当于这里使用forward函数-->继承关系
    # srt_feat-->源节点特征，dst_feat-->目标节点特征，res-->目标节点更新后的特征
    print("srt_feat =", srt_feat, srt_feat.shape)
    print("dst_feat =", dst_feat, dst_feat.shape)
    # -->一层图注意力机制
    res = gatconv(g, (srt_feat, dst_feat))
    print("3D res =", res, res.shape)
    # 只有获取od或pid到sid的信息，用一层图注意力机制应该就够了
    # if edge_flag == True:
    # print("edge_flag = True")
    res0 = gatconv0(g0, (dst_feat, srt_feat))

    res = res.mean(axis=1, keepdim=False)  # 均值后变为二维
    res0 = res0.mean(axis=1, keepdim=False)  # 均值后变为二维
    print("2D res =", res, res.shape)
    # -->线性变换-->降维-->降到目标维度
    dst_feat = LinearChange(res, dst_feat_D)
    srt_feat = LinearChange(res0, srt_feat_D)
    print("dst_feat =", dst_feat, dst_feat.shape)

    # -->两层图注意力机制
    res = gatconv(g, (srt_feat, dst_feat))
    print("3D res =", res, res.shape)
    '''
    res0 = gatconv0(g0, (dst_feat, srt_feat))
    res = res.mean(axis=1, keepdim=False)  # 均值后变为二维
    res0 = res0.mean(axis=1, keepdim=False)  # 均值后变为二维
    print("2D res =", res, res.shape)
    # -->线性变换-->降维-->降到目标维度
    dst_feat = LinearChange(res, dst_feat_D)
    srt_feat = LinearChange(res0, srt_feat_D)
    print("dst_feat =", dst_feat, dst_feat.shape)

    # -->三层图注意力机制
    res = gatconv(g, (srt_feat, dst_feat))
    print("3D res =", res, res.shape)
    '''
    # res = res.flatten(1)  # 对结果进行平铺，多头结果进行平铺
    # print("flatten res =", res, res.shape)
    # 三维tensor,转二维Tensor,https://blog.csdn.net/tailonh/article/details/105524789
    # tensor降维（从三维转换为二维），对1头注意力机制才有用，对多头，似乎用处不大
    # res = th.reshape(res, (-1, dst_feat_update))
    # print("2-D res =", res, res.shape)
    # 对多头注意力机制所得的值求均值:https://blog.csdn.net/qq_36810398/article/details/104845401
    # 其中的第[2,4]平均值后直接为二维的tensor，非常适合用于多头注意力机制，牛批！！
    res = res.mean(axis=1, keepdim=False)
    print("2D mean res =", res, res.shape)
    # tensor转为numpy
    res_a = res.detach().numpy()
    #     # if edge_flag == True:
        # 目标节点特征合并形式
        # res_a = np.concatenate((res_a, res_d_a), axis=1)
    # 将numpy转换为pd.DataFrame
    res_d = pd.DataFrame(res_a)
    # 将数据转换为DataFrame型
    # dst_feat_data = pd.DataFrame()
    # 删除为nan，即值为0的行
    print("dst_data_raw =", dst_data_raw, dst_data_raw.shape)
    dst_data_raw = dst_data_raw.dropna(axis=0, how='any')
    print("dst_data_raw =", dst_data_raw, dst_data_raw.shape)
    print("dst_name =", dst_name)
    # --->记得是用.values，而不是用pd.DataFrame
    res_d[dst_name] = dst_data_raw.values
    # res_d = res_d.dropna(axis=0, how='any')
    print("res_d =", res_d, res_d.shape)
    # np,show()
    # res_d.to_csv(path + who + '/beijing_' + dst_name + '_test.csv')
    # ------->将所得特征存到.csv文件中<--------#
    # 保存目标节点的特征到文件中
    filepath = path + who + '/beijing_' + dst_name + '_feature.csv'
    # 可存可不存，存的话，可一劳永逸，不存的话，每次都得训练？-->存起来好一点
    res_d.to_csv(filepath)
    # np,show()
    print("myattention over, return your data")
    # 返回 pd.DataFrame格式数据
    return res_d


def main():
    # 分别学习o和d节点的并合并为od的特征-->只包含更新部分，原始特征用坐标即可添加
    extract_od_feat()

    # 分别学习pid和od节点的特征-->只包含更新部分
    # extract_user_od_feat()

    # 通过od学习sid的特征
    # extract_sid_od_feat()
    # 通过pid学习sid的特征
    # extract_sid_pid_feat()

    # 按道理下面这两项用处不是很大

    # 不受pid影响od对sid学习
    # extract_sid_od_without_pid_feat()
    # 不受od影响的pid对sid的学习
    # extract_sid_pid_without_od_feat()

    # 添加边的权重：https://docs.dgl.ai/en/0.6.x/guide_cn/graph-feature.html
    # https://docs.dgl.ai/en/0.6.x/guide_cn/message-edge.html
    pass

if __name__ == '__main__':
    main()