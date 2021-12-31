import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dgl
from dgl.nn.pytorch import GATConv, GraphConv, SAGEConv
from dgl.nn.pytorch import HeteroGraphConv
import pandas as pd
import dgl.function as fn
# change your computer’s path
path = '../p38dglproject/dataset/output/'
who = 'beijing'
hid_dim = 8
# 线性变换-->降维
# 输入：需要降维的tensor，需要输出的维度
# 输出：返回修改维度的tensor
def LinearChange(res, dst_feat_D):
    fc = nn.Linear(res.shape[1], dst_feat_D, bias=True)
    Ba = nn.BatchNorm1d(dst_feat_D)
    Dr = nn.Dropout(0.2)

    dst_feat = fc(res)
    dst_feat = Ba(dst_feat)
    dst_feat = F.leaky_relu(dst_feat)
    dst_feat = Dr(dst_feat)
    return dst_feat

# -->语义级注意力
class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()
        # -->映射有点像单层MLP-->为了计算权重w
        self.project = nn.Sequential(
            # z
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            # q
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        # -->z来自语义嵌入semantic_embeddings
        # -->映射公式(7)-->求每条mata-path的权重
        w = self.project(z).mean(0)                    # (M, 1)
        # -->归一化操作公式(8)
        beta = torch.softmax(w, dim=0)                 # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape) # (N, M, 1)
        # -->语义级中的公式(9)
        return (beta * z).sum(1)                       # (N, D * K)

# --异构节点类型级注意力
class NodetypeAttention(nn.Module):
    def __init__(self, in_size, hidden_size=64):
        super(NodetypeAttention, self).__init__()
        # -->映射有点像单层MLP-->为了计算权重w
        self.project = nn.Sequential(
            # z
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            # q
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        # -->z来自语义嵌入semantic_embeddings
        # -->映射公式(7)-->求每条mata-path的权重
        w = self.project(z).mean(0)                    # (M, 1)
        # -->归一化操作公式(8)
        beta = torch.softmax(w, dim=0)                 # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape) # (N, M, 1)
        # -->语义级中的公式(9)
        return (beta * z).sum(1)                       # (N, D * K)

class HANLayer(nn.Module):
    def __init__(self, meta_paths, in_size, out_size, layer_num_heads, dropout):
        super(HANLayer, self).__init__()
        # One GAT layer for each meta path based adjacency matrix
        # --> 节点级注意力？
        self.gat_layers = nn.ModuleList()
        for i in range(len(meta_paths)):
            # HAN基于元路径的操作使用gat
            self.gat_layers.append(GATConv(in_size, out_size, layer_num_heads,
                                           dropout, dropout, residual=False, activation=F.elu,
                                           allow_zero_in_degree=True))

        # --> 语义级注意力？
        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)
        self.meta_paths = list(tuple(meta_path) for meta_path in meta_paths)
        self._cached_graph = None
        self._cached_coalesced_graph = {}


    def forward(self, g, h):
        # -->语义嵌入-->节点级注意力
        semantic_embeddings = []
        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for meta_path in self.meta_paths:
                self._cached_coalesced_graph[meta_path] = dgl.metapath_reachable_graph(
                        g, meta_path)

        for i, meta_path in enumerate(self.meta_paths):
            new_g = self._cached_coalesced_graph[meta_path]
            semantic_embeddings.append(self.gat_layers[i](new_g, h).flatten(1))
        # --> 语义嵌入
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)                  # (N, M, D * K)
        # --> 语义嵌入：semantic_embeddings-->用于语义注意力的输入
        # -->语义级注意力
        return self.semantic_attention(semantic_embeddings)                            # (N, D * K)

class THANLayer(nn.Module):
    def __init__(self, in_size, pid_size, od_size, out_size, layer_num_heads, dropout):
        super(THANLayer, self).__init__()
        # pid_gat
        # 二部图（源节点，目标节点），输出特征
        self.pid_gat = GATConv((pid_size, in_size), out_size, layer_num_heads,
                               dropout, dropout, residual=True, activation=F.leaky_relu,
                               allow_zero_in_degree=True)
        # gat
        self.od_gat = GATConv((od_size, in_size), out_size, layer_num_heads,
                              dropout, dropout, residual=True, activation=F.leaky_relu,
                              allow_zero_in_degree=True)

        # --> 语义级注意力？
        self.nodetype_attention = NodetypeAttention(in_size=out_size * layer_num_heads)

    def forward(self, temp_h, pid_g, pid_h, od_g, od_h):
        # -->节点类型嵌入-->节点级注意力
        nodetype_embeddings = []
        nodetype_embeddings.append(self.pid_gat(pid_g, (pid_h, temp_h)).flatten(1))
        nodetype_embeddings.append(self.od_gat(od_g, (od_h, temp_h)).flatten(1))
        # --> 语义嵌入
        nodetype_embeddings = torch.stack(nodetype_embeddings, dim=1)                  # (N, M, D * K)
        # --> 语义嵌入：semantic_embeddings-->用于语义注意力的输入
        # -->语义级注意力
        return self.nodetype_attention(nodetype_embeddings)
# --异构节点类型级注意力
class HybridAttention(nn.Module):
    def __init__(self, in_size, hidden_size=64):
        super(HybridAttention, self).__init__()
        # -->映射有点像单层MLP-->为了计算权重w
        self.project = nn.Sequential(
            # z
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            # q
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)                    # (M, 1)
        beta = torch.softmax(w, dim=0)                 # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape) # (N, M, 1)
        return (beta * z).sum(1)                       # (N, D * K)

class THAN(nn.Module):
    def __init__(self, meta_paths, in_size, pid_size, o_size, d_size, od_size, hidden_size, out_size, num_heads, dropout):
        super(THAN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(HANLayer(meta_paths, in_size, hidden_size, num_heads[0], dropout))
        # 多头注意力机制(公式5)，num_heads为次数
        for l in range(1, len(num_heads)):
            self.layers.append(HANLayer(meta_paths, hidden_size * num_heads[l-1],
                                        hidden_size, num_heads[l], dropout))

        # -->数据量x3075维度
        self.predict0 = nn.Linear(hidden_size * num_heads[-1], out_size, bias=True)
        self.Ba0 = nn.BatchNorm1d(out_size)
        self.Dr0 = nn.Dropout(0.2)

        # 异构节点注意力
        # hidden_size = 8, num_heads[0] = num_heads[-1] = 8
        self.nodetype_nn = THANLayer(in_size, pid_size, od_size, hidden_size, num_heads[0], dropout)
        # MLP# 预测<---利用多层吧？<--多层似乎不好

        self.predict1 = nn.Linear(hidden_size * num_heads[-1] * 2, out_size, bias=True)
        self.Ba1 = nn.BatchNorm1d(out_size)
        self.Dr1 = nn.Dropout(0.2)

        # 异构向量，同质向量
        self.Hybrid_attention = HybridAttention(in_size=hidden_size * num_heads[-1])

        self.predict2 = nn.Linear(in_size, out_size, bias=True)

        # 二部图（源节点，目标节点），输出特征-->out_size不能等于本身维度，其他都行
        # 源->目标-->gcn
        self.o_d_gat = GraphConv(o_size, o_size, norm='both', weight=True, bias=True)#, activation=torch.tanh)

        # 源->目标-->gcn
        self.d_o_gat = GraphConv(d_size, d_size, norm='both', weight=True, bias=True)#, activation=torch.tanh)

        # od_pid_gat
        # 二部图（源节点，目标节点），输出特征-->out_size不能等于本身维度，其他都行
        self.od_pid_gat = GATConv((od_size, pid_size), od_size, num_heads[0],
                                           dropout, dropout, residual=True, activation=F.leaky_relu,
                                           allow_zero_in_degree=True)

        # pid_od_gat
        self.pid_od_gat = GATConv((pid_size, od_size), pid_size, num_heads[0],
                               dropout, dropout, residual=True, activation=F.leaky_relu,
                               allow_zero_in_degree=True)

        # 恢复o维度
        self.recover_o_D = nn.Linear(d_size, o_size, bias=True)
        # 恢复d维度
        self.recover_d_D = nn.Linear(o_size, d_size, bias=True)
        # 恢复pid维度
        self.recover_pid_D = nn.Linear(od_size, pid_size, bias=True)
        # 恢复od维度
        self.recover_od_D = nn.Linear(pid_size, od_size, bias=True)

        # 将o+d的维度变成od维度
        self.recover_o_d_to_od_D = nn.Linear((o_size + d_size), od_size, bias=True)
        hidden = 128
        # 恢复pid维度-->in_size
        self.recover_pid_D_hetergcn = nn.Linear(pid_size, in_size, bias=True)
        # 恢复od维度-->in_size
        self.recover_od_D_hetergcn = nn.Linear(od_size, in_size, bias=True)
        # https://docs.dgl.ai/en/0.6.x/guide/nn-heterograph.html
        self.heterGcn = HeteroGraphConv({
            # p为源，a为目标节点
            'pa': GraphConv(in_size, hidden),
            'ap': GraphConv(in_size, hidden),
            'pf': GraphConv(in_size, hidden),
            'fp': GraphConv(in_size, hidden)},
            aggregate='sum')
        self.heterGcn1 = HeteroGraphConv({
            # p为源，a为目标节点
            'pa': GraphConv(hidden, hidden_size * num_heads[-1]),
            'ap': GraphConv(hidden, hidden_size * num_heads[-1]),
            'pf': GraphConv(hidden, hidden_size * num_heads[-1]),
            'fp': GraphConv(hidden, hidden_size * num_heads[-1])},
            aggregate='sum')
        self.Dr3 = nn.Dropout(0.6)

    # pid_h为pid特征
    # od_h为od特征
    # 完整的THAN：o+d->od, od与pid互相影响，分层注意力机制
    # THAN: 多二部图嵌入，异构节点：gat，元路径（同质节点）：gat，有残差！！！

    #  --->
    #  （1）分组验证“鲁棒性”，
    #  （2）超参数敏感性：同质和异构向量维度变化，多头注意力的头的数量
    # forward_TAHN
    def forward(self, g, h, pid_h, o_h, d_h, o_d_g, d_o_g, od_h, o_d_od_ID_data, o_d_count):
        # 节点原始特征做好备份
        temp_h = h
        # 同质网络中节点预测-->基于元路径
        for gnn in self.layers:
            # 输入的h(原)每次都一样，满足h(更新) = L*h(原)*w+b
            # 其实是在改变w，使得h(更新)更适合于预测标签
            h = gnn(g, h)

        # ---->上面主要做基于元路径的m预测，下面主要做基于异构节点的m预测

        # -------->重新将o + d -> od将o和d合并为od<--------
        # 更新o节点特征
        res0 = self.o_d_gat(o_d_g, (o_h, d_h))  # , edge_weight=o_d_count)
        # 降维操作-->gcn可不用-->无多头注意力机制
        # 更新d节点特征
        res1 = self.d_o_gat(d_o_g, (d_h, o_h))  # , edge_weight=o_d_count)
        # 降维操作-->gcn可不用-->无多头注意力机制

        # 线性变化：恢复o_h,d_h维度
        o_h = self.recover_o_D(res1)
        d_h = self.recover_d_D(res0)
        o_h = o_h.detach().numpy()
        d_h = d_h.detach().numpy()
        o_h = pd.DataFrame(o_h)
        d_h = pd.DataFrame(d_h)
        o_df = o_h.reset_index()
        d_df = d_h.reset_index()
        o_h['o_ID'] = range(len(o_df))
        d_h['d_ID'] = range(len(d_df))
        # 循环读操作，肯定存在固定耗时，看看如何放出去外面操作
        # o_d_od_ID_data = pd.read_csv((path + who + '/o_d_od_ID_data.csv'))
        o_d_od_ID_data = o_d_od_ID_data.detach().numpy()
        o_d_od_ID_data = pd.DataFrame(o_d_od_ID_data)
        o_d_od_ID_data_temp = pd.DataFrame()
        o_d_od_ID_data_temp['od_ID'] = o_d_od_ID_data[1]
        o_d_od_ID_data_temp['o_ID'] = o_d_od_ID_data[2]
        o_d_od_ID_data_temp['d_ID'] = o_d_od_ID_data[3]
        # 不用从文本中读入，自然会快很多
        o_d_od_ID_data = o_d_od_ID_data_temp
        o_d_od_ID_data = o_d_od_ID_data.merge(o_h, on='o_ID', how='left')
        o_d_od_ID_data = o_d_od_ID_data.merge(d_h, on='d_ID', how='left')
        # 按pid删除重复行
        o_d_od_ID_data = o_d_od_ID_data.drop_duplicates(subset=['od_ID'], keep='first', inplace=False)
        # print("od number =", g.num_nodes('field'))
        # -->需要的是OD的数量
        o_d_od_ID_data = o_d_od_ID_data[: g.num_nodes('field')]
        del o_d_od_ID_data['od_ID']
        del o_d_od_ID_data['o_ID']
        del o_d_od_ID_data['d_ID']

        # o&d特征拼接
        od_h = o_d_od_ID_data.values
        od_h = torch.FloatTensor(od_h)
        # 将o和d维度生成的特征转换为od维度的特征
        od_h = self.recover_o_d_to_od_D(od_h)
        #  -----> o + d -->od   <--------
        # 二部图节点序列
        pid_m = g.edges('all', etype='pa')
        od_m = g.edges('all', etype='pf')
        # 构建od_pid二部图,学习pid节点特征
        od_pid_g = dgl.heterograph({('srt_type', 'srt_dst_type', 'dst_type'): (od_m[1], pid_m[1])})
        # 构建pid_od二部图，学习od节点特征
        pid_od_g = dgl.heterograph({('srt_type', 'srt_dst_type', 'dst_type'): (pid_m[1], od_m[1])})

        # 更新pid节点特征:od-->pid
        res0 = self.od_pid_gat(od_pid_g, (od_h, pid_h))
        # gcn-->无需降维
        res0 = res0.mean(axis=1, keepdim=False)  # 均值后变为二维
        # 更新od节点特征:pid-->od
        res1 = self.pid_od_gat(pid_od_g, (pid_h, od_h))
        # gcn-->无需降维
        res1 = res1.mean(axis=1, keepdim=False)  # 均值后变为二维

        # 线性变化：恢复pid_h,od_h维度
        pid_h = self.recover_pid_D(res0)
        od_h = self.recover_od_D(res1)

        # 构建二部图-->重新利用图注意力学习mode节点特征
        pid_g = dgl.heterograph({('srt_type', 'srt_dst_type', 'dst_type'): (pid_m[1], pid_m[2])})
        # 构建二部图-->重新利用图注意力学习节点特征
        od_g = dgl.heterograph({('srt_type', 'srt_dst_type', 'dst_type'): (od_m[1], od_m[2])})
        # 异构网络中节点预测-->基于异构节点
        m_h = self.nodetype_nn(temp_h, pid_g, pid_h, od_g, od_h)

        # 目标节点特征合并形式-->简单拼接
        h = torch.cat((m_h, h), 1)
        # 实现MLP-->训练出来的两个信息进行连接
        h = self.predict1(h)  # 数据量
        # THAN是有这个操作的（激活函数，归一化等）
        h = self.Ba1(h)
        h = F.leaky_relu(h)
        # h = torch.tanh(h)
        h = F.softmax(h, dim=1)
        h = self.Dr1(h)

        return h
