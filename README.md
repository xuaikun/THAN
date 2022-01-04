# THAN
THAN: Multi-Modal Transportation Recommendation with Heterogeneous Graph Attention Networks

Data and codes will be provided after the paper is accepted......

dataset in web: 链接：链接：https://pan.baidu.com/s/18QYrYO5yXnamEfDpkTJKGA 提取码：zm4m 

tips:

def setup(args):
    args.update(default_configure)
    set_random_seed(args['seed'])
    args['dataset'] = 'ACMRaw' if args['hetero'] else 'ACM'
    args['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    args['log_dir'] = setup_log_dir(args)
    return args

def setup(args):
    args.update(default_configure)
    set_random_seed(args['seed'])
    args['dataset'] = 'ACMRaw' if args['hetero'] else 'ACM'
    args['device'] = 'cpu'
    args['log_dir'] = setup_log_dir(args)
    return args
