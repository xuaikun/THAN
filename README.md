# THAN
THAN: Multi-Modal Transportation Recommendation with Heterogeneous Graph Attention Networks

Data and codes will be provided after the paper is accepted......

tips:
change "args['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'"

to

args['device'] = 'cpu'

in def setup()

also in def setup_for_sampling()
