import os
import torch
import yaml
import torch.nn as nn
import parser
from model import ft_net, ft_net_dense, ft_net_EF4, ft_net_EF5, ft_net_EF6, ft_net_IR, ft_net_NAS, ft_net_SE, ft_net_DSE, PCB, CPB, ft_net_angle, ft_net_arc

def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1 # count the image number in every class
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight

# Get model list for resume
def get_model_list(dirname, key):
    if os.path.exists(dirname) is False:
        print('no dir: %s'%dirname)
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and key in f and ".pth" in f]
    if gen_models is None:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name

######################################################################
# Save model
#---------------------------
def save_network(network, dirname, epoch_label):
    if isinstance(epoch_label, int):
        save_filename = 'net_%03d.pth'% epoch_label
    else:
        save_filename = 'net_%s.pth'% epoch_label
    save_path = os.path.join('./data/outputs',dirname,save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available:
        network.cuda()


######################################################################
#  Load model for resume
#---------------------------
def load_network(name, opt):
    # Load config
    dirname = os.path.join('./data/outputs',name)
    last_model_name = os.path.basename(get_model_list(dirname, 'net'))
    epoch = last_model_name.split('_')[1]
    epoch = epoch.split('.')[0]
    if not epoch=='last':
       epoch = int(epoch)
    config_path = os.path.join(dirname,'opts.yaml')
    with open(config_path, 'r') as stream:
        config = yaml.load(stream)

    opt.name = config['name']
    opt.inputsize = config['inputsize']
    opt.data_dir = config['data_dir']
    opt.train_all = config['train_all']
    opt.train_veri = config['train_veri']
    opt.train_comp = config['train_comp']
    opt.train_comp_veri = config['train_comp_veri']
    opt.droprate = config['droprate']
    opt.color_jitter = config['color_jitter']
    opt.batchsize = config['batchsize']
    opt.inputsize = config['inputsize']
    opt.stride = config['stride']
  
    if 'pool' in config:
        opt.pool = config['pool']
    if 'use_DSE' in config:
        opt.use_DSE = config['use_DSE']
    if 'use_EF4' in config:
        opt.use_EF4 = config['use_EF4']
        opt.use_EF5 = config['use_EF5']
        opt.use_EF6 = config['use_EF6']
    if 'h' in config:
        opt.h = config['h']
        opt.w = config['w']
    if 'gpu_ids' in config:
        opt.gpu_ids = config['gpu_ids']
    opt.erasing_p = config['erasing_p']
    opt.lr = config['lr']
    opt.nclasses = config['nclasses']
    opt.erasing_p = config['erasing_p']
    opt.use_dense = config['use_dense']
    opt.use_NAS = config['use_NAS']
    opt.use_SE = config['use_SE']
    opt.use_IR = config['use_IR']
    opt.PCB = config['PCB']
    opt.CPB = config['CPB']
    opt.fp16 = config['fp16']
    opt.balance = config['balance']
    opt.angle = config['angle']
    opt.arc = config['arc']

    if opt.use_dense:
        model = ft_net_dense(opt.nclasses, opt.droprate, opt.stride, None, opt.pool)
    elif opt.use_NAS:
        model = ft_net_NAS(opt.nclasses, opt.droprate, opt.stride)
    elif opt.use_SE:
        model = ft_net_SE(opt.nclasses, opt.droprate, opt.stride, opt.pool)
    elif opt.use_DSE:
        model = ft_net_DSE(opt.nclasses, opt.droprate, opt.stride, opt.pool)
    elif opt.use_IR:
        model = ft_net_IR(opt.nclasses, opt.droprate, opt.stride)
    elif opt.use_EF4:
        model = ft_net_EF4(opt.nclasses, opt.droprate)
    elif opt.use_EF5:
        model = ft_net_EF5(opt.nclasses, opt.droprate)
    elif opt.use_EF6:
        model = ft_net_EF6(opt.nclasses, opt.droprate)
    else:
        model = ft_net(opt.nclasses, opt.droprate, opt.stride, None, opt.pool)

    if opt.PCB:
        model = PCB(opt.nclasses)

    if opt.CPB:
        model = CPB(opt.nclasses)

    if opt.angle:
        model = ft_net_angle(opt.nclasses, opt.droprate, opt.stride)
    elif opt.arc:
        model = ft_net_arc(opt.nclasses, opt.droprate, opt.stride)

    # load model
    if isinstance(epoch, int):
        save_filename = 'net_%03d.pth'% epoch
    else:
        save_filename = 'net_%s.pth'% epoch

    save_path = os.path.join('./data/outputs',name,save_filename)
    print('Load the model from %s'%save_path)
    network = model
    try:
        network.load_state_dict(torch.load(save_path))
    except:
        network = torch.nn.DataParallel(network)
        network.load_state_dict(torch.load(save_path))
        network = network.module
    return network, opt, epoch
