import shutil
import pickle
import time
from utils.utils import *
from utils.data import get_eval_loader, get_train_loader
from utils.opt import parse_opt
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
# from spherecluster import SphericalKMeans
import models
from models.encoder import Encoder
from models.decoder import Decoder
from models.capmodel import CapModel
import torch
import h5py
import os
import torch.nn as nn
import numpy as np
from evaluate import evaluate, convert_data_to_coco_scorer_format
from tensorboard_logger import configure, log_value
from torchtext.data import *
import torchtext

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(opt):
    configure(opt.log_environment, flush_secs=10)

    # load vocabulary
    filed = torchtext.legacy.data.Field(sequential=True, tokenize="spacy",
                                 eos_token="<eos>",
                                 include_lengths=True,
                                 batch_first=True,
                                 fix_length=opt.max_words,
                                 lower=True,
                                 unk_token="<unk>"
                                 )
    if opt.min_freq ==4:
        filed.vocab = pickle.load(open(opt.vocab_pkl_path, 'rb'))
    elif opt.min_freq == 2:
        filed.vocab = pickle.load(open(opt.vocab_pkl_path, 'rb'))
    vocab_size = len(filed.vocab)
    # if opt.dataset == 'msr-vtt':
    #     filed.vocab = np.array(filed.vocab)
    print(vocab_size)

    # print parameters
    print('Learning rate: %.5f' % opt.learning_rate)
    print('Learning rate decay: ', opt.learning_rate_decay)
    print('Batch size: %d' % opt.train_batch_size)
    print('results directory: ', opt.result_dir)

    # build model
    encoder = Encoder(opt)
    decoder = Decoder(opt, filed)
    net = CapModel(encoder, decoder)
    if opt.use_multi_gpu:
        net = torch.nn.DataParallel(net)
    print('Total parameters:', sum(param.numel() for param in net.parameters()))

    if os.path.exists(opt.model_pth_path) and opt.use_checkpoint:
        net.load_state_dict(torch.load(opt.model_pth_path))
    net.to(DEVICE)

    # initialize data loader
    train_loader = get_train_loader(opt.train_caption_pkl_path, opt.feature_h5_path, filed,
                                    opt.train_batch_size)
    total_step = len(train_loader)

    # eval_loader = get_eval_loader(opt.val_caption_pkl_path, opt.feature_h5_path, filed,
    #                                 opt.train_batch_size)
    # total_step = len(eval_loader)
    # prepare groundtruth
    reference = convert_data_to_coco_scorer_format(opt.test_reference_txt_path)

    # start training
    best_meteor = 0
    best_meteor_epoch = 0
    best_cider = 0
    best_cider_epoch = 0
    loss_count = 0
    count = 0
    saving_schedule = [int(x * total_step / opt.save_per_epoch) for x in list(range(opt.save_per_epoch, opt.save_per_epoch + 1))]
    print('total: ', total_step)
    print('saving_schedule: ', saving_schedule)
    cat_feat = []
    for epoch in range(opt.max_epoch):
        start_time = time.time()

        if opt.learning_rate_decay and epoch % opt.learning_rate_decay_every == 0 and epoch > 0:
            opt.learning_rate /= opt.learning_rate_decay_rate
        epsilon = max(0.6, opt.ss_factor / (opt.ss_factor + np.exp(epoch / opt.ss_factor)))
        print('epoch:%d\tepsilon:%.8f' % (epoch, epsilon))
        log_value('epsilon', epsilon, epoch)

        for i, (frames, captions, cap_lens, video_ids) in enumerate(train_loader, start=1):
            # convert data to DEVICE mode

            frames = frames.to(DEVICE)
            targets = captions.to(DEVICE)
            for k in range(frames.size(0)):
                if torch.rand(1) <= 0.5:
                    cat_feat.append(frames[k].unsqueeze(0).cpu())

            # cat_feat.append(frames.cpu())
            print("iter {}/{}".format(i,len(train_loader)))
        # test_feat = evaluate(opt, net, opt.test_range, opt.test_prediction_txt_path, reference)

        train_feat = torch.cat(cat_feat,dim=0)

        # np.save('msvd.npy',train_feat.view(-1,4096).numpy())
        print(train_feat.shape)

        # traintest_feat = torch.cat([train_feat, test_feat], dim=0)
        # print("begin kmeans...{}".format(traintest_feat.view(-1,4096).size()))
        # kmeans = KMeans(n_clusters=300, random_state=0).fit(cat_feat.view(-1,4096))
        kmeans = MiniBatchKMeans(n_clusters=1000, random_state=0, batch_size = 2048).fit(train_feat.view(-1,4096))
        # kmeans = SphericalKMeans(n_clusters=300).fit(cat_feat.view(-1,4096))
        print(kmeans.cluster_centers_.shape)
        end_time = time.time()
        print("*******One epoch time: %.3fs*******\n" % (end_time - start_time))
        print('best cider: %.3f' % best_cider)

        f = h5py.File('data/MSRVTT/msrvtt_concept1000_feat_train_50percent.h5','w')  
        f['concept_features'] = kmeans.cluster_centers_                         
        f.close()                          
        exit()

    with open(opt.test_score_txt_path, 'w') as f:
        f.write('MODEL: {}\n'.format(opt.model))
        f.write('best meteor epoch: {}\n'.format(best_meteor_epoch))
        f.write('best cider epoch: {}\n'.format(best_cider_epoch))
        f.write('best cider score: {}\n'.format(best_cider))
        f.write('Learning rate: {:6f}\n'.format(opt.learning_rate))
        f.write('Learning rate decay: {}\n'.format(opt.learning_rate_decay))
        f.write('Batch size: {}\n'.format(opt.train_batch_size))
        f.write('results directory: {}\n'.format(opt.result_dir))


if __name__ == '__main__':
    # opt = parse_opt()
    # main(opt)
    
    
    # For a bigger commensense video dictionary (MSVD + MSR-VTT + VATEX)
    # msvd = h5py.File('data/MSVD/msvd_concept1000_feat_train.h5','r')['concept_features'][:,:]
    # msrvtt = h5py.File('data/MSRVTT/msrvtt_concept1000_feat_train.h5','r')['concept_features'][:,:]
    # vatex = h5py.File('data/VATEX/vatex_concept1000_feat.h5','r')['concept_features'][:,:]
    
    # msvd = torch.from_numpy(msvd)
    # msrvtt = torch.from_numpy(msrvtt)
    # vatex = torch.from_numpy(vatex)

    # print(msvd.size(), msrvtt.size(), vatex.size())

    # all_f = torch.cat([msvd, msrvtt, vatex], dim=0)
    # print(all_f.size())
    # kmeans = MiniBatchKMeans(n_clusters=1000, random_state=0, batch_size = 2048).fit(all_f.view(-1,4096))
    # # kmeans = SphericalKMeans(n_clusters=300).fit(cat_feat.view(-1,4096))
    # print(kmeans.cluster_centers_.shape)
    # end_time = time.time()

    # f = h5py.File('data/three_dataset_concept1000_feat_train.h5','w')  
    # f['concept_features'] = kmeans.cluster_centers_                         
    # f.close()

    # combination test of train, val, test
    train = h5py.File('data/VATEX/vatex_concept1000_feat_train.h5','r')['concept_features'][:,:]
    val = h5py.File('data/VATEX/vatex_concept1000_feat_val.h5','r')['concept_features'][:,:]
    
    train = torch.from_numpy(train)
    val = torch.from_numpy(val)

    print(train.size(), val.size())

    all_f = torch.cat([train, val], dim=0)
    print(all_f.size())
    kmeans = MiniBatchKMeans(n_clusters=1000, random_state=0, batch_size = 2048).fit(all_f.view(-1,4096))
    print(kmeans.cluster_centers_.shape)
    end_time = time.time()

    f = h5py.File('data/VATEX/vatex_concept1000_feat_trainval.h5','w')  
    f['concept_features'] = kmeans.cluster_centers_                         
    f.close()