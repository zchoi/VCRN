import sys
sys.path.append('caption-eval')
# sys.path.insert(0, './caption-eval')
import torch
import pickle
import models
from utils.utils import Vocabulary
from utils.data import get_eval_loader
import h5py
from utils.opt import parse_opt
from tqdm import tqdm
import torchtext
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import models
from models.encoder import Encoder
from models.decoder import Decoder
from models.capmodel import CapModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(opt, net = None, eval_range = None, prediction_txt_path = None, reference = None):
    eval_loader = get_eval_loader(eval_range, opt.feature_h5_path, opt.test_batch_size)

    result = {}
    frame_list = []
    for i, (frames,  video_ids) in tqdm(enumerate(eval_loader)):
        # frames = frames.to(DEVICE)
        print("{}/{}".format(i,len(eval_loader)))
        frame_list.append(frames.reshape(-1, 4096).cpu())

    frame_feat = torch.cat(frame_list,dim=0)
    print("begin kmeans...{}".format(frame_feat.view(-1,4096).size()))
    kmeans = MiniBatchKMeans(n_clusters=1000, random_state=0, batch_size = 1024).fit(frame_feat.view(-1,4096))
    f = h5py.File('data/VATEX/vatex_concept1000_feat_test.h5','w') 
    # print(kmeans.cluster_centers_.shape) 
    f['concept_features'] = kmeans.cluster_centers_                         
    f.close()
    exit()


if __name__ == '__main__':
    opt = parse_opt()

    filed = torchtext.legacy.data.Field(sequential=True, tokenize="spacy",
                                 eos_token="<eos>",
                                 include_lengths=True,
                                 batch_first=True,
                                 tokenizer_language='en_core_web_sm',
                                 fix_length=opt.max_words,
                                 lower=True,
                                 )
    if opt.min_freq == 4:
        filed.vocab = pickle.load(open(opt.vocab_pkl_path, 'rb'))
    elif opt.min_freq == 2:
        filed.vocab = pickle.load(open(opt.vocab_pkl_path, 'rb'))
    vocab_size = len(filed.vocab)

    # build model
    # encoder = Encoder(opt)
    # decoder = Decoder(opt, filed)
    # net = CapModel(encoder, decoder)
    # if opt.use_multi_gpu:
    #     net = torch.nn.DataParallel(net)
    # if not opt.eval_metric:
    #     net.load_state_dict(torch.load(opt.model_pth_path))
    # elif opt.eval_metric == 'METEOR':
    #     net.load_state_dict(torch.load(opt.best_meteor_pth_path))
    # elif opt.eval_metric == 'CIDEr':
    #     net.load_state_dict(torch.load(opt.best_cider_pth_path))
    # else:
    #     raise ValueError('Please choose the metric from METEOR|CIDEr')
    # net.to(DEVICE)
    # net.eval()

    metrics = evaluate(opt = opt, eval_range = opt.test_range)
