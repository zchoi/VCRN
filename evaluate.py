from socket import NI_MAXHOST
import sys
sys.path.append('caption-eval')
# sys.path.insert(0, './caption-eval')
import torch
import pickle
import models
from utils.utils import Vocabulary
from utils.data import get_eval_loader
from cocoeval import COCOScorer, suppress_stdout_stderr
import h5py
from utils.opt import parse_opt
from tqdm import tqdm
import os
import torchtext
import json
from sklearn.cluster import MiniBatchKMeans
import models
from models.encoder import Encoder
from models.decoder import Decoder
from models.capmodel import CapModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def convert_data_to_coco_scorer_format(reference):
    reference_json = {}
    non_ascii_count = 0
    with open(reference, 'r') as f:
        lines = f.readlines()
        for line in lines:
            vid = line.split('\t')[0]
            sent = line.split('\t')[1].strip()
            try:
                sent.encode('ascii', 'ignore').decode('ascii')
            except UnicodeDecodeError:
                non_ascii_count += 1
                continue
            if vid in reference_json:
                reference_json[vid].append({u'video_id': vid, u'cap_id': len(reference_json[vid]),
                                    u'caption': sent.encode('ascii', 'ignore').decode('ascii')})
            else:
                reference_json[vid] = []
                reference_json[vid].append({u'video_id': vid, u'cap_id': len(reference_json[vid]),
                                    u'caption': sent.encode('ascii', 'ignore').decode('ascii')})
    if non_ascii_count:
        print("=" * 20 + "\n" + "non-ascii: " + str(non_ascii_count) + "\n" + "=" * 20)
    return reference_json

def convert_prediction(prediction):
    prediction_json = {}
    with open(prediction, 'r') as f:
        lines = f.readlines()
        for line in lines:
            vid = line.split('\t')[0]
            sent = line.split('\t')[1].strip()
            prediction_json[vid] = [{u'video_id': vid, u'caption': sent}]
    return prediction_json


def evaluate(opt, net, eval_range, prediction_txt_path, reference):
    eval_loader = get_eval_loader(eval_range, opt.feature_h5_path, opt.test_batch_size)

    result = {}
    for i, (frames,  video_ids) in tqdm(enumerate(eval_loader)):

        frames = frames.to(DEVICE)
        outputs, _ = net(frames, None)
        for (tokens, vid) in zip(outputs, video_ids):
            if opt.use_multi_gpu:
                s = net.module.decoder.decode_tokens(tokens.data)
            else:
                s = net.decoder.decode_tokens(tokens.data)
            result[vid] = s

    with open(prediction_txt_path, 'w') as f:
        for vid, s in result.items():
            f.write('%d\t%s\n' % (vid, s))

    prediction_json = convert_prediction(prediction_txt_path)

    ############### visualization ###############
    # vis = []
    # for k,v in prediction_json.items():
    #     vis.append(v[-1])

    # json.dump(vis, open(os.path.join(opt.result_dir,'visualize.json'),'w'))
    #############################################

    # compute scores
    scorer = COCOScorer()
    with suppress_stdout_stderr():
        scores, sub_category_score = scorer.score(reference, prediction_json, prediction_json.keys())
    for metric, score in scores.items():
        print('%s: %.6f' % (metric, score * 100))

    if sub_category_score is not None:
        print('Sub Category Score in Spice:')
        for category, score in sub_category_score.items():
            print('%s: %.6f' % (category, score * 100))
    return scores


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
    encoder = Encoder(opt)
    decoder = Decoder(opt, filed)
    net = CapModel(encoder, decoder)
    if opt.use_multi_gpu:
        net = torch.nn.DataParallel(net)
    if not opt.eval_metric:
        net.load_state_dict(torch.load(opt.model_pth_path))
    elif opt.eval_metric == 'METEOR':
        net.load_state_dict(torch.load(opt.best_meteor_pth_path))
    elif opt.eval_metric == 'CIDEr':
        net.load_state_dict(torch.load(opt.best_cider_pth_path))
    else:
        raise ValueError('Please choose the metric from METEOR|CIDEr')
    net.to(DEVICE)
    net.eval()

    # reference = convert_data_to_coco_scorer_format(opt.train_reference_txt_path)
    # metrics = evaluate(opt, net, opt.train_range, opt.train_prediction_txt_path, reference)

    reference = convert_data_to_coco_scorer_format(opt.test_reference_txt_path)
    metrics = evaluate(opt, net, opt.test_range, opt.test_prediction_txt_path, reference)
    with open(opt.test_score_txt_path, 'a') as f:
        f.write('\nBEST ' + str(opt.eval_metric) + '(beam size = {}):\n'.format(opt.beam_size))
        for k, v in metrics.items():
            f.write('\t%s: %.2f\n' % (k, 100 * v))

