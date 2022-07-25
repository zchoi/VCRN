import h5py
import torch
import sys
import numpy as np
import pickle
import torchtext
# sys.path.append('/home/zengpengpeng/projects/video_caption/Hierarchical_Gumbel/version1/')
import torch.utils.data as data
from .opt import parse_opt
opt = parse_opt()
import torchtext
import numpy as np
import json
import warnings
import os
import array
from tqdm import tqdm
import math


class BigFile:

    def __init__(self, datadir):
        self.nr_of_images, self.ndims = map(int, open(os.path.join(datadir,'shape.txt')).readline().split())
        id_file = os.path.join(datadir, "id.txt")
        # python 3
        self.names = open(id_file, 'rb').read().strip().split()
        for i in range(len(self.names)):
            self.names[i] = str(self.names[i], encoding='ISO-8859-1')
        
        # python 2
        # self.names = open(id_file).read().strip().split()

        assert(len(self.names) == self.nr_of_images)
        self.name2index = dict(zip(self.names, range(self.nr_of_images)))
        self.binary_file = os.path.join(datadir, "feature.bin")
        print ("[%s] %dx%d instances loaded from %s" % (self.__class__.__name__, self.nr_of_images, self.ndims, datadir))


    def read(self, requested, isname=True):
        requested = set(requested)
        if isname:
            index_name_array = [(self.name2index[x], x) for x in requested if x in self.name2index]
        else:
            assert(min(requested)>=0)
            assert(max(requested)<len(self.names))
            index_name_array = [(x, self.names[x]) for x in requested]
        if len(index_name_array) == 0:
            return [], []
       
        index_name_array.sort(key=lambda v:v[0])
        sorted_index = [x[0] for x in index_name_array]

        nr_of_images = len(index_name_array)
        vecs = [None] * nr_of_images
        offset = np.float32(1).nbytes * self.ndims
        
        res = array.array('f')
        fr = open(self.binary_file, 'rb')
        fr.seek(index_name_array[0][0] * offset)
        res.fromfile(fr, self.ndims)
        previous = index_name_array[0][0]
 
        for next in sorted_index[1:]:
            move = (next-1-previous) * offset
            #print next, move
            fr.seek(move, 1)
            res.fromfile(fr, self.ndims)
            previous = next

        fr.close()

        return [x[1] for x in index_name_array], [ res[i*self.ndims:(i+1)*self.ndims].tolist() for i in range(nr_of_images) ]


    def read_one(self, name):
        renamed, vectors = self.read([name])
        return vectors[0]    

    def shape(self):
        return [self.nr_of_images, self.ndims]

class V2TDataset(data.Dataset):
    def __init__(self, cap_pkl, frame_feature_h5, filed):

        # h5 = h5py.File(frame_feature_h5, 'r')
        # print(frame_feature_h5)

        # self.video_feats = h5[opt.feature_h5_feats]
        self.tokens = []
        self.video_ids = []
        self.filed = filed
        self.dataset = opt.dataset
        self.max_frames = opt.max_frames
        if opt.dataset == 'msr-vtt':
            # self.video_feats=np.load("data/MSRVTT/msrvtt_feat_res_rext.npy",allow_pickle=True)

            self.visual_feat = BigFile('data/MSRVTT/pyresnet-152_imagenet11k,flatten0_output,os_pyresnext-101_rbps13k,flatten0_output,os')
            self.video2frames = open('data/MSRVTT/pyresnet-152_imagenet11k,flatten0_output,os_pyresnext-101_rbps13k,flatten0_output,os/video2frames.txt','r')
            
            a = self.video2frames.read()  
            self.dict_data = eval(a)  
            self.video2frames.close()

            with open(opt.msrvtt_caption_train, 'r') as data:
                lines = data.readlines()
                for line in lines:
                    vid = line.split('\t')[0]
                    sent = line.split('\t')[1].strip()
                    self.video_ids.append(int(vid))
                    self.tokens.append(filed.preprocess(sent))
            self.captions, self.lengths = filed.process(self.tokens)

        elif opt.dataset=="msvd":
             # Res_Resnext Features
            # self.video_feats=np.load("data/MSVD/msvd_feat_res_rext.npy",allow_pickle=True)

            # MXNet Features
            self.visual_feat = BigFile('data/MSVD/pyresnext-101_rbps13k,flatten0_output,os+pyresnet-152_imagenet11k,flatten0_output,os')
            self.video2frames = open('data/MSVD/pyresnext-101_rbps13k,flatten0_output,os+pyresnet-152_imagenet11k,flatten0_output,os/video2frames.txt','r')
            
            a = self.video2frames.read()
            self.dict_data = eval(a)  
            self.video2frames.close()

            with open(opt.msvd_caption_train, 'r') as data:
                lines = data.readlines()
                for line in lines:
                    vid=int(line.split('\t')[0].split("vid")[1])-1
                    sent = line.split('\t')[1].strip()
                    self.video_ids.append(int(vid))
                    self.tokens.append(filed.preprocess(sent)) #48774
            self.captions, self.lengths = filed.process(self.tokens)
        elif opt.dataset == 'vatex':
            self.visual_feat = BigFile('data/VATEX/trainval/pyresnext-101_rbps13k,flatten0_output,os+pyresnet-152_imagenet11k,flatten0_output,os')
            self.video2frames = open('data/VATEX/trainval/pyresnext-101_rbps13k,flatten0_output,os+pyresnet-152_imagenet11k,flatten0_output,os/video2frames.txt','r')
            
            a = self.video2frames.read()
            self.dict_data = eval(a)  
            self.video2frames.close()
            

            self.name2id = json.load(open('data/VATEX/name2id.json'))
            self.id2name = {v:k for k,v in self.name2id.items()}

            data = json.load(open(opt.vatex_caption_train))
            for line in data:
                vid = self.name2id[line['videoID']]
                vid_name = line['videoID']
                for sent_i in line['enCap']:
                    sent = sent_i.strip().lower()[:-1]
                    self.video_ids.append(int(vid))
                    self.tokens.append(filed.preprocess(sent)) #48774
            
            # self.captions, self.lengths = filed.process(self.tokens)
            self.s2i = json.load(open('dddd/stoi.json'))

            self.captions, self.lengths = torch.zeros(len(self.tokens), opt.max_words,dtype=torch.long), torch.zeros(len(self.tokens),dtype=torch.long)
            
            print("Tokensize {}".format(opt.vatex_caption_train))
            for i in tqdm(range(len(self.tokens))):
                for j, token in enumerate(self.tokens[i][:opt.max_words]):
                    if token in self.s2i.keys():
                        self.captions[i,j] = self.s2i[token]
                    else:
                        self.captions[i,j] = self.s2i['<unk>']
                if j+1 == opt.max_words:
                    self.captions[i,j] = self.s2i['<eos>']
                else:
                    self.captions[i,j + 1] = self.s2i['<eos>']
                self.lengths[i] = len(self.tokens[i]) + 1

            # for i,c in enumerate(self.captions):
            #     for j,cc in enumerate(c):
            #         ggg= []
            #         for tt in self.tokens[i]:
            #             if tt in self.s2i.keys():
            #                 ggg.append(self.s2i[tt])
            #             else:
            #                 ggg.append(3)
            #     print(c)
            #     print(ggg)
            #     print(self.tokens[i])
            #     print(self.lengths[i])

            # exit()

    def __getitem__(self, index):

        caption = self.captions[index]
        video_id = self.video_ids[index]
        # video_feat = torch.from_numpy(self.video_feats[video_id])
        lengths = self.lengths[index]
        
        if self.dataset == 'msvd':
        # Res_Resnext Features
            # video_feat = torch.from_numpy(self.video_feats[video_id])

        # MXNet Features
            frame_list = self.dict_data['video'+str(video_id+1)]
            frame_vecs = []
            for frame_id in frame_list:
                frame_vecs.append(self.visual_feat.read_one(frame_id))
            frames_tensor = torch.Tensor(frame_vecs)
            
            video_feat = torch.zeros(self.max_frames, frames_tensor.size(-1), dtype=torch.float32)
            if frames_tensor.size(0) < self.max_frames:
                video_feat[:frames_tensor.size(0), :] = frames_tensor[:, :4096]
            
            elif frames_tensor.size(0) > self.max_frames:
                sample_id = np.linspace(0, frames_tensor.size(0), num=self.max_frames, endpoint=False, retstep=False)
                video_feat = frames_tensor[sample_id][:, :4096]
        elif self.dataset == 'msr-vtt':
        # Res_Resnext Features
            # video_feat = torch.from_numpy(self.video_feats[video_id])

        # MXNet Features
            frame_list = self.dict_data['video'+str(video_id)]
            frame_vecs = []
            for frame_id in frame_list:
                frame_vecs.append(self.visual_feat.read_one(frame_id))
            frames_tensor = torch.Tensor(frame_vecs)
            
            video_feat = torch.zeros(self.max_frames, frames_tensor.size(-1), dtype=torch.float32)
            if frames_tensor.size(0) < self.max_frames:
                video_feat[:frames_tensor.size(0), :] = frames_tensor[:, :4096]
            
            elif frames_tensor.size(0) > self.max_frames:
                sample_id = np.linspace(0, frames_tensor.size(0), num=self.max_frames, endpoint=False, retstep=False)
                video_feat = frames_tensor[sample_id][:, :4096]
        elif self.dataset == 'vatex':
        # MXNet Features
            frame_list = self.dict_data[self.id2name[video_id]]
            frame_vecs = []
            for frame_id in frame_list:
                frame_vecs.append(self.visual_feat.read_one(frame_id))
            frames_tensor = torch.Tensor(frame_vecs)
            
            video_feat = torch.zeros(self.max_frames, 4096, dtype=torch.float32)
            if frames_tensor.size(0) > 0 and frames_tensor.size(1) > 0:
                if frames_tensor.size(0) < self.max_frames:
                    video_feat[:frames_tensor.size(0), :] = frames_tensor[:, :4096]
                
                elif frames_tensor.size(0) > self.max_frames:
                    sample_id = np.linspace(0, frames_tensor.size(0), num=self.max_frames, endpoint=False, retstep=False)
                    video_feat = frames_tensor[sample_id][:, :4096]
            else:
                print('Empty Frames!!!')
                warnings.warn('Empty Frames from {}'.format(self.id2name[video_id]))
            if caption.data.sum() == 0:
                print('Empty Captions!!!')
                warnings.warn('Empty Captions')
        return video_feat, caption, lengths, video_id

    def __len__(self):
        return len(self.captions)

# 新特征
class VideoDataset(data.Dataset):
    def __init__(self, eval_range, frame_feature_h5):
        self.dataset = opt.dataset
        self.eval_list = tuple(range(*eval_range))
        self.max_frames = opt.max_frames
        # self.video_feats=np.load("data/MSVD/msvd_feat_res_rext.npy",allow_pickle=True)

        if opt.dataset == 'msvd':
            # Res_Rext Feature
            # self.video_feats=np.load("data/MSVD/msvd_feat_res_rext.npy",allow_pickle=True)

            # MXNet Features
            self.visual_feat = BigFile('data/MSVD/pyresnext-101_rbps13k,flatten0_output,os+pyresnet-152_imagenet11k,flatten0_output,os')
            self.video2frames = open('data/MSVD/pyresnext-101_rbps13k,flatten0_output,os+pyresnet-152_imagenet11k,flatten0_output,os/video2frames.txt','r')
            
            a = self.video2frames.read()  
            self.dict_data = eval(a)  
            self.video2frames.close() 

        elif opt.dataset == 'msr-vtt':
            # Res_Rext Feature
            self.video_feats=np.load("data/MSRVTT/msrvtt_feat_res_rext.npy",allow_pickle=True)
            # MXNet Features
            self.visual_feat = BigFile('data/MSRVTT/pyresnet-152_imagenet11k,flatten0_output,os_pyresnext-101_rbps13k,flatten0_output,os')
            self.video2frames = open('data/MSRVTT/pyresnet-152_imagenet11k,flatten0_output,os_pyresnext-101_rbps13k,flatten0_output,os/video2frames.txt','r')
            
            a = self.video2frames.read()  
            self.dict_data = eval(a)  
            self.video2frames.close()

        elif opt.dataset == 'vatex':
            self.visual_feat = BigFile('data/VATEX/test/pyresnext-101_rbps13k,flatten0_output,os+pyresnet-152_imagenet11k,flatten0_output,os')
            self.video2frames = open('data/VATEX/test/pyresnext-101_rbps13k,flatten0_output,os+pyresnet-152_imagenet11k,flatten0_output,os/video2frames.txt','r')
            
            a = self.video2frames.read()
            self.dict_data = eval(a)  
            self.video2frames.close()

            self.name2id = json.load(open('data/VATEX/name2id.json'))
            self.id2name = {v:k for k,v in self.name2id.items()}


    def __getitem__(self, index):
        video_id = self.eval_list[index]
        # video_feat = torch.from_numpy(self.video_feats[video_id])

        if self.dataset == 'msvd':

            # Res_Rext Feature
            # video_feat = torch.from_numpy(self.video_feats[video_id])

            # MXNet Features
            frame_list = self.dict_data['video'+str(video_id+1)]
            frame_vecs = []
            for frame_id in frame_list:
                frame_vecs.append(self.visual_feat.read_one(frame_id))
            frames_tensor = torch.Tensor(frame_vecs)
            
            video_feat = torch.zeros(self.max_frames, frames_tensor.size(-1), dtype=torch.float32)
            if frames_tensor.size(0) < self.max_frames:
                video_feat[:frames_tensor.size(0), :] = frames_tensor[:, :4096]
                
            elif frames_tensor.size(0) > self.max_frames:
                sample_id = np.linspace(0, frames_tensor.size(0), num=self.max_frames, endpoint=False, retstep=False)
                video_feat = frames_tensor[sample_id][:, :4096]

        elif self.dataset == 'msr-vtt':
            # Res_Rext Feature
            # video_feat = torch.from_numpy(self.video_feats[video_id])

            # MXNet Features
            frame_list = self.dict_data['video'+str(video_id)]
            frame_vecs = []
            for frame_id in frame_list:
                frame_vecs.append(self.visual_feat.read_one(frame_id))
            frames_tensor = torch.Tensor(frame_vecs)
            
            video_feat = torch.zeros(self.max_frames, frames_tensor.size(-1), dtype=torch.float32)
            if frames_tensor.size(0) < self.max_frames:
                video_feat[:frames_tensor.size(0), :] = frames_tensor[:, :4096]
                
            elif frames_tensor.size(0) > self.max_frames:
                sample_id = np.linspace(0, frames_tensor.size(0), num=self.max_frames, endpoint=False, retstep=False)
                video_feat = frames_tensor[sample_id][:, :4096]
                
        elif self.dataset == 'vatex':
            # MXNet Features
            frame_list = self.dict_data[self.id2name[video_id]]
            frame_vecs = []
            for frame_id in frame_list:
                frame_vecs.append(self.visual_feat.read_one(frame_id))
            frames_tensor = torch.Tensor(frame_vecs)
            
            video_feat = torch.zeros(self.max_frames, 4096, dtype=torch.float32)
            if frames_tensor.size(0) > 0 and frames_tensor.size(1)>0:
                if frames_tensor.size(0) < self.max_frames:
                    video_feat[:frames_tensor.size(0), :] = frames_tensor[:, :4096]
                
                elif frames_tensor.size(0) > self.max_frames:
                    sample_id = np.linspace(0, frames_tensor.size(0), num=self.max_frames, endpoint=False, retstep=False)
                    video_feat = frames_tensor[sample_id][:, :4096]
            else:
                warnings.warn('Empty Frames from {}'.format(self.id2name[video_id]))
        return video_feat, video_id

    def __len__(self):
        return len(self.eval_list)



def train_collate_fn(data):
    data.sort(key=lambda x: x[2], reverse=True)

    # videos, captions, lengths, video_ids, object_feat = zip(*data)
    videos, captions, lengths, video_ids = zip(*data)

    videos = torch.stack(videos, 0)
    captions = torch.stack(captions,0)
    # object_feat = torch.stack(object_feat,0)

    # return videos, captions, lengths, video_ids, object_feat
    return videos, captions, lengths, video_ids


def eval_collate_fn(data):

    data.sort(key=lambda x: x[-1], reverse=False)

    # videos, video_ids, object_feat = zip(*data)
    videos, video_ids = zip(*data)

    videos = torch.stack(videos, 0)

    # object_feat = torch.stack(object_feat, 0)

    # return videos, video_ids, object_feat
    return videos, video_ids


def get_train_loader(cap_pkl, frame_feature_h5, filed, batch_size=100, shuffle=True, num_workers=12, pin_memory=True):
    v2t = V2TDataset(cap_pkl, frame_feature_h5, filed)
    data_loader = torch.utils.data.DataLoader(dataset=v2t,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=train_collate_fn,
                                              pin_memory=pin_memory)
    return data_loader


def get_eval_loader(cap_pkl, frame_feature_h5,  batch_size=100, shuffle=False, num_workers=4, pin_memory=False):

    vd = VideoDataset(cap_pkl, frame_feature_h5)
    data_loader = torch.utils.data.DataLoader(dataset=vd,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=eval_collate_fn,
                                              pin_memory=pin_memory)
    return data_loader


if __name__ == '__main__':
    filed = torchtext.data.Field(sequential=True, tokenize="spacy",
                                eos_token="<eos>",
                                include_lengths=True,
                                batch_first=True,
                                fix_length=26,
                                lower=True,
                                )
    filed.vocab = pickle.load(open(opt.vocab_pkl_path, 'rb'))
    print(len(filed.vocab))
    print(filed.vocab.stoi['<eos>'])
    # train_loader = get_train_loader(opt.train_caption_pkl_path, opt.feature_h5_path, filed)
    # print(len(train_loader))

    # for i, data in enumerate(train_loader):
    #     print(data[0].size())
    #     print(data[1].size())
    #     print(data[2])

