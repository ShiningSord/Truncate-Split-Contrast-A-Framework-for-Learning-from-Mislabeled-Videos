# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import torch.utils.data as data

from PIL import Image
import json
import os
import numpy as np
from numpy.random import randint
from numpy.testing import assert_array_almost_equal
import copy

import pdb

class VideoRecord(object):
    def __init__(self, row, resume=False):
        if not resume:
           gt = row[2]
           row.append(gt)
        assert len(row) == 4, "error in VideoRecord initial, the length of row is not 4"
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])

    @property
    def groundtruth(self):
        return int(self._data[3])


class TSNDataSet(data.Dataset):
    def __init__(self, root_path, list_file,
                 num_segments=3, new_length=1, modality='RGB',
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 random_shift=True, test_mode=False,
                 remove_missing=False, dense_sample=False, twice_sample=False,
                 noise_type=None, noise_rate=0.0, random_state=0):

        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.remove_missing = remove_missing
        self.dense_sample = dense_sample  # using dense sample as I3D
        self.twice_sample = twice_sample  # twice sample for more validation

        # for noisy label version
        self.noise_type = noise_type
        self.noise_rate = noise_rate
        self.random_state = random_state

        if self.dense_sample:
            print('=> Using dense sample for the dataset...')
        if self.twice_sample:
            print('=> Using twice sample for the dataset...')

        if self.modality == 'RGBDiff':
            self.new_length += 1  # Diff needs one more image to calculate diff

        self._parse_list()

    def _load_image(self, directory, idx):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            try:
                return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx))).convert('RGB')]
            except Exception:
                print('error loading image:', os.path.join(self.root_path, directory, self.image_tmpl.format(idx)))
                return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(1))).convert('RGB')]
        elif self.modality == 'Flow':
            if self.image_tmpl == 'flow_{}_{:05d}.jpg':  # ucf
                x_img = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format('x', idx))).convert(
                    'L')
                y_img = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format('y', idx))).convert(
                    'L')
            elif self.image_tmpl == '{:06d}-{}_{:05d}.jpg':  # something v1 flow
                x_img = Image.open(os.path.join(self.root_path, '{:06d}'.format(int(directory)), self.image_tmpl.
                                                format(int(directory), 'x', idx))).convert('L')
                y_img = Image.open(os.path.join(self.root_path, '{:06d}'.format(int(directory)), self.image_tmpl.
                                                format(int(directory), 'y', idx))).convert('L')
            else:
                try:
                    # idx_skip = 1 + (idx-1)*5
                    flow = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx))).convert(
                        'RGB')
                except Exception:
                    print('error loading flow file:',
                          os.path.join(self.root_path, directory, self.image_tmpl.format(idx)))
                    flow = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(1))).convert('RGB')
                # the input flow file is RGB image with (flow_x, flow_y, blank) for each channel
                flow_x, flow_y, _ = flow.split()
                x_img = flow_x.convert('L')
                y_img = flow_y.convert('L')

            return [x_img, y_img]

    def _parse_list(self):
        if 'k200' in self.list_file:
            noise_path = '/apdcephfs/share_1290939/zixiaowang/research/dataset/k200/%.1f_%s.json'%(self.noise_rate,self.noise_type)
        elif 'k400' in self.list_file:
            noise_path = '/apdcephfs/share_1290939/zixiaowang/research/dataset/k400/%.1f_%s.json'%(self.noise_rate,self.noise_type)
        elif 'sthv1' in self.list_file:
            noise_path =  '/apdcephfs/share_1290939/zixiaowang/research/dataset/sthv1/%.1f_%s.json'%(self.noise_rate,self.noise_type)
        else:
            raise NotImplementedError
        if self.noise_type and self.noise_rate > 0 and os.path.exists(noise_path):
            print("exist %s, loading"%(noise_path))
            video_list_json = json.load(open(noise_path,"r"))
            self.video_list = [VideoRecord(item,True) for item in video_list_json]
            # obtain number of category
            self.num_category = -1
            for v in self.video_list:
                if self.num_category < int(v._data[2]):
                    self.num_category = int(v._data[2])
            self.num_category += 1
        else:
            # check the frame number is large >3:
            tmp = [x.strip().split(' ') for x in open(self.list_file)]
            if not self.test_mode or self.remove_missing:
                tmp = [item for item in tmp if int(item[1]) >= 3]
            self.video_list = [VideoRecord(item) for item in tmp]
    
            # obtain number of category
            self.num_category = -1
            for v in self.video_list:
                if self.num_category < int(v._data[2]):
                    self.num_category = int(v._data[2])
            self.num_category += 1
    
            if self.noise_type and self.noise_rate > 0:
                self._noisify() # change label of video sample based on settings

            if self.image_tmpl == '{:06d}-{}_{:05d}.jpg':
                for v in self.video_list:
                    v._data[1] = int(v._data[1]) / 2
            if self.noise_type and self.noise_rate > 0:
                video_list_json = [tmp._data for tmp in self.video_list]
                json.dump(video_list_json,open(noise_path,"w"))
                print("noisy file saved to %s"%(noise_path))
        self.raw_video_list = copy.deepcopy(self.video_list)

        print('video number:%d' % (len(self.video_list)))
 
    def _noisify(self, noise_type=None, noise_rate=0, random_state=0):

        n_cate = self.num_category
        n = self.noise_rate

        if self.noise_type == 'pairflip':
            P = np.eye(n_cate)

            if n > 0.0:
                # 0 -> 1
                P[0, 0], P[0, 1] = 1. - n, n
                for i in range(1, n_cate-1):
                    P[i, i], P[i, i + 1] = 1. - n, n
                P[n_cate-1, n_cate-1], P[n_cate-1, 0] = 1. - n, n

                self._multiclass_noisify(P)


        if self.noise_type == 'symmetric':
            P = np.ones((n_cate, n_cate))
            P = (n / (n_cate - 1)) * P

            if n > 0.0:
                # 0 -> 1
                P[0, 0] = 1. - n
                for i in range(1, n_cate-1):
                    P[i, i] = 1. - n
                P[n_cate-1, n_cate-1] = 1. - n

                self._multiclass_noisify(P)


        print(P)


    def _multiclass_noisify(self, P):

        assert P.shape[0] == P.shape[1]
        assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
        assert (P >= 0.0).all()

        flipper = np.random.RandomState(self.random_state)
        y_ori = []
        y_new = []
        for v in self.video_list:
            ori_label = int(v._data[2])
            flipped = flipper.multinomial(1, P[ori_label, :], 1)[0]
            new_label = int(np.where(flipped == 1)[0])
            # pdb.set_trace()
            y_ori.append(ori_label)
            y_new.append(new_label)

            v._data[2] = str(new_label)

        actual_noise = (np.array(y_ori) != np.array(y_new)).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)

    def _sample_indices(self, record):
        """

        :param record: VideoRecord
        :return: list
        """
        if self.dense_sample:  # i3d dense sample
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        else:  # normal sample
            average_duration = (record.num_frames - self.new_length + 1) // self.num_segments
            if average_duration > 0:
                offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration,
                                                                                                  size=self.num_segments)
            elif record.num_frames > self.num_segments:
                offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
            else:
                offsets = np.zeros((self.num_segments,))
            return offsets + 1

    def _get_val_indices(self, record):
        if self.dense_sample:  # i3d dense sample
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        else:
            if record.num_frames > self.num_segments + self.new_length - 1:
                tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
                offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
            else:
                offsets = np.zeros((self.num_segments,))
            return offsets + 1

    def _get_test_indices(self, record):
        if self.dense_sample:
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_list = np.linspace(0, sample_pos - 1, num=10, dtype=int)
            offsets = []
            for start_idx in start_list.tolist():
                offsets += [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        elif self.twice_sample:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)

            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)] +
                               [int(tick * x) for x in range(self.num_segments)])

            return offsets + 1
        else:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
            return offsets + 1

    def __getitem__(self, index):
        record = self.video_list[index]
        # check this is a legit video folder

        if self.image_tmpl == 'flow_{}_{:05d}.jpg':
            file_name = self.image_tmpl.format('x', 1)
            full_path = os.path.join(self.root_path, record.path, file_name)
        elif self.image_tmpl == '{:06d}-{}_{:05d}.jpg':
            file_name = self.image_tmpl.format(int(record.path), 'x', 1)
            full_path = os.path.join(self.root_path, '{:06d}'.format(int(record.path)), file_name)
        else:
            file_name = self.image_tmpl.format(1)
            full_path = os.path.join(self.root_path, record.path, file_name)

        while not os.path.exists(full_path):
            print('################## Not Found:', os.path.join(self.root_path, record.path, file_name))
            index = np.random.randint(len(self.video_list))
            record = self.video_list[index]
            if self.image_tmpl == 'flow_{}_{:05d}.jpg':
                file_name = self.image_tmpl.format('x', 1)
                full_path = os.path.join(self.root_path, record.path, file_name)
            elif self.image_tmpl == '{:06d}-{}_{:05d}.jpg':
                file_name = self.image_tmpl.format(int(record.path), 'x', 1)
                full_path = os.path.join(self.root_path, '{:06d}'.format(int(record.path)), file_name)
            else:
                file_name = self.image_tmpl.format(1)
                full_path = os.path.join(self.root_path, record.path, file_name)

        if not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)
        return self.get(record, segment_indices, index)

    def get(self, record, indices, index):

        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                seg_imgs = self._load_image(record.path, p)
                images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1

        process_data = self.transform(images)
        return process_data, record.label, record.groundtruth, index

    def __len__(self):
        return len(self.video_list)

    def adjust_split(self, new_split):
        assert new_split.shape[0] == len(self.raw_video_list), "{} not equal {}".format(new_split.shape[0], len(self.raw_video_list))
        self.video_list = []
        for i in range(new_split.shape[0]):
            if new_split[i]:
                self.video_list.append(self.raw_video_list[i])

        print('video number:%d' % (len(self.video_list)))
    
    def adjust_segments(self, seg):
        print('segments in dataloader is adjusted to {}'.format(seg))
        self.num_segments = seg

    def adjust_label(self, new_label):
        assert new_label.shape[0] == len(self.raw_video_list)
        self.video_list = []
        for i in range(new_label.shape[0]):
            if new_label[i] != -1:
                self.video_list.append(self.raw_video_list[i])
                self.video_list[-1]._data[2] = int(new_label[i])
        print('video number:%d' % (len(self.video_list)))
        print("relabel dataset finished!")

        
