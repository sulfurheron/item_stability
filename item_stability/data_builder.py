import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import re
import json
import cv2


class DataGen:
    def __init__(self,
                 datadir="/media/dmitriy/HDD/aruco_box_sim_02",
                 img_dim=(224, 224),
                 target='lin_v',
                 target_frame=10,
                 n_frames=3):
        self._datadir = datadir
        self.img_dim = img_dim
        self._episode_dict = {}
        self.load_data()
        self.build_rgb_dataset(
            target=target,
            target_frame=target_frame,
            n_frames=n_frames
        )

    def load_data(self):
        episodes = os.listdir(self._datadir)
        for k, episode in enumerate(episodes[78:]):
            if episode == "500":
                pass
            ep_dict = {}
            all_files = os.listdir(os.path.join(self._datadir, episode))
            ep_dict['rgb'] = [filename for filename in all_files if 'rgb' in filename]
            ep_dict['json'] = [filename for filename in all_files if 'json' in filename]
            ep_dict['depth'] = [filename for filename in all_files if 'depth' in filename]
            # sort filenames by timestep
            for mode in ep_dict:
                if mode == 'json':
                    ep_dict[mode] = sorted(ep_dict[mode], key=lambda x: int(re.search('img.(.*).info', x).group(1)))
                else:
                    ep_dict[mode] = sorted(ep_dict[mode], key=lambda x: int(re.search('img.(.*).' + mode, x).group(1)))
            self._episode_dict[episode] = ep_dict

    def build_rgb_dataset(self, target_frame=10, n_frames=5, val_split=0.1, target='lin_v'):
        """
        Args:
            target_frame: the timestep for which we make predictions
            n_frames: LSTM history size - how many frames prior to target_frame to use
            val_split: size of validation set
            target: lin_v, ang_v, pos or orn - target vector to predict
        """
        all_seq = []
        labels = []
        for ep, item in self._episode_dict.items():
            img_seq = []
            for i in range(n_frames):
                img = Image.open(os.path.join(self._datadir, ep, item['rgb'][target_frame - n_frames + i]))
                img = np.array(img)
                img = img[50:-150, 200:-200, :]
                img = cv2.resize(img, self.img_dim[:2])
                img = (img/255.0).astype('float32')
                img_seq.append(img)
            img_seq = np.array(img_seq)
            # plt.figure()
            # for j, img in enumerate(img_seq):
            #     plt.subplot(1, 5, j + 1)
            #     plt.imshow(img)
            #     plt.xticks([])
            #     plt.yticks([])
            # plt.show()
            all_seq.append(img_seq)
            info = json.load(open(os.path.join(self._datadir, ep, item['json'][target_frame - 1])))
            labels.append(info[target])
        data = np.array(all_seq)
        labels = np.array(labels)
        self.train_size = int(len(data) * (1 - val_split))
        self.x_train, self.x_val = data[:self.train_size], data[self.train_size:]
        self.y_train, self.y_val = labels[:self.train_size], labels[self.train_size:]

    def generate_batch(self, size, dataset='train'):
        self.progress_in_epoch = 0
        if dataset == "train":
            data, labels = self.x_train, self.y_train
        elif dataset == "val":
            data, labels = self.x_val, self.y_val
        ix  = np.random.permutation(len(data))
        for start in range(0, len(data), size):
            batch_data = data[ix[start: start + size]]
            batch_labels = labels[ix[start: start + size]]
            yield batch_data, batch_labels
            self.progress_in_epoch = start/len(ix)

if __name__ == "__main__":
    d = DataGen()
    d.load_data()
    d.build_rgb_dataset()