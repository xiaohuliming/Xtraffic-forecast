import os
import pickle
import torch
import numpy as np
import threading
import multiprocessing as mp
import gc
import json
from tqdm import tqdm

class DataLoader(object):
    def __init__(self, data, idx, seq_len, horizon, bs, logger, pad_last_sample=False):
        if pad_last_sample:
            num_padding = (bs - (len(idx) % bs)) % bs
            idx_padding = np.repeat(idx[-1:], num_padding, axis=0)
            idx = np.concatenate([idx, idx_padding], axis=0)
        
        self.data = data
        self.idx = idx
        self.size = len(idx)
        self.bs = bs
        self.num_batch = int(self.size // self.bs)
        self.current_ind = 0
        logger.info(f'Samples: {self.size}, Batches: {self.num_batch}')
        
        self.x_offsets = np.arange(-(seq_len - 1), 1, 1)
        self.y_offsets = np.arange(1, (horizon + 1), 1)
        self.seq_len = seq_len
        self.horizon = horizon


    def shuffle(self):
        perm = np.random.permutation(self.size)
        idx = self.idx[perm]
        self.idx = idx


    def write_to_shared_array(self, x, y, idx_ind, start_idx, end_idx):
        for i in range(start_idx, end_idx):
            x[i] = self.data[idx_ind[i] + self.x_offsets, :, :]
            if self.data.shape[-1] == 5:
                y[i] = self.data[idx_ind[i] + self.y_offsets, :, :3]
            else:
                y[i] = self.data[idx_ind[i] + self.y_offsets, :, :1]

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.bs * self.current_ind
                end_ind = min(self.size, self.bs * (self.current_ind + 1))
                idx_ind = self.idx[start_ind: end_ind, ...]

                x_shape = (len(idx_ind), self.seq_len, self.data.shape[1], self.data.shape[-1])
                try:
                    x_shared = mp.RawArray('f', int(np.prod(x_shape)))
                    x = np.frombuffer(x_shared, dtype='f').reshape(x_shape)
                except MemoryError:
                    raise MemoryError(f"Insufficient memory for shared array with shape: {x_shape}")

                y_shape = (len(idx_ind), self.horizon, self.data.shape[1], 1)
                y_shared = mp.RawArray('f', int(np.prod(y_shape)))
                y = np.frombuffer(y_shared, dtype='f').reshape(y_shape)

                array_size = len(idx_ind)
                num_threads = max(len(idx_ind) // 2, 1)
                chunk_size = array_size // num_threads
                threads = []
                for i in range(num_threads):
                    start_index = i * chunk_size
                    end_index = start_index + chunk_size if i < num_threads - 1 else array_size
                    thread = threading.Thread(target=self.write_to_shared_array, args=(x, y, idx_ind, start_index, end_index))
                    thread.start()
                    threads.append(thread)

                for thread in threads:
                    thread.join()

                yield (x, y)
                self.current_ind += 1

        return _wrapper()

class IncidentDataLoader(object):
    def __init__(self, samples, event, bs, logger, x_offsets=None, y_offsets=None, input_dim=None, incidents_sensor=None):
        self.event = event
        self.samples = samples
        self.size = len(samples)
        self.bs = bs
        self.num_batch = max(1, int(self.size // self.bs)) if self.size > 0 else 0
        self.current_ind = 0
        self.input_dim = input_dim
        self.incidents_sensor = incidents_sensor

        first_sample = samples[0]
        self.seq_len = first_sample['x_data'].shape[0]
        self.horizon = first_sample['y_data'].shape[0]
        self.num_nodes = first_sample['x_data'].shape[1]
        self.feature_dim = first_sample['x_data'].shape[2]
        
        if x_offsets is None:
            self.x_offsets = np.arange(-(self.seq_len - 1), 1, 1)
        else:
            self.x_offsets = x_offsets
            
        if y_offsets is None:
            self.y_offsets = np.arange(1, (self.horizon + 1), 1)
        else:
            self.y_offsets = y_offsets
        
        logger.info(f'Samples: {self.size}, Batches: {self.num_batch}')
        logger.info(f'Shape: x=({self.seq_len},{self.num_nodes},{self.input_dim}), y=({self.horizon},{self.num_nodes},1)')

    def shuffle(self):
        indices = np.random.permutation(self.size)
        self.samples = [self.samples[i] for i in indices]

    def write_to_shared_array(self, x, y, batch_samples, start_idx, end_idx):
        # BUGFIX: original code was `batch_samples[i-start_idx]`, which causes every
        # thread to read only the first chunk_size elements of batch_samples. With
        # batch_size=48 and num_threads=24, the effective batch was 2 unique samples
        # duplicated 24x. Correct indexing is `batch_samples[i]`.
        for i in range(start_idx, end_idx):
            sample = batch_samples[i]
            x[i] = sample['x_data'][..., :self.input_dim]
            y[i] = sample['y_data'][..., :1]

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            if self.size == 0:
                return
                
            while self.current_ind < self.num_batch:
                start_ind = self.bs * self.current_ind
                end_ind = min(self.size, self.bs * (self.current_ind + 1))
                batch_samples = self.samples[start_ind:end_ind]
                
                if len(batch_samples) == 0:
                    self.current_ind += 1
                    continue
                batch_size = len(batch_samples)
                feature_dim = self.input_dim
                x_shape = (batch_size, self.seq_len, self.num_nodes, feature_dim)
                if feature_dim == 5:
                    y_shape = (batch_size, self.horizon, self.num_nodes, 3)
                else:
                    y_shape = (batch_size, self.horizon, self.num_nodes, 1)

                try:
                    x_shared = mp.RawArray('f', int(np.prod(x_shape)))
                    x = np.frombuffer(x_shared, dtype='f').reshape(x_shape)
                    
                    y_shared = mp.RawArray('f', int(np.prod(y_shape)))
                    y = np.frombuffer(y_shared, dtype='f').reshape(y_shape)
                except MemoryError:
                    raise MemoryError(f"Insufficient memory for shapes: {x_shape}, {y_shape}")
                
                array_size = batch_size
                num_threads = max(batch_size // 2, 1)
                chunk_size = array_size // num_threads
                threads = []
                
                for i in range(num_threads):
                    start_index = i * chunk_size
                    end_index = start_index + chunk_size if i < num_threads - 1 else array_size
                    
                    thread = threading.Thread(
                        target=self.write_to_shared_array, 
                        args=(x, y, batch_samples, start_index, end_index)
                    )
                    thread.start()
                    threads.append(thread)
                
                for thread in threads:
                    thread.join()
                
                if self.event:
                    event_features_list = []
                    for sample in batch_samples:
                        # incident -> event
                        event_dict = sample['event_features']
                        event_array = [
                            event_dict.get('Event Time', 0.0),
                            event_dict.get('Description', 0),
                            event_dict.get('Type', 0),
                            event_dict.get('Holiday', 0)
                        ]
                        event_features_list.append(event_array)

                    batch_event_data = {
                        'x_data': x,
                        'y_data': y,
                        'incident_features': np.array(event_features_list),
                        'incident_position': np.array([sample['event_position'] for sample in batch_samples]),
                        'incident_distances': np.array([sample['event_distances'] for sample in batch_samples]),
                        'durations': np.array([sample['durations'] for sample in batch_samples]),
                    }
                    
                    if self.incidents_sensor is not None:
                        sensor_type = np.array(self.incidents_sensor['sensor_type'], dtype=np.int64)
                        surface = np.array(self.incidents_sensor['surface'], dtype=np.int64)
                        roadway_use = np.array(self.incidents_sensor['roadway_use'], dtype=np.int64)
                        road_width = np.array(self.incidents_sensor['road_width'], dtype=np.float32)
                        speed_limit = np.array(self.incidents_sensor['speed_limit'], dtype=np.float32)
                        
                        batch_event_data['sensor_data'] = {
                            'sensor_type': torch.tensor(sensor_type, dtype=torch.long).unsqueeze(0).expand(batch_size, -1),
                            'surface': torch.tensor(surface, dtype=torch.long).unsqueeze(0).expand(batch_size, -1),
                            'roadway_use': torch.tensor(roadway_use, dtype=torch.long).unsqueeze(0).expand(batch_size, -1),
                            'road_width': torch.tensor(road_width, dtype=torch.float).unsqueeze(0).expand(batch_size, -1),
                            'speed_limit': torch.tensor(speed_limit, dtype=torch.float).unsqueeze(0).expand(batch_size, -1),
                        }
                    
                    yield batch_event_data
                else:
                    yield (x, y)
                
                self.current_ind += 1
                
                del x_shared, y_shared, x, y
                gc.collect()

        return _wrapper()


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def transform(self, data):
        return (data - self.mean) / self.std


    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def load_dataset(data_path, args, logger):
    dataloader = {}

    sensor_info = None
    if args.use_sensor_info:
        sensor_path = os.path.join(data_path, 'sensor_info.npz')
        if os.path.exists(sensor_path):
            with np.load(sensor_path) as npz:
                sensor_info = {k: np.array(npz[k]) for k in npz.files}
            logger.info(f"Loaded sensor_info with keys: {list(sensor_info.keys())}")
        else:
            logger.warning(f"--use_sensor_info set but {sensor_path} missing; running without sensor info")

    for cat in ['train', 'val', 'test']:
        # incident -> event
        file_name = f"incident_data_{cat}.npy"
        file_path = os.path.join(data_path, file_name)

        samples = np.load(file_path, allow_pickle=True)

        if len(samples) == 0:
            logger.error(f"No samples in {file_path}")
            continue

        first_sample = samples[0]
        x_shape = first_sample['x_data'].shape
        y_shape = first_sample['y_data'].shape
        logger.info(f"{cat}: samples={len(samples)}, x={x_shape}, y={y_shape}")

        dataloader[cat + '_loader'] = IncidentDataLoader(
            samples, args.incident, args.bs, logger, input_dim=args.input_dim, incidents_sensor=sensor_info
        )
    
    stats_path = f'incident_data_stats.npz'
    stats_file = os.path.join(data_path, stats_path)
    
    if os.path.exists(stats_file):
        stats = np.load(stats_file, allow_pickle=True)
        logger.info(f"Stats: mean={stats['mean']}, std={stats['std']}")
        scaler = StandardScaler(mean=stats['mean'], std=stats['std'])
    else:
        logger.warning(f"Stats file not found, using default scaler")
        scaler = StandardScaler(mean=0, std=1)
    
    return dataloader, scaler


def load_adj_from_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def load_adj_from_numpy(numpy_file):
    return np.load(numpy_file)


def get_dataset_info(dataset):
    base_dir = os.getcwd() + '/data/'
    d = {
         'CA': [base_dir+'ca', base_dir+'ca/ca_rn_adj.npy', 8600],
         'GLA': [base_dir+'gla', base_dir+'gla/gla_rn_adj.npy', 3834],
         'GBA': [base_dir+'gba', base_dir+'gba/gba_rn_adj.npy', 2352],
         'SD': [base_dir+'sd', base_dir+'sd/sd_rn_adj.npy', 716],
         'xtraffic': [base_dir+'xtraffic', base_dir+'xtraffic/adj_matrix.npy', 16972],
         'Alameda': [base_dir+'xtraffic/Alameda', base_dir+'xtraffic/Alameda/adj_matrix.npy', 521],
         'Contra_Costa': [base_dir+'xtraffic/Contra_Costa', base_dir+'xtraffic/Contra_Costa/adj_matrix.npy', 496],
         'Orange': [base_dir+'xtraffic/Orange', base_dir+'xtraffic/Orange/adj_matrix.npy', 990]
        }
    assert dataset in d.keys()
    return d[dataset]