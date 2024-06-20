import h5py
import tqdm

def main():
    seq_dir = '/home/yilong/Documents/mimicgen_envs/datasets/core/data.hdf5'

    with h5py.File(seq_dir, 'r') as f:
        data = f['data']
        for demo in data:
            print(f['data'][demo]['next_obs'].keys())
            print(f['data'][demo]['obs'].keys())

if __name__ == "__main__":
    main()