import argparse
import os

shapenet_classes = ['ashtray0', 'bag0', 'bottle0', 'bottle1', 'bottle3', 
                    'bowl0', 'bowl1', 'bowl2', 'bowl3', 'bowl4', 
                    'bowl5', 'bucket0', 'bucket1', 'cap0', 'cap3', 
                    'cap4', 'cap5', 'cup0', 'cup1', 'eraser0', 
                    'headset0', 'headset1', 'helmet0', 'helmet1', 'helmet2', 
                    'helmet3', 'jar0', 'microphone0', 'shelf0', 'tap0', 
                    'tap1', 'vase0', 'vase1', 'vase2', 'vase3', 
                    'vase4', 'vase5', 'vase7', 'vase8', 'vase9']

real3dad_classes = ['airplane','car','candybar','chicken',
                   'diamond','duck','fish','gemstone',
                   'seahorse','shell','starfish','toffees']

def main(args):

    if args.dataset == 'shapenet':
        result_path = './results/shapenet/'
        for cate in shapenet_classes:
            if (args.vis):
                cmd = f"python test_shapenet.py --gpu {args.gpu} --seed {args.seed} --memory_size {args.memory_size} --anomaly_scorer_num_nn {args.anomaly_scorer_num_nn} --faiss_on_gpu --vis --snapshots './output_shapenet/' --dataset_name {cate} --result_path {result_path} --faiss_num_workers 16 sampler -p 0.1 approx_greedy_coreset"
            else:
                cmd = f"python test_shapenet.py --gpu {args.gpu} --seed {args.seed} --memory_size {args.memory_size} --anomaly_scorer_num_nn {args.anomaly_scorer_num_nn} --faiss_on_gpu --snapshots './output_shapenet/' --dataset_name {cate} --result_path {result_path} --faiss_num_workers 16 sampler -p 0.1 approx_greedy_coreset"
            os.system(cmd)
            
    elif args.dataset == 'real3dad':
        result_path = './results/real3dad/'
        for cate in real3dad_classes:
            if (args.vis):
                cmd = f"python test_real3dad.py --gpu {args.gpu} --seed {args.seed} --memory_size {args.memory_size} --anomaly_scorer_num_nn {args.anomaly_scorer_num_nn} --faiss_on_gpu --vis --snapshots './output_real3dad/' --dataset_name {cate} --result_path {result_path} --faiss_num_workers 16 sampler -p 0.1 approx_greedy_coreset"
            else:
                cmd = f"python test_real3dad.py --gpu {args.gpu} --seed {args.seed} --memory_size {args.memory_size} --anomaly_scorer_num_nn {args.anomaly_scorer_num_nn} --faiss_on_gpu --snapshots './output_real3dad/' --dataset_name {cate} --result_path {result_path} --faiss_num_workers 16 sampler -p 0.1 approx_greedy_coreset"
            os.system(cmd)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument('--vis', action='store_true', default=False)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--memory_size', type=int, default=10000)
    parser.add_argument("--anomaly_scorer_num_nn", type=int, default=1)
    args = parser.parse_args()
    main(args)