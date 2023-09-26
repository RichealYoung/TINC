import argparse
import os
from utils.ModelSave import load_tree_models
from utils.tool import save_img

def main():
    model_dir = args.d
    decompressed_path = args.p
    tree_mlp = load_tree_models(model_dir)
    decompressed_data = tree_mlp.predict(device='cuda', batch_size=16*256*256)
    save_img(decompressed_path, decompressed_data)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='single task for decompression')
    parser.add_argument('-d', type=str, default='outputs/default/compressed/models', help='models dir')
    parser.add_argument('-p', type=str, default='outputs/default/decompressed/predict.tif', help='decompressed data path')
    parser.add_argument('-g', help='availabel gpu list', default='0,1,2,3',
                        type=lambda s: [int(item) for item in s.split(',')])
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in args.g])
    main()