import os
import shutil
import sys
from tqdm import tqdm
import argparse
import time
import math
import json
from omegaconf import OmegaConf
from utils.logger import MyLogger, reproduc
from utils.OctTree import OctTreeMLP
from utils.tool import read_img, save_img, get_folder_size
from utils.metrics import eval_performance
from utils.ModelSave import save_tree_models
class CompressFramework:
    def __init__(self, opt, Log) -> None:
        self.opt = opt
        self.Log = Log
        self.compress_opt = opt.CompressFramwork
        self.data_path = self.compress_opt.Path
        self.origin_data = read_img(self.data_path)
            
    def compress(self):
        time_start = time.time()
        time_eval = 0
        tree_mlp = OctTreeMLP(self.compress_opt)
        f_structure = open(os.path.join(self.Log.info_dir,'structure.txt'), 'w+')
        for key in tree_mlp.net_structure:
            f_structure.write('*'*12+key+'*'*12+'\n')
            f_structure.write(str(tree_mlp.net_structure[key])+'\n')
        f_structure.close()
        self.Log.log_metrics({'ratio_set':self.compress_opt.Ratio}, 0)
        self.Log.log_metrics({'ratio_theory':tree_mlp.ratio}, 0)
        sampler = tree_mlp.sampler
        optimizer = tree_mlp.optimizer
        lr_scheduler = tree_mlp.lr_scheduler
        metrics = {'psnr_best':0, 'psnr_epoch':0, 'ssim_best':0, 'ssim_epoch':0, 'acc200_best':0, 'acc200_epoch':0, 'acc500_best':0, 'acc500_epoch':0}
        pbar = tqdm(sampler, desc='Training', leave=True, file=sys.stdout)
        for step, (sampled_idxs, sampled_coords) in enumerate(pbar): 
            optimizer.zero_grad()
            loss = tree_mlp.cal_loss(sampled_idxs, sampled_coords)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            pbar.set_postfix_str("loss={:.6f}".format(loss.item()))
            pbar.update(1)
            if sampler.judge_eval(self.compress_opt.Eval.epochs):
                time_eval_start = time.time()
                predict_data = tree_mlp.predict(device=self.compress_opt.Eval.device, batch_size=self.compress_opt.Eval.batch_size)
                metrics['decode_time'] = time.time()-time_eval_start
                psnr, ssim, acc200, acc500 = eval_performance(self.origin_data, predict_data)
                if psnr > metrics['psnr_best']:
                    metrics['psnr_best'] = psnr
                    metrics['psnr_epoch'] = sampler.epochs_count
                    save_tree_models(tree_mlp=tree_mlp, model_dir=os.path.join(self.Log.compressed_dir, 'models_psnr_best'))
                    save_img(os.path.join(self.Log.decompressed_dir, 'decompressed_psnr_best.tif'), predict_data)
                if ssim > metrics['ssim_best']:
                    metrics['ssim_best'] = ssim
                    metrics['ssim_epoch'] = sampler.epochs_count
                    save_tree_models(tree_mlp=tree_mlp, model_dir=os.path.join(self.Log.compressed_dir, 'models_ssim_best'))
                    save_img(os.path.join(self.Log.decompressed_dir, 'decompressed_ssim_best.tif'), predict_data)
                if acc200 > metrics['acc200_best']:
                    metrics['acc200_best'] = acc200
                    metrics['acc200_epoch'] = sampler.epochs_count
                if acc500 > metrics['acc500_best']:
                    metrics['acc500_best'] = acc500
                    metrics['acc500_epoch'] = sampler.epochs_count
                self.Log.log_metrics({'psnr':psnr,'ssim':ssim,'acc200':acc200,'acc500':acc500}, sampler.epochs_count)
                time_eval += (time.time() - time_eval_start)
        model_dir = os.path.join(self.Log.compressed_dir, 'models')
        save_tree_models(tree_mlp=tree_mlp, model_dir=model_dir)
        predict_path = os.path.join(self.Log.decompressed_dir, 'decompressed.tif')
        save_img(predict_path, predict_data)
        ratio_actual = os.path.getsize(self.data_path)/get_folder_size(model_dir)
        self.Log.log_metrics({'ratio_actual':ratio_actual}, 0)
        metrics['ratio_set'], metrics['ratio_theory'], metrics['ratio_actual'] = self.compress_opt.Ratio, tree_mlp.ratio, ratio_actual
 
        compress_time = int(time.time()-time_start-time_eval)
        print('Compression time: {}s={:.2f}min={:.2f}h'.format(compress_time, compress_time/60, compress_time/3600))
        metrics['time'] = compress_time
        with open(os.path.join(self.Log.info_dir,'metrics.json'), 'w+') as f_metrics:
            json.dump(metrics, f_metrics)
        f_metrics.close()
        self.Log.log_metrics({'time':compress_time}, 0)
        self.Log.close()

def main():
    opt = OmegaConf.load(args.p)
    Log = MyLogger(**opt['Log'])
    shutil.copy(args.p, Log.script_dir)
    shutil.copy(__file__, Log.script_dir)
    reproduc(opt["Reproduc"])

    compressor = CompressFramework(opt, Log)
    compressor.compress()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='single task for compression')
    parser.add_argument('-p', type=str, default='opt/SingleTask/default.yaml', help='config file path')
    parser.add_argument('-g', help='availabel gpu list', default='0,1,2,3',
                        type=lambda s: [int(item) for item in s.split(',')])
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in args.g])
    main()