import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'common')))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import argparse
import cv2
import glob
# conda_path = '/home/pchennur/.conda/envs/cent7/2020.11-py38/prateek/bin'
# os.environ['PATH'] = f'{conda_path}:{os.environ["PATH"]}'
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
from fvcore.nn import FlopCountAnalysis
import quiver_qis_input_args
import quiver_qis_dataloader
import selections, qis_utils


def main(args):
    os.makedirs(args.plotdir, exist_ok=True)
    os.makedirs(args.save_path, exist_ok=True)
    
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_count = torch.cuda.device_count()
    print('gpu count:', gpu_count)
    print("selected: ", args.device)

    model = selections.model_select(args, args.model_name).to(args.device)
    print(f"spat Model path: {args.weights_path}")
    
    input = torch.ones((1,11,1,256,256)).to(args.device)
    # flops = FlopCountAnalysis(model, input)
    # print('total flops', flops.total())

    checkpoint = torch.load(args.weights_path, map_location=args.device)
    model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint.keys() else checkpoint)
    print('weights initialized with saved model at location: %s' % args.weights_path)

    # if gpu_count > 1:
    #     model = torch.nn.DataParallel(model, device_ids=[i for i in range(gpu_count)]).to(args.device)

    model.eval()

    testset = quiver_qis_dataloader.spadtest_dataloader(args, 
                                                        file_list_path="/media/agarg54/Extreme SSD/dataset_txt_files/full_test_set.txt", 
                                                        bits=3)
    t_dataloader = DataLoader(dataset=testset, num_workers=16, batch_size=1, shuffle=False)
    psnr = test(args, t_dataloader, model)
    return 0

import piq
from tqdm import tqdm
import pyiqa

# from DeQAScore.src import Scorer
MODELmaniqa = pyiqa.create_metric('maniqa', device=torch.device("cuda"))
MODELclipiqa = pyiqa.create_metric('clipiqa', device=torch.device("cuda"))
MODELmusiq = pyiqa.create_metric('musiq', device=torch.device("cuda"))
def compute_no_reference_metrics(out_img):
    _, _, h, w = out_img.shape
    top = (h - 224) // 2
    left = (w - 224) // 2
    out_img = out_img[:, :, top:top+224, left:left+224]

    # ManIQA DeQA MUSIQ ClipIQA
    maniqa_score = MODELmaniqa(out_img).item()
    clipiqa_score = MODELclipiqa(out_img).item()
    musiq_score = MODELmusiq(out_img).item()

    return maniqa_score, clipiqa_score, musiq_score #, deqa_score


def test(args, dataloader, model):
    psnr = 0
    ssim = 0
    lpips = 0
    maniq = 0
    clipiq = 0
    musiq = 0
    count = 0
    fidx = (args.past_frames + args.future_frames + 1) // 2
    with torch.no_grad():
        pbar = tqdm(total=len(dataloader))
        for batch, data in enumerate(dataloader):
            qis_seq, gt_seq = data
            qis_seq = (qis_seq.to(torch.float32)).to(args.device)
            gt_seq = (gt_seq.to(torch.float32)).to(args.device)
            gt_seq = gt_seq[:, args.past_frames:args.num_frames - args.future_frames, ...]
            
            _, out_seq, _, _ = model(qis_seq)
            qis_seq = qis_seq[:, args.past_frames:args.num_frames - args.future_frames, ...]
            
            count = count + (out_seq.shape[0] * out_seq.shape[1])
            
            psnr += qis_utils.batch_psnr(out_seq.clamp(0.0, 1.0), gt_seq.clamp(0.0, 1.0), qis_seq.clamp(0.0, 1.0), data_range=1.0, plotdir=args.plotdir, iteration=batch, visualize=args.visualize)            
            ssim += qis_utils.batch_ssim(out_seq.clamp(0.0, 1.0), gt_seq.clamp(0.0, 1.0), data_range=1)
            lololo = out_seq.size(0)
            for sdofi in range(lololo):
                lpips += piq.LPIPS(reduction='none')(out_seq[sdofi, ...].clamp(0.0, 1.0), gt_seq[sdofi, ...].clamp(0.0, 1.0)).item()
                # print(out_seq[sdofi, ...].repeat(1,3,1,1).size())
                thisMan, thisCLIP, thisMUSIQ = compute_no_reference_metrics(out_seq[sdofi, ...].clamp(0.0, 1.0).repeat(1,3,1,1))
                maniq += thisMan
                clipiq += thisCLIP
                thisMUSIQ += thisMUSIQ
            lpips /= lololo
            maniq /= lololo
            clipiq /= lololo
            musiq /= lololo

            file_name = '%05d'% (batch + fidx)
            if not os.path.exists(os.path.join(args.save_path, args.folder_name + '_gt')):
                os.makedirs(os.path.join(args.save_path, args.folder_name + '_gt'))
                os.makedirs(os.path.join(args.save_path, args.folder_name + '_qis'))
                os.makedirs(os.path.join(args.save_path, args.folder_name + '_out'))

            # np.save(os.path.join(args.save_path, args.folder_name + '_out', file_name + '_quiverout.npy'), out_seq.cpu().numpy())
            out_seq = ((out_seq.clamp(0.0, 1.0).squeeze())).detach().cpu().numpy()
            qis_seq = ((qis_seq.clamp(0.0, 1.0).squeeze())).detach().cpu().numpy()
            gt_seq = ((gt_seq.clamp(0.0, 1.0).squeeze())).detach().cpu().numpy()
            out_seq = (out_seq - np.amin(out_seq)) / np.amax(out_seq)
            qis_seq = (qis_seq - np.amin(qis_seq)) / np.amax(qis_seq)
            gt_seq = (gt_seq - np.amin(gt_seq)) / np.amax(gt_seq)

            out_seq = out_seq * 255
            qis_seq = qis_seq * 255
            gt_seq = gt_seq * 255

            cv2.imwrite(os.path.join(args.save_path, args.folder_name + '_qis', file_name + '_qis.png'),
                        qis_seq.astype(np.uint8))
            cv2.imwrite(os.path.join(args.save_path, args.folder_name + '_gt', file_name + '_gt.png'),
                        gt_seq.astype(np.uint8))
            cv2.imwrite(os.path.join(args.save_path, args.folder_name + '_out', file_name + '_quiverout.png'),
                        out_seq.astype(np.uint8))
            del out_seq
            del qis_seq
            del gt_seq
            pbar.update(1)

    print('psnr: %.3f' % (psnr/count))
    print('ssim: %.4f' % (ssim/count))
    print('lpips: %.4f' % (lpips/count))

    print('--------------------------------')
    print('ManIQA: %.3f' % (maniq/count))
    print('CLIP-IQA: %.4f' % (clipiq/count))
    print('MUSIQ: %.4f' % (musiq/count))

    return psnr/count


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='video denoising parameters')
    quiver_qis_input_args.quiver_testing_args(parser)
    quiver_qis_input_args.sensor_args(parser)
    args = parser.parse_args()
    model = main(args)
