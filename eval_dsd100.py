import argparse
import os

import librosa
import museval
import numpy as np
import torch

from lib import nets
from lib import spec_utils

import inference


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', '-g', type=int, default=-1)
    p.add_argument('--pretrained_model', '-P', type=str, default='models/baseline.pth')
    p.add_argument('--input', '-i', required=True)
    p.add_argument('--sr', '-r', type=int, default=44100)
    p.add_argument('--n_fft', '-f', type=int, default=2048)
    p.add_argument('--hop_length', '-H', type=int, default=1024)
    p.add_argument('--batchsize', '-B', type=int, default=4)
    p.add_argument('--cropsize', '-c', type=int, default=256)
    p.add_argument('--output_image', '-I', action='store_true')
    p.add_argument('--postprocess', '-p', action='store_true')
    p.add_argument('--tta', '-t', action='store_true')
    args = p.parse_args()

    device = torch.device('cpu')
    model = nets.CascadedNet(args.n_fft, args.hop_length)
    model.load_state_dict(torch.load(args.pretrained_model, map_location=device))
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device('cuda:{}'.format(args.gpu))
        model.to(device)

    sp = inference.Separator(model, device, args.batchsize, args.cropsize, args.postprocess)

    SDR_list = []
    ISR_list = []
    SIR_list = []
    SAR_list = []

    dirs = os.listdir(args.input)
    for i, d in enumerate(dirs):
        print('{}: {}'.format(i + 1, d))

        bass_path = os.path.join(args.input, d, 'bass.wav')
        bass, sr = librosa.load(
            bass_path, sr=args.sr, mono=False, dtype=np.float32, res_type='kaiser_fast')

        drums_path = os.path.join(args.input, d, 'drums.wav')
        drums, sr = librosa.load(
            drums_path, sr=args.sr, mono=False, dtype=np.float32, res_type='kaiser_fast')

        other_path = os.path.join(args.input, d, 'other.wav')
        other, sr = librosa.load(
            other_path, sr=args.sr, mono=False, dtype=np.float32, res_type='kaiser_fast')

        vocal_path = os.path.join(args.input, d, 'vocals.wav')
        vocal, sr = librosa.load(
            vocal_path, sr=args.sr, mono=False, dtype=np.float32, res_type='kaiser_fast')

        y = bass + drums + other
        X = y + vocal

        if X.ndim == 1:
            # mono to stereo
            X = np.asarray([X, X])

        X_spec = spec_utils.wave_to_spectrogram(X, args.hop_length, args.n_fft)

        if args.tta:
            y_spec, v_spec = sp.separate_tta(X_spec)
        else:
            y_spec, v_spec = sp.separate(X_spec)

        est_y = spec_utils.spectrogram_to_wave(y_spec, hop_length=args.hop_length)
        # sf.write('{}_Instruments.wav'.format(d), est_y.T, sr)

        est_v = spec_utils.spectrogram_to_wave(v_spec, hop_length=args.hop_length)
        # sf.write('{}_Vocals.wav'.format(d), est_v.T, sr)

        references = np.asarray([y, vocal]).transpose(0, 2, 1)
        estimates = np.asarray([est_y, est_v]).transpose(0, 2, 1)

        SDR, ISR, SIR, SAR = museval.evaluate(references, estimates)

        SDR = np.nanmean(SDR, axis=1)
        ISR = np.nanmean(ISR, axis=1)
        SIR = np.nanmean(SIR, axis=1)
        SAR = np.nanmean(SAR, axis=1)

        SDR_list.append(SDR)
        ISR_list.append(ISR)
        SIR_list.append(SIR)
        SAR_list.append(SAR)

        print('SDR:', SDR)
        print('ISR:', ISR)
        print('SIR:', SIR)
        print('SAR:', SAR)

    print('* Summary')
    print('SDR:', np.mean(SDR_list, axis=0))
    print('ISR:', np.mean(ISR_list, axis=0))
    print('SIR:', np.mean(SIR_list, axis=0))
    print('SAR:', np.mean(SAR_list, axis=0))


if __name__ == '__main__':
    main()
