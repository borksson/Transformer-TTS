import torch as t
from utils import spectrogram2wav
from scipy.io.wavfile import write
import hyperparams as hp
from text import text_to_sequence
import numpy as np
from network import ModelPostNet, Model
from collections import OrderedDict
from tqdm import tqdm
import argparse
import logging

from spacy.lang.en import English

def load_checkpoint(step, model_name="transformer", device=t.device('mps')):
    state_dict = t.load('./checkpoint/checkpoint_%s_%d.pth.tar'% (model_name, step), map_location=device)   
    new_state_dict = OrderedDict()
    for k, value in state_dict['model'].items():
        key = k[7:]
        new_state_dict[key] = value

    return new_state_dict

def synthesis(text, path, args):
    device = t.device(args.device)
    m = Model()
    m_post = ModelPostNet()

    m.load_state_dict(load_checkpoint(args.restore_step1, "transformer", device))
    m_post.load_state_dict(load_checkpoint(args.restore_step2, "postnet", device))

    text = np.asarray(text_to_sequence(text, [hp.cleaners]))
    text = t.LongTensor(text).unsqueeze(0)
    text = text.to(device)
    mel_input = t.zeros([1,1, 80]).to(device)
    pos_text = t.arange(1, text.size(1)+1).unsqueeze(0)
    pos_text = pos_text.to(device)

    m=m.to(device)
    m_post = m_post.to(device)
    m.train(False)
    m_post.train(False)
    
    pbar = tqdm(range(args.max_len))
    with t.no_grad():
        for i in pbar:
            pos_mel = t.arange(1,mel_input.size(1)+1).unsqueeze(0).to(device)
            mel_pred, postnet_pred, attn, stop_token, _, attn_dec = m.forward(text, mel_input, pos_text, pos_mel)
            mel_input = t.cat([mel_input, mel_pred[:,-1:,:]], dim=1)

            if mel_input.size(1) > hp.window_size:
                mel_input = t.cat((mel_input[:1],mel_input[2:]))

        mag_pred = m_post.forward(postnet_pred)
        print(mag_pred)
        
    wav = spectrogram2wav(mag_pred.squeeze(0).cpu().numpy())
    write(hp.sample_path + path, hp.sr, wav)
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_step1', type=int, help='Global step to restore checkpoint', default=220500)
    parser.add_argument('--restore_step2', type=int, help='Global step to restore checkpoint', default=100000)
    parser.add_argument('--max_len', type=int, help='Synthesis steps', default=400)
    parser.add_argument('--device', type=str, help='device', default="mps")

    args = parser.parse_args()

    with open(hp.input_file, 'r') as f:
        text = f.read()
    
    spacy_en = English()
    spacy_en.add_pipe('sentencizer')
    sentences = spacy_en(text)
    sentences = [str(sentence) for sentence in sentences.sents]
    print(sentences)
    
    for i, sentence in enumerate(sentences): 
        synthesis(sentence, f'/npl_long/test_001_{i}.wav',args)
