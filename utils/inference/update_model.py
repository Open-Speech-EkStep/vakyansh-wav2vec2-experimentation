import torch
import argparse
import os


def update_pretrained_model_path(finetuned, pretrained_path):
    model = torch.load(finetuned)
    model['cfg']['model'].w2v_path = pretrained_path
    name = finetuned.split('/')[-1]
    path = "/".join(finetuned.split('/')[:-1])
    new_model_path = path+'/finetuned_new.pt'
    torch.save(model, new_model_path)
    cmd ="mv " + new_model_path + " " + finetuned
    os.system(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert default models to combined model')
    parser.add_argument('-f', '--finetuned_model', type=str, help="Fine-tuned Model path")
    parser.add_argument('-p', '--pretrained_model', type=str, help="Pre-tuned Model path")
    args_local = parser.parse_args()

    update_pretrained_model_path(finetuned=args_local.finetuned_model, 
                                 pretrained_path = args_local.pretrained_model)
