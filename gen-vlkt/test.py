import torch
ckpt = torch.load("G:\HICO_GEN_VLKT_S.pth")
ckpt = torch.load("G:\checkpoint_last.pth")
print(ckpt.keys()) #model optimizer lr_scheduler
print(ckpt['lr_scheduler'])
print(ckpt['optimizer'].keys())