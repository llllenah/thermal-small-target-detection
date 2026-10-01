"""Patches for BasicIRSTD (github.com/XinyiYing/BasicIRSTD), run once from its folder.

1. net.py: drop a broken import from skimage test files (crashes on new skimage).
2. train.py:
   * checkpoint every --save_every epochs (default 10, original 50) so Colab disconnects lose little;
   * LR drops at 60% and 85% of --nEpochs (original: fixed epochs 200/300, never reached at 100 epochs);
   * --lr sets the Adam learning rate (original hard-codes 5e-4), needed for fine-tuning;
   * --pretrained loads weights from any path (original checked the name and then loaded the wrong
     variable), used to fine-tune from our own best checkpoint on a new dataset;
   * torch.load(..., weights_only=False) for newer PyTorch.
"""
t = open('train.py').read()
rep = [
    ("opt.scheduler_settings = {'epochs':400, 'step': [200, 300], 'gamma': 0.1}",
     "opt.scheduler_settings = {'epochs':400, 'step': [max(1, int(opt.nEpochs*0.6)-epoch_state), max(2, int(opt.nEpochs*0.85)-epoch_state)], 'gamma': 0.1}"),
    ("opt.optimizer_settings = {'lr': 5e-4}", "opt.optimizer_settings = {'lr': opt.lr}"),
    ("(idx_epoch + 1) % 50 == 0", "(idx_epoch + 1) % opt.save_every == 0"),
    ("(idx_epoch + 1) % 50 != 0", "(idx_epoch + 1) % opt.save_every != 0"),
    ("""        for pretrained_pth in opt.pretrained:
            if opt.dataset_name in pretrained_pth and opt.model_name in pretrained_pth:
                ckpt = torch.load(resume_pth)
                net.load_state_dict(ckpt['state_dict'])""",
     """        for pretrained_pth in opt.pretrained:
            ckpt = torch.load(pretrained_pth, weights_only=False)
            net.load_state_dict(ckpt['state_dict'])
            print('loaded pretrained weights from', pretrained_pth)"""),
    ("torch.load(resume_pth)", "torch.load(resume_pth, weights_only=False)"),
    ("torch.load(save_pth)", "torch.load(save_pth, weights_only=False)"),
    ('parser.add_argument("--seed"',
     'parser.add_argument("--save_every", type=int, default=10)\n'
     'parser.add_argument("--lr", type=float, default=5e-4)\n'
     'parser.add_argument("--seed"'),
]
for a, b in rep:
    assert a in t, 'pattern not found: ' + a[:60]
    t = t.replace(a, b)
open('train.py', 'w').write(t)
n = open('net.py').read().replace('from skimage.feature.tests.test_orb import img\n', '')
open('net.py', 'w').write(n)
print('BasicIRSTD patched')
