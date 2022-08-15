"""
    NeRF helper functions
    @author Enigmatisms @date 2022.4.24
"""
import torch

def saveModel(model, path:str, other_stuff: dict = None, opt = None, amp = None):
    checkpoint = {'model': model.state_dict(),}
    if not amp is None:
        checkpoint['amp'] =  amp.state_dict()
    if not opt is None:
        checkpoint['optimizer'] = opt.state_dict()
    if not other_stuff is None:
        checkpoint.update(other_stuff)
    torch.save(checkpoint, path)
    
def makeMLP(in_chan, out_chan, act = torch.nn.ReLU(), batch_norm = False):
    modules = [torch.nn.Linear(in_chan, out_chan)]
    if batch_norm == True:
        modules.append(torch.nn.BatchNorm1d(out_chan))
    if not act is None:
        modules.append(act)
    return modules

# from pytorch.org https://discuss.pytorch.org/t/finding-source-of-nan-in-forward-pass/51153/2
def nan_hook(self, inp, output):
    if not isinstance(output, tuple):
        outputs = [output]
    else:
        outputs = output

    for i, out in enumerate(outputs):
        nan_mask = torch.isnan(out)
        if nan_mask.any():
            print("In", self.__class__.__name__)
            raise RuntimeError(f"Found NAN in output {i} at indices: ", nan_mask.nonzero(), "where:", out[nan_mask.nonzero()[:, 0].unique(sorted=True)])

def positional_encoding(x:torch.Tensor, freq_level:int) -> torch.Tensor:
    result = []
    for fid in range(freq_level):
        freq = 2. ** fid
        for func in (torch.sin, torch.cos):
            result.append(func(freq * x))
    encoded = torch.cat(result, dim = -1)
    if x.dim() > 2:
        ray_num, point_num = x.shape[0], x.shape[1]
        encoded = encoded.view(ray_num, point_num, -1)
    return encoded

def linear_to_srgb(linear: torch.Tensor, eps: float = None) -> torch.Tensor:
  """From JAX multiNeRF official repo: https://github.com/google-research/multinerf"""
  if eps is None:
    eps = torch.full((1, ), torch.finfo(torch.float32).eps, device = linear.device)
  srgb0 = 323 / 25 * linear
  srgb1 = (211 * torch.maximum(eps, linear)**(5 / 12) - 11) / 200
  return torch.where(linear <= 0.0031308, srgb0, srgb1)