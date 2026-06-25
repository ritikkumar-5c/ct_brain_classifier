"""
xai/gradcampp.py
Grad-CAM++ (Chattopadhyay et al., 2018) on a convolutional layer of the MaxViT
backbone. The paper (Sec 6.3) found the deep conv layer
`stages.3.blocks.1.conv.conv2_kxk` produced the most clinically focused maps.

Because the task is study-level MIL, we generate the map for an individual slice
by running a single-slice bag through the full model and back-propagating the
target-class (abnormal) study logit to that slice's conv activations. Pair this
with the MIL attention weights (which slices mattered) for full interpretability.
"""
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm


def _get_module(root, dotted):
    mod = root
    for name in dotted.split("."):
        mod = mod[int(name)] if name.isdigit() else getattr(mod, name)
    return mod


class GradCAMpp:
    def __init__(self, model, target_layer_name):
        self.model = model
        self.layer = _get_module(model.backbone, target_layer_name)
        self.acts = None
        self.grads = None
        self._fh = self.layer.register_forward_hook(self._fwd)
        self._bh = self.layer.register_full_backward_hook(self._bwd)

    def _fwd(self, _m, _i, out):
        self.acts = out.detach()

    def _bwd(self, _m, _gi, gout):
        self.grads = gout[0].detach()

    def remove(self):
        self._fh.remove(); self._bh.remove()

    @torch.enable_grad()
    def __call__(self, slice_tensor, target_class=1):
        """slice_tensor: 3xHxW -> cam HxW in [0,1] (numpy)."""
        self.model.zero_grad(set_to_none=True)
        bag = slice_tensor.unsqueeze(0).unsqueeze(0)              # 1 x 1 x 3 x H x W
        mask = torch.ones(1, 1, dtype=torch.bool, device=slice_tensor.device)
        logits = self.model(bag, mask)                            # 1 x C
        score = logits[0, target_class]
        score.backward()

        A = self.acts[0]                                          # C x h x w
        g = self.grads[0]                                         # C x h x w
        g2, g3 = g ** 2, g ** 3
        sum_a = A.sum(dim=(1, 2), keepdim=True)                   # C x 1 x 1
        denom = 2 * g2 + sum_a * g3
        alpha = g2 / (denom + 1e-7)
        weights = (alpha * F.relu(g)).sum(dim=(1, 2))             # C
        cam = F.relu((weights[:, None, None] * A).sum(0))         # h x w
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-7)
        cam = F.interpolate(cam[None, None], size=slice_tensor.shape[1:],
                            mode="bilinear", align_corners=False)[0, 0]
        return cam.cpu().numpy()


def denormalize(slice_tensor, mean, std):
    """Undo ImageNet normalization -> HxWx3 uint8 for display."""
    t = slice_tensor.detach().cpu().clone()
    for c in range(3):
        t[c] = t[c] * std[c] + mean[c]
    img = (t.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return img


def overlay(img_uint8, cam, alpha=0.45):
    """Blend a jet heatmap of `cam` over the grayscale CT slice."""
    heat = (cm.jet(cam)[..., :3] * 255).astype(np.uint8)         # HxWx3
    blended = (alpha * heat + (1 - alpha) * img_uint8).astype(np.uint8)
    return blended                                                # HxWx3 uint8
