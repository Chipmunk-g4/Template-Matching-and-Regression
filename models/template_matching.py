import math
import torch
import torch.nn.functional as F

from torch import nn
from torchvision.ops import roi_align

class TemplateMatching(nn.Module):
    def __init__(self, template_type, squeeze = False):
        super(TemplateMatching, self).__init__()

        self.squeeze = squeeze
        self.scale = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))

        template_types = {
            'roi_align': self.extract_template,
            'prototype': self.extract_prototype
        }
        self.extract_function = template_types[template_type]
        self.matching_algorithm = self.cross_correlation
    
    def cross_correlation(self, feature, template):
        # num of pixels in matching patch
        bs, c, h, w = template.shape
        epsilon = 1e-14

        # cross correlation
        feature = feature.flatten(0,1).unsqueeze(0)
        template = template.flatten(0,1)[:,None,...]
        f = F.conv2d(feature, template, bias=None, groups=template.size(0)) / (h*w + epsilon)

        # Squeeze dimension setting
        if self.squeeze:
            f = torch.sum(f, dim=1, keepdim=True)

        # Restore feature shape
        _, _, ph, pw = template.shape
        ph, pw = ph//2, pw//2
        f = F.pad(f, (pw,pw,ph,ph))
        return f

    def extract_prototype(self, f, exemplar_coord):
        _, _, Hf, Wf = f.shape
        x1, y1, x2, y2 = exemplar_coord
        x1, y1 = min(1., max(0., x1)), min(1., max(0., y1))
        x2, y2 = min(1., max(0., x2)), min(1., max(0., y2))

        x1, x2 = math.floor(x1*Wf), math.ceil(x2*Wf)
        y1, y2 = math.floor(y1*Hf), math.ceil(y2*Hf)

        prototype = self.avg_pool(f[:,:,y1:y2, x1:x2])
        return prototype
    
    def extract_template(self, f, exemplar_coord):
        _, _, Hf, Wf = f.shape
        x1, y1, x2, y2 = exemplar_coord
        x1, y1 = min(1., max(0., x1)), min(1., max(0., y1))
        x2, y2 = min(1., max(0., x2)), min(1., max(0., y2))

        x1, x2 = x1*Wf, x2*Wf
        y1, y2 = y1*Hf, y2*Hf
        now_roi = [torch.tensor([[x1, y1, x2, y2]], dtype = exemplar_coord.dtype, device = exemplar_coord.device)]

        # Minimum template that includes the exemplar
        xt1, xt2 = math.floor(x1), math.ceil(x2)
        yt1, yt2 = math.floor(y1), math.ceil(y2)

        # determine_template_size
        Wt = xt2 - xt1
        Ht = yt2 - yt1
        if Wt % 2 == 0: Wt -= 1
        if Ht % 2 == 0: Ht -= 1

        template = roi_align(f, now_roi, (Ht, Wt), aligned=True)
        return template
    
    # Template Matching process
    def matcher(self, sample, exemplars):
        bs, _, H, W = sample.shape

        matching_score_maps = []
        for B in range(bs):
            now_f = sample[B].unsqueeze(0)
            exemplar_coord = exemplars[B][0]

            template = self.extract_function(now_f, exemplar_coord)
            now_f = self.matching_algorithm(now_f, template)

            matching_score_maps.append(now_f)
        f = torch.concat(matching_score_maps, dim=0)

        return f

    def extract_template_only(self, sample, exemplars):
        """Extract template features from a sample containing the exemplar.
        Returns a list of template tensors (one per batch item).
        Used for two-phase tiled inference."""
        bs, _, H, W = sample.shape

        templates = []
        for B in range(bs):
            now_f = sample[B].unsqueeze(0)
            exemplar_coord = exemplars[B][0]
            template = self.extract_function(now_f, exemplar_coord)
            templates.append(template)
        return templates

    def match_with_template(self, sample, templates):
        """Run template matching using pre-extracted template tensors.
        `templates` is a list of template tensors (one per batch item).
        Used for two-phase tiled inference."""
        bs, _, H, W = sample.shape

        matching_score_maps = []
        for B in range(bs):
            now_f = sample[B].unsqueeze(0)
            now_f = self.matching_algorithm(now_f, templates[B])
            matching_score_maps.append(now_f)
        f = torch.concat(matching_score_maps, dim=0)
        return f

    def forward(self, feature, exemplars):
        f = self.matcher(feature, exemplars)
        f = f * self.scale

        return f