import torch
from torch import nn
import torch.nn.functional as F

from .template_matching import TemplateMatching
from .regression_head import Decoder_model, ObjectnessHead, BboxesHead
from .encoders import build_encoder

class matching_net(nn.Module):
    def __init__(self, backbone, args):
        super(matching_net, self).__init__()

        self.args = args
        self.emb_dim = args.emb_dim
        self.fusion = args.fusion
        self.box_reg = not args.ablation_no_box_regression
        self.encoder = build_encoder(args)(backbone, args.emb_dim)
        self.decoder_model = Decoder_model

        self.feature_upsample = args.feature_upsample

        if args.no_matcher:
            self.matcher = None
        else:
            self.matcher = TemplateMatching(args.template_type, args.squeeze)

        if isinstance(self.encoder.num_channels, list):
            self.input_proj = nn.ModuleList([nn.Conv2d(channel, self.emb_dim, kernel_size=1) for channel in self.encoder.num_channels])
        else:
            self.input_proj = nn.ModuleList([nn.Conv2d(self.encoder.num_channels, self.emb_dim, kernel_size=1)])

        decoder_num_layer = args.decoder_num_layer
        decoder_kernel_size = args.decoder_kernel_size
        if args.squeeze:
            self.decoder_o = self.decoder_model(1 + self.emb_dim if self.fusion else 1, decoder_num_layer, decoder_kernel_size)
            self.decoder_b = self.decoder_model(1 + self.emb_dim if self.fusion else 1, decoder_num_layer, decoder_kernel_size) if self.box_reg else None
        else:
            self.decoder_o = self.decoder_model(2 * self.emb_dim if self.fusion else self.emb_dim, decoder_num_layer, decoder_kernel_size)
            self.decoder_b = self.decoder_model(2 * self.emb_dim if self.fusion else self.emb_dim, decoder_num_layer, decoder_kernel_size) if self.box_reg else None

        self.objectness_head = ObjectnessHead(self.decoder_o.out_channels)
        self.ltrbs_head = BboxesHead(self.decoder_b.out_channels) if self.box_reg else None

    def _encode_and_project(self, sample):
        """Shared encoder + projection + optional upsample."""
        f = self.encoder(sample)
        if not isinstance(f, list):
            f = [f]
        if self.feature_upsample:
            f = [F.interpolate(f_, scale_factor=2, mode='bilinear', align_corners=False) for f_ in f]
        fps = [self.input_proj[i](f[i]) for i in range(len(f))]
        return f, fps

    def _heads(self, fp, f_TM):
        """Run fusion + decoder + objectness/regression heads on one feature level."""
        if self.fusion:
            f_cat = torch.cat([fp, f_TM], dim=1)
        else:
            f_cat = f_TM

        if self.box_reg:
            f_box = self.decoder_b(f_cat)
            b = self.ltrbs_head(f_box)
        else:
            b = None

        f_obj = self.decoder_o(f_cat)
        o = self.objectness_head(f_obj)
        return o, b, F.relu(f_TM)

    def forward(self, sample, exemplars, **kwargs):

        f, fps = self._encode_and_project(sample)

        os, bs, f_TMs = [], [], []
        for i in range(len(f)):
            fp = fps[i]

            if self.matcher is None:
                f_TM = fp
            else:
                f_TM = self.matcher(fp, exemplars)

            o, b, f_TM_relu = self._heads(fp, f_TM)
            os.append(o)
            bs.append(b)
            f_TMs.append(f_TM_relu)

        return os, bs, f_TMs, f[0]

    def extract_templates(self, sample, exemplars):
        """Phase 1 of tiled inference: extract template features from a crop
        that contains the exemplar. Returns a list (per feature level) of
        lists (per batch item) of template tensors."""
        if self.matcher is None:
            return None

        _, fps = self._encode_and_project(sample)
        templates_per_level = []
        for fp in fps:
            templates = self.matcher.extract_template_only(fp, exemplars)
            templates_per_level.append(templates)
        return templates_per_level

    def forward_with_templates(self, sample, templates_per_level):
        """Phase 2 of tiled inference: run matching + heads using
        pre-extracted templates. `templates_per_level` is the output of
        `extract_templates()`.  Returns the same tuple as `forward()`."""
        f, fps = self._encode_and_project(sample)

        os, bs, f_TMs = [], [], []
        for i in range(len(f)):
            fp = fps[i]

            if self.matcher is None or templates_per_level is None:
                f_TM = fp
            else:
                f_TM = self.matcher.match_with_template(fp, templates_per_level[i])
                f_TM = f_TM * self.matcher.scale

            o, b, f_TM_relu = self._heads(fp, f_TM)
            os.append(o)
            bs.append(b)
            f_TMs.append(f_TM_relu)

        return os, bs, f_TMs, f[0]