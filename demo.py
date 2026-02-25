import cv2
import torch
import gradio
import argparse
import numpy as np

from torch import nn
from PIL import Image
from gradio_bbox_annotator import BBoxAnnotator

from models import build_model
from utils.TM_utils import Get_pred_boxes, GT_map, NMS
from utils.box_refine import SAM_box_refiner
from models.backbone.sam.sam import Sam_Backbone

import albumentations as A
from albumentations.pytorch import ToTensorV2


def _normalize_transform():
    """ImageNet normalization + to tensor (no resize)."""
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def _resize_and_normalize_transform(size):
    """Resize to `size` x `size`, normalize, and convert to tensor."""
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def config_parser():
    parser = argparse.ArgumentParser(description="TMR Demo")
    
    parser.add_argument('--ckpt', default="", metavar="FILE", help='path to ckpt', required=True)
    parser.add_argument('--port', default=6099, type=int)

    # model setting
    parser.add_argument('--modeltype', type=str, default="matching_net", help='Type of model')

    parser.add_argument('--emb_dim', default=512, type=int, help='Embedding dimension')
    parser.add_argument("--no_matcher", type=bool, default=False, help="If true, we don't use matching module")
    parser.add_argument("--squeeze", type=bool, default=False, help="If true, we use matching feature with channel 1")
    parser.add_argument("--fusion", type=bool, default=True, help="If true, we use a fusion layer to combine the features from the backbone and the template matching module")
    parser.add_argument("--positive_threshold", default=0.5, type=float, help="Threshold for positive samples")
    parser.add_argument("--negative_threshold", default=0.5, type=float, help="Threshold for negative samples")
    parser.add_argument("--NMS_cls_threshold", default=0.7, type=float, help="Threshold for NMS classificaiton score")
    parser.add_argument("--NMS_iou_threshold", default=0.5, type=float, help="Threshold for NMS Iou")
    parser.add_argument("--ablation_no_box_regression", type=bool, default=False, help="If true, we don't regress box parameters. Insted we use template size as box width, height parameter")
    parser.add_argument('--template_type', type=str, default='roi_align', help='template extraction algorithm Type')
    parser.add_argument("--feature_upsample", type=bool, default=True, help="If true, feature upsample for template matching")
    parser.add_argument('--eval_multi_scale', type=bool, default=False, help='multi scale processing for evaluation')
    parser.add_argument('--regression_scaling_imgsize', type=bool, default=False)
    parser.add_argument('--regression_scaling_WH_only', type=bool, default=False)
    parser.add_argument("--focal_loss", type=bool, default=False, help='Flag to use focal loss')

    # model - backbone setting
    parser.add_argument("--backbone", default="sam", type=str, help="Name of the backbone to use")
    parser.add_argument("--encoder", default="original", type=str, help="Name of the encoder type to use")
    parser.add_argument("--dilation", default=True, help="If true, we replace stride with dilation in the last convolutional block (DC5)")

    # model - head setting
    parser.add_argument("--decoder_num_layer", default=1, type=int)
    parser.add_argument("--decoder_kernel_size", default=3, type=int)
    args = parser.parse_args()

    return args

class Inference(nn.Module):
    TILE_SIZE = 1024
    TILE_OVERLAP = 256

    def __init__(self, args):
        super(Inference, self).__init__()

        self.args = args
        self.model = build_model(args)
        self.is_cuda = torch.cuda.is_available()

        self.temp_sam = Sam_Backbone(requires_grad=False, model_type="vit_h")
        self.refiner = SAM_box_refiner()

    @staticmethod
    def generate_tiles(image_np, tile_size=1024, overlap=256):
        """Split *image_np* (H, W, 3) into overlapping tiles of
        ``tile_size x tile_size``.  Returns a list of
        ``(tile_np, offset_x, offset_y, tile_w, tile_h)``.

        Edge tiles are right/bottom-aligned so every pixel is covered.
        All tiles are padded to ``tile_size`` (multiple of 16) if needed.
        """
        H, W, _ = image_np.shape
        step = tile_size - overlap
        tiles = []

        # Compute start positions, ensuring coverage of the full extent
        ys = list(range(0, max(H - tile_size, 0) + 1, step))
        if ys[-1] + tile_size < H:
            ys.append(H - tile_size)
        xs = list(range(0, max(W - tile_size, 0) + 1, step))
        if xs[-1] + tile_size < W:
            xs.append(W - tile_size)

        # Deduplicate (e.g. when image is slightly larger than tile_size)
        ys = sorted(set(ys))
        xs = sorted(set(xs))

        for oy in ys:
            for ox in xs:
                crop = image_np[oy:oy + tile_size, ox:ox + tile_size]
                ch, cw, _ = crop.shape
                # Pad to tile_size if the crop is smaller (image edge)
                if ch < tile_size or cw < tile_size:
                    padded = np.zeros((tile_size, tile_size, 3), dtype=crop.dtype)
                    padded[:ch, :cw] = crop
                    tile_np = padded
                else:
                    tile_np = crop
                tiles.append((tile_np, ox, oy, cw, ch))

        return tiles

    @staticmethod
    def map_preds_to_full(pred_boxes, pred_logits, ref_points,
                          ox, oy, tw, th, full_w, full_h):
        """Convert tile-relative [0,1] predictions to
        full-image-relative [0,1] coordinates.

        pred_boxes are (N, 4) in [x1, y1, x2, y2] format, each in [0, 1]
        relative to the tile whose pixel size is (tw, th) at offset (ox, oy)
        in the full image of size (full_w, full_h).
        """
        tile_size = torch.tensor([tw, th, tw, th],
                                 dtype=pred_boxes.dtype,
                                 device=pred_boxes.device)
        offset = torch.tensor([ox, oy, ox, oy],
                              dtype=pred_boxes.dtype,
                              device=pred_boxes.device)
        full_res = torch.tensor([full_w, full_h, full_w, full_h],
                                dtype=pred_boxes.dtype,
                                device=pred_boxes.device)

        mapped_boxes = (pred_boxes * tile_size + offset) / full_res

        # Map ref_points similarly
        ref_tile = torch.tensor([tw, th],
                                dtype=ref_points.dtype,
                                device=ref_points.device)
        ref_offset = torch.tensor([ox, oy],
                                  dtype=ref_points.dtype,
                                  device=ref_points.device)
        ref_full = torch.tensor([full_w, full_h],
                                dtype=ref_points.dtype,
                                device=ref_points.device)
        mapped_refs = (ref_points * ref_tile + ref_offset) / ref_full

        return mapped_boxes, pred_logits, mapped_refs

    def preprocess(self, image_input):
        """Parse UI input and return:
        - img_url           : path to the image file
        - ori_image_np      : (H, W, 3) uint8 numpy array at original res
        - exemplars_px      : list of [x1,y1,x2,y2] in pixel coords
        """
        img_url = image_input[0]
        exemplars_px = [[int(p[0]), int(p[1]), int(p[2]), int(p[3])]
                        for p in image_input[1]]

        ori_image = Image.open(img_url).convert("RGB")
        ori_image_np = np.array(ori_image)

        return img_url, ori_image_np, exemplars_px

    def _make_exemplar_crop(self, ori_image_np, ex_box_px):
        """Create a 1024×1024 crop centred on the exemplar box.
        Returns the crop tensor and the exemplar's normalised coords
        within that crop (format expected by the model).
        """
        H, W, _ = ori_image_np.shape
        ts = self.TILE_SIZE
        x1, y1, x2, y2 = ex_box_px

        # Centre of the exemplar
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

        # Desired crop region (clamp to image bounds)
        crop_x1 = int(max(cx - ts / 2, 0))
        crop_y1 = int(max(cy - ts / 2, 0))
        crop_x2 = min(crop_x1 + ts, W)
        crop_y2 = min(crop_y1 + ts, H)
        # Re-adjust start if the crop was clamped at the right/bottom
        crop_x1 = max(crop_x2 - ts, 0)
        crop_y1 = max(crop_y2 - ts, 0)

        crop = ori_image_np[crop_y1:crop_y2, crop_x1:crop_x2]
        ch, cw, _ = crop.shape

        # If the image itself is smaller than tile_size, resize to 1024
        if ch < ts or cw < ts:
            crop_tensor = _resize_and_normalize_transform(ts)(image=crop)['image'].unsqueeze(0)
            # Normalised exemplar coords within the resized crop
            norm_ex = torch.tensor([
                (x1 - crop_x1) / cw,
                (y1 - crop_y1) / ch,
                (x2 - crop_x1) / cw,
                (y2 - crop_y1) / ch,
            ], dtype=torch.float32).clamp(0, 1)
        else:
            crop_tensor = _normalize_transform()(image=crop)['image'].unsqueeze(0)
            norm_ex = torch.tensor([
                (x1 - crop_x1) / ts,
                (y1 - crop_y1) / ts,
                (x2 - crop_x1) / ts,
                (y2 - crop_y1) / ts,
            ], dtype=torch.float32).clamp(0, 1)

        if self.is_cuda:
            crop_tensor = crop_tensor.cuda()
            norm_ex = norm_ex.cuda()

        # Wrap exemplar coords in the format expected by the model:
        # list[batch] of tensor(K, 4) where K = number of exemplars
        exemplars = [norm_ex.unsqueeze(0)]
        return crop_tensor, exemplars

    @torch.no_grad()
    def infer(self, image_input, refine_box, *args, **kwargs):
        img_url, ori_image_np, exemplars_px = self.preprocess(image_input)
        full_h, full_w, _ = ori_image_np.shape

        needs_tiling = full_w > self.TILE_SIZE or full_h > self.TILE_SIZE

        all_templates = []   # one entry per exemplar
        all_exemplar_norm = []  # normalised exemplar coords per tile
        for ex_px in exemplars_px:
            crop_tensor, crop_exemplars = self._make_exemplar_crop(ori_image_np, ex_px)
            templates = self.model.extract_templates(crop_tensor, crop_exemplars)
            all_templates.append(templates)

            # Also keep a full-image normalised version of the exemplar
            # (needed for Get_pred_boxes regression scaling)
            norm_ex = torch.tensor([
                ex_px[0] / full_w, ex_px[1] / full_h,
                ex_px[2] / full_w, ex_px[3] / full_h,
            ], dtype=torch.float32)
            if self.is_cuda:
                norm_ex = norm_ex.cuda()
            all_exemplar_norm.append(norm_ex)

        if needs_tiling:
            tiles = self.generate_tiles(ori_image_np,
                                        self.TILE_SIZE,
                                        self.TILE_OVERLAP)
        else:
            # Single-tile path: pad/resize small images to TILE_SIZE
            th, tw = ori_image_np.shape[:2]
            tiles = [(ori_image_np, 0, 0, tw, th)]

        all_logits, all_boxes, all_refs = [], [], []
        dummy = {
            'regression_ablation_b': False,
            'regression_ablation_c': False,
        }

        for tile_np, ox, oy, tw, th in tiles:
            # Normalise tile to tensor
            if tile_np.shape[0] == self.TILE_SIZE and tile_np.shape[1] == self.TILE_SIZE:
                tile_tensor = _normalize_transform()(image=tile_np)['image'].unsqueeze(0)
            else:
                tile_tensor = _resize_and_normalize_transform(self.TILE_SIZE)(image=tile_np)['image'].unsqueeze(0)
            if self.is_cuda:
                tile_tensor = tile_tensor.cuda()

            tile_logits, tile_boxes, tile_refs = [], [], []

            for idx, templates in enumerate(all_templates):
                # Tile-local normalised exemplar (needed for regression).
                # Only the SIZE matters for regression scaling and adaptive
                # kernel selection in Get_pred_boxes.  Compute the exemplar
                # width/height relative to the tile's model-input resolution
                # (TILE_SIZE) and place a synthetic box at the tile centre so
                # that the box is never degenerate — even when the real
                # exemplar lies outside this particular tile.
                full_norm = all_exemplar_norm[idx]
                ex_w_px = (full_norm[2].item() - full_norm[0].item()) * full_w
                ex_h_px = (full_norm[3].item() - full_norm[1].item()) * full_h
                ex_w_norm = ex_w_px / self.TILE_SIZE
                ex_h_norm = ex_h_px / self.TILE_SIZE
                tile_ex = torch.tensor([
                    0.5 - ex_w_norm / 2,
                    0.5 - ex_h_norm / 2,
                    0.5 + ex_w_norm / 2,
                    0.5 + ex_h_norm / 2,
                ], dtype=torch.float32)
                if self.is_cuda:
                    tile_ex = tile_ex.cuda()
                tile_exemplar_wrap = [tile_ex.unsqueeze(0)]

                pred_obj, pred_reg, _, _ = self.model.forward_with_templates(
                    tile_tensor, templates)
                _logits, _boxes, _refs = Get_pred_boxes(
                    pred_obj, pred_reg, tile_exemplar_wrap,
                    dummy, self.args.NMS_cls_threshold, True)

                tile_logits.append(_logits[0])
                tile_boxes.append(_boxes[0])
                tile_refs.append(_refs[0])

            tile_logits = [torch.concat(tile_logits)]
            tile_boxes = [torch.concat(tile_boxes)]
            tile_refs = [torch.concat(tile_refs)]

            # Optional per-tile SAM refinement
            if refine_box:
                backbone_feat = self.temp_sam(tile_tensor)
                tile_logits, tile_boxes, tile_refs = self.refiner(
                    tile_logits, tile_boxes, tile_refs,
                    tile_tensor, backbone_feat)

            # Map to full-image coordinates.
            # For the tiled path the model always operates on a
            # TILE_SIZE × TILE_SIZE canvas (padded if necessary), so its
            # [0,1] output coords span TILE_SIZE pixels, not tw × th.
            # For the non-tiled path the image was *resized* from (tw, th)
            # to TILE_SIZE, so [0,1] correctly maps back to tw × th.
            eff_tw = self.TILE_SIZE if needs_tiling else tw
            eff_th = self.TILE_SIZE if needs_tiling else th
            mapped_boxes, mapped_logits, mapped_refs = self.map_preds_to_full(
                tile_boxes[0], tile_logits[0], tile_refs[0],
                ox, oy, eff_tw, eff_th, full_w, full_h)

            all_logits.append(mapped_logits)
            all_boxes.append(mapped_boxes)
            all_refs.append(mapped_refs)

        pred_logits = [torch.concat(all_logits)]
        pred_boxes = [torch.concat(all_boxes)]
        ref_points = [torch.concat(all_refs)]

        pred_logits, pred_boxes, ref_points = NMS(
            pred_logits, pred_boxes, ref_points,
            self.args.NMS_iou_threshold)

        return self.visualize(img_url, pred_boxes[0].cpu().numpy())

    def visualize(self, img_url, pred_boxes):

        img = cv2.imread(img_url)
        H, W, _ = img.shape

        for box in pred_boxes:
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H)
            thickness = max(2, int(min(W, H) / 500))
            img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), thickness)

        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return img

def main(args):
    Infer = Inference(args)    
    state_dict = torch.load(args.ckpt, map_location='cpu')['state_dict']
    Infer.load_state_dict(state_dict, strict=False)
    Infer.eval()
    if torch.cuda.is_available():
        Infer = Infer.cuda()

    demo = gradio.Blocks()
    image_input = BBoxAnnotator(label="Target Image", categories=["support exemplars"])
    image_output = gradio.components.Image(label="Output Image", type="pil")

    with demo:
        gradio.Markdown("# Template Matching and Regression Demo")
        with gradio.Row():
            with gradio.Column(scale=5.0):
                image_input.render()
                refine_box_checkbox = gradio.Checkbox(label="SAM deocder box refinement")
                with gradio.Row(scale=2.0):
                    clearBtn = gradio.ClearButton(components=[image_input, refine_box_checkbox])
                    runBtn = gradio.Button("Run")
            with gradio.Column(scale=5.0):
                image_output.render()

                example = gradio.Examples(
                    examples=[
                        ["demo/1.jpg"],
                        ["demo/2.jpg"],
                        ["demo/3.jpg"],
                        ["demo/4.jpg"],
                        ["demo/5.jpg"],
                        ["demo/6.jpg"],
                    ],
                    inputs=image_input,
                    cache_examples=False,
                )

        runBtn.click(
            fn=lambda image_input, refine_box: Infer.infer(image_input, refine_box=refine_box),
            inputs=[image_input, refine_box_checkbox],
            outputs=[image_output]
        )

    demo.queue().launch(share=True,server_port=args.port)

if __name__ == "__main__":
    args = config_parser()
    main(args)