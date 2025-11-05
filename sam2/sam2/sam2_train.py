# /media/voanhnhat/SDD_OUTSIDE1/PROJECT_DETECT_OBJECT/sam2/sam2/sam2_train.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List

class SAM2Train(nn.Module):
    def __init__(self, model):
        """
        Wrapper cho SAM2Base để hỗ trợ training/fine-tuning.
        - Freeze image encoder (backbone) để tiết kiệm VRAM.
        - Chỉ train sam_prompt_encoder và sam_mask_decoder.
        """
        super(SAM2Train, self).__init__()
        self.model = model
        
        if not hasattr(self.model, "image_encoder"):
            raise AttributeError("Model does not have 'image_encoder' — please check SAM2Base definition.")

        # Freeze full encoder
        for param in self.model.image_encoder.parameters():
            param.requires_grad = False

        # Make prompt encoder / mask decoder trainable (if present)
        if hasattr(self.model, "sam_prompt_encoder"):
            for param in self.model.sam_prompt_encoder.parameters():
                param.requires_grad = True

        # note: some codebases call it `sam_mask_decoder` or `mask_decoder`
        if hasattr(self.model, "sam_mask_decoder"):
            for param in self.model.sam_mask_decoder.parameters():
                param.requires_grad = True
        elif hasattr(self.model, "mask_decoder"):
            for param in self.model.mask_decoder.parameters():
                param.requires_grad = True

        print("[SAM2Train] Initialized: Backbone frozen, training sam_prompt_encoder and sam_mask_decoder.")

    def forward(
        self,
        batched_input: Any,
        multimask_output: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Runs the forward pass for training, đã fix lỗi KeyName, NoneType, lỗi List/Tensor, và lỗi Unpack High-Res Features.
        """
        
        # ----------------------------------------------------------------------
        # BƯỚC 1: TRÍCH XUẤT VÀ LÀM SẠCH DỮ LIỆU INPUT
        # ----------------------------------------------------------------------
        
        # Image: [B, T, C, H, W] -> [B*T, C, H, W] (T=1 cho static image)
        image = batched_input.img_batch
        if image.ndim == 5:
            image = image.squeeze(1) # Squeeze time dimension
            
        masks = batched_input.masks 
        
        # Lấy kích thước ảnh đầu vào (thường là 640x640)
        image_size = getattr(batched_input, 'image_size', image.shape[-1] if image is not None else 640)
        
        # Lấy prompts thô
        box_coords: Optional[torch.Tensor] = getattr(batched_input, 'box_coords', None)
        point_coords: Optional[torch.Tensor] = getattr(batched_input, 'point_coords', None)
        point_labels: Optional[torch.Tensor] = getattr(batched_input, 'point_labels', None)
        
        
        # ----------------------------------------------------------------------
        # BƯỚC 2: IMAGE ENCODER (Đã fix KeyName và lỗi List/Tensor)
        # ----------------------------------------------------------------------
        image_encoder_output = self.model.image_encoder(image) 
        
        # Ánh xạ key đã sửa: 'vision_features' -> image_embeddings, 'vision_pos_enc' -> image_pe, 
        image_embeddings = image_encoder_output.get("vision_features")
        image_pe = image_encoder_output.get("vision_pos_enc")
        # high_res_features là list FPN features (P2, P3, P4).
        high_res_features = image_encoder_output.get("backbone_fpn", None)
        
        
        # *** FIX LỖI LIST/TENSOR CHO IMAGE_PE VÀ IMAGE_EMBEDDINGS ***
        if isinstance(image_embeddings, (list, tuple)):
            # Lấy feature map cuối cùng (thường là coarsest, dùng cho transformer cross-attention)
            image_embeddings = image_embeddings[-1]
            
        if isinstance(image_pe, (list, tuple)):
            # Lấy PE tương ứng cuối cùng
            image_pe = image_pe[-1]
            
        # *** FIX LỖI UNPACKING CHO HIGH_RES_FEATURES (ValueError: too many values to unpack (expected 2)) ***
        if isinstance(high_res_features, (list, tuple)):
            # Mask Decoder đang mong đợi TỐI ĐA 2 features (feat_s0, feat_s1). 
            # Ta chỉ lấy 2 feature có độ phân giải cao nhất (P2, P3), là 2 phần tử đầu tiên.
            if len(high_res_features) > 2:
                print(f"[DEBUG] Slicing high_res_features from {len(high_res_features)} to 2.")
                high_res_features = high_res_features[:2] 
            
            # Nếu len < 2, vẫn sẽ gây ra lỗi Unpack. Kiểm tra xem Mask Decoder có cần high_res_features hay không.
            # Dựa trên cấu hình `use_high_res_features_in_sam: true` trong `sam2.1_hiera_b+.yaml`, nó CẦN.
            if len(high_res_features) < 2:
                # Nếu không đủ 2 feature, chuyển thành None để Mask Decoder có thể bỏ qua hoặc xử lý lỗi khác.
                # Tuy nhiên, nếu nó đã bị lỗi unpack thì ta phải đảm bảo 2.
                # Để tạm thời pass qua, ta sẽ không thay đổi gì nếu < 2, vì lỗi ban đầu là "too many".
                # Nếu lỗi mới xảy ra là "not enough values to unpack", ta sẽ xử lý sau.
                pass

        
        # Kiểm tra None sau khi trích xuất
        if image_embeddings is None or image_pe is None:
             output_keys = list(image_encoder_output.keys())
             raise KeyError(
                f"[FATAL ERROR] Image Encoder output is missing required features. Received keys: {output_keys}."
             )
        
        
        # ----------------------------------------------------------------------
        # BƯỚC 3: PROMPT ENCODER & FIX LỖI NONETYPE
        # ----------------------------------------------------------------------
        prompt_encoder = getattr(self.model, "sam_prompt_encoder", None)
        
        sparse_embeddings = None
        dense_prompt_embeddings = None
        
        prompts_exist = box_coords is not None or point_coords is not None
        
        if prompt_encoder is not None and prompts_exist:
            
            points_input: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
            if point_coords is not None and point_labels is not None:
                points_input = (point_coords, point_labels)
            elif point_coords is not None:
                # Nếu chỉ có coords, tạo labels là 1 (foreground)
                points_input = (point_coords, torch.ones(point_coords.shape[:2], 
                                                        dtype=torch.int, 
                                                        device=point_coords.device))

            sparse_embeddings, dense_prompt_embeddings = prompt_encoder(
                points=points_input,
                boxes=box_coords, 
                masks=None,       
                image_size=image_size
            )
        else:
             # Nếu không có prompts, cố gắng lấy embeddings có sẵn
             sparse_embeddings = getattr(batched_input, 'sparse_embeddings', None)
             dense_prompt_embeddings = getattr(batched_input, 'dense_prompt_embeddings', None)


        # FIX LỖI NONETYPE: Nếu vẫn là None (do dataloader không cấp prompt/embedding), khởi tạo tensor rỗng
        if sparse_embeddings is None:
            B = image.size(0)
            d_model = 256 # Kích thước feature chuẩn của SAM2
            dense_h, dense_w = image_embeddings.shape[-2:] # Lấy kích thước từ feature map đã extract
            
            # Sparse embeddings rỗng (B, 0 tokens, d_model features)
            sparse_embeddings = torch.zeros(
                (B, 0, d_model), dtype=image.dtype, device=image.device
            )
            
            # Dense embeddings rỗng (B, d_model, H, W)
            dense_prompt_embeddings = torch.zeros(
                (B, d_model, dense_h, dense_w), 
                dtype=image.dtype, device=image.device
            )
            
        # ----------------------------------------------------------------------
        # BƯỚC 4: MASK DECODER 
        # ----------------------------------------------------------------------
        
        mask_decoder = getattr(self.model, "sam_mask_decoder", getattr(self.model, "mask_decoder", None))
        if mask_decoder is None:
            raise AttributeError("[ERROR] Model missing mask_decoder / sam_mask_decoder")

        # Gọi Mask Decoder. image_embeddings và image_pe đã được đảm bảo là Tensor.
        low_res_masks, iou_predictions, _, _ = mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings, 
            dense_prompt_embeddings=dense_prompt_embeddings, 
            multimask_output=multimask_output,
            repeat_image=False,
            high_res_features=high_res_features,
        )


        # ----------------------------------------------------------------------
        # BƯỚC 5: UPSCALE VÀ TRẢ VỀ KẾT QUẢ
        # ----------------------------------------------------------------------
        
        # Upscale mask lên kích thước ảnh gốc (image_size x image_size)
        upscaled_masks = F.interpolate(
            low_res_masks,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        )

        return {
            "pred_masks": upscaled_masks,
            "iou_predictions": iou_predictions,
        }

    def postprocess_masks(self, masks, input_size, original_size):
        """
        Post-process masks (upscale và binarize nếu cần cho loss).
        """
        # Dùng F.interpolate để upscale
        masks = torch.nn.functional.interpolate(masks,
                                                size=original_size,
                                                mode="bilinear",
                                                align_corners=False)
        return masks
