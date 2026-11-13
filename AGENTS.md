# LTX roto workflow

- Preserve original RGB, raw LTX alpha and derived alpha separately. Change copies of a run, never its source assets.
- Use a loopback ComfyUI API. Do not upload footage or call a cloud inference service. Record a prompt ID before submission and recover its status instead of blindly resubmitting.
- This profile uses an LTX-2.5 22B BF16 **FHD prior** followed by original-4K ViTMatte F32 refinement and ML foreground reconstruction. Do not describe it as native-4K LTX inference.
- Preserve original frame indices and timestamps. Do not use interpolation to invent matte frames. Trimap seed erosion is separate from final alpha; do not erode, threshold or feather the delivered alpha to hide defects.
- In AE, RGB-only foreground must have no alpha, and each roto branch applies the external alpha once. Coded-RGB EXR input requires verified Preserve RGB interpretation. Keep input and output color conversion settings explicit.
- Use a unique render directory per AE queue item. Native frame numbering can restart at zero; map outputs using the actual source-frame contract before assembling a canonical sequence.
- Technical success is not visual approval. Inspect hair at 1:1 on multiple backgrounds and all-frame motion. Report exactness separately from numerical tolerance.
- Keep credentials, model weights, source footage, EXRs, videos, AEPs, runtime receipts and machine-specific paths out of Git. Keep only small, curated validation summaries.
- Repository changes should use compile/help checks and focused regression tests. Do not start a new GPU run or native render merely to package or document an already validated result.
