# Dependencies and source material

This repository packages workflow code and configuration. Model weights, upstream packages and example footage are obtained separately; their own terms continue to apply. No blanket license for those assets is granted by this repository. A public open-source license for the repository's own orchestration code has not been selected.

| Component | Upstream source / terms |
|---|---|
| LTX-2.5 base weights | [Model card and LTX-2.x Community License links](https://huggingface.co/Lightricks/LTX-2.5) |
| Alpha Gen IC-LoRA | [Official model card](https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Alpha-Gen) |
| ComfyUI core | [Comfy-Org/ComfyUI](https://github.com/Comfy-Org/ComfyUI); pinned dependency information in `config/dependencies.json` |
| ComfyUI LTX nodes | [Lightricks/ComfyUI-LTXVideo](https://github.com/Lightricks/ComfyUI-LTXVideo), [upstream LICENSE](https://github.com/Lightricks/ComfyUI-LTXVideo/blob/master/LICENSE) |
| VideoHelperSuite | [Kosinkadink/ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) |
| ViTMatte implementation | [hustvl/ViTMatte](https://github.com/hustvl/ViTMatte), [MIT license](https://github.com/hustvl/ViTMatte/blob/main/LICENSE) |
| ViTMatte Base weights | [Pinned model revision](https://huggingface.co/hustvl/vitmatte-base-composition-1k/tree/bf486d01a7d9e3dbcc8400f7942835caf0eaf76e) |
| PyMatting | [Project](https://github.com/pymatting/pymatting), [MIT license](https://github.com/pymatting/pymatting/blob/master/LICENSE.md) |
| Example source video | Kaboompics, [Close-up of a Woman Posing on a Windy Day, Pexels 8370090](https://www.pexels.com/video/close-up-of-a-woman-posing-on-a-windy-day-8370090/), [Pexels license](https://www.pexels.com/license/) |

The source clip and model files are not redistributed. Access acceptance for the LTX base and Alpha Gen model may be separate. Authenticate using your own local Hugging Face setup; do not put access tokens in a config file or commit them.

After Effects is separately installed, licensed software. FFmpeg, OpenImageIO, PyTorch, Transformers, NumPy, SciPy and other installed packages retain their upstream licenses. No vendored copy of those packages is included.
