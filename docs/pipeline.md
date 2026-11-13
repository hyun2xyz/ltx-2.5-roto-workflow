# Local pipeline recipe

This is the path that produced the selected65-frame shot: LTX-2.5 distilled22B BF16 generates a1920×1088 **prior**, ViTMatte Base F32 refines that prior against the original4096×2160 RGB, and PyMatting reconstructs foreground RGB. It is not native4K LTX generation. Raw LTX RGB/alpha, refined alpha and RGB-only foreground remain separate.

## Install dependencies and obtain models

Use a separate environment or environments; do not alter an unrelated Comfy installation. Install a suitable CUDA-enabled [PyTorch](https://pytorch.org/get-started/locally/) separately, then the Python packages in `requirements.txt`. `config/runtime-versions.json` records the actual two production environments. A new unified installation has not been GPU-tested by the packaging step. FFmpeg/ffprobe must be on PATH or configured explicitly.

The required [ComfyUI](https://github.com/Comfy-Org/ComfyUI), [official LTX nodes](https://github.com/Lightricks/ComfyUI-LTXVideo) and [VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) commits are in `config/dependencies.json`. Install those upstream dependencies in your own Comfy installation and retain their licenses. There are no private exporter/streaming nodes or private skill dependencies. `SaveImageAdvanced`, `SaveConditioning`, `ConditioningLoader` and `GetImageSize` are core nodes at the recorded Comfy commit. The graph needs the recorded streamed-guide/Fusion support; an older installation is not assumed compatible. The CLI checks live node/model registrations with GET requests before a fresh submission.

Obtain these files from the official cards using your own local Hugging Face setup, after accepting any applicable terms. No weights, tokens or upstream source are redistributed:

- [Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5), revision `49b093a39c19169a8a81721dcc541e07e945232f`: distilled-transformer BF16, Gemma4 BF16 text encoder, and `video-vae-bf16` DiffVAE. The `conv` VAE and quantized checkpoints are different profiles.
- [Official Alpha Gen0.9](https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Alpha-Gen): the exact production filename and SHA are in `config/model-provenance.json`.
- [ViTMatte Base pinned revision](https://huggingface.co/hustvl/vitmatte-base-composition-1k/tree/bf486d01a7d9e3dbcc8400f7942835caf0eaf76e): `pytorch_model.bin`, `config.json`, `preprocessor_config.json`, `README.md`. Put all four in `vitmatte_model_dir` and compare every SHA against `config/model-provenance.json`. The HF model card and upstream research implementation have separate license statements; see `THIRD_PARTY_NOTICES.md`.

The worker loads the local tensor-only checkpoint with `torch.load(weights_only=True)`, a fixed Transformers ViTMatte class, strict state loading, F32 weights/input/output, autocast disabled and TF32 disabled. No arbitrary remote model code or automatic precision fallback is used.

## Dedicated BF16 Comfy server

Start a dedicated loopback server rather than sharing another user's queue. A representative command from a fresh Comfy checkout is:

```text
python main.py --listen 127.0.0.1 --port 8197 --fast-disk --disable-pinned-memory --bf16-unet --bf16-vae --bf16-text-enc --disable-auto-launch
```

Configure your model paths and output directory using ordinary Comfy options. The executable flags required by this profile are recorded in `config/example.json`. Leave the primary8188 service intact. The production16GB card used BF16 disk offload and sufficient host/pagefile storage; this is not a claim that69.75GB of base weights fit in16GB VRAM or32GB RAM. Storage/offload performance and other applications affect feasibility.

Copy `config/example.json` to a private, ignored local config and edit the paths/executables/model registrations. Relative configured paths resolve from this repository root. Registered model names must exactly match Comfy `/object_info`, including any registered subdirectory. `comfy_output` must be the real output root for that dedicated server. Do not use a junction to bypass Comfy's output containment checks.

## Make empty conditioning once

No private conditioning binary is required. `workflows/empty_conditioning_api.json` is the actual three-node path: BF16 Gemma `CLIPLoader(type=ltxv)` → empty `CLIPTextEncode` → native `SaveConditioning`.

```text
python -m pipeline conditioning-prepare --config config/local.json --out runs/empty
python -m pipeline submit --run runs/empty
python -m pipeline wait --run runs/empty
python -m pipeline conditioning-save --run runs/empty --dest ../ComfyUI/models/embeddings/empty_prompt_bf16_native.safetensors
python -m pipeline release --run runs/empty
```

`conditioning-save` preserves the complete safetensors file byte-for-byte, including tensor dtype/options. BF16 encoder inference in the production Comfy version returned native F32 conditioning; it was not cast back down. Refresh/restart only your dedicated service as appropriate to register the embeddings file, then set the exact `conditioning` registration name in your local config. The explicit `release` action requires this run's UUID to be successfully complete and the queue to be idle; it sends `/free` with `unload_models/free_memory`, does not kill the service, and refuses primary8188. It records a request, not an unmeasured guarantee of free VRAM.

## Run the source75..139 recipe

The profile intentionally accepts the demonstrated4096×2160/25fps/BT709-limited source contract. Obtain the [Pexels8370090 source](https://www.pexels.com/video/close-up-of-a-woman-posing-on-a-windy-day-8370090/) separately; this repository contains no footage. Keep source/output/model directories outside Git.

```text
python -m pipeline prepare --config config/local.json --source ../media/source.mp4 --out runs/shot
python -m pipeline submit --run runs/shot
python -m pipeline wait --run runs/shot
python -m pipeline collect --run runs/shot
python -m pipeline release --run runs/shot
python -m pipeline refine --run runs/shot --out runs/refined4k
```

For model/process reuse, use the opt-in `refine --execution batch` with the same arguments. It runs one GPU process with one model load, then one CPU foreground process, then one packing process. Frames and tiles remain sequential at unchanged precision. Completed assets require matching input/file hashes; intermediate reuse additionally requires input/config/code-bound receipts and matching NPY file/pixel hashes. Partial or changed files stop the job and remain preserved. Default `isolated` retains the original per-frame process behavior. See [next-shot timing and direct output](next-shot.md) for measured scope.

`prepare` runs CPU FFmpeg and local API GET/schema checks only. It extracts original4K RGB8 code values and creates the matched1920×1088 lossless-RGB reference. It writes an absolute local VHS reference path plus API/UI JSON. It does not POST. Default `start_frame=75`; zero-based local0..64 maps to source75..139 and file00001..00065, at25fps,2.6seconds. No optical flow creates intermediate matte frames.

The15-node graph uses empty prompt, LoRA/guide strength1, CFG1, eight explicit sigma intervals, nonancestral Euler, spatially full1920×1088 and temporal-only Fusion33 with a matching streamed33 guide/CPU canvas. This differs from the earlier17/33-frame ancestral pilot. Decode uses DiffVAE tile512/overlap64. The native exporter uses flattened `format.bit_depth=32-bit float` and `format.input_color_space=linear` keys. That setting preserves the model's numeric matte output; it is not a scene-RGB color conversion.

`submit` writes an exclusive durable UUID intent before POST and saves the reply. Repeating submit, recovering a saved UUID or an uncertain network reply performs GET-only observation; it never blindly posts the graph twice. If the server returns a different UUID, its reply is preserved and the run becomes uncertain for manual diagnosis. `wait` timeout also does not cancel or resubmit. These commands are regular local processes, not an app-owned durable scheduler; use a terminal or your own scheduler for long unattended jobs. This repository does not claim that closing the host of a foreground CLI keeps it alive.

`collect` requires successful completion of that owned UUID and exactly65 native F32 RGB EXRs under its unique output prefix. It preserves raw EXRs, derives F32 coverage using the recorded0.299/0.587/0.114 luma expression with range clipping, removes the known37/38-pixel prior padding and maps the prior to original4K by pixel-center bilinear interpolation. This derived prior is separate from the delivered refined alpha. It never derives a float master from alpha16 or a review MP4.

`refine` processes frames sequentially using the configured GPU and CPU executables. It classifies trimap seeds with raw prior FG≥.98/disk48 and BG≤.02/disk9, leaving unknown hair to ViTMatte. These are trimap seed operations, not final alpha erosion. It covers the native source with28 unresized1024 tiles at512 stride, including bottom origin1136. Positive pixel-center sin² overlap weights are normalized in F32. No final alpha threshold/erosion/feather is applied. At least8GiB free CUDA memory and4GiB free CPU RAM are preflight requirements; failure is preserved without automatic quantization/retry.

The CPU step uses [PyMatting1.1.16 `estimate_foreground_ml`](https://pymatting.github.io/foreground.html) on BT709 **code values**, F64 numerical working arrays, then saves F32 straight foreground RGB. It keeps the inferred F32 alpha byte-exact. Parameters are recorded in `pipeline/refine.py`; this selected coded-RGB branch is distinct from the rejected linear-light comparison and simple color-removal trials. The final foreground is not described as radiometrically linear RGB.

## Output and AE boundary

`frame-ledger.json` has top-level method/width/height/fps/frame_count and `frames[].local_index/source_frame_index/inputs`. Each of the four inputs has an absolute runtime path and SHA:

| Input | Native file contract |
|---|---|
| source | original4096×2160 RGB8 PNG, source clock unchanged |
| alpha | ZIP FLOAT32 EXR, RGB white plus derived linear-coverage A |
| foreground | ZIP FLOAT32 EXR, straight coded RGB3, **no alpha** |
| raw_prior | ZIP FLOAT32 EXR, RGB white plus mapped original LTX prior A |

EXR writes are read back by channel name and literal F32 bits are checked. Mixed HALF/FLOAT EXRs are rejected before `read_image(FLOAT)` can silently promote HALF channels. Keep all65 source/F/A/prior assets and the raw LTX EXRs. Technical completion alone is not temporal visual approval: inspect all-frame motion and1:1 hair on black/gray/white/contrast backgrounds.

See `docs/after-effects.md` for the independently packaged native host workflow. Foreground has no A and receives the external alpha once. Actual AE associated/black output is composited as `RGB + BG*(1-A)`; do not multiply its RGB by A again. Preserve RGB interpretation of coded EXR inputs must be proved before native render, and queue outputs need unique directories because native numbering can restart at0.

For a transparent file sequence without AE, run `python -m pipeline export-rgba --producer runs/refined4k --out runs/rgba --format exr` using the media environment with NumPy/OpenImageIO. It validates input SHAs, channel types, dimensions and contiguous frame clocks before joining restored RGB with the existing coverage A. This is straight RGBA, not associated RGB: do not multiply F by A in the stored file. EXR remains coded BT709 and needs an explicit straight-alpha/data-preserving interpretation in consumers. `--format png8` produces rounded/clamped 8-bit RGBA for review, not an F32 master. Both paths refuse existing output directories and write a clock/hash receipt. Neither starts inference or AE.

## Packaging verification boundary

```text
python -m pipeline --help
python -m unittest discover -s tests -p test_pipeline.py -v
```

The small regression suite uses mocked local API calls for no-duplicate POST behavior, saved uncertain/mismatched identities, loopback restrictions, native F32 low-bit preservation, an actual tiny mixed-channel EXR rejection when OpenImageIO is installed, fixed28-tile overlap hashes, graph settings and65-frame numbering/clock. All10 tests, CLI help and byte-compilation passed during packaging, including the actual OpenImageIO mixed-channel regression. No real POST, GPU inference or render was started by these tests. The selected production assets were generated and separately visually/native-verified before this code was made portable; a clean-machine end-to-end replay remains outside the packaging test scope.
