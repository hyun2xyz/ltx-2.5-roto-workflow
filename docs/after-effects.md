# Editable After Effects handoff

The checked profile is **4096×2160,25fps,65frames**. It preserves original RGB8, mapped raw LTX F32 prior, refined F32 alpha and reconstructed F32 foreground separately. This is original4K ViTMatte refinement of an HD LTX prior, not native4K LTX inference.

`ae/` contains project/queue JSX, input validation, a native probe and the100-export precision checker. It does not require a private orchestration harness and never starts an app/render automatically. Models, media, AEPs and runtime receipts are external and must stay out of Git. Third-party implementation code is not vendored; Python dependencies are listed in `ae/requirements.txt`.

## Input and project preparation

Copy `ae/settings.example.json` to a local settings file **outside Git**. Paths are resolved relative to that settings file. Select a new delivery directory; existing files are never overwritten.

The producer supplies `frame-ledger.json` with width/height/fps/frame_count and65 rows. Each row has local_index/source_frame_index and inputs{source,alpha,foreground,raw_prior}{path,sha256}. If time_seconds exists, it must equal local_index/25. Relative asset paths are resolved against the producer root. The four sequence folders are:

- `assets/source/source_00001.png`: original-grid RGB8,no A.
- `assets/foreground_rgb_float32/foreground_00001.exr`: coded-BT709 F32 RGB,no A.
- `assets/alpha_rgba_float32/alpha_00001.exr`: white RGB=1 plus refined F32 coverage A.
- `assets/raw_prior_rgba_float32/alpha_00001.exr`: white RGB=1 plus preserved mapped-prior F32 A.

All EXR channels must actually be FLOAT32/ZIP16, not HALF. Alpha is coverage/data; foreground RGB is coded BT709, not scene-linear merely because its container is EXR.

```text
python -B ae/prepare_ae.py --settings ../runs/ae-settings.json verify-inputs
python -B ae/prepare_ae.py --settings ../runs/ae-settings.json prepare-config --authorize-build
```

Inspect the generated `ae-build.filled.jsx`, then use AE **File → Scripts → Run Script File**. The standalone JSX writes its own host receipt. It requires a clean empty unsaved project and a new output AEP; it never closes or replaces user work. Raw-prior/source-refined/clean branches use native `setTrackMatte(ALPHA)` once each. Each composite receives the resulting associated cutout without another matte. The saved project and build receipt remain explicitly pending input interpretation/native QA.

## Required input/output color readback

On the **foreground RGB EXR only**, open Interpret Footage → Color and enable **Preserve RGB**. Save the base AEP. In the tested host, automatic Interpret As Linear Light was On for32bpc; Preserve RGB disabled that interpretation and retained the file numbers. Do not compensate with a gamma effect or alter A. Source and both matte interpretations were left unchanged. Set `foreground_input_preserve_rgb_confirmed` in local settings only after this actual UI readback.

Create/select your own output-module template: OpenEXR Sequence,RGBA,FLOAT32,ZIP16,premultiplied/black,no resize/crop. Output Color separately requires Preserve RGB=true and Convert To Linear Light=Off. A Depth label alone does not prove FLOAT32; the native header and pixel checks do.

```text
python -B ae/queue_tools.py --settings ../runs/ae-settings.json make-queue --phase probe
```

Run its generated `run-in-AE.jsx`, inspect the module, save, then render its saved `render.aep` with AE or your installed `aerender -project ... -mfr OFF 100`. Use a durable job/log mechanism for long work. Read the actual single EXR from the probe item directory:

```text
python -B ae/queue_tools.py --settings ../runs/ae-settings.json verify-probe --native-exr ACTUAL_EXR --receipt ../runs/probe-proof.json
python -B ae/queue_tools.py --settings ../runs/ae-settings.json make-queue --phase full --probe-proof ../runs/probe-proof.json
```

Full queue preparation refuses an absent/failed numeric probe or a changed base-AEP/input-validation SHA. The probe compares original F×A directly and A independently. Tolerance PASS is never labelled bit-exact.

## Unique directories and canonical100

The full queue exports clean65 plus seven auxiliary branches at local0/8/16/32/64:100 EXRs. **Every queue item has a distinct empty output directory.** In the tested AE host, file numbers restarted at0 for shared queue output patterns despite correct time spans and GUI starting numbers. Never infer a source frame solely from the filename or let queue items share a directory.

After rendering, collect unchanged native bytes and run the independent checker:

```text
python -B ae/queue_tools.py collect --validation INPUT_VALIDATION --queue-receipt HOST_QUEUE_RECEIPT --output NEW_CANONICAL_DIRECTORY
python -B ae/verify_4k_ae_native_v2.py --validation INPUT_VALIDATION --build-receipt HOST_BUILD_RECEIPT --render-root CANONICAL_DIRECTORY --receipt NEW_NATIVE_PROOF
```

Collection identifies each clean frame by the original F32 A bit hash; ambiguous hashes stop collection. Auxiliary samples use the actual queue time span and must then pass the v2 original-frame/branch pixel checks. Raw item folders are preserved. No native image is re-encoded during collection.

V2 checks all65 clean A and associatedRGB against **original input F×A**. It checks sourceRGB8 byte restoration, opaqueF import, raw/refined alpha, two source roto branches and gray composite on five samples only. It records literal F32 bit-exactness separately from maxerror/tolerance. Soft clean pixels detect A². Recompose native output as `associatedRGB + BG*(1-A)`, without multiplying native RGB by A again.

## Validation scope

The completed production run used AE26.5x89:100 actual4K ZIP/FLOAT32 exports; all65 clean A and original F×A were bit-exact with maxerror0. OriginalRGB8 and opaqueF native preservation were five-sample checks. Temporal visual PASS used all-frame contacts and selected1:1 hair/boundary patches; it is not a ground-truth-perfect-matte claim.

These repository copies have portable path/config adapters and a safer unique-directory queue. Their validation is syntax/help, clock and focused guard regression checks. They were **not re-rendered in a new native AE run** during packaging. See `ae/validation-summary.json`. Final visual approval remains separate from technical precision checks.
