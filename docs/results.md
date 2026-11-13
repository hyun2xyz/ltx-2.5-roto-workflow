# 검증 기록

## 실제 로컬 결과

검증 대상은 Pexels 8370090의 0-based 원본 프레임 75–139, 25fps, 65프레임입니다. 원본은 4096×2160 RGB8이고, 결과 구간은 2.6초입니다. Windows, RTX 4070 Ti SUPER 16GB, RAM 32GB에서 생성했습니다.

| 단계 | 확인한 내용 |
|---|---|
| LTX prior | 공식 22B distilled BF16 transformer, BF16 text encoder/DiffVAE, Alpha Gen LoRA. FHD 1920×1088, seed42, CFG1, 8-step, temporal fusion33 |
| 원본 좌표 | FHD alpha의 위37/아래38 padding을 제거한 1920×1013 영역을 원본 4096×2160에 매핑 |
| 4K 정밀화 | ViTMatte Base F32, autocast/TF32 off. 1024×1024 tile, stride512, 정규화한 겹침 가중치 |
| Trimap | FG≥0.98, 원본 기준 원형 erosion radius48. BG≤0.02, radius9. 최종 alpha에는 threshold/erosion/feather 미적용 |
| 전경 RGB | PyMatting ML, 원본 BT.709 coded RGB와 derived alpha 사용. RGB-only F32로 저장 |
| 자산 파일 | 65 source PNG + 65 refined alpha EXR + 65 foreground EXR + 65 raw-prior EXR = 260개 SHA·채널·프레임 검증 |
| AE | 32bpc, unmanaged `None`, 25fps, alpha 각 branch 1회, RGB-only 전경 입력 |
| Native 출력 | 실제 ZIP/FLOAT32 RGBA EXR 100개. clean 전체65 + 7개 보조 branch의 5개 표본 |
| Native alpha | 전체65 bit-exact, 최대 오차0 |
| Native RGB | 전체65에서 원래 전경 `F * A`와 최대 오차0 |
| 보조 표본 | local0/8/16/32/64의 원본 RGB8 byte 보존과 독립 전경 입력을 검사 |
| 이중 알파 방지 | soft-alpha 34,113,433개 픽셀에서 `A`와 `A²`를 구별하는 검사 수행 |
| 시각 검수 | 전65 연속 alpha/gray contact, 인코딩된 전65 모발 source/gray 1:1 비교, 전65 tile-boundary alpha/gray, 대표4배경 비교 |

ViTMatte 모델 revision은 `bf486d01a7d9e3dbcc8400f7942835caf0eaf76e`입니다. 모델 파일 SHA-256은 `b2521bcc4b719fb24611c39605b6642162fd7502e69b3cc846506ca921757b41`입니다.

실제 입력 검증 manifest SHA-256: `91ff7ea6b01ffddda9beac2280c266d7930e8447edf235bb213545e813851797`.

수치 exactness와 시각 품질은 별개입니다. 검수에서 큰 dropout, 불투명 몸통의 구멍, 직선 tile seam, 큰 색 번쩍임은 보이지 않았습니다. 약한 flyaway와 원본 motion blur의 한계는 남으며 다른 영상에서도 같은 품질을 보장하지 않습니다. 이 프로파일은 LTX native-4K 추론이 아닙니다.

## 수정해 반영한 실패 원인

1. **AE 입력 색상:** 출력 설정이 정상이어도 coded-RGB EXR 입력의 자동 linear-light 해석이 `F^(1/2.4)`에 해당하는 색 변화를 만들었습니다. 전경 EXR 입력의 `Preserve RGB`를 켠 뒤 실제 출력의 `F*A`가 오차0으로 맞았습니다.
2. **AE 출력 번호:** 여러 queue item이 같은 파일 패턴을 쓰면 aerender가 번호를 0부터 다시 써 앞선 파일을 덮었습니다. 실제 값으로 유효 파일을 회수하고 누락37개만 고유 폴더에 출력한 후 source-frame ledger로 최종100개를 구성했습니다. 재사용63개와 새37개 모두 최종 픽셀 검증에 포함했습니다.
3. **감독 프로세스와 실제 결과:** 종료 코드0만으로 성공을 판단하지 않습니다. 반대로 별도 로그 파일이 없다는 supervisor 오류만으로 정상 결과를 재렌더하지 않습니다. 필요한 산출물과 실제 값을 확인한 뒤 상태를 확정합니다.

실패 출력과 이전 시도는 원래 로컬 작업 폴더에 보존했습니다. 이 저장소에 큰 출력 파일이나 전체 로그를 복제하지 않습니다.

## 저장소 복사본 검증

실제 GPU/AE 결과는 위 로컬 실행에서 얻은 증거입니다. 개인 경로를 제거하고 설정화한 저장소 복사본은 별도의 compile/help/CPU 또는 mock 검증을 수행합니다. 복사본으로 전체 GPU 추론과 AE 렌더를 다시 실행한 것으로 표현하지 않습니다.

파이프라인 패키징 단계에서 Python compile, CLI help, 10개 회귀 검증이 통과했습니다. 중복 POST 방지·불확실한 응답 보존·loopback 제한·F32 하위 비트·실제 mixed HALF/FLOAT EXR 거부·28tile 겹침 hash·65frame clock을 검사했고, 실제 POST/GPU/렌더는 실행하지 않았습니다.

AE 패키징 단계에서는 Python AST, 3개 CLI help, standalone JSX 구문 분석과 11개 핵심 guard 검증이 통과했습니다. ledger의 4096×2160/25fps/65frames와 각 행의 시간, 비유한 값 거부, probe 및 고유 출력 폴더 계약을 검사했습니다. 상세 범위는 `ae/validation-summary.json`에 있습니다.

## 최종 브레이크다운

로컬 전달본 `LTX_4K_Hair_Breakdown_1080x1350.mp4`는 1080×1350, 30fps, 756프레임, 25.2초이며 전체 decode·clock·BT.709 matrix/primaries/transfer·TV range·레이아웃 검사를 통과했습니다. 실제 인코딩된 intro, ComfyUI capture, 결과 프레임과 전환 주변을 확인했습니다.

처음 출력에서 누락된 primaries/transfer 태그는 H.264 metadata stream-copy로 보완했습니다. 전후 756개 decoded frame hash와 timestamp가 일치하여 픽셀 재인코딩이 없음을 확인했습니다. 최종 MP4 SHA-256은 `fa284b3c657a3455b8139e31c82c9c7672e34ef0835bc42f0e082c56de47b3dd`입니다. 영상은 저장소에 포함하지 않으며 네이티브 F32 master를 대신하지 않습니다.
