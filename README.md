# realtime-vlm

![realtime-vlm UI](images/ui.png)

RTSP 카메라와 MP4 영상을 최대 4개까지 동시에 받아서, **자유 텍스트 프롬프트**(예: `넘어지는 사람`, `걸어가는 사람`)에 해당하는 대상을 VLM(Vision-Language Model)으로 찾고 **위치(bbox)를 실시간으로** 보여주는 시스템입니다. 감지된 장면은 스냅샷과 함께 이벤트 로그로 남습니다.

- **입력**: RTSP / MP4 (웹 업로드 또는 `videos/` 폴더), 최대 4개 동시
- **모델**: Qwen3-VL-4B-Instruct-FP8 (vLLM 서빙, RTX 4060 8GB에서 동작)
- **출력**: 프롬프트에 해당하는 대상의 bbox, 감지 점수, 이벤트 로그(스냅샷)
- **UI**: 웹 페이지 (2x2 그리드 + 이벤트 패널)

---

## 목차
1. [구조](#구조)
2. [요구 사항](#요구-사항)
3. [빠른 시작](#빠른-시작)
4. [UI 사용법](#ui-사용법)
5. [추론 방식](#추론-방식)
6. [설정](#설정-configsconfigyaml)
7. [개발](#개발)
8. [API](#api)
9. [성능과 제약](#성능과-제약)
10. [폴더 구조](#폴더-구조)
11. [샘플 영상 출처](#샘플-영상-출처)

---

## 구조

```
 브라우저 ── http://127.0.0.1:8000 ──▶ frontend (Vite, React)
                                         │  /api, /ws, /event-files 프록시
                                         ▼
                                       backend (FastAPI)            ◀── RTSP 카메라 / MP4
                                         │  스트림 디코딩 · 프레임 샘플링 · 이벤트 저장
                                         │  HTTP (OpenAI 호환 API)
                                         ▼
                                       serving (vLLM, GPU)
                                         Qwen3-VL-4B-Instruct-FP8
```

| 컨테이너 | 역할 | 이미지 | 포트 (127.0.0.1) |
|---|---|---|---|
| `realtime-vlm-serving` | VLM 서빙 (GPU) | vllm/vllm-openai v0.33.0 | 9000 |
| `realtime-vlm-backend` | 스트림 입력, 추론 요청, 이벤트, REST/WS/MJPEG | python:3.12-slim + ffmpeg | 8001 (`/docs`) |
| `realtime-vlm-frontend` | 웹 UI (Vite 개발 서버) | node:22-slim | **8000** |

- **소스 코드는 이미지에 넣지 않고 볼륨으로 마운트합니다.** 코드를 고쳐도 다시 빌드할 필요가 없습니다(backend는 자동 재시작, frontend는 HMR).
- backend는 serving이 준비(healthy)된 뒤에 시작합니다.
- serving 백엔드는 `vllm`과 `triton`(예정) 중에서 고를 수 있습니다(`configs/config.yaml`의 `serving.backend`).

## 요구 사항

- NVIDIA GPU 8GB 이상 (RTX 4060 8GB에서 확인)
- Docker + Docker Compose v2 + NVIDIA Container Toolkit
- WSL2를 쓰면 `.wslconfig`에 `memory=12GB` 이상을 권장합니다(모델 로딩 시 메모리 부족 방지).

## 빠른 시작

### 1. 모델 준비 (`model_files/`)
모델은 로컬 폴더에서 읽고, 실행 중에는 인터넷에 접속하지 않습니다(`serving.offline: true`). 처음 한 번만 받아 두면 됩니다.

```bash
sh build_docker.sh   # 다운로드에 serving 이미지를 쓰므로 먼저 빌드

# 기본 모델 (약 5.7GB)
docker run --rm --entrypoint python3 -u $(id -u):$(id -g) -e HF_HOME=/tmp/hf \
  -v $(pwd)/model_files:/workspace/model_files realtime-vlm-serving:latest -c \
  "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-VL-4B-Instruct-FP8', local_dir='/workspace/model_files/Qwen3-VL-4B-Instruct-FP8')"

# (선택) 가벼운 대안 모델 Qwen3-VL-2B-Instruct (약 4GB): 위 명령에서 모델 이름만 바꿉니다.
```

### 2. 빌드 및 실행
```bash
sh build_docker.sh   # docker compose build (의존성이 바뀔 때만)
sh run_docker.sh     # docker compose up -d
```
첫 실행 시 vLLM이 모델을 올리고 컴파일하는 데 1~2분 걸립니다. 진행 상황은 `docker compose logs -f serving`으로 볼 수 있습니다.

### 3. 접속
- UI: **http://127.0.0.1:8000**
- API 문서: http://127.0.0.1:8001/docs

### 4. 종료
```bash
docker compose down
```

## UI 사용법

| 기능 | 방법 |
|---|---|
| 스트림 추가 | 상단 입력칸에 `rtsp://...` 또는 `videos/...mp4`를 입력(클릭하면 파일 목록이 뜸)하고, 프롬프트를 쓴 뒤 **추가** |
| 여러 개 한 번에 추가 | **MP4 업로드 (여러 개)** 에서 파일을 여러 개 선택하면, 빈 슬롯 수만큼 바로 스트림으로 추가됩니다. 남는 파일은 업로드만 됩니다 |
| 빈 슬롯에서 추가 | **＋ 빈 슬롯** 을 클릭하면 그 자리에 추가 폼이 뜹니다 |
| 프롬프트 변경 | 타일 아래 입력칸에서 바꾸고 **적용** (실행 중에 바로 반영) |
| 세부 설정 | 타일의 ⚙ 버튼: 프레임 수, 구간(초), 요청 간격(ms), 판단 임계값 |
| 감지 표시 | 빨간 테두리 + `감지 0.xx` 배지, 영상 위에 bbox와 라벨 |
| 이벤트 로그 | 우측 패널. 스트림별 필터, 지우기, 클릭하면 스냅샷 상세 |
| 스트림 삭제 | 타일의 ✕ |

- 마지막으로 쓴 프롬프트는 브라우저에 저장되어 다음에 기본값으로 들어갑니다.
- 업로드한 파일은 `videos/uploads/`에 저장됩니다. 같은 이름이면 `name_1.mp4`처럼 번호가 붙습니다.

### 테스트용 RTSP 카메라
`videos/` 안의 mp4를 RTSP 카메라처럼 송출합니다(compose 네트워크 안에서 동작).
```bash
sh tools/rtsp_sim.sh videos/samples/urfd-fall-01-rgb.mp4 cam1   # UI에 rtsp://rtsp-sim:8554/cam1 입력
sh tools/rtsp_sim.sh videos/samples/people-detection.mp4 cam2
sh tools/rtsp_sim.sh stop
```
실제 카메라 주소 형식의 예:
- Hikvision: `rtsp://user:pw@IP:554/Streaming/Channels/101`
- Dahua: `rtsp://user:pw@IP:554/cam/realmonitor?channel=1&subtype=0`

## 추론 방식

기본값은 **two_stage**입니다(`inference.strategy`).

1. **판단 (judge)**: 최근 2초에서 고른 프레임 4장을 시간 순서대로 보내 "프롬프트에 해당하는가"를 yes/no로 묻습니다. yes의 확률(logprob)이 **점수**가 됩니다.
2. **위치 (grounding)**: 점수가 `judge_threshold`(0.5) 이상이면, 마지막 프레임 1장으로 bbox를 받습니다.

- 판단 결과는 bbox보다 먼저 WebSocket으로 보냅니다. 그래서 프레임 후 약 0.2초 만에 경고가 뜨고, bbox는 약 1초 뒤에 따라옵니다.
- 같은 스트림에서 5초 안에 다시 감지되면 하나의 이벤트로 합칩니다(`events.cooldown_sec`).

### 평가 결과 (UR Fall 샘플 4개, 0.25초 간격, `넘어지는 사람`)

| 모델 | 방식 | 정밀도 | 재현율 | F1 | 오탐 | bbox IoU |
|---|---|---|---|---|---|---|
| 2B | multi_image (4장 → bbox 한 번에) | 0.28 | 0.31 | 0.29 | 21 | 0.12 |
| 2B | two_stage, 4장 | 0.60 | 0.96 | 0.74 | 17 | 0.98 |
| 4B-FP8 | two_stage, 1장 | 0.92 | 0.85 | 0.88 | 2 | 0.98 |
| **4B-FP8** | **two_stage, 4장 (기본값)** | **1.00** | **0.85** | **0.92** | **0** | **0.99** |

- 4장을 한 번에 보내고 bbox를 받으면 좌표가 깨집니다. 그래서 4장은 판단에만 쓰고, bbox는 1장으로 받습니다.
- 놓친 경우는 모두 넘어지기 시작하는 첫 0.25~0.5초입니다. 감지가 그만큼 늦다고 보면 됩니다.
- 표본이 영상 4개뿐이라 실제 CCTV 영상으로 다시 검증해야 합니다.
- 다시 실행하려면: `docker exec -w /workspace realtime-vlm-backend python3 backend/examples/eval_strategies.py`

## 설정 (`configs/config.yaml`)

serving과 backend가 함께 쓰는 설정 파일입니다. backend 쪽 값은 저장하면 자동으로 반영되고, serving 쪽 값은 `docker compose restart serving`이 필요합니다.

| 섹션 | 키 | 기본값 | 설명 |
|---|---|---|---|
| serving | `backend` | `vllm` | `vllm` \| `triton`(예정) |
| | `model_id` | `model_files/Qwen3-VL-4B-Instruct-FP8` | 로컬 모델 폴더 |
| | `gpu_memory_utilization` | `0.85` | 4B 기준. 2B는 0.78 |
| | `max_model_len` | `2048` | 4B 기준. 2B는 8192 |
| | `scheduling_policy` | `priority` | bbox 요청을 판단 요청보다 먼저 처리 |
| preprocess | `max_side` | `448` | 서버로 보내는 프레임의 긴 변(px). 서버 쪽 메모리 계산과 이미지 크기 상한에도 같이 쓰임 |
| inference | `strategy` | `two_stage` | `two_stage` \| `multi_image` \| `single_frame` |
| | `judge_threshold` | `0.5` | 판단 점수 임계값 |
| | `num_frames` / `window_sec` | `4` / `2.0` | 판단에 쓰는 프레임 수 / 구간 |
| stream | `max_streams` | `4` | 동시 스트림 수 |
| | `interval_ms` | `1000` | 스트림별 요청 간격 (아래 성능 참고) |
| | `buffer_fps` / `buffer_max_side` | `10` / `960` | 링버퍼 저장 속도 / 해상도 (스냅샷 해상도) |
| events | `cooldown_sec` | `5.0` | 이 시간 안의 재감지는 같은 이벤트로 합침 |
| backend | `reload` | `true` | 코드 수정 시 자동 재시작 |

환경 변수 `SERVING`, `MODEL_ID`로 serving 값을 덮어쓸 수 있습니다(예: `MODEL_ID=model_files/Qwen3-VL-2B-Instruct sh run_docker.sh`). 이때 메모리 관련 값(`gpu_memory_utilization`, `max_model_len`)은 config에서 같이 바꿔야 합니다.

## 개발

| 바꾼 것 | 해야 할 일 |
|---|---|
| `backend/app/**`, `configs/` | 없음 (uvicorn 자동 재시작) |
| `frontend/src/**` | 없음 (HMR, 브라우저에 바로 반영) |
| `serving/*.sh`, config의 serving 값 | `docker compose restart serving` |
| `backend/requirements.txt`, `frontend/package.json`, vLLM 버전 | `sh build_docker.sh && sh run_docker.sh` |

```bash
docker compose ps                         # 상태
docker compose logs -f backend            # 로그 (serving, frontend도 동일)
docker exec -it realtime-vlm-backend bash # 컨테이너 접속 (vim, net-tools, alias 포함)

# 서버에 직접 추론 (이미지 / 영상 / 영상 전체 훑기)
sh backend/examples/run_vlm_inference.sh --video videos/samples/urfd-fall-01-rgb.mp4 --at 4 --prompt "넘어지는 사람"
sh backend/examples/run_vlm_inference.sh --video videos/samples/urfd-fall-01-rgb.mp4 --sweep 0.5 --prompt "넘어지는 사람"
#   → 결과 이미지: data/inference/

# WebSocket 결과 통계 (스트림별 감지율, 지연)
docker exec realtime-vlm-backend python3 examples/ws_monitor.py --sec 20

# frontend 타입 체크
docker exec realtime-vlm-frontend npx tsc --noEmit
```

- 스트림 목록은 메모리에만 있습니다. backend가 재시작(코드 수정 포함)되면 스트림은 사라지고, 이벤트는 `data/events/`에 남습니다.
- backend와 frontend는 호스트 사용자 UID로 실행되므로 `data/`, `videos/uploads/` 파일은 사용자 소유로 생깁니다.

## API

backend: `http://127.0.0.1:8001` (UI에서는 `http://127.0.0.1:8000/api/...`로도 접근 가능)

| Method | Path | 설명 |
|---|---|---|
| GET | `/api/health` | 서빙 상태, 모델, 추론 설정 |
| GET / POST | `/api/streams` | 목록 / 추가 `{url, prompt, name?, num_frames?, window_sec?, interval_ms?, judge_threshold?}` |
| GET / PATCH / DELETE | `/api/streams/{id}` | 조회 / 설정 변경 / 삭제 |
| GET | `/api/streams/{id}/mjpeg` | 라이브 영상 (`?overlay=1`: 서버가 박스를 그림) |
| GET | `/api/streams/{id}/snapshot` | 최신 프레임 JPEG |
| GET | `/api/videos` | `videos/` 안의 영상 목록 |
| POST | `/api/upload` | 영상 업로드 (multipart `file`) |
| GET / DELETE | `/api/events` | 이벤트 목록 (`?stream_id=&limit=`) / 전체 삭제 |
| GET | `/event-files/{YYYYMMDD}/{id}.jpg` | 이벤트 스냅샷 |
| WS | `/ws` | `status`(1초마다), `judge`, `result`, `event`, `error` 메시지 |

## 성능과 제약

RTX 4060 8GB + Qwen3-VL-4B-FP8 기준, 4스트림 동시(RTSP 1 + MP4 3):

| 스트림별 요청 간격 | 판단만 | 감지 시 판단+bbox (p90) |
|---|---|---|
| 250ms | 약 300ms | 약 4초 |
| 500ms | 약 300ms | 2.5~4초 |
| **1000ms (기본값)** | 200~300ms | **약 1.7초** |

- **bbox 생성(약 44토큰)이 느린 게 병목입니다.** 디코딩 각 단계가 다른 스트림의 판단 요청과 같이 처리되기 때문입니다. 1스트림만 돌릴 때는 약 1초 걸립니다.
- **GPU 메모리가 빠듯합니다.** 8GB 중 약 7.2GB를 쓰고 KV cache는 약 3,200토큰입니다. Windows 쪽에서 GPU를 많이 쓰면 serving 기동이 실패할 수 있습니다.
- **화면의 bbox는 추론한 프레임 기준이라**, 움직이는 대상에서는 약 1초 뒤처져 보입니다. 이벤트 스냅샷은 그 프레임에 직접 그리므로 정확합니다.
- **스냅샷 폴더 날짜는 컨테이너 시간대(UTC) 기준입니다.** UI의 시각은 브라우저 시간으로 표시됩니다.

## 폴더 구조

```
realtime-vlm/
├── README.md, PLAN.md
├── docker-compose.yml
├── build_docker.sh              # docker compose build
├── run_docker.sh                # docker compose up -d
├── configs/config.yaml          # 전체 설정
├── serving/
│   ├── Dockerfile               # vllm/vllm-openai 기반
│   ├── entrypoint.sh            # SERVING 값에 따라 vllm | triton 실행
│   ├── gen_env.py               # config.yaml → vLLM 인자
│   └── vllm.sh
├── backend/
│   ├── Dockerfile, requirements.txt, start.sh
│   ├── app/
│   │   ├── main.py              # REST, WS, MJPEG, 업로드
│   │   ├── config.py, schemas.py
│   │   ├── inference/           # preprocess, prompts, parser, vllm_client (strategy)
│   │   ├── streams/             # source(RTSP/MP4, 링버퍼), worker(추론 루프), manager
│   │   └── events/store.py      # 이벤트, 스냅샷, jsonl
│   └── examples/                # run_vlm_inference, eval_strategies, ws_monitor
├── frontend/
│   ├── Dockerfile, package.json, vite.config.ts
│   └── src/                     # App, useLive(WS), components/(AddStreamForm, StreamTile, EventPanel)
├── tools/rtsp_sim.sh            # 테스트용 RTSP 카메라
├── images/                      # README 이미지
├── model_files/                 # 모델 (git 제외)
├── videos/                      # 입력 영상, uploads/ (git 제외)
└── data/                        # events/, inference/ (git 제외)
```

## 샘플 영상 출처

`videos/samples/`는 git에 포함되지 않습니다. 출처와 라이선스는 `videos/samples/SOURCES.md`에 있습니다.

| 파일 | 출처 | 라이선스 |
|---|---|---|
| `urfd-fall-01/02/05-rgb.mp4`, `urfd-adl-01-rgb.mp4` | [UR Fall Detection Dataset](http://fenix.ur.edu.pl/~mkepski/ds/uf.html) (RGB 부분만 잘라냄) | CC BY-NC-SA 4.0 (비상업적 학술 용도) |
| `people-detection.mp4` | [intel-iot-devkit/sample-videos](https://github.com/intel-iot-devkit/sample-videos) | CC BY 4.0 |

## 로드맵
- [x] P0 Docker + vLLM 서빙
- [x] P1 추론 방식·모델 평가 (4B-FP8, two_stage)
- [x] P2 스트림 입력, worker, API
- [x] P3 이벤트 로그
- [x] P4 웹 UI
- [x] 컨테이너 분리 (serving / backend / frontend)
- [ ] P5 안정화·튜닝: bbox 출력 토큰 줄이기, 4스트림 부하 측정
- [ ] P6 Triton 서빙 비교
