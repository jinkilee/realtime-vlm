# realtime-vlm 개발 계획

RTSP/MP4 영상 최대 4개를 입력받아, 중앙 VLM 서빙 서버(vLLM)에 자유 텍스트 프롬프트(예: "넘어지는 사람")로
질의하고, 해당 대상의 위치(bbox)를 웹 UI에 실시간 오버레이 + 이벤트 로그로 보여주는 시스템.

## 0. 확정된 결정 사항

| 항목 | 결정 |
|---|---|
| 입력 | RTSP URL, MP4 (웹 업로드 + `./videos` 마운트) |
| 스트림 수 | 최대 4개 |
| localization | 자유 텍스트 프롬프트 → bbox (open-vocabulary, 한국어 가능) |
| 시간 정보 | **멀티 프레임 two_stage**: 최근 2초에서 4장 → yes/no 판단(P(yes)=점수) → yes면 마지막 프레임 1장으로 bbox |
| 서빙 | **vLLM** (기본). Triton은 나중에 비교 목적으로 추가 |
| 모델 | **`Qwen3-VL-4B-Instruct-FP8`** (P1 평가로 결정), 대안 `Qwen3-VL-2B-Instruct`. 둘 다 `model_files/`에 로컬 보관. **4B보다 큰 모델은 쓰지 않음** |
| 검출 처리 | bbox 오버레이 + **이벤트 로그 패널** |
| 목표 성능 | 일단 동작 우선, 이후 측정/튜닝 |
| 접속 범위 | 이 PC에서만 (`127.0.0.1` 바인딩) |
| 환경 | Docker 단일 이미지, `build_docker.sh` / `run_docker.sh` 한 줄 명령 |
| 형상관리 | git, 단계(Phase)별 커밋 |

## 1. 하드웨어 환경

| 항목 | 값 | 영향 |
|---|---|---|
| GPU | RTX 4060 8GB (유휴 시 Windows가 약 1.1GB 사용) | vLLM에 실제로 쓸 수 있는 VRAM은 약 6.8GB → `gpu_memory_utilization` 0.75~0.8 |
| RAM | WSL 11.7GB (`.wslconfig` memory=12GB) | 모델 로딩 여유 확보됨 |
| OS | WSL2 Ubuntu 24.04, Docker 29.8 (WSL 안에 직접 설치, nvidia runtime) | |

## 2. 아키텍처

```
 ┌──────────── Browser (frontend: React + Vite + TS) ─────────────┐
 │  URL 추가/업로드 · 2x2 그리드(MJPEG + canvas bbox) · 이벤트 로그  │
 └──────┬──────────────────────────────▲──────────────────────────┘
        │ REST                          │ WebSocket (bbox, event)
 ┌──────▼──────────────────────────────┴──────────────────────────┐
 │  app (FastAPI, 127.0.0.1:8000)                                  │
 │   StreamManager ─ 스트림마다 worker (최대 4)                     │
 │     decode 스레드 → 프레임 링버퍼(최근 수 초)                     │
 │     infer 루프  → N프레임 샘플 → InferenceClient ──HTTP──┐       │
 │   EventStore   → 검출 시 스냅샷 + 메타데이터 저장          │       │
 └───────────────────────────────────────────────────────────┼─────┘
                                                             ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │  serving (컨테이너 내부 :9000)  SERVING=vllm (기본) | triton(추후)  │
 │   vllm serve $MODEL_ID  (OpenAI 호환 /v1/chat/completions)        │
 └─────────────────────────────────────────────────────────────────┘
```

- 서빙 서버는 하나만 띄우고, 4개 스트림이 모두 HTTP API로 요청을 보낸다. 동시 요청은 vLLM의 continuous batching이 묶어서 처리한다.
- app은 서빙 서버와 URL(`INFER_URL`)로만 통신한다. 서빙을 다른 머신으로 옮겨도 코드는 바꾸지 않는다.
- 컨테이너는 1개다. `entrypoint.sh`가 vLLM을 띄우고, `/health`를 통과하면 uvicorn을 띄운다.

## 3. 멀티 프레임 추론 설계

- **링버퍼**: decode 스레드가 최근 `WINDOW_SEC`(기본 2초) 동안의 프레임을 타임스탬프와 함께 보관한다.
- **샘플링**: 추론할 때 윈도우 안에서 균등 간격으로 `NUM_FRAMES`장(기본 4)을 뽑는다. 마지막 장은 항상 최신 프레임이다.
- **요청 형식**: vLLM chat completions에 이미지 N장을 시간 순서대로 넣는다. 서버는 `--limit-mm-per-prompt '{"image":4}'`로 띄운다.
  - 프레임마다 짧은 변(또는 긴 변)을 384~448px로 축소해서 비전 토큰 수와 지연을 줄인다.
  - 대안으로 vLLM의 video 입력(`video_url`)도 P1에서 비교한다.
- **프롬프트 템플릿** (사용자 문구 `{query}`를 끼워 넣음):
  > 다음은 시간 순서대로 나열된 CCTV 프레임 N장이다. "{query}"에 해당하는 대상이 있으면 **마지막 프레임** 기준으로
  > 위치를 JSON으로만 출력하라: `{"detected": bool, "objects": [{"bbox_2d": [x1,y1,x2,y2], "label": str, "reason": str}]}`
- **좌표**: Qwen3-VL은 0~1000 정규화 좌표를 쓴다. `parser.py`에서 0~1 범위로 바꾸고 모델별 형식 차이를 흡수한다. JSON이 깨지면 정규식으로 복구를 시도한다.
- **동시성**: 스트림마다 이전 요청이 끝나야 다음 요청을 보낸다. 처리가 밀려 큐가 쌓이지 않는다.
- 스트림별 설정: `prompt`, `num_frames`(1이면 단일 프레임), `window_sec`, `interval_ms`.

### 3-1. P1 평가 결과 (2026-11-18, `examples/eval_strategies.py`, UR Fall 4개 영상, 0.25초 간격)

| 모델 | 방식 | 정밀도 | 재현율 | F1 | 오탐 | 좌표 IoU |
|---|---|---|---|---|---|---|
| 2B | multi_image (N장 → bbox 한 번에) | 0.28 | 0.31 | 0.29 | 21 | 0.12 (좌표 깨짐) |
| 2B | single_frame | 0.39 | 1.00 | 0.57 | 40 | 0.98 |
| 2B | two_stage, 4장, th 0.5 | 0.60 | 0.96 | 0.74 | 17 | 0.98 |
| 4B-FP8 | two_stage, 1장, th 0.5 | 0.92 | 0.85 | 0.88 | 2 | 0.98 |
| **4B-FP8** | **two_stage, 4장, th 0.5** | **1.00** | **0.85** | **0.92** | **0** | **0.99** |

- multi_image는 좌표가 깨져서 쓸 수 없다. grounding은 항상 마지막 프레임 1장으로 한다.
- grounding 프롬프트는 Qwen 학습 형식(`Locate every instance ... Report bbox coordinates in JSON format.`)을 써야 깨진 출력이 없다. 대신 상황과 관계없이 대상을 찾으므로 판단은 judge 단계가 맡는다.
- 2B는 프레임 수와 관계없이 쪼그려 앉기를 넘어짐으로 판단(점수 0.4~0.9)해서 threshold를 0.9 근처로 아슬아슬하게 잡아야 한다.
- 4B는 음성 점수 최대 0.04, 양성 대부분 0.9 이상으로 여유가 크다. 4장이 1장보다 오탐이 적다(2 → 0).
- 놓친 4건은 모두 넘어지기 시작하는 첫 0.25~0.5초다. 감지가 그만큼 늦는다는 뜻이다.
- 한계: 영상 4개라 표본이 작다. 실제 CCTV 영상으로 다시 검증해야 한다.
- 4B 메모리: 가중치 5.85GB, KV cache 3,248토큰. 4스트림 동시 처리는 P5에서 확인한다.

### 3-2. P2 측정 결과 (2026-11-18, 4스트림 동시: RTSP 1 + MP4 3, 4B-FP8, two_stage)

| 스트림별 요청 간격 | 판단만 (p50) | 검출 시 판단+bbox (p90) | GPU |
|---|---|---|---|
| 250ms | 약 300ms | 약 4초 | 100% |
| 500ms | 약 300ms | 2.5~4초 | 100% |
| **1000ms (기본값)** | 약 200~300ms | **약 1.7초** | 100% |

- bbox 생성(약 44토큰)은 토큰을 하나씩 만드는데, 그 각 단계가 다른 스트림의 판단 요청 프롬프트 처리와 같이 묶여 처리돼서 느려진다. 1스트림 단독일 때는 약 1초.
- vLLM priority scheduling(bbox 우선)은 켜 두었지만 측정상 효과는 없었다(같은 단계 안에 섞이는 문제라 순서 문제가 아님).
- P5 개선 후보: ① 판단 결과(점수)를 bbox보다 먼저 WS로 보내서 알람 지연을 판단 시간(약 0.3초)으로 줄이기 ② structured output으로 bbox 출력 토큰 줄이기(라벨/코드블록 제거, 44 → 약 20토큰) ③ 스트림 수에 따라 간격 자동 조절.

## 4. 이벤트 로그

- **발생 조건**: `detected=true`이고 bbox가 1개 이상일 때.
- **중복 억제**: 같은 스트림에서는 `EVENT_COOLDOWN_SEC`(기본 5초) 동안 새 이벤트를 만들지 않는다. 그 사이에 나온 검출은 기존 이벤트의 `last_seen`과 `count`만 갱신한다.
- **저장 내용**: `id, stream_id, stream_name, prompt, ts_start, last_seen, count, boxes, reason, snapshot_path`
  - 스냅샷: bbox를 그린 JPEG를 `/data/events/{date}/{id}.jpg`에 저장한다.
  - 메타데이터: `/data/events/events.jsonl`에 append한다. 메모리에는 최근 500건을 유지한다.
  - `./data`는 볼륨으로 마운트해서 컨테이너를 재시작해도 남게 한다.
- **UI**: 우측 패널에 최신순으로 시각, 스트림, 프롬프트, 썸네일을 표시한다. 클릭하면 확대 스냅샷과 reason을 보여준다. 스트림별 필터와 지우기 기능을 둔다. 새 이벤트는 WebSocket으로 즉시 반영한다.

## 5. 백엔드 API

| Method | Path | 설명 |
|---|---|---|
| POST | `/api/streams` | `{name?, url, prompt, num_frames?, window_sec?, interval_ms?}` → 스트림 추가. 4개를 넘으면 409 |
| GET | `/api/streams` | 목록과 상태(connected, src fps, infer latency, infer fps) |
| PATCH | `/api/streams/{id}` | 프롬프트나 파라미터 변경 |
| DELETE | `/api/streams/{id}` | 스트림 삭제 |
| POST | `/api/upload` | mp4 업로드 → `/videos/uploads/...` 경로를 반환 |
| GET | `/api/videos` | `/videos` 안의 mp4 목록 (UI 드롭다운용) |
| GET | `/api/streams/{id}/mjpeg` | 라이브 영상 (MJPEG) |
| GET | `/api/events?stream_id=&limit=` | 이벤트 목록 |
| DELETE | `/api/events` | 이벤트 로그 지우기 |
| GET | `/events/...` | 스냅샷 이미지 (정적 파일) |
| WS | `/ws` | `{type:"result", stream_id, frame_ts, boxes, latency_ms}` / `{type:"event", ...}` / `{type:"status", ...}` |
| GET | `/api/health` | app 상태, 서빙 서버 상태, 모델명, 백엔드 종류 |

입력 소스:
- **RTSP**: PyAV(FFmpeg)로 TCP transport를 쓴다. 연결이 끊기면 지수 백오프로 재연결한다.
- **MP4**: 원본 FPS에 맞춰 재생 속도를 조절하고, 끝나면 처음부터 반복 재생한다.

## 6. Frontend (`frontend/`)

- 기술 스택: React + Vite + TypeScript. 빌드 결과는 FastAPI가 정적 파일로 제공한다.
- 레이아웃:
  - **상단**: URL 입력(rtsp:// 또는 `/videos/...`), `./videos` 목록 드롭다운, mp4 업로드, 프롬프트 입력, [Add] 버튼. 스트림이 4개가 되면 비활성화한다.
  - **중앙**: 2x2 그리드. 타일마다 MJPEG 영상(`<img>`)과 `<canvas>` bbox 오버레이, 상태 배지(연결 상태, latency, infer fps), 프롬프트 수정, [Remove] 버튼을 둔다. 검출 중인 타일은 테두리를 강조한다.
  - **우측**: 이벤트 로그 패널.
  - **헤더**: 서빙 백엔드, 모델명, 서버 상태.
- bbox는 영상이 그려진 실제 영역(object-fit 레터박스 고려)에 맞춰 그린다. 결과가 너무 오래되면(예: 3초 이상) 흐리게 표시하거나 지운다.

## 7. 폴더 구조

```
realtime-vlm/
├── PLAN.md
├── README.md
├── Dockerfile               # multi-stage: node(frontend build) → vllm 런타임
├── build_docker.sh          # docker build -t realtime-vlm:latest .
├── run_docker.sh            # docker run --gpus all ... realtime-vlm:latest
├── entrypoint.sh            # 서빙 기동 → health 대기 → app 기동
├── serving/
│   ├── vllm.sh              # vllm serve 인자
│   └── triton/              # (P4) model_repository
├── app/
│   ├── main.py              # FastAPI: REST, WS, MJPEG, 정적 파일
│   ├── config.py            # 환경변수 설정
│   ├── schemas.py
│   ├── streams/
│   │   ├── manager.py
│   │   ├── source.py        # RTSP / MP4 디코더, 링버퍼
│   │   └── worker.py        # 샘플링 → 추론 → 결과/이벤트 발행
│   ├── inference/
│   │   ├── base.py          # InferenceClient 인터페이스
│   │   ├── vllm_client.py
│   │   ├── triton_client.py # (P4)
│   │   ├── prompts.py
│   │   └── parser.py
│   └── events/
│       └── store.py         # 쿨다운, 스냅샷, jsonl
├── frontend/
│   └── src/{App.tsx, api.ts, ws.ts, components/{AddStreamBar, StreamGrid, StreamTile, BoxOverlay, EventPanel}.tsx}
├── videos/                  # 테스트용 mp4 (git 제외)
├── data/                    # 이벤트 스냅샷 (git 제외)
└── tools/
    ├── test_infer.py        # 이미지 N장 → 서빙 서버 직접 호출
    └── rtsp_sim.sh          # mediamtx + ffmpeg로 mp4를 RTSP로 송출
```

## 8. Docker

- **베이스 이미지**: `vllm/vllm-openai:<Qwen3-VL을 지원하는 버전>`. Triton은 P4에서 비교할 때 별도 Dockerfile이나 build-arg로 추가한다.
- **multi-stage 빌드**: ① `node:20`에서 `frontend`를 빌드하고 ② vllm 이미지에 ffmpeg, PyAV, opencv-headless, fastapi, uvicorn을 설치한 뒤 빌드된 frontend를 복사한다.
- **스크립트** (각각 한 줄):
  ```bash
  # build_docker.sh
  docker build -t realtime-vlm:latest .
  # run_docker.sh
  docker run --rm -it --gpus all --shm-size=4g -p 127.0.0.1:8000:8000 -v $HOME/.cache/huggingface:/root/.cache/huggingface -v $(pwd)/videos:/videos -v $(pwd)/data:/data -e MODEL_ID=${MODEL_ID:-Qwen/Qwen3-VL-2B-Instruct} realtime-vlm:latest
  ```
- **주요 환경변수**: `SERVING`(vllm), `MODEL_ID`, `GPU_MEM_UTIL`(0.78), `MAX_MODEL_LEN`(8192), `NUM_FRAMES`(4), `WINDOW_SEC`(2), `FRAME_SIZE`(448), `EVENT_COOLDOWN_SEC`(5)

## 9. 개발 단계

| 단계 | 내용 | 완료 기준 |
|---|---|---|
| **P0** | git init, Dockerfile, build/run 스크립트, entrypoint로 vLLM 기동 | 컨테이너가 뜨고 `/v1/models` 응답, VRAM 사용량 기록 |
| **P1** | `test_infer.py`로 이미지 1장 / N장 grounding 테스트, 프롬프트·parser 확정, 이미지 N장 방식과 video 입력 방식 비교 | 샘플 영상에서 bbox가 맞음, latency 기록 |
| **P2** | app: source(RTSP/MP4), 링버퍼, worker, manager, vLLM client, REST/WS/MJPEG | mp4 1개 → WS로 bbox 수신 |
| **P3** | 이벤트 스토어: 쿨다운, 스냅샷, jsonl, API | 검출 시 이벤트와 스냅샷 생성 |
| **P4-UI** | frontend: 추가/업로드/그리드/오버레이/이벤트 패널 | 브라우저에서 4스트림과 이벤트 확인 |
| **P5** | 안정화: RTSP 재연결, 4스트림 부하 측정, 파라미터 튜닝, 2B와 4B 비교, README | 성능 표, 모델 비교 결과 |
| **P6** | Triton(vLLM backend) 추가, `SERVING=triton`, vLLM과 비교 | 같은 UI로 동작, 비교 표 |

각 단계가 끝날 때마다 git 커밋한다.

## 10. 위험 요소와 대응

- **VRAM 여유가 약 6.8GB**: 이미지 4장 × 비전 토큰 수 때문에 KV cache가 빠듯하다. 프레임 해상도, `MAX_MODEL_LEN`, `max_num_seqs`(4)로 조절한다.
- **2B 모델의 행동 인식 정확도**: "넘어지는 사람" 같은 동작 질의는 2B가 약할 수 있다. P5에서 4B FP8과 비교한다.
- **지연**: 멀티 프레임은 요청당 1~3초 걸릴 수 있다. 스트림이 4개면 서로 GPU를 나눠 쓴다. 우선 동작하게 만들고 측정 후 튜닝한다.
- **bbox와 화면의 시간차**: bbox는 추론 시점 프레임 기준이라 실제 화면보다 늦다. 결과에 `frame_ts`를 함께 보내 얼마나 오래된 결과인지 UI에 표시한다.
- **이미지 크기**: vLLM 이미지가 10GB 이상이라 첫 빌드가 오래 걸린다.

## 11. 진행 현황 (2026-11-18)

| 단계 | 상태 | 커밋 |
|---|---|---|
| P0 Docker + vLLM 서빙 | 완료 | d93a4a9 |
| P1 추론 방식/모델 평가 | 완료 (4B-FP8, two_stage) | a330ccc |
| P2 스트림/worker/API | 완료 | a266168 |
| P3 이벤트 로그 | 완료 | 4ea0d5f |
| P4-UI frontend | 완료 | 4ea0d5f |
| 구조 개편 (컨테이너 3개) | 완료 | (이번 커밋) |
| P5 안정화/튜닝 | 예정 | |
| P6 Triton 비교 | 예정 | |

## 12. 컨테이너 구조 (2026-11-18 변경)

컨테이너 3개, 소스 코드는 이미지에 넣지 않고 볼륨으로 마운트한다(수정할 때마다 docker build 하지 않기 위해).

```
realtime-vlm/
├── docker-compose.yml
├── build_docker.sh          # docker compose build (의존성 바뀔 때만)
├── run_docker.sh            # docker compose up -d
├── configs/config.yaml      # serving·backend 공유
├── serving/   Dockerfile, entrypoint.sh, gen_env.py, vllm.sh, (triton/ P6)
├── backend/   Dockerfile, requirements.txt, start.sh, app/, examples/
├── frontend/  Dockerfile, package.json, src/, vite.config.ts
├── tools/rtsp_sim.sh        # 테스트용 RTSP 카메라 (compose 네트워크)
└── model_files/ videos/ data/
```

| 컨테이너 | 이미지 | 마운트 | 포트 (127.0.0.1) | 코드 수정 시 |
|---|---|---|---|---|
| realtime-vlm-serving | vllm/vllm-openai v0.33.0 | serving/, configs/, model_files/ (읽기 전용) | 9000 | 컨테이너 재시작 (모델 로드 1~2분) |
| realtime-vlm-backend | python:3.12-slim + ffmpeg | backend/, configs/, videos/, data/ | 8001 (API, /docs) | uvicorn 자동 재시작 |
| realtime-vlm-frontend | node:22-slim (Vite 개발 서버) | frontend/ (node_modules는 이미지) | **8000 (UI)** | HMR로 즉시 반영 |

- backend는 serving이 healthy가 된 뒤에 시작한다(compose healthcheck). backend → serving 주소는 `http://serving:9000`.
- frontend(Vite)가 `/api`, `/ws`, `/event-files`를 backend로 프록시한다.
- backend·frontend는 호스트 UID로 실행되어 `data/`, `videos/uploads/` 파일이 사용자 소유가 된다.
- 모든 이미지에 vim, net-tools, alias(`old`, `fn`, `wd`, `vi`), vimrc 설정이 들어 있다(`/etc/bash.bashrc`, `/etc/vim/vimrc.local`).
- 스트림 목록은 메모리에만 있다. backend가 재시작되면 스트림은 사라지고, 이벤트는 `data/events/`에 남는다.
- 실시간 박스는 추론에 쓴 프레임 기준이라 움직이는 대상에서는 약 1초 뒤처져 보인다. 이벤트 스냅샷은 같은 프레임에 그리므로 정확하다.

자주 쓰는 명령:
```bash
sh build_docker.sh && sh run_docker.sh                       # 빌드 + 실행 → http://127.0.0.1:8000
docker compose logs -f backend                               # 로그
docker compose restart serving                               # config의 serving 값 변경 후
sh tools/rtsp_sim.sh videos/samples/urfd-fall-01-rgb.mp4 cam1  # UI에 rtsp://rtsp-sim:8554/cam1 입력
sh backend/examples/run_vlm_inference.sh --video videos/samples/urfd-fall-01-rgb.mp4 --at 4 --prompt "넘어지는 사람"
docker exec realtime-vlm-backend python3 examples/ws_monitor.py --sec 20
```
