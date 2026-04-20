# Tamil Voice → English Name

A single-page app that turns a spoken Tamil name into a clean English spelling
the user can click to copy. Designed for hospital front-desk use: a nurse taps
the mic, says a patient's name in Tamil, and picks the correct English
romanisation from four suggestions.

Pipeline:

```
Mic (browser)
   ↓  PCM 16 kHz → WAV blob (client-side, plain JS)
POST /api/transcribe
   ↓
Sarvam saarika:v2.5 (ta-IN, translit mode) → English spelling
   ↓
Suggester:
   ├─ ASR echo as suggestion #1 (always shown)
   ├─ Sarvam-m LLM → 2–3 alternate English spellings
   └─ Rule-based variants (th↔t, ee↔i, v↔w …) fill any remaining slots
   ↓
Response: { transcript, suggestions: [{name, source, score} × 4] }
```

Auto-stops on ~1 s silence so the user never has to click stop.

## Endpoints

- `GET  /api/health` — backend + Sarvam availability.
- `POST /api/transcribe` — multipart WAV upload; returns transcript + suggestions.
- `POST /api/suggest?q=<text>&k=4` — text-only query (skip ASR).

## Local setup

```bash
python -m venv .venv
# Windows:   .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # then paste your SARVAM_API_KEY
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Open http://localhost:8000 — tap the mic, speak a Tamil name, pick a spelling.

## Deploy to Railway

Railway handles WebSocket/HTTP, env vars, and Python apps natively — a good
fit for this backend.

1. **Push this repo to GitHub** (see the `git` section below).
2. Create an account at https://railway.app and click **New Project →
   Deploy from GitHub repo**, select this repo.
3. Railway auto-detects Python via `requirements.txt` + `Procfile` (or
   `railway.json`). No extra config needed.
4. Under the service's **Variables** tab, add:
   ```
   SARVAM_API_KEY=sk_...
   SARVAM_MODEL=saarika:v2.5
   SARVAM_LANG=ta-IN
   SARVAM_MODE=translit
   SARVAM_LLM_MODEL=sarvam-m
   ```
   **Do NOT commit `.env`** — it's gitignored. Variables are set only in the
   Railway dashboard.
5. Under **Settings → Networking**, click **Generate Domain**. Your app is
   live at `https://<project>.up.railway.app`.
6. Visit the health check: `https://<domain>/api/health` — should return
   `{"ok": true, "sarvam_stt_available": true, ...}`.

Railway's free tier gives ~$5 of usage credit/month, comfortably enough for
this POC. The service cold-starts in ~10 s if idle; first mic click after a
cold start will take a bit longer.

## git setup (first-time push)

```bash
cd hospital-voice-poc
git init
git add .
git commit -m "Initial commit: Tamil voice → English name (Sarvam STT + LLM)"
git branch -M main
# on github.com, create a new empty repo called e.g. hospital-voice-poc
# then:
git remote add origin https://github.com/<you>/hospital-voice-poc.git
git push -u origin main
```

Double-check `git status` shows `.env` as ignored before pushing. The included
`.gitignore` excludes `.env`, `__pycache__/`, and `backend/cache/`.

## Security — before you deploy

- **Rotate any API keys that appear in your local `.env`** before pushing.
  They are safe once set as Railway Variables, but not if they ever slipped
  into a commit.
- Railway assigns an HTTPS domain by default; keep CORS wide-open only if
  you're OK with other origins hitting your backend. Restrict via
  `allow_origins` in `backend/main.py` for production.

## Repo layout

```
hospital-voice-poc/
├── backend/
│   ├── main.py            # FastAPI — /api/{health,suggest,transcribe}
│   ├── sarvam_client.py   # Sarvam STT wrapper (saarika:v2.5, translit)
│   ├── sarvam_llm.py      # Sarvam chat.completions wrapper (sarvam-m)
│   ├── suggester.py       # ASR echo + LLM + rule-based variants
│   ├── variants.py        # Rule-based spelling mutations
│   ├── phonetic.py        # Tamil-aware consonant-skeleton encoder
│   └── romanize.py        # Tamil script → readable English (fallback)
├── frontend/
│   └── index.html         # Mic UI, VAD auto-stop, suggestion cards
├── requirements.txt       # Slim deps: fastapi + sarvamai + utils
├── Procfile               # Railway / Heroku start command
├── railway.json           # Railway build + healthcheck config
├── runtime.txt            # Python 3.12
├── .env.example           # Template — copy to .env, paste keys locally
└── .gitignore             # Excludes .env, __pycache__, backend/cache/
```
