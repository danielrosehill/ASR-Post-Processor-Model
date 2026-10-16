# Dictation Scenarios (recording app)

A static, mobile-first web app for recording the **Phase 1 real evaluation set** (40 spontaneous
dictations, see [`docs/development-plan.md`](../docs/development-plan.md)). It shows one fictional
scenario card at a time from [`recording/scenarios.json`](../recording/scenarios.json), records you
dictating it, keeps the takes on the phone, and exports them as a zip.

**URL:** <https://danielrosehill.github.io/ASR-Post-Processor-Model/app/> (GitHub Pages, `main` branch,
repo root; `.nojekyll` at the root). HTTPS is required for microphone access.

The recordings are evaluation data only. They are never trained on (see the repo `CLAUDE.md`).

## Files

| File | What |
|---|---|
| `index.html`, `styles.css`, `app.js` | The app. No build step, no framework. |
| `sw.js` | Network-first service worker: fresh files when online, last good copy offline. Bump `CACHE` to drop old entries. |
| `sample-scenarios.json` | Test data (ids `X01`–`X05`, `version: 0`). Used **only** when `../recording/scenarios.json` returns 404; a yellow banner then says SAMPLE. Any other fetch failure shows an error instead. |

Only runtime dependency: JSZip 3.10.1 from cdnjs (pinned).

## How to use (Android Chrome)

1. Open the URL. Allow the microphone on the first recording.
2. Read the card: situation, the details to mention (tick them off if you like), and the highlighted
   delivery cues. **Do not read the card aloud; say it the way you would naturally dictate it.** The cast
   (fictional people) is under "Cast"; the taxonomy ids the scenario targets are under "why", for the
   researcher.
3. Tap **Record**, speak, tap **Stop**. The timer goes amber below the target range, green inside it,
   red above it. The thin bar above the buttons is the input level.
4. Play the take back, then **Keep**, **Retake** (discard and record again at once) or **Discard**.
   A scenario can hold several kept takes; the newest kept take is the **primary**.
5. Move with the arrows, or tap the `x/40` counter for the overview (done / needs review / skipped).
   **Skip for now** marks a card skipped and jumps to the next unrecorded one.
6. **Export** → **Export zip** → **Share zip** (Android share sheet, e.g. Drive) or **Download zip**.
   Single takes can be shared/saved from each card.
7. After the zip is safely off the phone: **Export → Clear all recordings… → Delete everything**.

Takes are saved to IndexedDB (database `dictation-scenarios`, stores `takes` and `meta`) the moment you
press Stop, so closing the tab loses nothing, including takes not yet reviewed. The app asks for
persistent storage; the export page says whether the browser granted it. Clearing site data in Chrome
deletes the recordings.

Recording settings: `getUserMedia({audio: {echoCancellation: false, noiseSuppression: false,
autoGainControl: false, channelCount: 1}})`, MediaRecorder at 128 kbps, first supported of
`audio/webm;codecs=opus`, `audio/webm`, `audio/ogg;codecs=opus`, `audio/mp4`. What the device actually
applied is recorded per take in `audio_settings`.

## Export format

```
dictation-takes-YYYYMMDD-HHMMSS.zip
├── manifest.json
├── scenarios.json          # byte-for-byte copy of the scenarios file the app loaded
└── takes/
    ├── S01_t1.webm         # <scenario id>_t<take number>.<ext>
    └── ...
```

Only **kept** takes are exported; takes awaiting review are not. Audio is stored uncompressed in the
zip (it is already Opus).

`manifest.json` (`format_version: 1`):

```json
{
  "format": "dictation-scenarios-export",
  "format_version": 1,
  "app_version": "1.0.0",
  "exported_at": "2026-10-16T13:36:00.234Z",
  "scenarios_version": 1,
  "scenarios_source": "recording/scenarios.json",
  "scenarios_file": "scenarios.json",
  "scenario_count": 40,
  "scenarios_with_takes": 38,
  "scenarios_missing": ["S17", "S33"],
  "takes_count": 39,
  "takes": [
    {
      "file": "takes/S01_t1.webm",
      "scenario_id": "S01",
      "take": 1,
      "primary": true,
      "mimeType": "audio/webm;codecs=opus",
      "duration_s": 21.4,
      "size_bytes": 342811,
      "recorded_at": "2026-10-16T13:30:02.118Z",
      "kept_at": "2026-10-16T13:30:40.551Z",
      "user_agent": "Mozilla/5.0 (Linux; Android ...) Chrome/...",
      "requested_constraints": {"echoCancellation": false, "noiseSuppression": false, "autoGainControl": false, "channelCount": 1},
      "audio_settings": {"echoCancellation": false, "noiseSuppression": false, "autoGainControl": false, "sampleRate": 48000, "channelCount": 1},
      "requested_bitrate": 128000
    }
  ]
}
```

- `primary`: exactly one per scenario id: the kept take with the highest take number. Use primaries for
  the evaluation set; other takes are spares.
- `duration_s`: wall-clock time between Start and Stop (WebM from Chrome has no duration header).
- `recorded_at`: when recording started. Join scenario text via `scenario_id` → `scenarios.json`.
- `scenarios_missing`: ids in the loaded scenarios file with no kept take.

## After recording: raw Parakeet transcripts (not implemented here)

The evaluation input is what **stock** Parakeet v2 produces. Convert each primary take to 16 kHz mono
WAV, then transcribe with `nvidia/parakeet-tdt-0.6b-v2`:

```bash
unzip dictation-takes-*.zip -d export
mkdir -p export/wav
jq -r '.takes[] | select(.primary) | .file' export/manifest.json | while read -r f; do
  ffmpeg -nostdin -loglevel error -i "export/$f" -ac 1 -ar 16000 -c:a pcm_s16le \
    "export/wav/$(basename "${f%.*}").wav"
done
```

```python
# uv venv .venv && uv pip install "nemo_toolkit[asr]"   (needs a GPU host; the laptop has no accelerator)
import glob, json, nemo.collections.asr as nemo_asr
model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2")
files = sorted(glob.glob("export/wav/*.wav"))
out = model.transcribe(files)
with open("raw-transcripts.jsonl", "w") as fh:
    for f, hyp in zip(files, out):
        fh.write(json.dumps({"file": f, "raw": hyp.text}) + "\n")
```

Keep the raw output unedited: it is the model input. The hand-cleaned layer-1 and layer-1+2 targets
are written separately (Phase 1, step 1).

## Status and known limits (2026-10-16)

- Confirmed on desktop Chrome via a local server: loads the 40-scenario file, card/overview/export
  rendering, IndexedDB persistence across reload, zip build (contents and manifest checked), clear-all.
- **Not yet confirmed on a phone:** microphone capture, whether Android honours the disabled
  echo-cancellation/noise-suppression/AGC (check `audio_settings` in the first export), the share sheet,
  and the screen wake lock while recording.
- Chrome's Web Share only accepts certain file types; a `.zip` may not be shareable, in which case only
  **Download zip** appears (saves to Downloads). Single `.webm` takes usually are.
- The service worker only registers over HTTPS (or `localhost`), so offline use starts after the first
  online visit to the Pages URL.
