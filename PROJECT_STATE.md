# PROJECT_STATE.md

Context-handoff for a fresh Claude Code session, zero prior context. Read this before touching code. Reflects `backend/main.py` and `frontend/app/page.tsx` as read in full and verified line-by-line as of commit `6b2260a` ("perspective-correct keyboard calibration with 2D key-shape modeling (Stage 1.5)"), then updated for commit `6914904` (proximity measurement functions) and for the consolidation pass that followed it (§3 item 15). If reading this later, run `git log --oneline -10` and sanity-check the commit history still matches before trusting anything below over the actual code.

---

## 1. Goal & Architecture

**What it does:** bird's-eye-view video of a piano performance in → engraved, print-ready sheet music PDF out.

| Layer | Technology | File |
|---|---|---|
| Backend | FastAPI, single file | `backend/main.py` (1178 lines) |
| Frontend | Next.js / React, single file, client component | `frontend/app/page.tsx` (861 lines) |
| Audio pitch detection | `basic-pitch` (Spotify neural model) | via `basic_pitch.inference.predict` |
| Notation rendering | `abjad` → LilyPond → PDF | — |
| Video hand tracking | MediaPipe Tasks API (`HandLandmarker`) | requires downloaded `.task` model |

**Intended final architecture:** audio pipeline detects notes/rhythm (as now) AND video pipeline (keyboard calibration + hand-motion press detection) cross-references which physical key was actually struck at each audio-detected onset, resolving ambiguities the audio-only pipeline can't (octave errors, phantom notes, missed/extra notes). Two independent signal sources converging on one answer.

**Built vs. unbuilt, precisely:**
- Audio pipeline: **fully built and live**, wired end-to-end into `/api/transcribe`.
- Video/CV layer: **built as isolated, individually-validated, completely unwired functions.** `build_keyboard_calibration()` and `detect_key_press_at_timestamp()` exist, work, and are validated against real footage — but have **zero call sites** anywhere in `analyze_audio()` or `/api/transcribe` (verified via grep). The only CV-adjacent code that actually *runs* in the live pipeline is a 3-line diagnostic print loop (see §2).
- Frontend calibration UI: **fully built, fully interactive, gates the Transcribe button** — but the calibration data it captures (4 corners, leftmost/rightmost note) is **never sent to the backend**. Confirmed by reading `uploadFile()` in full: only `file`, `time_signature`, `tempo_bpm` go into the `FormData`. The UI *looks* wired (blocks submission until filled in) but isn't connected to anything server-side. Easy trap for a future session to miss.
- Cross-referencing video against audio (the actual point of having both) — **not started**. This is Stage 3, unplanned in code.
- Music theory engine (chord naming, key detection) — **not started at all**.

---

## 2. Current Pipeline (verified against actual code)

### `analyze_audio(audio_path, video_path, time_signature=None, tempo_bpm_override=None)` — `main.py:1015`

| # | Step | Function |
|---|---|---|
| 1 | Load waveform, native sample rate | `librosa.load(sr=None)` |
| 2 | Duration | `librosa.get_duration()` |
| 3 | Parse time signature or default 4/4 (auto never attempts detection — a separate, harder, unbuilt problem) | `parse_time_signature`, `time_signature_bar_length_beats` |
| 4 | Tempo: user override (×1.5 if compound meter, since compound tempo markings are dotted-quarter) or `librosa.beat.beat_track` | inline in `analyze_audio` |
| 5 | Raw polyphonic pitch detection, unfiltered — becomes API's `raw_notes` | `detect_notes_basic_pitch` |
| 6 | Merge same-pitch fragments (gap ≤0.05s, span cap 4.0 **beats**, converted to seconds using this piece's tempo — see §3 item 6) | `deduplicate_notes` |
| 7 | Chain onsets into chord/roll/single events (window 0.15s) | `group_notes_into_events` |
| 8 | Group-relative confidence filter (≥0.35 absolute AND ≥0.6 of event's own max) | `filter_event_notes` |
| 9 | Drop short (<0.4s) notes whose pitch class echoes one of the last 2 events within 0.5s | `suppress_decay_tail_notes` |
| 10 | Onset→beat, pick coarsest non-colliding quantization grid, snap, assign bar/beat | `convert_seconds_to_beats` → `select_best_quantization_resolution` → `quantize_beats` → `calculate_bar_structures` |
| 11 | Duration = gap to next quantized event's beat (NOT acoustic offset — see §4), classify note type | `calculate_note_durations` |
| 12 | Attach bar/beat_in_bar/duration_beats/note_type onto each event dict | inline |
| 13 | **Diagnostic only**, not real integration: reads a video frame at each of the first 3 raw onsets, prints success/failure. Does not cross-reference calibration or hand-tracking in any way. | `extract_frame_at_time` |
| 14 | Return dict: `duration_seconds, sample_rate, tempo_bpm, raw_notes, events, quantization_resolution, time_signature, tempo_source` | — |

### `POST /api/transcribe` — `main.py:1102`

1. Validate content-type (`video/mp4`, `video/quicktime`, `video/x-m4v`).
2. Save upload → `backend/uploads/<uuid>.<ext>`.
3. Extract audio via `moviepy.VideoFileClip` → 44.1kHz/16-bit PCM WAV.
4. Call `analyze_audio(...)` with `time_signature`/`tempo_bpm` from `Form(...)` params (both optional).
5. `generate_notation_pdf(events, pdf_path, time_signature=<parsed tuple>)` — real PDF via abjad/LilyPond.
6. `generate_placeholder_musicxml(musicxml_path)` — **still a static, hardcoded single-measure placeholder**, deliberately disconnected from real notes (§4).
7. Return JSON: upload metadata + `pdf_url`/`musicxml_url` + everything from `analyze_audio`'s dict spread in.

### CV-layer functions that exist but are NOT wired into the endpoint

| Function | Purpose | Wired? |
|---|---|---|
| `build_keyboard_calibration(leftmost_note, rightmost_note, front_left_px, front_right_px, back_left_px, back_right_px)` | 4-corner perspective homography → per-key pixel quads + `pixel_to_note(x,y)` reverse lookup | No — 0 call sites outside its own definition |
| `detect_key_press_at_timestamp(video_path, timestamp, calibration, window_seconds=0.2, fps=None)` | MediaPipe fingertip z-trajectory dip detection → ranked note candidates | No — 0 call sites outside its own definition |
| `extract_frame_at_time` | Single-frame extraction | Yes, but only for the diagnostic loop above |

### Key tunable constants (current values)

| Constant | Value | Where |
|---|---|---|
| `DEDUP_GAP_SECONDS` / `DEDUP_MAX_SPAN_BEATS` | 0.05s / 4.0 beats | dedup |
| `GROUPING_WINDOW_SECONDS` / `SOLID_THRESHOLD_SECONDS` | 0.15 / 0.03 | event grouping |
| `RELATIVE_CONFIDENCE_FRACTION` / `ABSOLUTE_CONFIDENCE_FLOOR` | 0.6 / 0.35 | confidence filter |
| `DECAY_TAIL_DURATION_THRESHOLD` / lookback / margin | 0.4s / 2 events / 0.5s | decay-tail suppression |
| quantization candidates / tolerance | `[1.0, 0.5, 0.25]` / 0.3 | `select_best_quantization_resolution` |
| `KEY_PRESS_WINDOW_SECONDS` / `KEY_PRESS_DIP_THRESHOLD` | 0.2 / 0.006 | CV press detection (unwired) |
| `WHITE_KEY_WIDTH_MM` | 23.6 (DIN 8995) | CV calibration |
| `WHITE_KEY_TOTAL_DEPTH_MM` / `BLACK_KEY_DEPTH_MM` | 145.0 / 100.0 | CV calibration depth |

**Frontend flow note:** the "Optional: help us get the rhythm right" card (time signature / tempo / pickup) is visible whenever `!isBusy`, **independent of file selection** — it's shown both before and after a file is chosen. The dropzone and the "Mark the keyboard" 4-corner calibration card are mutually exclusive, gated by `selectedFile`.

---

## 3. Chronological History — What Was Tried, What Happened, Why It Changed

**Do not re-propose any of the rejected approaches below without new evidence.**

1. **Monophonic detection (librosa YIN + piptrack)** → abandoned: real piano playing is polyphonic, single-pitch-per-frame detection can't represent chords at all.

2. **CQT (Constant-Q Transform) + peak-picking, polyphonic** → discovered pervasive harmonic-confusion phantoms: a struck note's overtone at +12/+19 semitones frequently peak-picked as a separately "played" note. Also a catastrophic failure mode where a near-silent tail produced an 8-note noise-floor "phantom chord." **Fundamentally unfixable with peak-picking** — nothing in the peak-picking step can distinguish a real fundamental from a strong harmonic peak; the approach was abandoned entirely, not patched.

3. **Model evaluation: `basic-pitch` (Spotify) vs. `piano_transcription_inference` (ByteDance)** — three-way bake-off against the CQT approach:
   - `piano_transcription_inference`: piano-specific, has pedal detection, but ~1x realtime on CPU (vs. basic-pitch's ~15-30x realtime) and its velocity/confidence showed **no separation** between real notes and phantoms. Install friction was also real: automatic checkpoint download via `wget` failed on macOS; manual staging would have required downloading and pickle-deserializing an external binary, which Claude Code's safety layer correctly declined without explicit user authorization.
   - `basic-pitch`: also produces phantom octave-harmonic notes (not fully solved by any approach tried), but its confidence score showed real, exploitable separation (~0.5-0.85 real vs. ~0.29-0.53 phantom) — this became the foundation for group-relative filtering (next item). **basic-pitch won** on this separation plus the 15-30x speed advantage.

4. **Global confidence threshold** (`CONFIDENCE_THRESHOLD = 0.6`, single cutoff) → catastrophic failure on real chord data: basic-pitch **splits its confidence "attention" across simultaneously-sounding notes**, so a real chord tone can legitimately score *lower* than an unrelated phantom elsewhere. Erased three entire 4-note chords in `chords-notes-mix.mp4`. **Replaced by `filter_event_notes`**: group notes into events first (chord-aware), then filter each note against its *own event's* strongest note (`RELATIVE_CONFIDENCE_FRACTION`) plus a low absolute floor. Immediately rescued all three erased chords.

5. **Onset-activation strength as a decay-tail discriminator** → tested and **rejected**: same simultaneity-splitting problem as global confidence (activation also gets divided across co-sounding notes). Replaced with the duration + pitch-recency heuristic now in `suppress_decay_tail_notes`.

6. **Dedup adjacency bug**: first implementation merged only *adjacent* entries in one global onset-sorted list → missed real merges when an unrelated note's onset sorted between two same-pitch fragments (confirmed: an E4 held across a chord change never merged, appeared as a spurious extra event). **Fix #1** (group by pitch first) → **over-corrected**: same-pitch fragments from entirely separate re-attacks (different chords, seconds apart) chain-merged into one absurd ~3s fake note whenever consecutive fragments happened to have near-zero gaps. **Fix #2**: `DEDUP_MAX_SPAN_SECONDS` = 2.0s cap, justified by the longest genuine single-attack note actually observed in testing (~1.65s). This regression was caught only because the fix was re-validated against known-good ground truth before acceptance, not just spot-checked — see §7.
   **Fix #3 (consolidation pass): the 2.0s cap was itself a real bug — tempo-dependent reasoning encoded as an absolute duration.** "Longest note observed is ~1.65s" is only true at these two clips' tempos; a whole note at 60 BPM lasts 4.0s and would have been incorrectly split. Replaced by `DEDUP_MAX_SPAN_BEATS = 4.0`, converted to seconds in `analyze_audio` using the tempo already resolved there (user override or auto-detect) before the value reaches `deduplicate_notes`. **4.0 beats was chosen empirically, not guessed**: sweeping the cap shows `chords-notes-mix.mp4` keeps byte-identical output only for 1.57–2.06s (= 3.22–4.22 beats at its 123 BPM, bounded below by a genuine 1.569s two-fragment merge), while `test.wav.mp4` is unconstrained (nothing in it merges >1 fragment, so the cap never binds there). 4.0 beats — a whole note in 4/4, the longest value `NOTATION_STANDARD_VALUES` can spell — is the only musically meaningful value in that intersection; **3.0 beats would be 1.46s at 123 BPM and would have split that real merge.** Note the residual asymmetry: 4.0 beats is 2.60s at `test.wav.mp4`'s 92.3 BPM, i.e. *more* permissive than the old 2.0s, so the chain-merge guard is looser than before at tempos below ~120 BPM — unobserved on these clips only because that clip never exercises the cap. `deduplicate_notes`'s `max_span_seconds` is now keyword-only with **no default**, so no future caller can silently reintroduce a tempo-blind constant.

7. **Notation rendering**: LilyPond/abjad chosen specifically for PDF/print engraving quality — LilyPond isn't primarily a MusicXML-authoring tool, so **MusicXML deliberately stays a static placeholder** rather than building a second half-supported export path (real scope, not yet planned). Rhythm spelling: an early **greedy duration-decomposition** approach (Stage B) produced musically illegal notation (didn't correctly split at every beat boundary crossed, no dotted/double-dotted support) → **replaced by `spell_rhythm`** (Stage B2), a beat-respecting recursive decomposition with double-dot support (see `main.py:145` docstring for the exact 3-rule priority).

8. **abjad accidental-language bug**: `abjad.Staff()` defaults to `language="english"` (`cs`/`ef` suffixes), not the Dutch-style `cis`/`es` the code originally assumed. Went undetected through two full validation stages because neither hardcoded test dataset happened to include an accidental — surfaced only when real detected data (which has plenty) was wired in.

9. **Adaptive multi-resolution quantization + collision guard**: originally a fixed 16th-note (0.25-beat) grid regardless of actual timing, producing heavily fragmented/tied notation. Replaced with `select_best_quantization_resolution` (tries 1.0 → 0.5 → 0.25, coarsest-first, average-deviation tolerance 0.3). **First version had a real bug**: a resolution could pass the average-deviation check while still collapsing two *distinct real onsets* onto the same beat (actual data loss, not just uglier notation). Fixed by adding an explicit collision check that disqualifies any resolution causing a beat collision, falling back to a finer grid.

10. **Time signature / tempo / pickup measure overrides**: added user-specifiable `time_signature` and `tempo_bpm` (replacing pure auto-detection where provided; compound-meter tempo interpreted as dotted-quarter and converted ×1.5 internally). **Pickup measure was fully implemented then deliberately removed from the backend** — `calculate_bar_structures` originally took a `pickup_beats` param that shortened bar 1, but this exposed an unresolved phase-shift bug in `spell_rhythm`'s bar-crossing logic (it assumes uniform bar length from position 0, which breaks when bar 1 is a different length). The backend logic was stripped out entirely; **the frontend pickup UI (yes/no + beat-count input) was deliberately left visible and interactive but functionally inert** — captured in state, never sent to the backend.

11. **CV layer, Stage 1** — 2-point (left/right pixel edge) keyboard calibration using real key-width geometry sourced from Wikimedia Commons "Pianoteilung.svg" (a to-scale technical diagram), cross-validated against DIN 8995 (white key 23.6mm, octave 165.2mm ≈ known 165.1mm standard). Validated by rendering computed key boundaries as an overlay on a real frame and visually confirming alignment. Known limitation flagged at the time: assumed a pure overhead camera, no perspective correction.

12. **CV layer, Stage 2** — MediaPipe hand-landmark motion-based key-press detection. **mediapipe 0.10.35: `mp.solutions.hands` is completely removed** (not merely deprecated) — only the Tasks API (`HandLandmarker`) exists, requiring a separately-downloaded `.task` model file (same pattern as basic-pitch's checkpoint; see §4). Detects a "dip and recover" pattern in fingertip z across a time window, not just proximity at one instant. Validated against `test.wav.mp4`'s 15 known onsets: **4/15 (27%) correct**. A real calibration bug was caught and fixed mid-stage: the initial hand-estimated calibration was geometrically self-consistent but had the *absolute* note-to-pixel mapping wrong by ~1.5-2 octaves (the thumb playing C4 was nowhere near the labeled C4 position) — caught by cross-referencing real fingertip x-positions against known ground-truth notes plus standard scale fingering (thumb=1, index=2, middle=3, thumb-under=1, index=2, middle=3, ring=4, pinky=5), not by trusting the initial visual estimate.

13. **CV layer, Stage 1.5** — rebuilt calibration around a proper 4-corner perspective homography (`cv2.getPerspectiveTransform`) replacing Stage 1's 2-point horizontal-only model, plus 2D key-shape modeling: black keys are physically shorter and set back from the front edge. Real depth dimensions researched from the same Pianoteilung.svg source (white key total visible depth 145mm, black key depth 100mm, white-only front zone 45mm) and cross-validated against independent real grand-piano forum measurements (142.9-149.2mm range). **Result: geometry visually verified correct** (overlay shows white/black key boxes aligned with real keys, correct asymmetric offsets) **but accuracy did not improve — 3/15, a slight regression from Stage 2's 4/15.** The *set* of correct notes shuffled almost entirely between the two runs even though the hit rate stayed flat — see §4 for the interpretation.

14. **CV layer, Stage 2.5 — lateral proximity measurement (measurement only, never wired).** Dropped motion/z entirely on the grounds that Stages 2 and 1.5 had shown MediaPipe's z signal cannot resolve a ~10mm key press, and measured only 2D lateral distance from each audio-detected note's key centre to the nearest hand landmark, at the single frame nearest each onset. Added `detect_hand_canonical_positions` and `lateral_distance_to_nearest_landmark_mm` (commit `6914904`) — **still zero pipeline call sites.** Findings that survive:
    - **All 21 hand landmarks discriminate better than the 5 fingertips**, and materially so — REAL-note median distance drops 6.9mm → 2.2mm while phantom distances are essentially unchanged. That was the difference between an overlapping distribution and a separated one. Do not assume fingertips are the right landmark set.
    - On `test.wav.mp4`, the three phantoms (F5, F5, and a spurious trailing C5 event) measured 139.8 / 85.2 / 72.9mm against a REAL max of ~29mm — a wide, clean gap.
    - **Proximity structurally cannot catch decay-tail phantoms.** The one decay-tail phantom measured 1.3mm, because the finger genuinely *is* still resting on the key it just played. This approach is orthogonal to `suppress_decay_tail_notes`, not a replacement for it.
    - **Camera framing bounds what is verifiable at all.** `chords-notes-mix.mp4` only shows C4–F5, so its three F3 phantoms are off-camera and unverifiable by construction. Widening the calibrated range cannot fix this — the keys are not in shot.
    - **Supporting evidence is thin: 3 phantoms in a single clip** (`test.wav.mp4`). See item 15 for why the other clip's measurements had to be discarded entirely.

15. **Consolidation pass — a false "basic-pitch octave error" finding, retracted.** Stage 2.5 reported that basic-pitch had mis-registered the final chord of `chords-notes-mix.mp4`, claiming the real notes were C#5/E5 because the video put fingers 0.9mm from C#5 and 73.8mm from C#4. **That finding is WRONG and is retracted.** The user has since confirmed the ground truth by watching the video: the final chord is a **root-position C#dim7 — C#4, E4, G4, A#4 — exactly as basic-pitch reported.** basic-pitch was correct; the video mapping was wrong.
    - **What actually happened:** the video read that chord as G4/A#4/C#5/E5 (first inversion). Every note is displaced by a consistent **+6 semitones (a tritone, ~80mm)** from the confirmed truth — the signature of a systematic lateral calibration offset in that clip's calibration, not a detection error.
    - **Root cause, and the general lesson:** that clip's absolute octave anchor was inferred by matching finger spacing against the burned-in "Fmaj7" caption. **Any single-chord spacing anchor is a weak constraint**, for two reasons worth remembering: (a) fingers commonly rest on keys they are not playing, so the fingers being matched may not be the notes sounding; (b) a diminished seventh is stacked minor thirds and therefore has *identical* finger spacing in every inversion — spacing alone cannot distinguish them. Anchor against a known *sequence* (as `test.wav.mp4`'s scale allows), not one chord.
    - **Consequence — all `chords-notes-mix.mp4` fingertip-to-key distance measurements are INVALID** and must not be cited: both claimed "octave errors" and the ev18 B4 measurement. `test.wav.mp4`'s measurements are unaffected; that clip's calibration was fitted and validated against its known 15-note scale sequence.
    - **Consequence — the proposed "octave correction" mechanism is CANCELLED.** Its entire evidence base was this bug. Do not re-propose it without new evidence.
    - The proximity-veto approach itself is **not** cancelled, but its evidence now rests on 3 phantoms in one clip (item 14) — thin.
    - **This failure mode does not exist in production.** The calibration UI has leftmost/rightmost note dropdowns, so a real user supplies the absolute anchor directly. Octave *inference* was only ever necessary because those dropdowns are not yet wired to the backend (§1). This is a test-harness limitation, not a product one.

---

## 4. Key Findings That Must Not Be Re-Litigated

- **basic-pitch splits confidence across simultaneous notes → a global threshold structurally cannot work; filtering must be group-relative.** Evidence: a single 0.6 global cutoff erased three entire 4-note chords in `chords-notes-mix.mp4`; switching to per-event relative filtering immediately rescued all three.
- **Notation duration comes from the RHYTHMIC gap between quantized onsets** (`calculate_note_durations`: next event's quantized beat − this event's), **NOT basic-pitch's acoustic offset**. A piano note's acoustic decay tail is often much longer than its intended notated value (a staccato-to-legato quarter note still rings well past the next beat) — using acoustic offsets would produce wildly overlong note values that don't reflect what the performer intended.
- **Perspective correction + 2D key-shape modeling was implemented correctly and did NOT close the CV accuracy gap** (4/15 → 3/15). Verified not a bug: the exact captured calibration was rendered as an overlay and visually confirmed correct (real key alignment, correct asymmetric black-key offsets). The specific *set* of correct notes changed almost completely between Stage 2 and Stage 1.5 even though the total stayed flat, and near-miss candidates routinely differ from the winning candidate by hundredths of a confidence point — this points to **z-signal noise / candidate discrimination as the real bottleneck, not calibration geometry**. Caveat, not fully closed: Stage 1.5's corner placement was a fresh visual judgment against the frame, not cross-checked against the same empirical finger-position ground truth that fixed Stage 2's calibration bug — so a residual absolute lateral offset in Stage 1.5 specifically isn't 100% ruled out either. Don't treat "geometry is fully exonerated" as more certain than that.
- **basic-pitch was NOT shown to make octave errors — that finding was retracted (§3 item 15).** A video-derived claim that it mis-registered a chord traced back to a ~80mm lateral calibration offset in the video, not to the detector. Treat any future "the audio got the register wrong" claim as a calibration suspect first, and confirm against user-stated ground truth before recording it.
- **An absolute-octave anchor must come from a known note SEQUENCE, not a single chord.** Fingers rest on keys they aren't playing, and symmetric chords (diminished sevenths especially — stacked minor thirds) have identical spacing in every inversion, so a one-chord spacing match cannot pin the register. This is the same class of mistake as the Stage 2 anchor bug (§3 item 12), caught the same way and worth the same caution.
- **MediaPipe 0.10.35: `mp.solutions.hands` is GONE, not deprecated.** Only `mp.tasks.vision.HandLandmarker` exists. Requires a separately-downloaded `.task` model at `backend/models/hand_landmarker.task` (gitignored, ~7.8MB, official Google storage URL — see `main.py:551-556` comment).
- **Library version drift has repeatedly caused real, non-obvious bugs**: abjad's English (`cs`/`ef`) vs. Dutch (`cis`/`es`) accidental naming; a librosa hop-length/frame-alignment issue during the earlier CQT-era phase (noted in prior project history, predates the current codebase which no longer has CQT code to re-verify this against); the MediaPipe solutions→Tasks API migration; `setuptools>=81` dropping the `pkg_resources` shim that `resampy` (a basic-pitch dependency) still imports, requiring a `setuptools==80.10.2` pin. **Always verify installed library APIs by direct introspection in the actual venv — never trust memory or training data about a library's current interface.**

---

## 5. Known Open Issues

Each checked against current code/most recent results, not assumed.

- **Surviving decay-tail phantom**: an F3 in the `test.wav.mp4` reference clip has acoustic duration (~0.66s) that clears `DECAY_TAIL_DURATION_THRESHOLD` (0.4s), so the pitch-recency check never even runs against it. Accepted gap at the current threshold.
- **"Single C" in `chords-notes-mix.mp4` never detected** — absent from `raw_notes` itself, not a filtering casualty. No pipeline-side fix possible without changing the detection model.
- **B4 cut by a dominant neighbor**: in an E-G-B-D chord, an unusually loud co-sounding E4 drags the relative-confidence bar high enough to exclude the real, quieter B4. Inherent tension in group-relative filtering.
- **Two phantom F3 leaks** in specific chords (F-A-C-E-type, D-F-A-C-type) survive relative-confidence filtering — their confidence doesn't fall low enough relative to the chord's strongest note.
- **CV accuracy 3-4/15 (~25-27%)**, not yet improved by geometry work — see §3/§4.
- **Single global tempo/time-signature**: no support for tempo drift or a mid-piece meter change. `tempo_bpm` is one scalar; `bar_length_beats`/`time_signature` apply uniformly to the entire piece.
- **Compound-meter and pickup-measure notation have never been visually validated against real audio** — no test recording with either feature exists (checked: neither `backend/uploads/` nor the user's Desktop has one). Only validated via direct math/unit-style checks in isolation (see prior session's temp validation scripts, since deleted per cleanup convention).
- **MusicXML still a static placeholder**, deliberately disconnected from real notes — see §3 item 7.
- **Music theory engine not started**: no key-signature detection, no chord-symbol/roman-numeral labeling. PDF renders in a hardcoded C-major key signature; accidentals are notated individually per note.
- **Frontend calibration drag handles use Mouse Events, not Pointer Events** (`onMouseDown` + window `mousemove`/`mouseup`) — chosen deliberately because the browser-automation tool used for testing didn't reliably synthesize Pointer Events, but this means the 4-corner calibration UI **won't work on touch-only devices** (tablets, phones). Real UX gap, not yet flagged to the user as a tradeoff at the time it was made.
- **`opencv-python` / `opencv-contrib-python` dual install**: mediapipe hard-requires `opencv-contrib-python`; the project already had `opencv-python`. Both coexist at the same version (5.0.0.93) and work in current testing, but installing both together is a known general OpenCV footgun — flagged in a `requirements.txt` comment, not resolved.
- **⚠️ EVERY numeric constant in this project is tuned against only two recordings.** `test.wav.mp4` and `chords-notes-mix.mp4` are the entire empirical basis, and several constants were fitted so closely to them that the safe range is razor-thin — the dedup cap sweep (§3 item 6) found `chords-notes-mix.mp4` tolerates only 1.57–2.06s before its output changes. **There is currently no way to tell a principled value from a coincidence of these two clips.** Specifically at risk:
  - `RELATIVE_CONFIDENCE_FRACTION` (0.6) and `ABSOLUTE_CONFIDENCE_FLOOR` (0.35) — both derived from one clip's chord confidences.
  - `DECAY_TAIL_DURATION_THRESHOLD` (0.4s) — see the seconds-constant audit note below.
  - `KEY_PRESS_DIP_THRESHOLD` (0.006, the MediaPipe z threshold) — explicitly a first-pass value, already known not to generalise.
  - `DEDUP_MAX_SPAN_BEATS` (4.0) — now tempo-correct, but its *value* still comes from these two clips.
  - Any future proximity-veto distance threshold — supporting evidence is 3 phantoms in one clip (§3 item 14).
  **Additional test recordings at varied tempos, rooms, and camera angles are the single highest-value thing that could be added to this project** — worth more than any further tuning of the above.
- **Seconds-valued constants audited (consolidation pass), none changed except the dedup cap.** Classification: `DEDUP_GAP_SECONDS` (0.05), `GROUPING_WINDOW_SECONDS` (0.15), `SOLID_THRESHOLD_SECONDS` (0.03), `KEY_PRESS_WINDOW_SECONDS` (0.2) are **PHYSICAL/PERCEPTUAL** — they describe detector fragmentation, human motor simultaneity, and hand motion, none of which scale with tempo; correctly left in seconds. `DECAY_TAIL_DURATION_THRESHOLD` (0.4s) and `DECAY_TAIL_DECAY_MARGIN_SECONDS` (0.5s) are **defensible as acoustic** (piano decay is a property of the instrument, not the tempo) **but are the weakest of the physical claims** — 0.4s is doing double duty as "short enough to be a decay artifact", which is partly a musical judgement and does drift with tempo. Flagged, deliberately not changed.
- **`note_type` labels are dead weight in the API response** — `calculate_note_durations` produces `"quarter"`/`"half"`/etc. and `analyze_audio` attaches them to each event (`main.py:1151`), but **nothing reads them**: the notation path (`spell_rhythm` / `_build_staff_input`) works purely from `duration_beats` in sixteenth-note integer units, and `frontend/app/page.tsx` neither displays the field nor declares it on its `EventData` type. Verified by grep across backend and frontend. Left in place pending a decision — removing them would change the public API response shape.
- **`generate_placeholder_pdf` (`main.py:37`) confirmed still present and still fully dead code** — re-verified in the consolidation pass: exactly one occurrence in the entire file, its own `def`, zero call sites. Safe to delete if asked; left alone because not explicitly requested.
- **CV layer entirely unwired** — see §1/§2. Not a bug, just unfinished integration.
- **Frontend calibration UI captures data it never sends** — see §1. Not a bug in the sense of crashing, but a real trap: the UI enforces completion of fields that currently do nothing server-side.

---

## 6. Test Assets & Ground Truth

Both clips came directly from the **user**, not from any detection output — treat as authoritative, do not regenerate or infer from pipeline output.

**`test.wav.mp4`** (~13.5s) — C major scale, single notes only. Ground truth, in order:
`C4 D4 E4 F4 G4 A4 B4 C5 B4 A4 G4 F4 E4 D4 C4` (15 notes, ascending then descending).
Known onset timestamps (seconds, at auto-detected tempo 92.3 BPM, used across Stage 2/1.5 CV validation): `1.660, 2.346, 2.950, 3.624, 4.298, 4.960, 5.622, 6.273, 6.912, 7.539, 8.202, 8.840, 9.479, 10.154, 10.850`. Possible additional unconfirmed trailing content past note 15 (a C5-ish artifact) — never confirmed against the source recording; open question, not a known bug.

**`chords-notes-mix.mp4`** (~12.15s) — mixed chords and single notes, 14 events (8 four-note chords + 6 single notes), in order:
1. F-A-C-E chord
2. F-G-B-D chord
3. single B
4. single C
5. E-G-B-D chord
6. E-G-A-C chord
7. single A
8. single B
9. D-F-A-C chord
10. D-F-G-B chord
11. single G
12. single A
13. C-E-G-B chord
14. C#-E-G-Bb chord — **registers user-confirmed: root-position C#dim7 = `C#4, E4, G4, A#4`** (the user verified this by watching the video; it also matches the clip's own burned-in "C#dim7" caption). This is the one event in this clip whose octaves are known.

Octave registers for the **other 13 events remain unconfirmed** by the user, beyond "mostly around octave 4" — judge pitch-class correctness first, register plausibility second. Do not infer this clip's registers from video/CV output: that mapping is known to carry a ~80mm lateral offset (§3 item 15).

**⚠️ `backend/uploads/` is NOT a stable fixture location.** The live endpoint always re-saves uploads under a fresh UUID, and old uploads rotate out over time — confirmed twice (Stage 2 and Stage 1.5 sessions both had to recover these exact files from the user's own Desktop after they'd disappeared from `backend/uploads/`). As of this writing both files are present at `~/Desktop/test.wav.mp4` and `~/Desktop/chords-notes-mix.mp4`, but don't assume permanence there either. **Identify by exact duration** (13.5s / 12.15s, via `ffprobe` or `moviepy`), never by filename or UUID — check fresh every session.

---

## 7. Working Process

- **Repo health check** (`git status`; confirm `backend/venv/` and `backend/models/` untracked; no stray files; `requirements.txt` captures the actual resolved dependency tree, not just top-level packages) **before AND after** any code-changing task.
- **Validate through the real live endpoint** (`/api/transcribe`), not standalone scripts, for final confirmation. Standalone scripts are fine for isolated diagnosis but are temporary and get deleted after use.
- **Validate against known ground truth BEFORE implementing a fix and AGAIN after**, before declaring success. Not optional process theater — two real regressions were only caught this way: the dedup span-cap over-correction (§3 item 6) and the Stage 2 calibration-anchor bug (§3 item 12), both of which looked fine on casual inspection and were wrong.
- **Visually inspect rendered output** — PDFs, CV calibration overlays — never just check for a successful exit code or a 200 response.
- **Prefer structural fixes over tunable-number patches** when the underlying problem is architectural (e.g. global→relative confidence filtering was a structural fix, not a threshold retune, because the problem — confidence isn't comparable across a whole piece — was architectural).
- **Report negative/failed results honestly.** The Stage 1.5 "3/15, no improvement" finding is in this document precisely because it was reported instead of buried — that's the expected standard going forward.
- **Git commits only when the user explicitly asks.** Validated, working changes sitting uncommitted across sessions is expected and fine.

---

## 8. Immediate Next Steps

All items below are **planned, not started** — zero code exists for any of them yet.

The CV layer's bottleneck is z-signal discrimination, not calibration geometry (§4). In order:

1. **Switch MediaPipe from per-frame IMAGE mode to VIDEO tracking mode.** Current `detect_key_press_at_timestamp` runs independent per-frame `IMAGE`-mode detection and stitches fingertip identity across frames with a nearest-position heuristic (`main.py:683-687`) — a likely noise source, since there's no persistent hand/finger identity the way MediaPipe's own `VIDEO`/`LIVE_STREAM` tracker would provide.
2. **Improve candidate scoring.** Currently threshold-then-rank by dip magnitude alone. Combine dip magnitude with lateral (u) proximity into one weighted score instead — observed failures frequently have the *correct* note present in the candidate list, just losing to an adjacent key by hundredths of a confidence point.
3. **Then Stage 3**: cross-reference video-detected key presses against audio-detected events — the actual point of building both pipelines (§1). Not started; no design work done yet either.

After the CV layer stabilizes: the music theory engine (chord naming, key detection), then the remaining minor issues in §5 (decay-tail phantom, phantom F3 leaks, compound-meter/pickup real-audio validation, real MusicXML generation).
