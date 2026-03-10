import csv
from collections import deque
import time

import numpy as np
import serial
from pythonosc.udp_client import SimpleUDPClient

# =========================
# CONFIG
# =========================
SERIAL_PORT = "COM4"
BAUD = 115200
print("it is pulled again and again!")

TD_IP = "127.0.0.1"
TD_PORT = 7000

CALIB_SECONDS = 20
WINDOW_SECONDS = 60
SEND_HZ = 150

# ---- CREATE PARTICIPANT ID ----
participant_id = int(time.strftime("%Y%m%d%H%M%S"))

# ---- Normalization drift-adaptation ----
DRIFT_UPDATE_MIN_SAMPLES = 200
LOW_PCT = 5
HIGH_PCT = 95
DRIFT_ALPHA = 0.02

# ---- Baseline-relative breath detection ----
BASELINE_ALPHA = 0.005

# Dynamic hysteresis (x = sig - baseline)
MIN_HYST = 0.010
HYST_GAIN = 0.35
AMP_ALPHA = 0.05

# Slope gating
RISE_SLOPE_TH = 0.0012
FALL_SLOPE_TH = 0.0012
DSIG_EMA_ALPHA = 0.25

# Refractory
MIN_BREATH_INTERVAL = 0.8

# ---- BPM window ----
BPM_WINDOW_SEC = 20.0
MINUTE_SEC = 60.0

# ---- Opacity smoothing ----
OPACITY_ATTACK = 0.20
OPACITY_RELEASE = 0.20

# ---- Visual smoothing ----
VISUAL_ALPHA = 0.15

# ---- Stabil post-calibration guard ----
POST_CALM_SECONDS = 1.0
USE_DRIFT_ADAPTATION = False


def ema(prev, x, alpha):
    return x if prev is None else (alpha * x + (1 - alpha) * prev)


def exp_smooth(prev, x, a):
    return x if prev is None else (a * x + (1 - a) * prev)


def bpm_to_opacity(bpm):
    if 5.0 <= bpm <= 7.0:
        return 1.0
    elif bpm < 5.0:
        return max(0.3, 0.3 + (bpm / 5.0) * 0.7)
    elif bpm <= 15.0:
        return 1.0 - (bpm - 7.0) * (0.2 / 8.0)
    elif bpm <= 20.0:
        return 0.8 - (bpm - 15.0) * (0.2 / 5.0)
    elif bpm <= 25.0:
        return 0.6 - (bpm - 20.0) * (0.3 / 5.0)
    else:
        return max(0.1, 0.3 - (bpm - 25.0) * 0.04)


def main():
    ser = serial.Serial(SERIAL_PORT, BAUD, timeout=1)
    time.sleep(2)

    osc = SimpleUDPClient(TD_IP, TD_PORT)
    t0 = time.time()

    # ---------- Calibration ----------
    calib_buf = []
    calib_start = time.time()
    calibrating = True
    low, high = None, None
    calib_end_time = None

    # ---------- Filters ----------
    fast_ema = None
    slow_ema = None
    visual_sm = None

    # ---------- Window for drift-adaptation ----------
    window = deque(maxlen=WINDOW_SECONDS * SEND_HZ)

    # ---------- Baseline + amplitude ----------
    base_ema = None
    amp_ema = None

    # ---------- Breath detection ----------
    breath_state = "exhale"
    last_inhale_event_t = None

    last_x = None
    dx_ema = None

    # inhale-start event timestamps for BPM window
    inhale_events = deque()

    # ---------- Minute tracking ----------
    current_minute_index = 0
    current_minute_count = 0
    minute_records = []
    last_completed_minute_bpm = 0.0

    # ---------- OSC send timing ----------
    last_send = 0.0
    send_period = 1.0 / SEND_HZ

    # ---------- Opacity ----------
    bpm_win = 0.0
    opacity_sm = None
    opacity = 1.0

    # ---------- Debug print timing ----------
    last_debug_print = 0.0
    DEBUG_PRINT_EVERY = 0.25

    # ---------- New debug-data storage ----------
    debug_records = []

    try:
        while True:
            line = ser.readline().decode(errors="ignore").strip()
            if not line:
                continue

            try:
                raw = float(line)
            except ValueError:
                continue

            t = time.time()
            t_sec = t - t0

            # -------- Finalize completed minute bins --------
            minute_index = int(t_sec // MINUTE_SEC)
            if minute_index > current_minute_index:
                minute_records.append({
                    "participant_id": participant_id,
                    "minute_index": current_minute_index + 1,
                    "start_sec": current_minute_index * MINUTE_SEC,
                    "end_sec": (current_minute_index + 1) * MINUTE_SEC,
                    "breath_count": current_minute_count,
                    "bpm_minute": float(current_minute_count)
                })
                last_completed_minute_bpm = float(current_minute_count)

                for skipped_idx in range(current_minute_index + 1, minute_index):
                    minute_records.append({
                        "participant_id": participant_id,
                        "minute_index": skipped_idx + 1,
                        "start_sec": skipped_idx * MINUTE_SEC,
                        "end_sec": (skipped_idx + 1) * MINUTE_SEC,
                        "breath_count": 0,
                        "bpm_minute": 0.0
                    })
                    last_completed_minute_bpm = 0.0

                current_minute_index = minute_index
                current_minute_count = 0

            # -------- CALIBRATION --------
            if calibrating:
                elapsed = t - calib_start
                calib_buf.append(raw)

                osc.send_message("/calib/elapsed", float(elapsed))
                osc.send_message("/calib/active", 1)

                tmp = (raw - 40.0) / (300.0 - 40.0)
                tmp = max(0.0, min(1.0, tmp))

                fast_ema = ema(fast_ema, tmp, alpha=0.35)
                slow_ema = ema(slow_ema, tmp, alpha=0.05)
                visual_sm = ema(visual_sm, tmp, alpha=VISUAL_ALPHA)

                if t - last_send >= send_period:
                    last_send = t
                    osc.send_message("/fsr/raw", float(tmp))
                    osc.send_message("/fsr/fast", float(fast_ema))
                    osc.send_message("/fsr/slow", float(slow_ema))
                    osc.send_message("/fsr/visual", float(visual_sm))

                    osc.send_message("/breath/state", 0)
                    osc.send_message("/breath/inhale_event", 0)
                    osc.send_message("/breath/bpm", 0.0)
                    osc.send_message("/breath/bpm_minute", float(last_completed_minute_bpm))
                    osc.send_message("/breath/opacity", 1.0)

                if elapsed >= CALIB_SECONDS and len(calib_buf) > 50:
                    low = float(np.percentile(calib_buf, LOW_PCT))
                    high = float(np.percentile(calib_buf, HIGH_PCT))

                    print(f"[CALIB DONE] low={low:.2f}, high={high:.2f}, range={high-low:.2f}")

                    if high - low < 5:
                        print("[CALIB] range too small, recalibrating...")
                        calib_buf.clear()
                        calib_start = time.time()
                        fast_ema = None
                        slow_ema = None
                        visual_sm = None
                        continue

                    calibrating = False
                    calib_end_time = t

                    osc.send_message("/calib/active", 0)
                    osc.send_message("/calib/low", float(low))
                    osc.send_message("/calib/high", float(high))

                    fast_ema = None
                    slow_ema = None
                    visual_sm = None
                    opacity_sm = None

                    base_ema = None
                    amp_ema = None
                    last_x = None
                    dx_ema = None
                    breath_state = "exhale"
                    inhale_events.clear()
                    last_inhale_event_t = None

                    window.clear()
                    for v in calib_buf[-window.maxlen:]:
                        window.append(v)

                continue

            # -------- SHORT POST-CALIBRATION CALM PERIOD --------
            if calib_end_time is not None and (t - calib_end_time) < POST_CALM_SECONDS:
                norm = (raw - low) / max(1e-6, (high - low))
                norm = max(0.0, min(1.0, norm))

                fast_ema = ema(fast_ema, norm, alpha=0.35)
                slow_ema = ema(slow_ema, norm, alpha=0.12)
                visual_sm = ema(visual_sm, norm, alpha=0.20)

                if t - last_send >= send_period:
                    last_send = t
                    osc.send_message("/fsr/raw", float(norm))
                    osc.send_message("/fsr/fast", float(fast_ema))
                    osc.send_message("/fsr/slow", float(slow_ema))
                    osc.send_message("/fsr/visual", float(visual_sm))

                    osc.send_message("/breath/state", 0)
                    osc.send_message("/breath/inhale_event", 0)
                    osc.send_message("/breath/bpm", 0.0)
                    osc.send_message("/breath/bpm_minute", float(last_completed_minute_bpm))
                    osc.send_message("/breath/opacity", 1.0)

                continue

            # -------- DRIFT-ADAPTATION WINDOW --------
            if USE_DRIFT_ADAPTATION:
                window.append(raw)
                if len(window) >= DRIFT_UPDATE_MIN_SAMPLES:
                    low_new = float(np.percentile(window, LOW_PCT))
                    high_new = float(np.percentile(window, HIGH_PCT))
                    low = (1.0 - DRIFT_ALPHA) * low + DRIFT_ALPHA * low_new
                    high = (1.0 - DRIFT_ALPHA) * high + DRIFT_ALPHA * high_new

            denom = high - low
            if denom < 1e-6:
                continue

            # -------- NORMALIZE 0..1 --------
            norm = (raw - low) / denom
            norm = max(0.0, min(1.0, norm))

            fast_ema = ema(fast_ema, norm, alpha=0.35)
            slow_ema = ema(slow_ema, norm, alpha=0.12)
            visual_sm = ema(visual_sm, norm, alpha=VISUAL_ALPHA)

            sig = float(slow_ema)

            # -------- BASELINE + RELATIVE SIGNAL x --------
            base_ema = ema(base_ema, sig, BASELINE_ALPHA)
            x = sig - float(base_ema)

            # -------- DYNAMIC HYSTERESIS --------
            amp_ema = ema(amp_ema, abs(x), AMP_ALPHA)
            amp = float(amp_ema if amp_ema is not None else 0.0)
            hyst = max(MIN_HYST, HYST_GAIN * amp)

            # -------- SLOPE on x --------
            if last_x is None:
                dx = 0.0
            else:
                dx = x - last_x
            last_x = x

            dx_ema = ema(dx_ema, dx, DSIG_EMA_ALPHA)
            dxs = float(dx_ema if dx_ema is not None else 0.0)

            # -------- STATE + INHALE-START EVENT --------
            inhale_event = False

            refractory_ok = True
            if last_inhale_event_t is not None and (t - last_inhale_event_t) < MIN_BREATH_INTERVAL:
                refractory_ok = False

            if breath_state == "exhale":
                if refractory_ok and x > +hyst and dxs > RISE_SLOPE_TH:
                    breath_state = "inhale"
                    inhale_event = True
                    last_inhale_event_t = t
                    inhale_events.append(t)
                    current_minute_count += 1
            else:
                if x < -hyst and dxs < -FALL_SLOPE_TH:
                    breath_state = "exhale"

            # -------- BPM WINDOW --------
            while inhale_events and (t - inhale_events[0]) > BPM_WINDOW_SEC:
                inhale_events.popleft()

            bpm_win = float(len(inhale_events) * 60.0 / BPM_WINDOW_SEC)

            # -------- OPACITY --------
            opacity_target = bpm_to_opacity(bpm_win)

            if opacity_sm is None:
                opacity_sm = opacity_target
            else:
                a = OPACITY_ATTACK if opacity_target > opacity_sm else OPACITY_RELEASE
                opacity_sm = exp_smooth(opacity_sm, opacity_target, a)

            opacity = float(opacity_sm)

            # -------- Save every processed sample to debug CSV buffer --------
            debug_records.append({
                "participant_id": participant_id,
                "t_sec": t_sec,
                "raw": raw,
                "low": low,
                "high": high,
                "norm": norm,
                "visual": float(visual_sm),
                "x": float(x),
                "hyst": float(hyst),
                "dx": float(dxs),
                "breath_state": breath_state,
                "inhale_event": int(inhale_event),
                "bpm_window": float(bpm_win),
                "opacity": float(opacity)
            })

            # -------- DEBUG PRINT --------
            if t - last_debug_print >= DEBUG_PRINT_EVERY:
                last_debug_print = t
                print(
                    f"raw={raw:.1f} | low={low:.1f} high={high:.1f} | "
                    f"norm={norm:.3f} | visual={visual_sm:.3f} | "
                    f"x={x:.3f} | hyst={hyst:.3f} | dx={dxs:.4f}"
                )

            # -------- OSC --------
            if t - last_send >= send_period:
                last_send = t

                osc.send_message("/fsr/raw", float(norm))
                osc.send_message("/fsr/fast", float(fast_ema))
                osc.send_message("/fsr/slow", float(slow_ema))
                osc.send_message("/fsr/visual", float(visual_sm))

                osc.send_message("/breath/state", 1 if breath_state == "inhale" else 0)
                osc.send_message("/breath/inhale_event", 1 if inhale_event else 0)

                osc.send_message("/breath/bpm", float(bpm_win))
                osc.send_message("/breath/bpm_minute", float(last_completed_minute_bpm))
                osc.send_message("/breath/opacity", float(opacity))

                osc.send_message("/breath/baseline", float(base_ema))
                osc.send_message("/breath/x", float(x))
                osc.send_message("/breath/hyst", float(hyst))
                osc.send_message("/breath/dx", float(dxs))

    except KeyboardInterrupt:
        print("Stopping...")

    finally:
        final_t_sec = time.time() - t0
        minute_records.append({
            "participant_id": participant_id,
            "minute_index": current_minute_index + 1,
            "start_sec": current_minute_index * MINUTE_SEC,
            "end_sec": final_t_sec,
            "breath_count": current_minute_count,
            "bpm_minute": float(current_minute_count)
        })

        debug_filename = f"breath_debug_{participant_id}.csv"
        with open(debug_filename, "w", newline="") as df:
            writer = csv.writer(df)
            writer.writerow([
                "participant_id", "t_sec", "raw", "low", "high", "norm",
                "visual", "x", "hyst", "dx", "breath_state",
                "inhale_event", "bpm_window", "opacity"
            ])
            for rec in debug_records:
                writer.writerow([
                    rec["participant_id"],
                    rec["t_sec"],
                    rec["raw"],
                    rec["low"],
                    rec["high"],
                    rec["norm"],
                    rec["visual"],
                    rec["x"],
                    rec["hyst"],
                    rec["dx"],
                    rec["breath_state"],
                    rec["inhale_event"],
                    rec["bpm_window"],
                    rec["opacity"]
                ])

        minutes_filename = f"breath_minute_bpm_{participant_id}.csv"
        with open(minutes_filename, "w", newline="") as mf:
            writer = csv.writer(mf)
            writer.writerow(["participant_id", "minute_index", "start_sec", "end_sec", "breath_count", "bpm_minute"])
            for rec in minute_records:
                writer.writerow([
                    rec["participant_id"],
                    rec["minute_index"],
                    rec["start_sec"],
                    rec["end_sec"],
                    rec["breath_count"],
                    rec["bpm_minute"]
                ])

        ser.close()
        print(f"Saved debug data to {debug_filename}")
        print(f"Saved minute BPM summary to {minutes_filename}")


if __name__ == "__main__":
    main()