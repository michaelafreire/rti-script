# import csv
from fileinput import filename
import time
from collections import deque
from supabase import create_client

import numpy as np
import serial
from pythonosc.udp_client import SimpleUDPClient

# =========================
# CONFIG
# =========================
SERIAL_PORT = "COM4"
BAUD = 9600

TD_IP = "127.0.0.1"
TD_PORT = 7000

CALIB_SECONDS = 20
WINDOW_SECONDS = 60
SEND_HZ = 50

# ---- CONNECT TO SUPABASE ----
SUPABASE_URL = "https://npqjepvydeaauwsymkbu.supabase.co"
# SUPABASE_KEY = ""

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---- CREATE PARTICIPANT ID ----
participant_id = int(time.strftime("%Y%m%d%H%M%S"))
print("Participant ID:", participant_id)


# ---- Normalization drift-adaptation ----
DRIFT_UPDATE_MIN_SAMPLES = 200
LOW_PCT = 5
HIGH_PCT = 95
DRIFT_ALPHA = 0.02

# ---- Baseline-relative breath detection ----
# Very slow baseline (removes "stuck at high level" issue)
BASELINE_ALPHA = 0.005   # smaller = slower baseline tracking

# Dynamic hysteresis (in "x" domain where x = sig - baseline)
MIN_HYST = 0.015         # don't go too tiny (noise immunity)
HYST_GAIN = 0.45         # how much of amplitude becomes hysteresis
AMP_ALPHA = 0.05         # EMA for amplitude estimate

# Slope gating (on x, not on absolute sig)
RISE_SLOPE_TH = 0.0025
FALL_SLOPE_TH = 0.0025
DSIG_EMA_ALPHA = 0.25

MIN_BREATH_INTERVAL = 0.8  # refractory (small breaths need shorter)

# ---- BPM window ----
BPM_WINDOW_SEC = 60.0

# ---- Rhythm scoring target ----
TARGET_BPM = 6.0
TARGET_PERIOD = 60.0 / TARGET_BPM

PERIOD_TOL_SEC = 1.2
STAB_WINDOW = 5
STAB_TOL_SEC = 1.0

BPM_TOL = 3.0
SCORE_POWER = 1.2
OPACITY_POWER = 1.6

# ---- Opacity smoothing ----
OPACITY_ATTACK = 0.20
OPACITY_RELEASE = 0.20
MIN_OPACITY = 0.0


def ema(prev, x, alpha):
    return x if prev is None else (alpha * x + (1 - alpha) * prev)


def clamp01(x):
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def exp_smooth(prev, x, a):
    return x if prev is None else (a * x + (1 - a) * prev)


def main():
    ser = serial.Serial(SERIAL_PORT, BAUD, timeout=1)
    time.sleep(2)

    osc = SimpleUDPClient(TD_IP, TD_PORT)

    t0 = time.time()

    # filename = f"breath_log_{int(time.time())}.csv"
    # f = open(filename, "w", newline="")
    # writer = csv.writer(f)
    # writer.writerow([
    #     "t_sec", "raw", "norm", "fast", "slow",
    #     "baseline", "x", "hyst",
    #     "state", "bpm_win", "bpm_inst", "period", "score", "opacity"
    # ])

    # ---------- Calibration ----------
    calib_buf = []
    calib_start = time.time()
    calibrating = True
    low, high = None, None

    # ---------- Filters ----------
    fast_ema = None
    slow_ema = None

    # ---------- Window for drift-adaptation ----------
    window = deque(maxlen=WINDOW_SECONDS * SEND_HZ)

    # ---------- Baseline + amplitude ----------
    base_ema = None
    amp_ema = None  # amplitude estimate of |x|

    # ---------- Breath detection ----------
    breath_state = "exhale"
    last_inhale_event_t = None

    last_x = None
    dx_ema = None

    # Peak/trough tracking in x domain
    peak_x = None
    peak_t = None
    trough_x = None
    trough_t = None

    periods = deque(maxlen=STAB_WINDOW)
    inhale_events = deque()

    # ---------- OSC send timing ----------
    last_send = 0.0
    send_period = 1.0 / SEND_HZ

    # ---------- Score / opacity ----------
    bpm_win = 0.0
    bpm_inst = 0.0
    period_mean = 0.0
    period_std = 0.0
    rhythm_score = 1.0
    opacity_sm = None
    opacity = 1.0

    db_buffer = []
    last_db_send = time.time()
    DB_SEND_INTERVAL = 0.2  # send once per 0.2 second

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

            db_buffer.append(raw)

            if time.time() - last_db_send >= DB_SEND_INTERVAL and db_buffer:
                try:
                    avg_raw = float(np.mean(db_buffer))  # average all readings in buffer
                    t_flush = time.time() - t0            # timestamp for this interval
                    row = {
                        "participant_id": participant_id,
                        "t_sec": t_flush,
                        "raw": avg_raw
                    }
                    supabase.table("breath_raw").insert([row]).execute()
                    print("Sent to DB:", row)  # optional for debugging
                    db_buffer.clear()
                    last_db_send = time.time()
                except Exception as e:
                    print("Database error:", e)


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

                if elapsed >= CALIB_SECONDS and len(calib_buf) > 50:
                    low = float(np.percentile(calib_buf, LOW_PCT))
                    high = float(np.percentile(calib_buf, HIGH_PCT))

                    if high - low < 5:
                        calib_buf.clear()
                        calib_start = time.time()
                        continue

                    calibrating = False
                    osc.send_message("/calib/active", 0)
                    osc.send_message("/calib/low", float(low))
                    osc.send_message("/calib/high", float(high))

                    # reset trackers
                    base_ema = None
                    amp_ema = None
                    last_x = None
                    dx_ema = None
                    peak_x = peak_t = None
                    trough_x = trough_t = None
                    breath_state = "exhale"
                    inhale_events.clear()
                    periods.clear()
                    last_inhale_event_t = None

                continue

            # -------- DRIFT-ADAPTATION WINDOW --------
            window.append(raw)
            if len(window) >= DRIFT_UPDATE_MIN_SAMPLES:
                low_new = float(np.percentile(window, LOW_PCT))
                high_new = float(np.percentile(window, HIGH_PCT))
                low = (1.0 - DRIFT_ALPHA) * low + DRIFT_ALPHA * low_new
                high = (1.0 - DRIFT_ALPHA) * high + DRIFT_ALPHA * high_new

            denom = (high - low)
            if denom < 1e-6:
                continue

            # -------- NORMALIZE 0..1 --------
            norm = (raw - low) / denom
            norm = max(0.0, min(1.0, norm))

            fast_ema = ema(fast_ema, norm, alpha=0.35)
            slow_ema = ema(slow_ema, norm, alpha=0.05)
            sig = float(slow_ema)

            # -------- BASELINE + RELATIVE SIGNAL x --------
            base_ema = ema(base_ema, sig, BASELINE_ALPHA)
            x = sig - float(base_ema)  # baseline-relative

            # If direction is inverted in your setup, flip here:
            # x = -x

            # amplitude estimate -> dynamic hysteresis
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

            # -------- EVENT + STATE --------
            inhale_event = False
            inhale_event_t = None

            refractory_ok = True
            if last_inhale_event_t is not None and (t - last_inhale_event_t) < MIN_BREATH_INTERVAL:
                refractory_ok = False

            if breath_state == "exhale":
                # track trough in x
                if trough_x is None or x < trough_x:
                    trough_x = x
                    trough_t = t

                # enter inhale when x is sufficiently positive + rising
                if refractory_ok and x > +hyst and dxs > RISE_SLOPE_TH:
                    breath_state = "inhale"
                    peak_x = x
                    peak_t = t

            else:  # inhale
                # track peak in x
                if peak_x is None or x > peak_x:
                    peak_x = x
                    peak_t = t

                # peak-based inhale_event when slope turns down
                if refractory_ok and peak_t is not None and dxs < -FALL_SLOPE_TH:
                    inhale_event = True
                    inhale_event_t = peak_t

                # exit inhale when x is sufficiently negative + falling
                if x < -hyst and dxs < -FALL_SLOPE_TH:
                    breath_state = "exhale"
                    trough_x = x
                    trough_t = t
                    peak_x = peak_t = None

            # -------- BPM WINDOW / PERIOD --------
            if inhale_event and inhale_event_t is not None:
                inhale_events.append(inhale_event_t)

                if last_inhale_event_t is not None:
                    p = inhale_event_t - last_inhale_event_t
                    if 1.0 < p < 30.0:
                        periods.append(p)
                last_inhale_event_t = inhale_event_t

            while inhale_events and (t - inhale_events[0]) > BPM_WINDOW_SEC:
                inhale_events.popleft()

            bpm_win = float(len(inhale_events) * 60.0 / BPM_WINDOW_SEC)

            if len(periods) >= 2:
                period_mean = float(np.mean(periods))
                period_std = float(np.std(periods))
                bpm_inst = float(60.0 / period_mean) if period_mean > 1e-6 else 0.0
            else:
                period_mean = 0.0
                period_std = 0.0
                bpm_inst = 0.0

            # -------- SCORE --------
            if len(periods) >= 2 and period_mean > 1e-6:
                err_period = abs(period_mean - TARGET_PERIOD)
                err_bpm = abs(bpm_inst - TARGET_BPM)

                period_score = 1.0 - clamp01(err_period / PERIOD_TOL_SEC)
                bpm_score = 1.0 - clamp01(err_bpm / BPM_TOL)
                stab_score = 1.0 - clamp01(period_std / STAB_TOL_SEC)

                rhythm_score = (0.55 * period_score + 0.30 * bpm_score + 0.15 * stab_score)
                rhythm_score = clamp01(rhythm_score ** SCORE_POWER)
            else:
                rhythm_score = 1.0

            # -------- OPACITY --------
            opacity_target = MIN_OPACITY + (1.0 - MIN_OPACITY) * (rhythm_score ** OPACITY_POWER)
            if opacity_sm is None:
                opacity_sm = opacity_target
            else:
                a = OPACITY_ATTACK if opacity_target > opacity_sm else OPACITY_RELEASE
                opacity_sm = exp_smooth(opacity_sm, opacity_target, a)
            opacity = float(opacity_sm)

            # -------- OSC --------
            if t - last_send >= send_period:
                last_send = t

                osc.send_message("/fsr/raw", float(norm))
                osc.send_message("/fsr/fast", float(fast_ema))
                osc.send_message("/fsr/slow", float(slow_ema))

                osc.send_message("/breath/state", 1 if breath_state == "inhale" else 0)
                osc.send_message("/breath/inhale_event", 1 if inhale_event else 0)

                osc.send_message("/breath/bpm", float(bpm_win))
                osc.send_message("/breath/bpm_inst", float(bpm_inst))
                osc.send_message("/breath/period", float(period_mean))
                osc.send_message("/breath/period_std", float(period_std))

                osc.send_message("/breath/score", float(rhythm_score))
                osc.send_message("/breath/opacity", float(opacity))

                # debug (çok işe yarıyor)
                osc.send_message("/breath/baseline", float(base_ema))
                osc.send_message("/breath/x", float(x))
                osc.send_message("/breath/hyst", float(hyst))
                osc.send_message("/breath/dx", float(dxs))

            # # -------- CSV --------
            # writer.writerow([
            #     t_sec, raw, norm,
            #     float(fast_ema), float(slow_ema),
            #     float(base_ema), float(x), float(hyst),
            #     1 if breath_state == "inhale" else 0,
            #     bpm_win, bpm_inst, period_mean, rhythm_score, opacity
            # ])

    except KeyboardInterrupt:
        print("Stopping...")

    finally:
        # f.close()
        ser.close()
        # print("Saved:", filename)


if __name__ == "__main__":
    main()
