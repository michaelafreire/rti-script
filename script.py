from collections import deque
import time

from supabase import create_client
import numpy as np
import serial
from pythonosc.udp_client import SimpleUDPClient

# =========================
# CONFIG
# =========================
SERIAL_PORT = "COM4"
BAUD = 9600
print("it is pulled!")

TD_IP = "127.0.0.1"
TD_PORT = 7000

CALIB_SECONDS = 20
WINDOW_SECONDS = 60
SEND_HZ = 50

# ---- CONNECT TO SUPABASE ----
SUPABASE_URL = "https://mgcesnjdswtzcnxsqovi.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im1nY2Vzbmpkc3d0emNueHNxb3ZpIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzI2MzY3OTYsImV4cCI6MjA4ODIxMjc5Nn0.BTBtWstvyDeJj5L_0_5tfoAQvNN2lYlWAwH8-lfzX3Q"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

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
BPM_WINDOW_SEC = 20.0   # rolling window for opacity feedback

# ---- Opacity smoothing ----
OPACITY_ATTACK = 0.20
OPACITY_RELEASE = 0.20

# ---- Visual smoothing (separate from detection) ----
VISUAL_ALPHA = 0.025


def ema(prev, x, alpha):
    return x if prev is None else (alpha * x + (1 - alpha) * prev)


def exp_smooth(prev, x, a):
    return x if prev is None else (a * x + (1 - a) * prev)


def bpm_to_opacity(bpm):
    """
    Desired mapping:
    - 5 to 7 BPM -> 1.0
    - 15 BPM -> ~0.8
    - 20 BPM -> ~0.6
    - 25 BPM -> ~0.3
    - higher -> approaches 0.1
    Also gives lower opacity for very slow breathing (<5 BPM).
    """
    if 5.0 <= bpm <= 7.0:
        return 1.0
    elif bpm < 5.0:
        # 0 BPM -> 0.3, 5 BPM -> 1.0
        return max(0.3, 0.3 + (bpm / 5.0) * 0.7)
    elif bpm <= 15.0:
        # 7 -> 1.0, 15 -> 0.8
        return 1.0 - (bpm - 7.0) * (0.2 / 8.0)
    elif bpm <= 20.0:
        # 15 -> 0.8, 20 -> 0.6
        return 0.8 - (bpm - 15.0) * (0.2 / 5.0)
    elif bpm <= 25.0:
        # 20 -> 0.6, 25 -> 0.3
        return 0.6 - (bpm - 20.0) * (0.3 / 5.0)
    else:
        # 25+ -> continue dimming, but not below 0.1
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

    # ---------- OSC send timing ----------
    last_send = 0.0
    send_period = 1.0 / SEND_HZ

    # ---------- Opacity ----------
    bpm_win = 0.0
    opacity_sm = None
    opacity = 1.0

    # ---------- DB buffer ----------
    db_buffer = []
    db_buffer_norm = []
    db_buffer_wave = []
    last_db_send = time.time()
    DB_SEND_INTERVAL = 0.1

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

            # -------- CALIBRATION --------
            if calibrating:
                elapsed = t - calib_start
                calib_buf.append(raw)

                osc.send_message("/calib/elapsed", float(elapsed))
                osc.send_message("/calib/active", 1)

                # Temporary normalized signal during calibration
                tmp = (raw - 40.0) / (300.0 - 40.0)
                tmp = max(0.0, min(1.0, tmp))

                fast_ema = ema(fast_ema, tmp, alpha=0.35)
                slow_ema = ema(slow_ema, tmp, alpha=0.05)
                visual_sm = ema(visual_sm, tmp, alpha=VISUAL_ALPHA)

                # Send temporary data so visuals can still react
                if t - last_send >= send_period:
                    last_send = t
                    osc.send_message("/fsr/raw", float(visual_sm))
                    osc.send_message("/fsr/fast", float(fast_ema))
                    osc.send_message("/fsr/slow", float(slow_ema))
                    osc.send_message("/fsr/visual", float(visual_sm))

                    osc.send_message("/breath/state", 0)
                    osc.send_message("/breath/inhale_event", 0)
                    osc.send_message("/breath/bpm", 0.0)
                    osc.send_message("/breath/opacity", 1.0)

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

                    # Reset trackers after calibration
                    base_ema = None
                    amp_ema = None
                    last_x = None
                    dx_ema = None
                    breath_state = "exhale"
                    inhale_events.clear()
                    last_inhale_event_t = None

                continue

            # -------- DRIFT-ADAPTATION WINDOW --------
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
            slow_ema = ema(slow_ema, norm, alpha=0.05)
            visual_sm = ema(visual_sm, norm, alpha=VISUAL_ALPHA)

            sig = float(slow_ema)

            # -------- BASELINE + RELATIVE SIGNAL x --------
            base_ema = ema(base_ema, sig, BASELINE_ALPHA)
            x = sig - float(base_ema)

            # If direction is inverted in your setup, uncomment this:
            # x = -x

            # -------- DB WRITE --------
            db_buffer.append(raw)
            db_buffer_norm.append(sig)
            db_buffer_wave.append(x)

            if time.time() - last_db_send >= DB_SEND_INTERVAL and db_buffer:
                try:
                    avg_raw = float(np.mean(db_buffer))
                    avg_norm = float(np.mean(db_buffer_norm))
                    avg_wave = float(np.mean(db_buffer_wave))
                    t_flush = time.time() - t0
                    row = {
                        "participant_id": participant_id,
                        "t_sec": t_flush,
                        "raw": avg_raw,
                        "norm": avg_norm,
                        "wave": avg_wave
                    }
                    supabase.table("breath_raw").insert([row]).execute()
                    db_buffer.clear()
                    db_buffer_norm.clear()
                    db_buffer_wave.clear()
                    last_db_send = time.time()
                except Exception as e:
                    print("Database error:", e)

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
                # Inhale start = exhale -> inhale transition
                if refractory_ok and x > +hyst and dxs > RISE_SLOPE_TH:
                    breath_state = "inhale"
                    inhale_event = True
                    last_inhale_event_t = t
                    inhale_events.append(t)

            else:  # inhale
                if x < -hyst and dxs < -FALL_SLOPE_TH:
                    breath_state = "exhale"

            # -------- BPM WINDOW (20s rolling) --------
            while inhale_events and (t - inhale_events[0]) > BPM_WINDOW_SEC:
                inhale_events.popleft()

            bpm_win = float(len(inhale_events) * 60.0 / BPM_WINDOW_SEC)

            # -------- OPACITY (directly from BPM) --------
            opacity_target = bpm_to_opacity(bpm_win)

            if opacity_sm is None:
                opacity_sm = opacity_target
            else:
                a = OPACITY_ATTACK if opacity_target > opacity_sm else OPACITY_RELEASE
                opacity_sm = exp_smooth(opacity_sm, opacity_target, a)

            opacity = float(opacity_sm)

            # -------- OSC --------
            if t - last_send >= send_period:
                last_send = t

                # visual signal is now separate
                osc.send_message("/fsr/raw", float(visual_sm))
                osc.send_message("/fsr/fast", float(fast_ema))
                osc.send_message("/fsr/slow", float(slow_ema))
                osc.send_message("/fsr/visual", float(visual_sm))

                osc.send_message("/breath/state", 1 if breath_state == "inhale" else 0)
                osc.send_message("/breath/inhale_event", 1 if inhale_event else 0)

                osc.send_message("/breath/bpm", float(bpm_win))
                osc.send_message("/breath/opacity", float(opacity))

                # Debug channels
                osc.send_message("/breath/baseline", float(base_ema))
                osc.send_message("/breath/x", float(x))
                osc.send_message("/breath/hyst", float(hyst))
                osc.send_message("/breath/dx", float(dxs))

    except KeyboardInterrupt:
        print("Stopping...")

    finally:
        ser.close()


if __name__ == "__main__":
    main()