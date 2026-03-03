# import csv
from fileinput import filename
import time
from collections import deque
from supabase import create_client

import numpy as np
import serial
from pythonosc.udp_client import SimpleUDPClient

# ---- CONFIG ----
SERIAL_PORT = "COM4"
BAUD = 9600

TD_IP = "127.0.0.1"
TD_PORT = 7000

CALIB_SECONDS = 20
WINDOW_SECONDS = 60
SEND_HZ = 50

# ---- CONNECT TO SUPABASE ----
SUPABASE_URL = "https://npqjepvydeaauwsymkbu.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5wcWplcHZ5ZGVhYXV3c3lta2J1Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzIwMjg4NzAsImV4cCI6MjA4NzYwNDg3MH0.5S2yGI80P9xsnMK8fWv1ErSE4aPxLL3Qqk0flVThyq0"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- BPM (60s rolling window) ---
BPM_WINDOW_SEC = 60.0
UPPER_TH = 0.60
LOWER_TH = 0.40
MIN_BREATH_INTERVAL = 1.0

# --- REAL-TIME TARGET (0.1 Hz) ---
TARGET_BPM = 6.0
TARGET_PERIOD = 60.0 / TARGET_BPM  # 10 sec

# Still strict, but not binary
PERIOD_TOL_SEC = 1.0     # ±1 sec
STAB_WINDOW = 3          # last 3 periods
STAB_TOL_SEC = 0.6       # std tolerance

# Opacity dynamics
OPACITY_ATTACK = 0.25
OPACITY_RELEASE = 0.25
MIN_OPACITY = 0.0        # can fully disappear

# Soft penalty shaping
BPM_TOL = 1.5            # ±1.5 bpm tolerance for smooth drop
SCORE_POWER = 1.2        # score shaping
OPACITY_POWER = 1.5      # opacity shaping


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
    # writer.writerow(["t_sec", "raw"])

    calib_buf = []
    calib_start = time.time()
    calibrating = True
    low, high = None, None

    fast_ema = None
    slow_ema = None

    breath_state = "exhale"
    breath_times = deque()
    last_breath_t = 0.0
    bpm = 0.0

    window = deque(maxlen=WINDOW_SECONDS * SEND_HZ)

    last_send = 0.0
    send_period = 1.0 / SEND_HZ

    # --- Rhythm State ---
    last_inhale_t = None
    periods = deque(maxlen=STAB_WINDOW)
    bpm_inst = 0.0
    period_mean = 0.0
    period_std = 0.0
    rhythm_score = 1.0
    opacity_sm = None
    opacity = 1.0

    # --- Buffer to save data to SupaBase ---
    db_buffer = []
    last_db_send = time.time()
    DB_SEND_INTERVAL = 0.5  # send once per 0.5 second

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
            participant_id = int(time.strftime("%Y%m%d%H%M%S"))
            print("Participant ID:", participant_id)
            # writer.writerow([t_sec, raw])
            db_buffer.append({
                "participant_id": participant_id,
                "t_sec": t_sec,
                "raw": raw,
            })
            if time.time() - last_db_send >= DB_SEND_INTERVAL and db_buffer:
                try:
                    supabase.table("breath_raw").insert(db_buffer).execute()
                except Exception as e:
                    print("Database error:", e)

            db_buffer.clear()
            last_db_send = time.time()

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

                # end calibration
                if elapsed >= CALIB_SECONDS and len(calib_buf) > 50:
                    low = float(np.percentile(calib_buf, 5))
                    high = float(np.percentile(calib_buf, 95))

                    if high - low < 5:
                        calib_buf.clear()
                        calib_start = time.time()
                        continue

                    calibrating = False
                    osc.send_message("/calib/active", 0)
                    osc.send_message("/calib/low", float(low))
                    osc.send_message("/calib/high", float(high))

                continue

            # -------- NORMALIZED SIGNAL --------
            window.append(raw)

            if len(window) > 200:
                low_new = float(np.percentile(window, 5))
                high_new = float(np.percentile(window, 95))
                low = 0.98 * low + 0.02 * low_new
                high = 0.98 * high + 0.02 * high_new

            denom = (high - low)
            if denom < 1e-6:
                continue

            norm = (raw - low) / denom
            norm = max(0.0, min(1.0, norm))

            fast_ema = ema(fast_ema, norm, alpha=0.35)
            slow_ema = ema(slow_ema, norm, alpha=0.05)

            # -------- BREATH DETECTION --------
            sig = float(slow_ema)
            now = t
            inhale_event = False

            if breath_state == "exhale" and sig > UPPER_TH and (now - last_breath_t) > MIN_BREATH_INTERVAL:
                breath_state = "inhale"
                last_breath_t = now
                breath_times.append(now)
                inhale_event = True
            elif breath_state == "inhale" and sig < LOWER_TH:
                breath_state = "exhale"

            while breath_times and (now - breath_times[0]) > BPM_WINDOW_SEC:
                breath_times.popleft()

            bpm = float(len(breath_times))

            # -------- PERIOD CALC --------
            if inhale_event:
                if last_inhale_t is not None:
                    p = now - last_inhale_t
                    if 2.0 < p < 30.0:
                        periods.append(p)
                last_inhale_t = now

            if len(periods) >= 2:
                period_mean = float(np.mean(periods))
                period_std = float(np.std(periods))
                bpm_inst = 60.0 / period_mean if period_mean > 1e-6 else 0.0
            else:
                period_mean = 0.0
                period_std = 0.0
                bpm_inst = 0.0

            # -------- SOFT RHYTHM SCORE --------
            if len(periods) >= 2 and period_mean > 1e-6:
                err_period = abs(period_mean - TARGET_PERIOD)
                err_bpm = abs(bpm_inst - TARGET_BPM)

                period_score = 1.0 - clamp01(err_period / PERIOD_TOL_SEC)
                bpm_score = 1.0 - clamp01(err_bpm / BPM_TOL)
                stab_score = 1.0 - clamp01(period_std / STAB_TOL_SEC)

                # Weighted blend (prevents "min()" cliff)
                rhythm_score = (
                    0.5 * period_score +
                    0.3 * bpm_score +
                    0.2 * stab_score
                )

                # Non-linear shaping
                rhythm_score = clamp01(rhythm_score ** SCORE_POWER)
            else:
                rhythm_score = 1.0

            # -------- OPACITY (SMOOTH + NON-LINEAR) --------
            # non-linear mapping: small errors -> small dim, big errors -> strong dim
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

                osc.send_message("/breath/bpm", float(bpm))
                osc.send_message("/breath/bpm_inst", float(bpm_inst))
                osc.send_message("/breath/period", float(period_mean))
                osc.send_message("/breath/period_std", float(period_std))

                osc.send_message("/breath/score", float(rhythm_score))
                osc.send_message("/breath/opacity", float(opacity))

                osc.send_message("/breath/state", 1 if breath_state == "inhale" else 0)

    except KeyboardInterrupt:
        print("Stopping...")

    finally:
        # f.close()
        ser.close()
        # print("Saved:", filename)

if __name__ == "__main__":
    main()
