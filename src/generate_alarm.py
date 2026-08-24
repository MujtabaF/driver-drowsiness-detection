"""Synthesizes a simple alarm tone so the project doesn't depend on an external audio file."""
import numpy as np
import wave
import os

OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "alarm.wav")

SAMPLE_RATE = 44100
DURATION = 0.6
FREQ_A = 950
FREQ_B = 700


def build_tone():
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    half = len(t) // 2
    tone = np.concatenate([
        np.sin(2 * np.pi * FREQ_A * t[:half]),
        np.sin(2 * np.pi * FREQ_B * t[half:]),
    ])
    fade = min(2000, len(tone) // 10)
    envelope = np.ones_like(tone)
    envelope[:fade] = np.linspace(0, 1, fade)
    envelope[-fade:] = np.linspace(1, 0, fade)
    tone = tone * envelope
    return np.int16(tone * 32767)


def main():
    audio = build_tone()
    with wave.open(OUT_PATH, "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SAMPLE_RATE)
        f.writeframes(audio.tobytes())
    print(f"Wrote alarm tone to {os.path.abspath(OUT_PATH)}")


if __name__ == "__main__":
    main()
