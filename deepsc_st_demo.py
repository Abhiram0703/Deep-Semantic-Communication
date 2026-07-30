"""
DeepSC-ST vs. Classical Text Transceiver — Tkinter Desktop Demo
================================================================

A local GUI that lets you:
  1. Pick a .wav audio file
  2. Type the reference sentence (ground-truth transcript of that audio)
  3. Choose a wireless channel (AWGN / Rayleigh / Rician) and SNR (dB)
  4. Run BOTH systems through that channel and compare CER / WER:
       - DeepSC-ST   : audio -> semantic encoder -> channel -> decoder -> text
       - Classical   : reference text -> 8-bit ASCII -> BPSK -> channel -> text

Run locally with:
    pip install torch torchaudio jiwer numpy matplotlib
    python deepsc_st_tkinter_demo.py

Before running, edit CHECKPOINT_PATH below to point at your trained
DeepSC-ST weights (a .pth file saved with `torch.save(model.state_dict(), ...)`).
If no checkpoint is found, the DeepSC-ST panel will still run but with an
untrained (randomly initialized) model — clearly flagged in the UI — so you
can still exercise the full app end to end.
"""

import itertools
import math
import os
import threading
import tkinter as tk
from dataclasses import dataclass
from tkinter import filedialog, messagebox, ttk

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchaudio
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from jiwer import cer as compute_cer, wer as compute_wer
except ImportError:
    def compute_cer(ref, hyp):
        ref, hyp = ref or " ", hyp or " "
        return _levenshtein(list(ref), list(hyp)) / max(len(ref), 1)

    def compute_wer(ref, hyp):
        ref_w, hyp_w = (ref or "").split(), (hyp or "").split()
        return _levenshtein(ref_w, hyp_w) / max(len(ref_w), 1)

    def _levenshtein(a, b):
        dp = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            prev, dp[0] = dp[0], i
            for j, cb in enumerate(b, 1):
                cur = dp[j]
                dp[j] = prev if ca == cb else 1 + min(prev, dp[j], dp[j - 1])
                prev = cur
        return dp[-1]

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ──────────────────────────────────────────────────────────────────────────
# Configuration — EDIT THESE for your environment
# ──────────────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = "/Users/madamabhiram/Downloads/DeepSCST/saved_weights_60.pth"   # trained DeepSC-ST weights, or None
CHANNELS = ["AWGN", "Rayleigh", "Rician"]
DEFAULT_SNR_DB = 10

VOCAB_CHARS = list("abcdefghijklmnopqrstuvwxyz") + ["'", " "]
BLANK_INDEX = len(VOCAB_CHARS)          # 28
NUM_CLASSES = len(VOCAB_CHARS) + 1      # 29
CHAR2IDX = {c: i for i, c in enumerate(VOCAB_CHARS)}


# ──────────────────────────────────────────────────────────────────────────
# Classical text transceiver (text -> bits -> BPSK -> channel -> text)
# ──────────────────────────────────────────────────────────────────────────
def text_to_bits(text: str) -> np.ndarray:
    bits = []
    for c in text:
        bits.extend(int(b) for b in format(ord(c), "08b"))
    return np.array(bits)


def bits_to_text(bits: np.ndarray) -> str:
    bits = bits[: len(bits) // 8 * 8]
    chars = []
    for i in range(0, len(bits), 8):
        value = int("".join(map(str, bits[i:i + 8])), 2)
        if value == 0:
            continue
        try:
            chars.append(chr(value))
        except ValueError:
            chars.append(" ")
    return "".join(chars)


def bpsk_modulation(bits: np.ndarray) -> np.ndarray:
    return 2 * bits - 1


def awgn_channel(signal: np.ndarray, snr_db: float) -> np.ndarray:
    snr = 10 ** (snr_db / 10)
    noise_std = np.sqrt(1 / (2 * snr))
    return signal + noise_std * np.random.randn(len(signal))


def rayleigh_channel(signal: np.ndarray, snr_db: float) -> np.ndarray:
    snr = 10 ** (snr_db / 10)
    h = np.random.rayleigh(scale=1, size=len(signal))
    noise_std = np.sqrt(1 / (2 * snr))
    y = h * signal + noise_std * np.random.randn(len(signal))
    return y / h


def rician_channel(signal: np.ndarray, snr_db: float, K: float = 5) -> np.ndarray:
    snr = 10 ** (snr_db / 10)
    los = np.sqrt(K / (K + 1))
    nlos = np.sqrt(1 / (2 * (K + 1))) * np.random.randn(len(signal))
    h = los + nlos
    noise_std = np.sqrt(1 / (2 * snr))
    y = h * signal + noise_std * np.random.randn(len(signal))
    return y / h


def bpsk_demodulation(signal: np.ndarray) -> np.ndarray:
    return (signal > 0).astype(np.int32)


def text_transceiver(text: str, channel: str, snr_db: float) -> str:
    bits = text_to_bits(text)
    symbols = bpsk_modulation(bits)
    if channel == "AWGN":
        rx = awgn_channel(symbols, snr_db)
    elif channel == "Rayleigh":
        rx = rayleigh_channel(symbols, snr_db)
    elif channel == "Rician":
        rx = rician_channel(symbols, snr_db)
    else:
        raise ValueError(f"Unknown channel: {channel}")
    return bits_to_text(bpsk_demodulation(rx))


# ──────────────────────────────────────────────────────────────────────────
# DeepSC-ST model (only defined if torch is available)
# ──────────────────────────────────────────────────────────────────────────
if TORCH_AVAILABLE:

    @dataclass
    class Config:
        sample_rate: int = 16_000
        frame_length_ms: int = 20
        frame_stride_ms: int = 10
        feature_norm: bool = True
        freq_bins: int = 161
        conv_filters: int = 32
        num_rnn_layers: int = 7
        rnn_hidden_size: int = 800
        rnn_type: str = "gru"
        is_bidirectional: bool = True
        use_bias: bool = True
        num_channel_units: int = 40
        num_classes: int = NUM_CLASSES
        bn_eps: float = 1e-5
        bn_momentum: float = 0.003

    CFG = Config()

    def compute_spectrogram_feature(audio_samples, sample_rate=16000,
                                     frame_length=20, frame_stride=10,
                                     max_freq=None, eps=1e-14):
        if max_freq is None:
            max_freq = sample_rate / 2
        stride_size = int(0.001 * sample_rate * frame_stride)
        frame_size = int(0.001 * sample_rate * frame_length)

        truncate_size = (len(audio_samples) - frame_size) % stride_size
        audio_samples = audio_samples[: len(audio_samples) - truncate_size]

        nshape = (frame_size, (len(audio_samples) - frame_size) // stride_size + 1)
        nstrides = (audio_samples.strides[0], audio_samples.strides[0] * stride_size)
        frames = np.lib.stride_tricks.as_strided(audio_samples, shape=nshape, strides=nstrides)

        weighting = np.hanning(frame_size)[:, None]
        fft = np.fft.rfft(frames * weighting, axis=0)
        fft = np.absolute(fft) ** 2
        scale = np.sum(weighting ** 2) * sample_rate
        fft[1:-1, :] *= (2.0 / scale)
        fft[(0, -1), :] /= scale

        freqs = float(sample_rate) / frame_size * np.arange(fft.shape[0])
        ind = np.where(freqs <= max_freq)[0][-1] + 1
        specgram = np.log(fft[:ind, :] + eps)
        return np.transpose(specgram, (1, 0)).astype(np.float32)

    def normalize_audio_feature(features):
        mean = np.mean(features, axis=0)
        var = np.var(features, axis=0)
        return ((features - mean) / (np.sqrt(var) + 1e-6)).astype(np.float32)

    def preprocess_wav(wav_path, cfg=CFG):
        waveform, sr = torchaudio.load(wav_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != cfg.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, cfg.sample_rate)
        audio_np = waveform.squeeze().numpy().astype(np.float64)
        feat = compute_spectrogram_feature(audio_np, cfg.sample_rate,
                                            cfg.frame_length_ms, cfg.frame_stride_ms)
        if cfg.feature_norm:
            feat = normalize_audio_feature(feat)
        feat = feat[np.newaxis, :, :, np.newaxis]
        return torch.from_numpy(feat)

    class ConvBNBlock(nn.Module):
        def __init__(self, in_ch, out_ch, kernel, stride, pad, cfg):
            super().__init__()
            self.pad_t, self.pad_f = pad
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel,
                                   stride=stride, padding=0, bias=False)
            self.bn = nn.BatchNorm2d(out_ch, eps=cfg.bn_eps, momentum=cfg.bn_momentum)

        def forward(self, x):
            x = F.pad(x, (self.pad_f, self.pad_f, self.pad_t, self.pad_t))
            return F.relu6(self.bn(self.conv(x)))

    class RNNBNBlock(nn.Module):
        _RNN_TYPES = {"gru": nn.GRU, "lstm": nn.LSTM, "rnn": nn.RNN}

        def __init__(self, input_size, hidden_size, rnn_type="gru",
                     bidirectional=True, cfg=CFG):
            super().__init__()
            rnn_cls = self._RNN_TYPES[rnn_type]
            self.rnn = rnn_cls(input_size, hidden_size, batch_first=True,
                                bidirectional=bidirectional)
            out_size = hidden_size * (2 if bidirectional else 1)
            self.bn = nn.BatchNorm1d(out_size, eps=cfg.bn_eps, momentum=cfg.bn_momentum)

        def forward(self, x):
            out, _ = self.rnn(x)
            B, T, Fdim = out.shape
            return self.bn(out.reshape(B * T, Fdim)).reshape(B, T, Fdim)

    class ChannelLayer(nn.Module):
        def forward(self, x_norm, channel="AWGN", snr_db=10.0):
            B, T, _ = x_norm.shape
            dev = x_norm.device
            snr_linear = 10 ** (snr_db / 10.0)
            std = math.sqrt(1.0 / snr_linear)

            if channel == "AWGN":
                awgn_f, ray_f, rice_f = 1.0, 0.0, 0.0
            elif channel == "Rayleigh":
                awgn_f, ray_f, rice_f = 0.0, 1.0, 0.0
            elif channel == "Rician":
                awgn_f, ray_f, rice_f = 0.0, 0.5, 0.5
            else:
                raise ValueError(f"Unknown channel: {channel}")

            h_raw = torch.randn(B, T, 1, 2, device=dev)
            h = (math.sqrt(rice_f) + math.sqrt(ray_f) * h_raw) / math.sqrt(2.0)
            noise = torch.randn(B, T, 1, 2, device=dev) * std
            x_e = x_norm.unsqueeze(2)

            xr, xi = x_e[..., 0], x_e[..., 1]
            hr, hi = h[..., 0], h[..., 1]
            nr, ni = noise[..., 0], noise[..., 1]

            hxr = xr * (hr + awgn_f) - xi * hi
            hxi = xr * hi + xi * (hr + awgn_f)
            yr, yi = hxr + nr, hxi + ni

            rr = yr * (hr + awgn_f) + yi * hi
            ri = yi * (hr + awgn_f) - yr * hi
            hh = hr * hr + hi * hi

            x_hat_r = rr / (hh + awgn_f)
            x_hat_i = ri / (hh + awgn_f)
            return torch.cat([x_hat_r, x_hat_i], dim=-1).squeeze(2)

    class DeepSCST(nn.Module):
        def __init__(self, cfg=CFG):
            super().__init__()
            self.cfg = cfg
            C, NC, NCU = cfg.conv_filters, cfg.num_classes, cfg.num_channel_units

            self.cnn1 = ConvBNBlock(1, C, (41, 11), (2, 2), (20, 5), cfg)
            self.cnn2 = ConvBNBlock(C, C, (21, 11), (2, 1), (10, 5), cfg)

            self._rnn_input_size = None
            self._rnn_layers = nn.ModuleList()

            rnn_out = cfg.rnn_hidden_size * (2 if cfg.is_bidirectional else 1)
            self.sem_out = nn.Linear(rnn_out, NC, bias=cfg.use_bias)

            self.chan_enc1 = nn.Linear(NC, NCU, bias=True)
            self.chan_enc2 = nn.Linear(NCU, NCU, bias=True)
            self.channel = ChannelLayer()
            self.chan_dec1 = nn.Linear(NCU, NCU, bias=True)
            self.chan_dec2 = nn.Linear(NCU, NCU, bias=True)
            self.chan_dec3 = nn.Linear(NCU, NC, bias=True)

        def _build_rnn_stack(self, input_size):
            cfg = self.cfg
            rnn_out = cfg.rnn_hidden_size * (2 if cfg.is_bidirectional else 1)
            layers = [RNNBNBlock(input_size if i == 0 else rnn_out, cfg.rnn_hidden_size,
                                  cfg.rnn_type, cfg.is_bidirectional, cfg)
                      for i in range(cfg.num_rnn_layers)]
            self._rnn_layers = nn.ModuleList(layers)
            self._rnn_input_size = input_size
            self._rnn_layers.to(next(self.parameters()).device)

        def forward(self, x, channel="AWGN", snr_db=10.0):
            x = x.permute(0, 3, 1, 2)
            x = self.cnn1(x)
            x = self.cnn2(x)

            B, C, T2, F2 = x.shape
            x = x.permute(0, 2, 3, 1).reshape(B, T2, F2 * C)

            if self._rnn_input_size is None:
                self._build_rnn_stack(F2 * C)
            for rnn_blk in self._rnn_layers:
                x = rnn_blk(x)

            p = F.softmax(self.sem_out(x), dim=-1)
            u = F.relu6(self.chan_enc1(p))
            u = self.chan_enc2(u)

            B2, L, _ = u.shape
            sym = u.reshape(B2, L * (self.cfg.num_channel_units // 2), 2)
            n_sym = sym.shape[1]
            sym_norm = math.sqrt(n_sym / 2.0) * F.normalize(sym, p=2, dim=1)
            sym_hat = self.channel(sym_norm, channel, snr_db)

            v = sym_hat.reshape(B2, L, self.cfg.num_channel_units)
            v = F.relu6(self.chan_dec1(v))
            v = F.relu6(self.chan_dec2(v))
            return F.softmax(self.chan_dec3(v), dim=-1)

    class GreedyDecoder:
        def __init__(self, labels, blank_index=BLANK_INDEX):
            self.int2char = {i: c for i, c in enumerate(labels)}
            self.blank_index = blank_index

        def decode(self, logits: np.ndarray) -> str:
            best = list(np.argmax(logits, axis=1))
            merged = [k for k, _ in itertools.groupby(best)]
            return "".join(self.int2char[k] for k in merged if k != self.blank_index)

        def decode_tensor(self, logits) -> str:
            if logits.dim() == 3:
                logits = logits[0]
            return self.decode(logits.cpu().detach().numpy())


# ──────────────────────────────────────────────────────────────────────────
# GUI Application
# ──────────────────────────────────────────────────────────────────────────
class DeepSCDemoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("DeepSC-ST vs. Classical Transceiver — Comparison Demo")
        self.root.geometry("880x720")
        self.audio_path = tk.StringVar(value="No file selected")
        self.channel_var = tk.StringVar(value=CHANNELS[0])
        self.snr_var = tk.IntVar(value=DEFAULT_SNR_DB)
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") \
            if TORCH_AVAILABLE else None
        self.model = None
        self.decoder = None
        self.model_status = "not loaded"

        self._build_layout()
        if TORCH_AVAILABLE:
            threading.Thread(target=self._load_model, daemon=True).start()
        else:
            self.status_label.config(
                text="PyTorch/torchaudio not installed — DeepSC-ST panel disabled. "
                     "Install with: pip install torch torchaudio")

    # ── model loading ────────────────────────────────────────────────
    def _load_model(self):
        self.model = DeepSCST(CFG).to(self.device)
        with torch.no_grad():
            _ = self.model(torch.randn(1, 100, 161, 1).to(self.device), channel="AWGN", snr_db=10)
        if CHECKPOINT_PATH and os.path.exists(CHECKPOINT_PATH):
            self.model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=self.device))
            self.model_status = f"loaded pretrained weights ({CHECKPOINT_PATH})"
        else:
            self.model_status = "UNTRAINED (no checkpoint found — results will be meaningless)"
        self.model.eval()
        self.decoder = GreedyDecoder(VOCAB_CHARS)
        self.status_label.config(text=f"DeepSC-ST model: {self.model_status}  |  Device: {self.device}")

    # ── layout ───────────────────────────────────────────────────────
    def _build_layout(self):
        pad = {"padx": 10, "pady": 6}

        file_frame = ttk.LabelFrame(self.root, text="1. Audio input")
        file_frame.pack(fill="x", **pad)
        ttk.Button(file_frame, text="Browse .wav...", command=self._browse_audio).pack(side="left", padx=8, pady=8)
        ttk.Label(file_frame, textvariable=self.audio_path, foreground="#444").pack(side="left")

        ref_frame = ttk.LabelFrame(self.root, text="2. Reference sentence (ground-truth transcript of the audio)")
        ref_frame.pack(fill="x", **pad)
        self.ref_entry = tk.Text(ref_frame, height=3, wrap="word")
        self.ref_entry.pack(fill="x", padx=8, pady=8)
        self.ref_entry.insert("1.0", "the quick brown fox jumps over the lazy dog")

        chan_frame = ttk.LabelFrame(self.root, text="3. Channel & SNR")
        chan_frame.pack(fill="x", **pad)
        ttk.Label(chan_frame, text="Channel:").pack(side="left", padx=(8, 4))
        ttk.Combobox(chan_frame, textvariable=self.channel_var, values=CHANNELS,
                     state="readonly", width=12).pack(side="left", padx=4)
        ttk.Label(chan_frame, text="SNR (dB):").pack(side="left", padx=(16, 4))
        ttk.Spinbox(chan_frame, from_=-10, to=30, textvariable=self.snr_var, width=6).pack(side="left")

        run_frame = ttk.Frame(self.root)
        run_frame.pack(fill="x", **pad)
        self.run_button = ttk.Button(run_frame, text="Run Comparison", command=self._run_comparison_thread)
        self.run_button.pack(side="left")
        self.status_label = ttk.Label(run_frame, text="Loading DeepSC-ST model...", foreground="#666")
        self.status_label.pack(side="left", padx=16)

        results_frame = ttk.LabelFrame(self.root, text="4. Results")
        results_frame.pack(fill="both", expand=True, **pad)

        text_frame = ttk.Frame(results_frame)
        text_frame.pack(fill="x", padx=8, pady=8)

        ttk.Label(text_frame, text="DeepSC-ST predicted text:", font=("Segoe UI", 9, "bold")).grid(
            row=0, column=0, sticky="w")
        self.deep_text_out = tk.Text(text_frame, height=2, wrap="word", bg="#f5faff")
        self.deep_text_out.grid(row=1, column=0, sticky="ew", pady=(0, 8))

        ttk.Label(text_frame, text="Classical transceiver predicted text:", font=("Segoe UI", 9, "bold")).grid(
            row=2, column=0, sticky="w")
        self.classic_text_out = tk.Text(text_frame, height=2, wrap="word", bg="#fff5f5")
        self.classic_text_out.grid(row=3, column=0, sticky="ew")
        text_frame.columnconfigure(0, weight=1)

        self.metrics_label = ttk.Label(results_frame, text="", font=("Segoe UI", 10))
        self.metrics_label.pack(pady=(4, 8))

        self.chart_container = ttk.Frame(results_frame)
        self.chart_container.pack(fill="both", expand=True, padx=8, pady=8)

    # ── actions ──────────────────────────────────────────────────────
    def _browse_audio(self):
        path = filedialog.askopenfilename(filetypes=[("WAV audio", "*.wav")])
        if path:
            self.audio_path.set(path)

    def _run_comparison_thread(self):
        threading.Thread(target=self._run_comparison, daemon=True).start()

    def _run_comparison(self):
        self.run_button.config(state="disabled")
        self.status_label.config(text="Running...")
        try:
            ref_text = self.ref_entry.get("1.0", "end").strip().lower()
            channel = self.channel_var.get()
            snr_db = self.snr_var.get()

            if not ref_text:
                messagebox.showwarning("Missing input", "Please enter a reference sentence.")
                return

            # ---- Classical transceiver ----
            classic_pred = text_transceiver(ref_text, channel, snr_db)
            classic_cer = compute_cer(ref_text, classic_pred)
            classic_wer = compute_wer(ref_text, classic_pred)

            # ---- DeepSC-ST ----
            deep_pred, deep_cer, deep_wer = "", None, None
            if TORCH_AVAILABLE and self.model is not None:
                audio_path = self.audio_path.get()
                if os.path.exists(audio_path):
                    x = preprocess_wav(audio_path).to(self.device)
                    with torch.no_grad():
                        probs = self.model(x, channel=channel, snr_db=snr_db)
                    deep_pred = self.decoder.decode_tensor(probs)
                    deep_cer = compute_cer(ref_text, deep_pred)
                    deep_wer = compute_wer(ref_text, deep_pred)
                else:
                    deep_pred = "(select a valid .wav file to run DeepSC-ST)"

            self.root.after(0, self._update_results, deep_pred, deep_cer, deep_wer,
                             classic_pred, classic_cer, classic_wer, channel, snr_db)
        except Exception as exc:
            self.root.after(0, messagebox.showerror, "Error", str(exc))
        finally:
            self.root.after(0, lambda: self.run_button.config(state="normal"))
            self.root.after(0, lambda: self.status_label.config(
                text=f"DeepSC-ST model: {self.model_status}" if TORCH_AVAILABLE else "PyTorch not installed"))

    def _update_results(self, deep_pred, deep_cer, deep_wer,
                         classic_pred, classic_cer, classic_wer, channel, snr_db):
        self.deep_text_out.delete("1.0", "end")
        self.deep_text_out.insert("1.0", deep_pred or "(DeepSC-ST unavailable)")
        self.classic_text_out.delete("1.0", "end")
        self.classic_text_out.insert("1.0", classic_pred)

        deep_str = f"CER {deep_cer:.3f} / WER {deep_wer:.3f}" if deep_cer is not None else "n/a"
        metrics_text = (f"Channel: {channel}   SNR: {snr_db} dB\n"
                         f"DeepSC-ST  →  {deep_str}\n"
                         f"Classical  →  CER {classic_cer:.3f} / WER {classic_wer:.3f}")
        self.metrics_label.config(text=metrics_text)

        self._draw_chart(deep_cer, deep_wer, classic_cer, classic_wer)

    def _draw_chart(self, deep_cer, deep_wer, classic_cer, classic_wer):
        for w in self.chart_container.winfo_children():
            w.destroy()

        fig, ax = plt.subplots(figsize=(6, 3.2), dpi=100)
        labels = ["CER", "WER"]
        deep_vals = [deep_cer or 0, deep_wer or 0]
        classic_vals = [classic_cer, classic_wer]

        x = np.arange(len(labels))
        width = 0.32
        ax.bar(x - width / 2, deep_vals, width, label="DeepSC-ST", color="#3b82f6")
        ax.bar(x + width / 2, classic_vals, width, label="Classical", color="#ef4444")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Error rate")
        ax.set_ylim(0, max(1.0, *deep_vals, *classic_vals) * 1.15)
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.chart_container)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        plt.close(fig)


def main():
    root = tk.Tk()
    app = DeepSCDemoApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
