#!/usr/bin/env python3
"""
VocalView – Analyseur vocal temps réel
---------------------------------------
Affiche :
  • Oscilloscope (forme d'onde déclenchée)
  • Spectre de fréquences (FFT, échelle log, lissage exponentiel)
  • Harmoniques H1..H8 annotées sur le spectre
  • Note chantée (ex. A4), fréquence en Hz, écart en cents
  • Indicateur de justesse (barre ±50 ¢)
  • Volume RMS en dB
  • Sélecteur d'entrée audio (tous les périphériques d'entrée disponibles)
"""

import sys
import threading

import numpy as np
import sounddevice as sd
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication, QComboBox, QFrame, QHBoxLayout, QLabel,
    QMainWindow, QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)
import pyqtgraph as pg

# ── Paramètres audio ──────────────────────────────────────────────────────────
SAMPLE_RATE    = 44100
CHUNK_SIZE     = 4096
MIN_PITCH_HZ   = 60
MAX_PITCH_HZ   = 1200
DISP_FREQ_MAX  = 8000
N_HARMONICS    = 8

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# ── Couleurs des harmoniques (cycle de 4) ─────────────────────────────────────
HARM_COLORS = ['#ffa657', '#d2a8ff', '#79c0ff', '#56d364'] * 2


# ══════════════════════════════════════════════════════════════════════════════
#  DSP
# ══════════════════════════════════════════════════════════════════════════════

def freq_to_note(freq):
    """
    Convertit une fréquence (Hz) en nom de note et écart en cents.
    Retourne (nom: str, cents: float, midi: int) ou (None, None, None).
    """
    if freq <= 0:
        return None, None, None
    midi   = 12.0 * np.log2(freq / 440.0) + 69.0
    midi_i = int(round(midi))
    cents  = (midi - midi_i) * 100.0
    octave = midi_i // 12 - 1
    name   = NOTE_NAMES[midi_i % 12]
    return f"{name}{octave}", cents, midi_i


def detect_pitch(signal, sr):
    """
    Détecte la fréquence fondamentale par autocorrélation (méthode FFT).
    Retourne la fréquence en Hz, ou None si le signal est trop faible / ambigu.
    """
    x   = signal - signal.mean()
    rms = float(np.sqrt((x ** 2).mean()))
    if rms < 0.005:
        return None
    x = x / (rms + 1e-12)

    n   = len(x)
    F   = np.fft.rfft(x, n=n * 2)
    acf = np.fft.irfft(F * F.conj())[:n].real        # autocorrélation

    lo = max(1, int(sr / MAX_PITCH_HZ))
    hi = min(int(sr / MIN_PITCH_HZ), n - 2)
    if lo >= hi or acf[0] <= 0:
        return None

    chunk = acf[lo:hi]
    thr   = 0.35 * acf[0]

    best_lag = -1
    for i in range(1, len(chunk) - 1):
        if (chunk[i] > thr
                and chunk[i] > chunk[i - 1]
                and chunk[i] > chunk[i + 1]):
            best_lag = i
            break                                    # premier pic = fondamentale

    if best_lag < 0:
        return None

    lag = best_lag + lo
    # Interpolation parabolique (précision sub-sample)
    if 0 < lag < n - 1:
        a, b, c = acf[lag - 1], acf[lag], acf[lag + 1]
        denom = 2.0 * b - a - c
        if denom > 0:
            lag = lag + (a - c) / (2.0 * denom)

    return sr / lag


# ══════════════════════════════════════════════════════════════════════════════
#  Fenêtre principale
# ══════════════════════════════════════════════════════════════════════════════

class VocalView(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("VocalView – Analyseur Vocal")
        self.resize(1340, 780)

        self._lock        = threading.Lock()
        self._buf         = np.zeros(CHUNK_SIZE, dtype=np.float32)
        self._stream      = None
        self._running     = False
        self._spec_smooth = None

        self._apply_theme()
        self._build_ui()
        self._build_curves()
        self._populate_devices()

        self._timer = QTimer(self)
        self._timer.setInterval(16)          # ≈ 60 fps
        self._timer.timeout.connect(self._refresh)

    # ── Thème ─────────────────────────────────────────────────────────────────

    def _apply_theme(self):
        pg.setConfigOption('background', '#0d1117')
        pg.setConfigOption('foreground', '#c9d1d9')
        self.setStyleSheet("""
            QWidget {
                background: #0d1117;
                color: #c9d1d9;
                font-family: 'Segoe UI', sans-serif;
            }
            QComboBox {
                background: #161b22;
                color: #c9d1d9;
                border: 1px solid #30363d;
                border-radius: 5px;
                padding: 5px 10px;
                min-width: 300px;
            }
            QComboBox QAbstractItemView {
                background: #161b22;
                color: #c9d1d9;
                selection-background-color: #1f6feb;
            }
            QComboBox::drop-down { border: none; width: 20px; }
            QPushButton {
                background: #21262d;
                color: #c9d1d9;
                border: 1px solid #30363d;
                border-radius: 5px;
                padding: 6px 18px;
                font-weight: bold;
            }
            QPushButton:hover  { background: #30363d; border-color: #58a6ff; }
            QPushButton:checked {
                background: #b91c1c;
                border-color: #f87171;
            }
            QPushButton:checked:hover { background: #dc2626; }
        """)

    # ── Construction UI ───────────────────────────────────────────────────────

    def _build_ui(self):
        root   = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(10)
        self.setCentralWidget(root)

        # ── Barre de contrôle ──────────────────────────────────────────────
        bar = QHBoxLayout()
        bar.setSpacing(8)

        lbl = QLabel("Entrée audio :")
        lbl.setFont(QFont("Segoe UI", 10))
        bar.addWidget(lbl)

        self._combo = QComboBox()
        self._combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._combo.currentIndexChanged.connect(self._on_device_changed)
        bar.addWidget(self._combo)

        btn_r = QPushButton("↻")
        btn_r.setFixedWidth(34)
        btn_r.setToolTip("Rafraîchir la liste des périphériques")
        btn_r.clicked.connect(self._populate_devices)
        bar.addWidget(btn_r)

        bar.addSpacing(14)

        self._btn = QPushButton("▶  Démarrer")
        self._btn.setCheckable(True)
        self._btn.setMinimumWidth(140)
        self._btn.clicked.connect(self._toggle)
        bar.addWidget(self._btn)

        bar.addStretch()

        title = QLabel("VocalView")
        title.setFont(QFont("Segoe UI", 14, QFont.Bold))
        title.setStyleSheet("color: #58a6ff;")
        bar.addWidget(title)

        layout.addLayout(bar)

        # ── Séparateur ────────────────────────────────────────────────────
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("background: #21262d;")
        sep.setFixedHeight(1)
        layout.addWidget(sep)

        # ── Panneau informations (note, freq, cents, justesse, volume) ─────
        panel = QFrame()
        panel.setStyleSheet("""
            QFrame {
                background: #161b22;
                border: 1px solid #30363d;
                border-radius: 8px;
            }
        """)
        panel.setFixedHeight(118)
        row = QHBoxLayout(panel)
        row.setContentsMargins(24, 8, 24, 8)
        row.setSpacing(0)

        # Note musicale (grande)
        self._note_lbl = QLabel("--")
        self._note_lbl.setFont(QFont("Segoe UI", 54, QFont.Bold))
        self._note_lbl.setAlignment(Qt.AlignCenter)
        self._note_lbl.setFixedWidth(140)
        self._note_lbl.setStyleSheet("color: #3fb950;")
        row.addWidget(self._note_lbl)

        row.addWidget(self._vsep())
        row.addLayout(self._info_box("FRÉQUENCE",     "---.- Hz", "_freq_lbl"))
        row.addWidget(self._vsep())
        row.addLayout(self._info_box("PRÉCISION",     "±0 ¢",     "_cents_lbl"))
        row.addWidget(self._vsep())

        # Indicateur de justesse (barre ±50 ¢)
        m_col = QVBoxLayout()
        m_col.setSpacing(2)
        m_lbl = QLabel("JUSTESSE")
        m_lbl.setFont(QFont("Segoe UI", 8))
        m_lbl.setStyleSheet("color: #8b949e;")
        m_lbl.setAlignment(Qt.AlignCenter)
        m_col.addWidget(m_lbl)

        self._meter = pg.PlotWidget()
        self._meter.setFixedSize(240, 56)
        self._meter.setBackground('#0d1117')
        self._meter.hideAxis('left')
        self._meter.hideAxis('bottom')
        self._meter.setXRange(-55, 55)
        self._meter.setYRange(0, 1)
        self._meter.getPlotItem().setMenuEnabled(False)
        self._meter.getViewBox().setMouseEnabled(x=False, y=False)
        m_col.addWidget(self._meter, alignment=Qt.AlignCenter)
        row.addLayout(m_col)

        row.addStretch()
        row.addWidget(self._vsep())
        row.addLayout(self._info_box("VOLUME (RMS)", "-∞ dB", "_vol_lbl"))

        layout.addWidget(panel)

        # ── Zone des graphiques ────────────────────────────────────────────
        plots = QHBoxLayout()
        plots.setSpacing(10)

        # Oscilloscope (colonne gauche, largeur fixe)
        oc = QVBoxLayout()
        oc.setSpacing(3)
        ot = QLabel("OSCILLOSCOPE")
        ot.setFont(QFont("Segoe UI", 8, QFont.Bold))
        ot.setStyleSheet("color: #8b949e;")
        ot.setAlignment(Qt.AlignCenter)
        oc.addWidget(ot)

        self._osc = pg.PlotWidget()
        self._osc.setBackground('#0d1117')
        self._osc.showGrid(x=True, y=True, alpha=0.13)
        self._osc.setYRange(-1.15, 1.15)
        self._osc.setLabel('bottom', 'ms')
        self._osc.getPlotItem().setMenuEnabled(False)
        self._osc.getViewBox().setMouseEnabled(x=False, y=False)
        oc.addWidget(self._osc)

        ow = QWidget()
        ow.setLayout(oc)
        ow.setFixedWidth(310)
        plots.addWidget(ow)

        # Spectre (colonne droite, extensible)
        sc = QVBoxLayout()
        sc.setSpacing(3)
        st = QLabel("SPECTRE DE FRÉQUENCES  ·  HARMONIQUES")
        st.setFont(QFont("Segoe UI", 8, QFont.Bold))
        st.setStyleSheet("color: #8b949e;")
        st.setAlignment(Qt.AlignCenter)
        sc.addWidget(st)

        self._spec = pg.PlotWidget()
        self._spec.setBackground('#0d1117')
        self._spec.showGrid(x=True, y=True, alpha=0.13)
        self._spec.setLabel('bottom', 'Hz')
        self._spec.setLabel('left', 'dB')
        self._spec.setLogMode(x=True, y=False)
        self._spec.setXRange(np.log10(50), np.log10(DISP_FREQ_MAX))
        self._spec.setYRange(-90, 5)
        self._spec.getPlotItem().setMenuEnabled(False)
        self._spec.getViewBox().setMouseEnabled(x=False, y=False)
        sc.addWidget(self._spec)

        plots.addLayout(sc)
        layout.addLayout(plots)

    # ── Helpers UI ────────────────────────────────────────────────────────────

    def _vsep(self):
        f = QFrame()
        f.setFrameShape(QFrame.VLine)
        f.setStyleSheet("background: #30363d; max-width: 1px;")
        f.setContentsMargins(10, 0, 10, 0)
        return f

    def _info_box(self, title, initial, attr):
        """Crée une colonne titre + valeur et stocke le QLabel dans self.<attr>."""
        col = QVBoxLayout()
        col.setSpacing(2)
        col.setContentsMargins(18, 0, 18, 0)

        tl = QLabel(title)
        tl.setFont(QFont("Segoe UI", 8))
        tl.setStyleSheet("color: #8b949e;")
        tl.setAlignment(Qt.AlignCenter)

        vl = QLabel(initial)
        vl.setFont(QFont("Segoe UI", 19, QFont.Bold))
        vl.setAlignment(Qt.AlignCenter)
        setattr(self, attr, vl)

        col.addWidget(tl)
        col.addWidget(vl)
        return col

    # ── Courbes pyqtgraph ─────────────────────────────────────────────────────

    def _build_curves(self):
        # Oscilloscope
        self._osc_c = self._osc.plot(pen=pg.mkPen('#3fb950', width=1.5))

        # Spectre
        self._spec_c = self._spec.plot(
            pen=pg.mkPen('#f78166', width=1.5),
            fillLevel=-90,
            brush=(247, 129, 102, 35),
        )

        # Marqueurs harmoniques H1..H8
        self._hlines  = []
        self._hlabels = []
        for i in range(N_HARMONICS):
            ln = pg.InfiniteLine(
                angle=90,
                pen=pg.mkPen(HARM_COLORS[i], width=1.1, style=Qt.DashLine),
            )
            ln.setVisible(False)
            self._spec.addItem(ln)
            self._hlines.append(ln)

            lb = pg.TextItem(
                f'H{i + 1}', color=HARM_COLORS[i], anchor=(0.5, 1.2),
            )
            lb.setFont(QFont("Segoe UI", 7))
            lb.setVisible(False)
            self._spec.addItem(lb)
            self._hlabels.append(lb)

        # Barre de justesse + repères
        self._mbar = pg.BarGraphItem(
            x=[0], height=[0.7], width=4, brush='#3fb950',
        )
        self._meter.addItem(self._mbar)
        self._meter.addItem(
            pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen('#c9d1d9', width=1))
        )
        for v, t in ((-50, '−50¢'), (0, '0'), (50, '+50¢')):
            ti = pg.TextItem(t, color='#8b949e', anchor=(0.5, 0))
            ti.setFont(QFont("Segoe UI", 7))
            ti.setPos(v, 0.76)
            self._meter.addItem(ti)

    # ── Périphériques audio ───────────────────────────────────────────────────

    def _populate_devices(self):
        self._combo.blockSignals(True)
        self._combo.clear()
        try:
            for i, d in enumerate(sd.query_devices()):
                if d['max_input_channels'] > 0:
                    try:
                        api = sd.query_hostapis(d['hostapi'])['name']
                    except Exception:
                        api = ''
                    self._combo.addItem(f"{d['name']}  [{api}]", userData=i)
            # Pré-sélectionne le périphérique d'entrée par défaut
            default_in = sd.default.device[0]
            for k in range(self._combo.count()):
                if self._combo.itemData(k) == default_in:
                    self._combo.setCurrentIndex(k)
                    break
        except Exception as e:
            self._combo.addItem(f"Erreur : {e}")
        self._combo.blockSignals(False)

    def _on_device_changed(self):
        """Redémarre la capture si on change de périphérique en cours de route."""
        if self._running:
            self._stop()
            self._start()

    # ── Démarrage / Arrêt ─────────────────────────────────────────────────────

    def _toggle(self, checked):
        if checked:
            self._start()
        else:
            self._stop()

    def _start(self):
        dev = self._combo.currentData()
        if dev is None:
            self._btn.setChecked(False)
            return
        try:
            self._stream = sd.InputStream(
                device=dev,
                channels=1,
                samplerate=SAMPLE_RATE,
                blocksize=CHUNK_SIZE,
                dtype='float32',
                callback=self._audio_cb,
            )
            self._stream.start()
            self._running = True
            self._btn.setText("⏹  Arrêter")
            self._btn.setChecked(True)
            self._timer.start()
        except Exception as e:
            self._btn.setChecked(False)
            self._note_lbl.setStyleSheet("color: #f85149;")
            self._note_lbl.setText("ERR")
            self._freq_lbl.setText(str(e)[:32])

    def _stop(self):
        self._timer.stop()
        self._running = False
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        self._btn.setText("▶  Démarrer")
        self._btn.setChecked(False)
        self._reset_display()

    def _audio_cb(self, indata, frames, time_info, status):
        """Callback sounddevice — appelé depuis le thread audio."""
        with self._lock:
            self._buf = indata[:, 0].copy()

    # ── Mise à jour des graphiques (~60 fps) ──────────────────────────────────

    def _refresh(self):
        with self._lock:
            data = self._buf.copy()

        # ── Oscilloscope déclenché ─────────────────────────────────────────
        N_OSC = 1024
        seg   = self._find_trigger(data, N_OSC)
        self._osc_c.setData(
            np.linspace(0.0, N_OSC / SAMPLE_RATE * 1000.0, N_OSC), seg
        )

        # ── FFT / Spectre ──────────────────────────────────────────────────
        win = data * np.hanning(len(data))
        mag = np.abs(np.fft.rfft(win))
        f   = np.fft.rfftfreq(len(data), 1.0 / SAMPLE_RATE)
        db  = 20.0 * np.log10(np.maximum(mag, 1e-12) / (len(data) / 2.0))

        # Lissage exponentiel (évite le scintillement)
        if self._spec_smooth is None or len(self._spec_smooth) != len(db):
            self._spec_smooth = db.copy()
        else:
            self._spec_smooth = 0.65 * self._spec_smooth + 0.35 * db

        mask = (f >= 50) & (f <= DISP_FREQ_MAX)
        self._spec_c.setData(f[mask], self._spec_smooth[mask])

        # ── Volume RMS ────────────────────────────────────────────────────
        rms = float(np.sqrt((data ** 2).mean()))
        self._vol_lbl.setText(f"{20.0 * np.log10(rms + 1e-12):.1f} dB")

        # ── Détection de la hauteur ────────────────────────────────────────
        pitch = detect_pitch(data, SAMPLE_RATE)

        if pitch and MIN_PITCH_HZ <= pitch <= MAX_PITCH_HZ:
            note, cents, _ = freq_to_note(pitch)
            c = int(round(cents))

            if abs(c) <= 10:
                color = '#3fb950'    # vert  — bien accordé
            elif abs(c) <= 25:
                color = '#e3b341'    # jaune — légèrement faux
            else:
                color = '#f85149'    # rouge — faux

            self._note_lbl.setText(note or '--')
            self._note_lbl.setStyleSheet(f"color: {color};")
            self._freq_lbl.setText(f"{pitch:.1f} Hz")
            self._cents_lbl.setText(f"{c:+d} ¢")
            self._cents_lbl.setStyleSheet(f"color: {color};")
            self._mbar.setOpts(
                x=[cents],
                width=max(3, min(abs(cents), 12)),
                brush=pg.mkBrush(color),
            )

            # ── Harmoniques ───────────────────────────────────────────────
            for i, (ln, lb) in enumerate(zip(self._hlines, self._hlabels)):
                hf = pitch * (i + 1)
                if hf <= DISP_FREQ_MAX:
                    log_pos = float(np.log10(hf))   # coordonnée en mode log
                    ln.setValue(log_pos)
                    ln.setVisible(True)
                    # Amplitude à cette fréquence (pour placer l'étiquette)
                    idx = int(np.searchsorted(f, hf))
                    idx = min(idx, len(self._spec_smooth) - 1)
                    lb.setPos(log_pos, float(self._spec_smooth[idx]) + 7)
                    lb.setText(f"H{i + 1}\n{hf:.0f} Hz")
                    lb.setVisible(True)
                else:
                    ln.setVisible(False)
                    lb.setVisible(False)
        else:
            self._clear_pitch()

    # ── Utilitaires ───────────────────────────────────────────────────────────

    def _find_trigger(self, data, n):
        """
        Cherche un passage par zéro montant pour stabiliser l'oscilloscope.
        Retourne exactement n échantillons.
        """
        limit = len(data) - n
        if limit <= 0:
            out = np.zeros(n, dtype=data.dtype)
            out[:len(data)] = data
            return out
        for i in range(1, limit):
            if data[i - 1] < 0.0 <= data[i]:
                return data[i: i + n]
        return data[:n]

    def _clear_pitch(self):
        self._note_lbl.setText('--')
        self._note_lbl.setStyleSheet('color: #484f58;')
        self._cents_lbl.setText('±0 ¢')
        self._cents_lbl.setStyleSheet('color: #c9d1d9;')
        self._mbar.setOpts(x=[0], width=2, brush='#484f58')
        for ln, lb in zip(self._hlines, self._hlabels):
            ln.setVisible(False)
            lb.setVisible(False)

    def _reset_display(self):
        self._clear_pitch()
        self._freq_lbl.setText('---.- Hz')
        self._vol_lbl.setText('-∞ dB')
        self._spec_smooth = None

    def closeEvent(self, event):
        self._stop()
        event.accept()


# ══════════════════════════════════════════════════════════════════════════════
#  Point d'entrée
# ══════════════════════════════════════════════════════════════════════════════

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    win = VocalView()
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
