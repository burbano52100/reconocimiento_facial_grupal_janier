"""Graba del microfono de Windows (winmm) con ctypes: no hace falta instalar nada."""
import ctypes
import time
from ctypes import wintypes

import numpy as np

WAVE_MAPPER = 0xFFFFFFFF  # el microfono predeterminado de Windows
WAVE_FORMAT_PCM = 1
WHDR_DONE = 1             # el bloque ya esta lleno


class _Formato(ctypes.Structure):  # WAVEFORMATEX
    _fields_ = [("wFormatTag", wintypes.WORD), ("nChannels", wintypes.WORD),
                ("nSamplesPerSec", wintypes.DWORD), ("nAvgBytesPerSec", wintypes.DWORD),
                ("nBlockAlign", wintypes.WORD), ("wBitsPerSample", wintypes.WORD), ("cbSize", wintypes.WORD)]


class _Bloque(ctypes.Structure):  # WAVEHDR
    pass


_Bloque._fields_ = [("lpData", ctypes.c_void_p), ("dwBufferLength", wintypes.DWORD),
                    ("dwBytesRecorded", wintypes.DWORD), ("dwUser", ctypes.c_size_t),
                    ("dwFlags", wintypes.DWORD), ("dwLoops", wintypes.DWORD),
                    ("lpNext", ctypes.POINTER(_Bloque)), ("reserved", ctypes.c_size_t)]


class Microfono:
    """Audio mono de 16 bits a 'frecuencia' Hz. Windows va llenando 'bloques' de 'tamano' muestras."""

    def __init__(self, frecuencia=44100, tamano=1024, bloques=32):
        self.winmm = w = ctypes.WinDLL("winmm")
        w.waveInOpen.argtypes = [ctypes.POINTER(wintypes.HANDLE), wintypes.UINT, ctypes.POINTER(_Formato),
                                 ctypes.c_size_t, ctypes.c_size_t, wintypes.DWORD]
        for nombre in ("waveInPrepareHeader", "waveInUnprepareHeader", "waveInAddBuffer"):
            getattr(w, nombre).argtypes = [wintypes.HANDLE, ctypes.POINTER(_Bloque), wintypes.UINT]
        for nombre in ("waveInStart", "waveInStop", "waveInReset", "waveInClose"):
            getattr(w, nombre).argtypes = [wintypes.HANDLE]

        formato = _Formato(WAVE_FORMAT_PCM, 1, frecuencia, frecuencia * 2, 2, 16, 0)
        self.dispositivo = wintypes.HANDLE()
        error = w.waveInOpen(ctypes.byref(self.dispositivo), WAVE_MAPPER, ctypes.byref(formato), 0, 0, 0)
        if error:
            raise RuntimeError(f"No se pudo abrir el micrófono (error {error} de Windows).")

        self.memoria = [ctypes.create_string_buffer(tamano * 2) for _ in range(bloques)]
        self.bloques = [_Bloque(ctypes.cast(m, ctypes.c_void_p), tamano * 2) for m in self.memoria]
        for bloque in self.bloques:
            w.waveInPrepareHeader(self.dispositivo, bloque, ctypes.sizeof(bloque))
            w.waveInAddBuffer(self.dispositivo, bloque, ctypes.sizeof(bloque))
        self.siguiente = 0  # los bloques se llenan en orden
        w.waveInStart(self.dispositivo)

    def leer(self):
        """Muestras nuevas (float32 entre -1 y 1). Espera si todavia no hay ningun bloque lleno."""
        partes = []
        while True:
            bloque = self.bloques[self.siguiente]
            if not bloque.dwFlags & WHDR_DONE:
                if partes:
                    return np.concatenate(partes).astype(np.float32) / 32768
                time.sleep(0.005)
                continue
            partes.append(np.frombuffer(self.memoria[self.siguiente], np.int16, bloque.dwBytesRecorded // 2).copy())
            bloque.dwFlags &= ~WHDR_DONE
            self.winmm.waveInAddBuffer(self.dispositivo, bloque, ctypes.sizeof(bloque))  # se vuelve a usar
            self.siguiente = (self.siguiente + 1) % len(self.bloques)

    def cerrar(self):
        self.winmm.waveInStop(self.dispositivo)
        self.winmm.waveInReset(self.dispositivo)  # devuelve los bloques pendientes
        for bloque in self.bloques:
            self.winmm.waveInUnprepareHeader(self.dispositivo, bloque, ctypes.sizeof(bloque))
        self.winmm.waveInClose(self.dispositivo)
