"""Mide cuanto acierta cada modelo con fotos y grabaciones de cada persona.

Uso:
  python probar_modelos.py fotos <persona> [cantidad]    toma fotos con la camara  -> muestras/<persona>/
  python probar_modelos.py grabar <persona> [segundos]   graba la voz              -> muestras/<persona>/
  python probar_modelos.py [carpeta]                     prueba los tres modelos con todas las muestras

En muestras/ va una carpeta por persona (burbano, kevin, julian, salgado...). La carpeta "desconocidos"
es para personas que NO deben entrar; si no existe, se usan las fotos de "otros" que trae modelos.zip.
Al grabar, di tu palabra secreta varias veces con una pausa despues de cada una: cada frase cuenta
como un intento de entrar. Se analiza todo con las mismas funciones y reglas de la app.
"""
import collections
import glob
import os
import sys
import time
import tkinter as tk
import wave
import zipfile

import cv2
import numpy as np

import app
from microfono import Microfono
from modelos import DetectorFrase, Escucha, recorte_cara, recorte_cuadrado
from seguridad import NIVEL_MINIMO_VOZ, NO_PERSONAS, UMBRAL_CONFIANZA, decidir_palabra

MUESTRAS = os.path.join(app.CARPETA, "muestras")
DESCONOCIDOS = "desconocidos"
RECHAZOS = (None, "otros", "falso")


# ---------------- tomar muestras ----------------

def _carpeta(persona):
    carpeta = os.path.join(MUESTRAS, persona)
    os.makedirs(carpeta, exist_ok=True)
    return carpeta


def tomar_fotos(persona, cantidad=20):
    carpeta = _carpeta(persona)
    camara = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not camara.isOpened():
        sys.exit("No se pudo abrir la cámara.")
    print(f"Se tomarán {cantidad} fotos de {persona}, una cada medio segundo. Muévete un poco entre fotos.")
    for n in (3, 2, 1):
        print(f"  {n}...")
        time.sleep(1)
    inicio = len(glob.glob(os.path.join(carpeta, "*.jpg")))
    for i in range(cantidad):
        ok, cuadro = camara.read()
        if not ok:
            sys.exit("No se pudo leer la cámara.")
        foto = recorte_cuadrado(cv2.flip(cuadro, 1))  # lo mismo que ve la app
        cv2.imencode(".jpg", foto)[1].tofile(os.path.join(carpeta, f"foto_{inicio + i + 1:03d}.jpg"))
        print(f"  foto {i + 1}/{cantidad}")
        time.sleep(0.5)
    camara.release()
    print(f"Fotos guardadas en {carpeta}")


def grabar(persona, segundos=10):
    carpeta = _carpeta(persona)
    microfono = Microfono()
    print(f"Grabando {segundos} s: {persona}, di tu palabra secreta varias veces, con una pausa después de cada una...")
    partes, total = [], 0
    while total < segundos * 44100:
        partes.append(microfono.leer())
        total += len(partes[-1])
    microfono.cerrar()
    audio = (np.clip(np.concatenate(partes), -1, 1) * 32767).astype(np.int16)
    ruta = os.path.join(carpeta, f"voz_{len(glob.glob(os.path.join(carpeta, '*.wav'))) + 1:03d}.wav")
    with wave.open(ruta, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(44100)
        w.writeframes(audio.tobytes())
    print(f"Volumen máximo: {20 * np.log10(np.abs(audio).max() / 32768 + 1e-9):.0f} dB. Guardado en {ruta}")


# ---------------- leer muestras ----------------

def leer_fotos(carpeta, persona):
    rutas = sorted(glob.glob(os.path.join(carpeta, persona, "*.jpg")) + glob.glob(os.path.join(carpeta, persona, "*.png")))
    fotos = [cv2.imdecode(np.fromfile(r, np.uint8), cv2.IMREAD_COLOR) for r in rutas]
    if persona == DESCONOCIDOS and not fotos:  # fotos de "otros" que vienen en los zip de Teachable Machine
        for ruta_zip in glob.glob(os.path.join(app.CARPETA, "modelos.zip", "*.zip")):
            with zipfile.ZipFile(ruta_zip) as z:
                fotos += [cv2.imdecode(np.frombuffer(z.read(n), np.uint8), cv2.IMREAD_COLOR)
                          for n in z.namelist() if "-samples/" in n and n.lower().endswith(".jpg")]
    return fotos


def leer_audios(carpeta, persona):
    audios = []
    for ruta in sorted(glob.glob(os.path.join(carpeta, persona, "*.wav"))):
        with wave.open(ruta) as w:
            if w.getframerate() != 44100 or w.getsampwidth() != 2:
                print(f"  (se omite {ruta}: debe ser WAV de 16 bits a 44100 Hz)")
                continue
            datos = np.frombuffer(w.readframes(w.getnframes()), np.int16).reshape(-1, w.getnchannels())
        audios.append(datos.mean(axis=1).astype(np.float32) / 32768)
    return audios


# ---------------- analisis (mismas funciones que la app) ----------------

def analizar_foto(a, bgr):
    """Devuelve {analisis: resultado} para una foto: el nombre de la persona o None/'otros'."""
    bgr = recorte_cuadrado(bgr)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    vista, escala = np.zeros((app.LADO_VIDEO, app.LADO_VIDEO, 3), np.uint8), app.LADO_VIDEO / bgr.shape[0]
    caras = a._analizador_caras().detectar(bgr)
    if len(caras) == 1:  # solo el modelo de Teachable Machine (sin perfiles)
        resultados = [(a.modelo_imagen.etiquetas[p.argmax()], p.max()) for p in
                      (a.modelo_imagen.predecir(recorte_cara(rgb, caras[0])), a.modelo_imagen.predecir(rgb))]
        nombres = {n for n, _ in resultados}
        modelo_rostro = nombres.pop() if len(nombres) == 1 and min(p for _, p in resultados) >= UMBRAL_CONFIANZA else "otros"
    else:
        modelo_rostro = None
    probabilidades, _, _, confianzas = a.modelo_pose.predecir(rgb)
    modelo_pose = (None if min(confianzas[0], confianzas[5], confianzas[6]) < 0.5 else
                   a.modelo_pose.etiquetas[probabilidades.argmax()] if probabilidades.max() >= UMBRAL_CONFIANZA else "otros")
    modelo_rostro, modelo_pose = ["otros" if n in NO_PERSONAS else n for n in (modelo_rostro, modelo_pose)]
    return {"rostro: solo el modelo": modelo_rostro,
            "rostro: app completa": a._identidad_rostro(bgr, rgb, vista, escala)[0],
            "pose: solo el modelo": modelo_pose,
            "pose: app completa": a._identidad_pose(bgr, rgb, vista, escala)[0]}


def analizar_audio(a, audio):
    """Recorta cada frase de la grabacion (como la app) y devuelve [(voz que reconoce el modelo,
    perfil que entraria con voz + palabra secreta)], uno por frase."""
    voz, salto = a.modelo_voz, a.modelo_voz.SALTO
    escucha, detector = Escucha(voz, app.ESPECTROS_POR_PREDICCION), DetectorFrase(NIVEL_MINIMO_VOZ)
    analisis, frases = collections.deque(maxlen=60), []
    audio = np.concatenate([audio, np.zeros(voz.FRECUENCIA, np.float32)])  # 1 s de silencio al final
    for i in range(0, len(audio), salto):
        if escucha.agregar(audio[i:i + salto]):
            analisis.append((escucha.llenos, voz.predecir(escucha.espectros)))
        frase = detector.actualizar(escucha)
        if frase is not None and analisis:
            persona, _, huella = a._voz_y_palabra(frase, analisis)
            frases.append((persona, decidir_palabra(persona, a.perfiles, huella)[0]))
    return frases


# ---------------- informe ----------------

def resumen(persona, resultados, personas_modelo):
    """Texto con el % de aciertos. Una persona que no esta en el modelo deberia ser rechazada."""
    n = len(resultados)
    if not n:
        return "sin muestras"
    debe_entrar = persona in personas_modelo
    conteo = collections.Counter("rechaza" if r in RECHAZOS else r for r in resultados)
    rechaza = conteo.pop("rechaza", 0)
    if debe_entrar:
        acierta = conteo.pop(persona, 0)
        texto = f"acierta {acierta}/{n} ({acierta / n:.0%})   rechaza {rechaza} ({rechaza / n:.0%})"
    else:
        texto = f"rechaza bien {rechaza}/{n} ({rechaza / n:.0%})"
    if conteo:
        texto += "   SE CONFUNDE con: " + ", ".join(f"{k} {c} ({c / n:.1%})" for k, c in conteo.most_common())
    return texto


def fiabilidad(todos, personas_modelo):
    """Cuando el analisis dice 'es X', que % de las veces era X de verdad."""
    dichos = collections.defaultdict(collections.Counter)
    for persona, resultados in todos.items():
        for r in resultados:
            if r not in RECHAZOS:
                dichos[r][persona] += 1
    return "   ".join(f"cuando dice «{x}» acierta {c[x]}/{sum(c.values())} ({c[x] / sum(c.values()):.0%})"
                       for x, c in sorted(dichos.items()) if x in personas_modelo) or "nunca dio acceso"


def evaluar(carpeta):
    raiz = tk.Tk()
    raiz.withdraw()
    a = app.App(raiz)
    personas = sorted(set(os.listdir(carpeta) if os.path.isdir(carpeta) else []) | {DESCONOCIDOS})
    for tipo in ("rostro", "pose"):
        print(f"Perfiles de {tipo}: " + ", ".join(f"{n} ({a.perfiles.cantidad(n, tipo)} fotos)"
                                                for n in a.personas[tipo]))

    fotos = {}
    for persona in personas:
        lista = leer_fotos(carpeta, persona)
        if lista:
            print(f"Analizando {len(lista)} fotos de {persona}...")
            fotos[persona] = [analizar_foto(a, f) for f in lista]
    for analisis, tipo in [("rostro: solo el modelo", "rostro"), ("rostro: app completa", "rostro"),
                           ("pose: solo el modelo", "pose"), ("pose: app completa", "pose")]:
        print(f"\n=== {analisis.upper()}  (personas del modelo: {', '.join(a.personas[tipo])}) ===")
        todos = {p: [r[analisis] for r in rs if r[analisis] is not None] for p, rs in fotos.items()}
        for persona, resultados in todos.items():
            sin = len(fotos[persona]) - len(resultados)
            print(f"  {persona:13s} {len(resultados):4d} fotos  {resumen(persona, resultados, a.personas[tipo])}"
                  + (f"   ({sin} sin {'cara' if tipo == 'rostro' else 'cabeza y hombros'} visibles)" if sin else ""))
        print("  Fiabilidad: " + fiabilidad(todos, a.personas[tipo]))

    frases = {}
    for persona in personas:
        audios = leer_audios(carpeta, persona)
        if audios:
            print(f"\nAnalizando {len(audios)} grabaciones de {persona}...")
            frases[persona] = [f for audio in audios for f in analizar_audio(a, audio)]
    if frases:
        con_palabra = [n for n in a.personas["voz"] if a.perfiles.palabra(n)]
        print(f"\nPalabra secreta registrada: {', '.join(con_palabra) or 'nadie'}")
        for indice, analisis in [(0, "VOZ: SOLO EL MODELO"), (1, "VOZ: APP COMPLETA (voz + palabra secreta)")]:
            print(f"\n=== {analisis}  (personas del modelo: {', '.join(a.personas['voz'])}) ===")
            todos = {p: [f[indice] for f in fs] for p, fs in frases.items()}
            for persona, resultados in todos.items():
                print(f"  {persona:13s} {len(resultados):4d} frases  {resumen(persona, resultados, a.personas['voz'])}")
            print("  Fiabilidad: " + fiabilidad(todos, a.personas["voz"]))
    else:
        print("\n(No hay grabaciones de voz: usa  python probar_modelos.py grabar <persona>)")
    raiz.destroy()


if __name__ == "__main__":
    argumentos = sys.argv[1:]
    if argumentos[:1] == ["fotos"] and len(argumentos) >= 2:
        tomar_fotos(argumentos[1], int(argumentos[2]) if len(argumentos) > 2 else 20)
    elif argumentos[:1] == ["grabar"] and len(argumentos) >= 2:
        grabar(argumentos[1], int(argumentos[2]) if len(argumentos) > 2 else 10)
    else:
        evaluar(argumentos[0] if argumentos else MUESTRAS)
