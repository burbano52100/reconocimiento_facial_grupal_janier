"""Vuelve a registrar cada perfil con las fotos que ya estan guardadas en su carpeta.

perfiles/<nombre>/fotos_rostro/*.jpg -> perfiles/<nombre>/rostro.npy
perfiles/<nombre>/fotos_pose/*.jpg   -> perfiles/<nombre>/pose.npy

El perfil queda con exactamente esas fotos: si se borra una foto mala y se ejecuta este script, su
huella sale del perfil. El archivo anterior se guarda como <tipo>.npy.bak.

Ejecutar:  python registrar_fotos.py
"""
import glob
import os
import shutil

import cv2
import numpy as np

from modelos import AnalizadorCaras, ClasificadorPose

CARPETA = os.path.dirname(os.path.abspath(__file__))
PERFILES = os.path.join(CARPETA, "perfiles")


def leer(ruta):
    return cv2.imdecode(np.fromfile(ruta, np.uint8), cv2.IMREAD_COLOR)


def main():
    caras = AnalizadorCaras(os.path.join(CARPETA, "modelo_caras"))
    pose = None  # el modelo de pose solo se carga si hay fotos de pose
    for nombre in sorted(os.listdir(PERFILES)):
        for tipo in ("rostro", "pose"):
            fotos = sorted(glob.glob(os.path.join(PERFILES, nombre, f"fotos_{tipo}", "*.jpg")))
            if not fotos:
                continue
            if tipo == "rostro":  # las fotos de rostro ya son la cara alineada de 112x112
                huellas = [caras.huella(leer(f)) for f in fotos]
            else:  # las fotos de pose son el cuadro completo de la camara (ya en espejo y cuadrado)
                pose = pose or ClasificadorPose(os.path.join(CARPETA, "modelo_pose"))
                huellas = [pose.predecir(cv2.cvtColor(leer(f), cv2.COLOR_BGR2RGB))[1] for f in fotos]

            ruta = os.path.join(PERFILES, nombre, f"{tipo}.npy")
            antes = len(np.load(ruta)) if os.path.exists(ruta) else 0
            if os.path.exists(ruta):
                shutil.copy(ruta, ruta + ".bak")
            np.save(ruta, np.array(huellas, np.float32))
            print(f"{nombre:10} {tipo:6}: {len(huellas)} fotos registradas (antes {antes} huellas)")


if __name__ == "__main__":
    main()
