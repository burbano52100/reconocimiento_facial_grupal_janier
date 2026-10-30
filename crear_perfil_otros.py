"""Crea el perfil de la clase "otros": personas que NO estan en los perfiles.

Toma las fotos de otras personas (las mismas "otros-samples" de Teachable Machine), guarda la huella
de cada cara en perfiles/otros/rostro.npy y reemplaza el perfil anterior. Al reconocer un rostro, si la
cara se parece mas a "otros" que al perfil de la persona, el resultado es "otros" (desconocido).

Ejecutar:  python crear_perfil_otros.py                 (usa las fotos de modelos.zip)
           python crear_perfil_otros.py <carpeta o .zip> (otras fotos)
"""
import os
import shutil
import sys
import zipfile

import cv2
import numpy as np

from modelos import AnalizadorCaras
from seguridad import OTROS, Perfiles

CARPETA = os.path.dirname(os.path.abspath(__file__))
FOTOS_POR_DEFECTO = os.path.join(CARPETA, "modelos.zip", "converted_keras (1).zip")


def leer_fotos(origen):
    """Devuelve (nombre, imagen BGR) de cada .jpg / .png de una carpeta o de un .zip."""
    extensiones = (".jpg", ".jpeg", ".png")
    if zipfile.is_zipfile(origen):
        with zipfile.ZipFile(origen) as z:
            for nombre in sorted(z.namelist()):
                if nombre.lower().endswith(extensiones) and "otros" in nombre.lower():
                    yield nombre, cv2.imdecode(np.frombuffer(z.read(nombre), np.uint8), cv2.IMREAD_COLOR)
    else:
        for nombre in sorted(os.listdir(origen)):
            if nombre.lower().endswith(extensiones):
                yield nombre, cv2.imdecode(np.fromfile(os.path.join(origen, nombre), np.uint8), cv2.IMREAD_COLOR)


def main():
    origen = sys.argv[1] if len(sys.argv) > 1 else FOTOS_POR_DEFECTO
    caras = AnalizadorCaras(os.path.join(CARPETA, "modelo_caras"))
    huellas, fotos, descartadas = [], [], 0
    for nombre, imagen in leer_fotos(origen):
        encontradas = [] if imagen is None else caras.detectar(imagen)
        if len(encontradas) != 1:  # igual que al registrar: debe verse una sola cara
            descartadas += 1
            continue
        alineada = caras.recorte_alineado(imagen, encontradas[0])
        huellas.append(caras.huella(alineada))
        fotos.append(alineada)
    if not huellas:
        sys.exit(f"No se encontraron caras en {origen}")

    perfiles = Perfiles(os.path.join(CARPETA, "perfiles"))
    shutil.rmtree(os.path.join(perfiles.carpeta, OTROS), ignore_errors=True)  # se reemplaza el perfil anterior
    total = perfiles.agregar(OTROS, "rostro", huellas, fotos)
    print(f"Perfil «{OTROS}» creado con {total} caras ({descartadas} fotos descartadas por no tener una sola cara).")


if __name__ == "__main__":
    main()
