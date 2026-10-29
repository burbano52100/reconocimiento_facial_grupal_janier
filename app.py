"""Control de acceso con los modelos de Teachable Machine (rostro y pose).

Ejecutar:  python app.py
"""
import math
import os
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
import numpy as np

from modelos import (ESQUELETO, AnalizadorCaras, ClasificadorImagen, ClasificadorPose,
                     recorte_cara, recorte_cuadrado)
from seguridad import (AZUL, MAX_INTENTOS, NO_PERSONAS, ROJO, UMBRAL_GIRO, UMBRAL_POSE,
                       UMBRAL_ROSTRO, VERDE, Perfiles, Verificador, decidir_identidad,
                       detectar_gesto, mira_de_frente, orientacion_cara)

CARPETA = os.path.dirname(os.path.abspath(__file__))
LADO_VIDEO = 480
FOTOS_REGISTRO = 10
VERDE_RGB, ROJO_RGB, AZUL_RGB = (0, 200, 0), (220, 0, 0), (30, 120, 255)
AMARILLO_RGB, GRIS_RGB, BLANCO_RGB = (255, 200, 0), (170, 170, 170), (255, 255, 255)
METAS = {"arriba": (0, -1), "abajo": (0, 1), "izquierda": (-1, 0), "derecha": (1, 0)}

BOTONES = [  # (modo, texto, fila, columna)
    ("imagen", "Reconocimiento facial", 0, 0),
    ("registrar_imagen", "Registrar rostro", 0, 1),
    ("pose", "Reconocimiento por pose", 1, 0),
    ("registrar_pose", "Registrar pose", 1, 1),
]


def dibujar_esqueleto(imagen, puntos, confianzas, minimo=0.4):
    for a, b in ESQUELETO:
        if confianzas[a] > minimo and confianzas[b] > minimo:
            cv2.line(imagen, tuple(map(int, puntos[a])), tuple(map(int, puntos[b])), (0, 255, 0), 3)
    for (x, y), c in zip(puntos, confianzas):
        if c > minimo:
            cv2.circle(imagen, (int(x), int(y)), 5, (255, 0, 0), -1)


def dibujar_cara(imagen, cara, escala, color):
    x, y, w, h = (cara[:4] * escala).astype(int)
    cv2.rectangle(imagen, (x, y), (x + w, y + h), color, 2)
    for i in range(4, 14, 2):
        cv2.circle(imagen, (int(cara[i] * escala), int(cara[i + 1] * escala)), 3, color, -1)


def dibujar_guia(imagen, cara, escala, desplazamiento, reto):
    """Metas alrededor de la cara (la pedida en amarillo) y el punto verde que se mueve con la cara."""
    x, y, w, h = cara[:4] * escala
    centro = np.array([x + w / 2, y + h / 2])
    distancia = max(0.8 * max(w, h), 90)  # el punto llega a la meta justo cuando el movimiento basta
    for direccion, vector in METAS.items():
        meta = tuple((centro + distancia * np.array(vector)).astype(int))
        if direccion == reto:
            cv2.arrowedLine(imagen, tuple(centro.astype(int)), meta, AMARILLO_RGB, 2, cv2.LINE_AA, tipLength=0.15)
            cv2.circle(imagen, meta, 18, AMARILLO_RGB, 3, cv2.LINE_AA)
        else:
            cv2.circle(imagen, meta, 12, GRIS_RGB, 1, cv2.LINE_AA)
    punto = tuple((centro + distancia * np.clip(desplazamiento, -1.3, 1.3)).astype(int))
    cv2.circle(imagen, punto, 14, VERDE_RGB, -1, cv2.LINE_AA)
    cv2.circle(imagen, punto, 14, BLANCO_RGB, 2, cv2.LINE_AA)


class App:
    def __init__(self, raiz):
        self.raiz = raiz
        self.modelo_imagen = ClasificadorImagen(os.path.join(CARPETA, "modelo_imagen", "keras_model.h5"),
                                                os.path.join(CARPETA, "modelo_imagen", "labels.txt"))
        self.modelo_pose = ClasificadorPose(os.path.join(CARPETA, "modelo_pose"))
        self.caras = None  # detector de caras: se carga la primera vez que se usa
        self.perfiles = Perfiles(os.path.join(CARPETA, "perfiles"))
        self.verificador = Verificador(self.perfiles)

        self.modo = None        # modo elegido (ver BOTONES) o None si la camara esta apagada
        self.modo_pintado = None
        self.registro = None    # datos del registro en curso
        self.resultado = None   # (imagen, info) del ultimo cuadro procesado
        self.estado = "Elige una opción para encender la cámara."
        self.error = None
        self.aviso = None
        self.hilo = None
        self.alerta = None
        self.mensaje_acceso = None   # ventana de "acceso concedido"
        self.acceso_mostrado = 0     # para mostrar cada acceso una sola vez
        self.etiquetas_barras = None

        raiz.title("Control de acceso con Teachable Machine")
        raiz.configure(padx=12, pady=12)
        raiz.protocol("WM_DELETE_WINDOW", self.cerrar)

        self.imagen_vacia = tk.PhotoImage(width=LADO_VIDEO, height=LADO_VIDEO)
        self.video = tk.Label(raiz, image=self.imagen_vacia, bg="black")
        self.video.grid(row=0, column=0)

        panel = tk.Frame(raiz, padx=12)
        panel.grid(row=0, column=1, sticky="n")
        marco_botones = tk.Frame(panel)
        marco_botones.pack(fill="x")
        self.botones = {}
        for modo, texto, fila, columna in BOTONES:
            boton = tk.Button(marco_botones, text=texto, width=21, height=2, font=("Segoe UI", 10, "bold"),
                              command=lambda m=modo: self.pulsar(m))
            boton.grid(row=fila, column=columna, padx=3, pady=3)
            self.botones[modo] = boton
        self.colores_normales = boton.cget("bg"), boton.cget("fg")

        self.texto_titulo = tk.Label(panel, font=("Segoe UI", 14, "bold"), width=30, height=3,
                                     wraplength=380, justify="left", anchor="w")
        self.texto_titulo.pack(fill="x", pady=(10, 0))
        self.texto_detalle = tk.Label(panel, font=("Segoe UI", 9), fg="gray25", width=60, height=4,
                                      wraplength=380, justify="left", anchor="nw")
        self.texto_detalle.pack(fill="x")
        self.marco_barras = tk.Frame(panel)
        self.marco_barras.pack(fill="x", pady=6)
        self.texto_intentos = tk.Label(panel, font=("Segoe UI", 10, "bold"), anchor="w")
        self.texto_intentos.pack(fill="x")
        self.texto_estado = tk.Label(raiz, fg="gray", anchor="w")
        self.texto_estado.grid(row=1, column=0, columnspan=2, sticky="we", pady=(8, 0))

        self._refrescar()

    # ---------------- botones ----------------

    def pulsar(self, modo):
        if self.modo == modo:  # pulsar el boton activo apaga la camara
            self.modo = None
            self.estado = "Cámara apagada. Elige una opción."
            return
        if modo.startswith("registrar"):
            nombre = self._dialogo_registro(modo)
            if not nombre:
                return
            self.registro = dict(nombre=nombre, inicio=None, ultima=0, huellas=[], fotos=[])
        self.modo = modo

    def _dialogo_registro(self, modo):
        """Pide el perfil (persona) a registrar. Devuelve su nombre o None."""
        tipo = "rostro" if modo == "registrar_imagen" else "pose"
        modelo = self.modelo_imagen if tipo == "rostro" else self.modelo_pose
        personas = [e for e in modelo.etiquetas if e not in NO_PERSONAS]

        ventana = tk.Toplevel(self.raiz, padx=14, pady=12)
        ventana.title(f"Registrar {tipo}")
        ventana.transient(self.raiz)
        ventana.resizable(False, False)
        tk.Label(ventana, text="Perfil:").grid(row=0, column=0, sticky="w")
        nombre = ttk.Combobox(ventana, values=personas, state="readonly", width=26)
        nombre.grid(row=0, column=1, pady=4)
        consejo = (f"Se tomarán {FOTOS_REGISTRO} fotos de tu cara. Mira a la cámara y mueve un poco la cabeza."
                   if tipo == "rostro" else
                   f"Se tomarán {FOTOS_REGISTRO} fotos de tu cuerpo. Colócate como cuando entrenaste el modelo de pose.")
        tk.Label(ventana, text=consejo, fg="gray25", wraplength=300, justify="left").grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(10, 0))

        respuesta = []

        def aceptar():
            if not nombre.get():
                messagebox.showwarning("Falta el perfil", "Elige el perfil a registrar.", parent=ventana)
                return
            respuesta.append(nombre.get())
            ventana.destroy()

        marco = tk.Frame(ventana)
        marco.grid(row=2, column=0, columnspan=2, pady=(12, 0))
        tk.Button(marco, text="Tomar fotos", command=aceptar).pack(side="left", padx=4)
        tk.Button(marco, text="Cancelar", command=ventana.destroy).pack(side="left", padx=4)
        ventana.grab_set()
        self.raiz.wait_window(ventana)
        return respuesta[0] if respuesta else None

    # ---------------- camara y modelos (hilo aparte) ----------------

    def _analizador_caras(self):
        if self.caras is None:
            anterior, self.estado = self.estado, "Cargando el detector de caras..."
            self.caras = AnalizadorCaras(os.path.join(CARPETA, "modelo_caras"))
            self.estado = anterior
        return self.caras

    def _trabajar(self):
        self.estado = "Abriendo la cámara..."
        camara = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not camara.isOpened():
            camara = cv2.VideoCapture(0)
        modo_anterior = None
        try:
            if not camara.isOpened():
                raise RuntimeError("No se pudo abrir la cámara.")
            while self.modo:
                modo = self.modo
                if modo != modo_anterior:
                    self.verificador.iniciar(modo)
                    self.estado = "Pulsa el mismo botón para apagar la cámara."
                    modo_anterior = modo

                ok, cuadro = camara.read()
                if not ok:
                    raise RuntimeError("No se pudo leer la imagen de la cámara.")
                # Espejo + recorte cuadrado, igual que la camara de Teachable Machine
                bgr = recorte_cuadrado(cv2.flip(cuadro, 1))
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                vista = cv2.resize(rgb, (LADO_VIDEO, LADO_VIDEO))
                escala = LADO_VIDEO / bgr.shape[0]
                ahora = time.monotonic()

                if ahora < self.verificador.bloqueado_hasta:
                    vista = vista // 3  # imagen oscurecida mientras dura el bloqueo
                    info = {"titulo": "Aplicación bloqueada", "color": ROJO, "detalle": ""}
                elif modo.startswith("registrar"):
                    info = self._paso_registro(modo, bgr, rgb, vista, escala, ahora)
                else:
                    info = self._paso_verificacion(modo, bgr, rgb, vista, escala, ahora)
                if self.modo == modo:
                    self.resultado = (vista, info)
        except Exception as e:
            self.error = str(e)
        finally:
            camara.release()

    def _paso_verificacion(self, modo, bgr, rgb, vista, escala, ahora):
        v = self.verificador
        info = {}
        confianza = None
        if v.necesita == "identidad":
            if modo == "imagen":
                dato, detalle, probabilidades, confianza = self._identidad_rostro(bgr, rgb, vista, escala)
                info["etiquetas"] = self.modelo_imagen.etiquetas
            else:
                dato, detalle, probabilidades, confianza = self._identidad_pose(rgb, vista, escala)
                info["etiquetas"] = self.modelo_pose.etiquetas
            info["probabilidades"] = probabilidades
        elif v.necesita == "gesto":
            _, _, puntos, confianzas = self.modelo_pose.predecir(rgb)
            dibujar_esqueleto(vista, puntos * escala, confianzas)
            dato = detectar_gesto(puntos, confianzas)
            detalle = "Gesto detectado: " + (dato or "no se ven los hombros, aléjate un poco.")
        else:  # orientacion de la cara
            caras = self._analizador_caras().detectar(bgr)
            dato = orientacion_cara(caras[0]) if len(caras) == 1 else None
            if dato is None:
                for cara in caras:
                    dibujar_cara(vista, cara, escala, ROJO_RGB)
                detalle = "Debe verse una sola cara."
            elif v.fase == "centrar":
                dibujar_guia(vista, caras[0], escala, (0, 0), None)
                detalle = ("Quédate quieto mirando de frente: enseguida te diré qué movimiento hacer."
                           if mira_de_frente(dato) else "Mira de frente a la cámara.")
            elif v.fase == "movimiento":
                dh, dv = v.desplazamiento(dato)
                dibujar_guia(vista, caras[0], escala, (dh, dv), v.reto)
                detalle = ("Mueve la cabeza para llevar el punto verde a la meta amarilla.\n"
                           f"Izquierda / derecha: {dh:+.1f}   ·   Arriba / abajo: {dv:+.1f}   (meta: ±1)")
            else:  # giro de cabeza del modo pose
                dibujar_cara(vista, caras[0], escala, AZUL_RGB)
                detalle = f"Giro de la cabeza: {dato[0]:+.2f} (hace falta pasar de ±{UMBRAL_GIRO})"

        v.paso(ahora, dato, confianza)
        info.update(titulo=v.titulo, color=v.color, detalle=detalle,
                    intentos=f"Intentos fallidos: {v.fallos}/{MAX_INTENTOS}")
        return info

    def _identidad_rostro(self, bgr, rgb, vista, escala):
        """1) detecta la cara, 2) analiza solo la cara, 3) analiza la foto completa, 4) compara con el perfil."""
        caras = self._analizador_caras().detectar(bgr)
        if not caras:
            return None, "No se detecta ninguna cara.", None, None
        if len(caras) > 1:
            for cara in caras:
                dibujar_cara(vista, cara, escala, ROJO_RGB)
            return "otros", "Hay más de una cara: solo debe haber una persona.", None, None

        cara = caras[0]
        etiquetas = self.modelo_imagen.etiquetas
        prob_cara = self.modelo_imagen.predecir(recorte_cara(rgb, cara))
        prob_foto = self.modelo_imagen.predecir(rgb)
        resultados = [(etiquetas[p.argmax()], float(p.max())) for p in (prob_cara, prob_foto)]
        dato, motivo = decidir_identidad(
            resultados, self.perfiles, "rostro",
            lambda: self.caras.huella(self.caras.recorte_alineado(bgr, cara)), UMBRAL_ROSTRO)

        dibujar_cara(vista, cara, escala, ROJO_RGB if dato in ("otros", "falso") else VERDE_RGB)
        (n1, p1), (n2, p2) = resultados
        detalle = f"Solo la cara: {n1} {p1:.0%}   ·   Foto completa: {n2} {p2:.0%}\n{motivo}"
        return dato, detalle, prob_cara, min(p1, p2)

    def _identidad_pose(self, rgb, vista, escala):
        probabilidades, huella, puntos, confianzas = self.modelo_pose.predecir(rgb)
        dibujar_esqueleto(vista, puntos * escala, confianzas)
        if min(confianzas[0], confianzas[5], confianzas[6]) < 0.5:
            return None, "No se detecta una persona (deben verse la cabeza y los hombros).", probabilidades, None
        nombre = self.modelo_pose.etiquetas[probabilidades.argmax()]
        p = float(probabilidades.max())
        dato, motivo = decidir_identidad([(nombre, p)], self.perfiles, "pose", lambda: huella, UMBRAL_POSE)
        return dato, f"Pose: {nombre} {p:.0%}\n{motivo}", probabilidades, p

    def _paso_registro(self, modo, bgr, rgb, vista, escala, ahora):
        r = self.registro
        tipo = "rostro" if modo == "registrar_imagen" else "pose"
        if r["inicio"] is None:
            r["inicio"] = ahora

        if tipo == "rostro":
            caras = self._analizador_caras().detectar(bgr)
            valido = len(caras) == 1
            for cara in caras:
                dibujar_cara(vista, cara, escala, VERDE_RGB if valido else ROJO_RGB)
            consejo = "Mira a la cámara y mueve un poco la cabeza." if valido else "Debe verse una sola cara."
        else:
            _, huella, puntos, confianzas = self.modelo_pose.predecir(rgb)
            dibujar_esqueleto(vista, puntos * escala, confianzas)
            valido = min(confianzas[0], confianzas[5], confianzas[6]) >= 0.5
            consejo = "Muévete un poco entre fotos." if valido else "Deben verse la cabeza y los hombros."

        espera = 3 - (ahora - r["inicio"])
        if espera > 0:
            return {"titulo": f"Registro de {r['nombre']}: prepárate... {math.ceil(espera)}",
                    "color": AZUL, "detalle": consejo}

        if valido and ahora - r["ultima"] >= 0.2:  # una foto cada 0,2 s
            if tipo == "rostro":
                foto = self.caras.recorte_alineado(bgr, caras[0])
                r["huellas"].append(self.caras.huella(foto))
            else:
                foto = bgr.copy()
                r["huellas"].append(huella)
            r["fotos"].append(foto)
            r["ultima"] = ahora

        n = len(r["huellas"])
        if n >= FOTOS_REGISTRO:
            total = self.perfiles.agregar(r["nombre"], tipo, r["huellas"], r["fotos"])
            self.aviso = (f"Se añadieron {n} fotos al perfil de {tipo} de {r['nombre']}.\n"
                          f"Total en el perfil: {total} fotos.")
            self.modo = None
            self.estado = "Registro completado."
        return {"titulo": f"Registro de {r['nombre']}: foto {n}/{FOTOS_REGISTRO}", "color": AZUL,
                "detalle": consejo}

    # ---------------- interfaz (hilo principal) ----------------

    def _refrescar(self):
        if self.error:
            error, self.error = self.error, None
            self.modo = None
            messagebox.showerror("Error", error)
        if self.aviso:
            aviso, self.aviso = self.aviso, None
            messagebox.showinfo("Registro completado", aviso)

        if self.modo != self.modo_pintado:
            self._pintar_modo()
        # Arranca el hilo de la camara si hay un modo elegido y no esta corriendo
        if self.modo and not (self.hilo and self.hilo.is_alive()):
            self.hilo = threading.Thread(target=self._trabajar, daemon=True)
            self.hilo.start()

        resultado, self.resultado = self.resultado, None
        if resultado and self.modo:
            self._mostrar(*resultado)
        self._revisar_bloqueo()
        self._revisar_acceso()
        self.texto_estado.configure(text=self.estado)
        self.tarea_refresco = self.raiz.after(30, self._refrescar)

    def _pintar_modo(self):
        self.modo_pintado = self.modo
        for modo, boton in self.botones.items():
            fondo, texto = (VERDE, "white") if modo == self.modo else self.colores_normales
            boton.configure(bg=fondo, fg=texto)
        self.video.configure(image=self.imagen_vacia)
        self.texto_titulo.configure(text="")
        self.texto_detalle.configure(text="")
        self._crear_barras([])

    def _mostrar(self, vista, info):
        alto, ancho = vista.shape[:2]
        self.foto = tk.PhotoImage(data=b"P6 %d %d 255 " % (ancho, alto) + vista.tobytes())
        self.video.configure(image=self.foto)
        self.texto_titulo.configure(text=info["titulo"], fg=info["color"])
        self.texto_detalle.configure(text=info["detalle"])
        self.texto_intentos.configure(text=info.get("intentos", ""))

        probabilidades = info.get("probabilidades")
        if probabilidades is not None:
            if info["etiquetas"] != self.etiquetas_barras:
                self._crear_barras(info["etiquetas"])
            for (barra, porcentaje), p in zip(self.barras, probabilidades):
                barra.configure(value=p * 100)
                porcentaje.configure(text=f"{p:.0%}")

    def _crear_barras(self, etiquetas):
        for hijo in self.marco_barras.winfo_children():
            hijo.destroy()
        self.barras = []
        for fila, etiqueta in enumerate(etiquetas):
            tk.Label(self.marco_barras, text=etiqueta, width=10, anchor="w").grid(row=fila, column=0)
            barra = ttk.Progressbar(self.marco_barras, length=200, maximum=100)
            barra.grid(row=fila, column=1, pady=2)
            porcentaje = tk.Label(self.marco_barras, width=5, anchor="e")
            porcentaje.grid(row=fila, column=2)
            self.barras.append((barra, porcentaje))
        self.etiquetas_barras = etiquetas

    def _revisar_acceso(self):
        """Muestra automaticamente el mensaje de acceso concedido con el perfil, y lo cierra solo."""
        perfil, hasta = self.verificador.ultimo_acceso
        if hasta != self.acceso_mostrado and time.monotonic() < hasta:
            self.acceso_mostrado = hasta
            self._cerrar_mensaje_acceso()
            self.raiz.bell()
            ventana = tk.Toplevel(self.raiz, bg=VERDE, padx=40, pady=24)
            ventana.title("Acceso concedido")
            ventana.transient(self.raiz)
            ventana.resizable(False, False)
            ventana.protocol("WM_DELETE_WINDOW", self._cerrar_mensaje_acceso)
            tk.Label(ventana, text="✔  ACCESO CONCEDIDO", font=("Segoe UI", 22, "bold"),
                     fg="white", bg=VERDE).pack()
            tk.Label(ventana, text=f"Perfil: {perfil}", font=("Segoe UI", 16), fg="white", bg=VERDE).pack(pady=(10, 0))
            ventana.update_idletasks()
            x = self.raiz.winfo_rootx() + (self.raiz.winfo_width() - ventana.winfo_width()) // 2
            y = self.raiz.winfo_rooty() + (self.raiz.winfo_height() - ventana.winfo_height()) // 2
            ventana.geometry(f"+{x}+{y}")
            self.mensaje_acceso = ventana
        elif self.mensaje_acceso is not None and time.monotonic() >= hasta:
            self._cerrar_mensaje_acceso()

    def _cerrar_mensaje_acceso(self):
        if self.mensaje_acceso is not None:
            self.mensaje_acceso.destroy()
            self.mensaje_acceso = None

    def _revisar_bloqueo(self):
        """Muestra la alerta y bloquea los botones mientras dure el bloqueo."""
        restante = self.verificador.bloqueado_hasta - time.monotonic()
        if restante > 0 and self.alerta is None:
            self.raiz.bell()
            self.alerta = tk.Toplevel(self.raiz, bg=ROJO, padx=30, pady=20)
            self.alerta.title("Alerta de seguridad")
            self.alerta.transient(self.raiz)
            self.alerta.resizable(False, False)
            self.alerta.protocol("WM_DELETE_WINDOW", lambda: None)  # no se puede cerrar
            tk.Label(self.alerta, text="⚠  LÍMITE DE INTENTOS FALLIDOS", font=("Segoe UI", 18, "bold"),
                     fg="white", bg=ROJO).pack()
            self.texto_alerta = tk.Label(self.alerta, font=("Segoe UI", 12), fg="white", bg=ROJO)
            self.texto_alerta.pack(pady=(10, 0))
            for boton in self.botones.values():
                boton.configure(state="disabled")
            self.alerta.update_idletasks()
            x = self.raiz.winfo_rootx() + (self.raiz.winfo_width() - self.alerta.winfo_width()) // 2
            y = self.raiz.winfo_rooty() + (self.raiz.winfo_height() - self.alerta.winfo_height()) // 2
            self.alerta.geometry(f"+{x}+{y}")
            self.alerta.grab_set()

        if self.alerta is not None:
            if restante > 0:
                self.texto_alerta.configure(text=f"La aplicación está bloqueada. Espera {math.ceil(restante)} s.")
            else:
                self.alerta.grab_release()
                self.alerta.destroy()
                self.alerta = None
                for boton in self.botones.values():
                    boton.configure(state="normal")

    def cerrar(self):
        self.raiz.after_cancel(self.tarea_refresco)
        self.modo = None
        if self.hilo:
            self.hilo.join(timeout=2)
        self.raiz.destroy()


if __name__ == "__main__":
    raiz = tk.Tk()
    App(raiz)
    raiz.mainloop()
