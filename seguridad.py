"""Reglas de seguridad para dar acceso.

Cada intento de acceso empieza igual:
  1. identidad  -> los modelos reconocen a la misma persona con >= 85 % (UMBRAL_CONFIANZA) durante 1 segundo
                   y coincide con su perfil registrado, y ese perfil es el que mas se parece de
                   todos los registrados. Si no: "otros" (desconocido).
                   Si el modelo confunde a las personas, decide el perfil cuando el parecido es muy
                   alto (UMBRAL_CORRECCION_ROSTRO / UMBRAL_CORRECCION_POSE).
                   Pose: ademas la cara debe coincidir con el perfil de rostro de esa persona
                   (probado con 610 fotos de desconocidos: el modelo de pose solo dice "burbano"
                   al 90 % en el 79 % de ellas).
Despues, segun el modo:
  Rostro: 2. centrar     -> sin hacer nada, se guarda la posicion de la cara mirando de frente
                            como referencia (medio segundo).
          3. movimiento  -> la app pide NUM_MOVIMIENTOS movimientos cortos de la cabeza al azar
                            (arriba, abajo, izquierda o derecha).
  Pose:   2. gesto       -> hace un gesto de brazos pedido al azar.
          3. cabeza      -> gira la cabeza al lado pedido al azar.
La voz funciona distinto: la persona dice su palabra secreta (una frase entre dos silencios) y entra
si se cumplen las dos cosas a la vez:
  - el modelo de voz reconoce a la persona con >= 90 % (UMBRAL_VOZ) al menos VECES_VOZ veces durante la frase
    (y a nadie mas), y el volumen supera NIVEL_MINIMO_VOZ;
  - la frase se parece a la palabra secreta registrada de esa persona (distancia <= UMBRAL_PALABRA)
    mas que a la de cualquier otro perfil.
  El modelo de Teachable Machine solo distingue personas, no palabras: la palabra se compara con las
  grabaciones del registro (MFCC + DTW). Se guarda solo esa huella, nunca el audio ni el texto.
Si falla MAX_INTENTOS veces, la app se bloquea SEGUNDOS_BLOQUEO segundos.
"""
import itertools
import os
import random

import cv2
import numpy as np

from modelos import distancia_frases

# ---------------- Ajustes ----------------
UMBRAL_CONFIANZA = 0.85     # probabilidad minima de los modelos de rostro y pose de Teachable Machine
UMBRAL_VOZ = 0.90           # probabilidad minima del modelo de voz
SEGUNDOS_CONSTANCIA = 1.0   # tiempo que debe mantenerse el mismo resultado
UMBRAL_ROSTRO = 0.40        # parecido minimo con el perfil de rostro registrado (0 a 1)
UMBRAL_POSE = 0.85          # parecido minimo con el perfil de pose registrado (0 a 1)
# Parecido con el perfil que corrige al modelo cuando confunde a las personas (ver decidir_identidad).
# Medido con las fotos registradas: rostro propio 0,81-0,98, de otra persona 0,50 como mucho;
# pose propia 0,99, de otra persona 0,78 como mucho.
UMBRAL_CORRECCION_ROSTRO = 0.60
UMBRAL_CORRECCION_POSE = 0.92
UMBRAL_GIRO = 0.15          # cuanto hay que girar la cabeza a izquierda / derecha (unos 20 grados)
UMBRAL_VERTICAL = 0.07      # cuanto hay que mover la cabeza arriba / abajo (unos 12 grados)
NUM_MOVIMIENTOS = 2         # movimientos de la cabeza que se piden en el reconocimiento facial
SEGUNDOS_GESTO = 0.6        # tiempo que hay que mantener cada gesto
NIVEL_MINIMO_VOZ = -55      # volumen minimo (dBFS) para analizar la voz: menos es silencio o microfono apagado
VECES_VOZ = 2               # analisis del modelo de voz que deben reconocer a la persona durante la frase
UMBRAL_PALABRA = 2.5        # diferencia maxima con la palabra secreta registrada (misma frase: ~0,6-1;
                            # otra frase: 3,6 o mas, medido con 90 grabaciones)
REPETICIONES_PALABRA = 3    # veces que se graba la palabra secreta al registrarla
TIEMPO_IDENTIDAD = 10       # segundos para confirmar la identidad
TIEMPO_RETO = 8             # segundos para cada reto
MAX_INTENTOS = 3
SEGUNDOS_BLOQUEO = 5

RUIDO = "Ruido de fondo"    # clase del modelo de voz que significa que nadie habla
# Clases que no son una persona concreta ("falso" = foto o pantalla, si se entrena esa clase)
NO_PERSONAS = {"otros", "nada", "fondo", "falso", RUIDO, "palabra al azar"}
GESTOS = ["mano derecha arriba", "mano izquierda arriba", "ambas manos arriba",
          "brazo derecho en cruz", "brazo izquierdo en cruz", "brazos en cruz"]
NEUTRO = "brazos abajo"
MOVIMIENTOS = {"arriba": "Levanta un poco la cabeza (ARRIBA)",
               "abajo": "Baja un poco la cabeza (ABAJO)",
               "izquierda": "Gira un poco la cabeza a la IZQUIERDA",
               "derecha": "Gira un poco la cabeza a la DERECHA"}

VERDE, ROJO, AZUL, NEGRO = "#2e7d32", "#c62828", "#1565c0", "black"


# ---------------- Patrones del cuerpo y la cara ----------------

def detectar_gesto(puntos, confianzas, minimo=0.3):
    """Gesto de los brazos a partir de los puntos de PoseNet.

    La imagen esta en espejo, asi que la derecha de la pantalla es la derecha de la persona.
    Devuelve un nombre de GESTOS, NEUTRO, "otro" o None si no se ven los hombros.
    """
    if confianzas[5] < 0.4 or confianzas[6] < 0.4:
        return None
    centro_x, altura_hombros = puntos[[5, 6]].mean(axis=0)
    ancho = max(abs(puntos[5, 0] - puntos[6, 0]), 1)
    altura_cabeza = puntos[0, 1] if confianzas[0] >= 0.4 else altura_hombros - 0.8 * ancho

    brazos = {"derecho": "abajo", "izquierdo": "abajo"}  # una muneca que no se ve cuenta como abajo
    for i in (9, 10):  # munecas
        if confianzas[i] < minimo:
            continue
        x, y = puntos[i]
        lado = "derecho" if x > centro_x else "izquierdo"
        if y < altura_cabeza:
            brazos[lado] = "arriba"
        elif abs(y - altura_hombros) < 0.6 * ancho and abs(x - centro_x) > 1.1 * ancho:
            brazos[lado] = "cruz"

    derecho, izquierdo = brazos["derecho"], brazos["izquierdo"]
    if derecho == izquierdo:
        return {"abajo": NEUTRO, "arriba": "ambas manos arriba", "cruz": "brazos en cruz"}[derecho]
    if izquierdo == "abajo":
        return {"arriba": "mano derecha arriba", "cruz": "brazo derecho en cruz"}[derecho]
    if derecho == "abajo":
        return {"arriba": "mano izquierda arriba", "cruz": "brazo izquierdo en cruz"}[izquierdo]
    return "otro"


def orientacion_cara(cara):
    """(horizontal, vertical) de la cara segun donde esta la nariz respecto a los ojos y la boca.

    horizontal: 0 de frente, > 0 girada a la derecha, < 0 a la izquierda (imagen en espejo).
    vertical:   posicion de la nariz entre los ojos (0) y la boca (1); baja al mirar arriba
                y sube al mirar abajo. Se mide en el eje de los ojos, por si la cabeza esta inclinada.
    """
    ojo_a, ojo_b, nariz, boca_a, boca_b = cara[4:14].reshape(5, 2)
    izquierdo, derecho = sorted([ojo_a, ojo_b], key=lambda p: p[0])
    ojos = (izquierdo + derecho) / 2
    distancia_ojos = max(np.linalg.norm(derecho - izquierdo), 1)
    eje_x = (derecho - izquierdo) / distancia_ojos
    eje_y = np.array([-eje_x[1], eje_x[0]])  # perpendicular, hacia abajo
    boca = (boca_a + boca_b) / 2 - ojos
    return (float((nariz - ojos) @ eje_x / distancia_ojos),
            float((nariz - ojos) @ eje_y / max(boca @ eje_y, 1)))


def mira_de_frente(orientacion):
    return orientacion is not None and abs(orientacion[0]) < UMBRAL_GIRO


def desplazamiento_cara(horizontal, vertical):
    """Movimiento de la cara dividido por su umbral: 1 = llega a la meta."""
    return horizontal / UMBRAL_GIRO, vertical / UMBRAL_VERTICAL


def direccion_cara(horizontal, vertical):
    """'arriba', 'abajo', 'izquierda', 'derecha' o 'centro' (respecto a la posicion de referencia)."""
    dh, dv = desplazamiento_cara(horizontal, vertical)
    if max(abs(dh), abs(dv)) < 1:
        return "centro"
    if abs(dh) >= abs(dv):
        return "derecha" if dh > 0 else "izquierda"
    return "abajo" if dv > 0 else "arriba"


# ---------------- Perfiles registrados ----------------

class Perfiles:
    """Guarda en disco: perfiles/<nombre>/{rostro.npy, pose.npy, fotos_rostro/, fotos_pose/, palabra.npz}."""

    def __init__(self, carpeta):
        self.carpeta = carpeta
        self._cache = {}

    def crear(self, nombres):
        """Crea la carpeta de cada persona, aunque todavia no tenga fotos registradas."""
        for nombre in nombres:
            os.makedirs(os.path.join(self.carpeta, nombre), exist_ok=True)

    def cantidad(self, nombre, tipo):
        """Numero de fotos registradas en el perfil."""
        huellas = self.huellas(nombre, tipo)
        return 0 if huellas is None else len(huellas)

    def mas_parecido(self, tipo, huella):
        """(nombre, parecido) del perfil registrado de ese tipo que mas se parece a la huella."""
        nombres = os.listdir(self.carpeta) if os.path.isdir(self.carpeta) else []
        parecidos = [(self.similitud(n, tipo, huella), n) for n in nombres]
        return max(((s, n) for s, n in parecidos if s is not None), default=(None, None))[::-1]

    def huellas(self, nombre, tipo):
        if (nombre, tipo) not in self._cache:
            ruta = os.path.join(self.carpeta, nombre, f"{tipo}.npy")
            self._cache[nombre, tipo] = np.load(ruta) if os.path.exists(ruta) else None
        return self._cache[nombre, tipo]

    def similitud(self, nombre, tipo, huella):
        """Parecido (0 a 1) con el perfil: promedio de las 3 fotos registradas mas parecidas."""
        huellas = self.huellas(nombre, tipo)
        if huellas is None:
            return None
        return float(np.sort(huellas @ huella)[-3:].mean())

    def agregar(self, nombre, tipo, huellas, fotos):
        """Suma las fotos nuevas al perfil (cada registro lo mejora). Devuelve el total de fotos."""
        carpeta = os.path.join(self.carpeta, nombre)
        carpeta_fotos = os.path.join(carpeta, f"fotos_{tipo}")
        os.makedirs(carpeta_fotos, exist_ok=True)
        anteriores = self.huellas(nombre, tipo)
        nuevas = np.array(huellas, np.float32)
        todas = nuevas if anteriores is None else np.concatenate([anteriores, nuevas])
        np.save(os.path.join(carpeta, f"{tipo}.npy"), todas)
        for i, foto in enumerate(fotos, len(todas) - len(nuevas) + 1):
            cv2.imencode(".jpg", foto)[1].tofile(os.path.join(carpeta_fotos, f"{i:03d}.jpg"))
        self._cache[nombre, tipo] = todas
        return len(todas)

    def palabra(self, nombre):
        """Huellas (MFCC) de las grabaciones de la palabra secreta, o None si no tiene."""
        if (nombre, "palabra") not in self._cache:
            ruta = os.path.join(self.carpeta, nombre, "palabra.npz")
            if os.path.exists(ruta):
                with np.load(ruta) as datos:
                    self._cache[nombre, "palabra"] = [datos[k] for k in sorted(datos.files)]
            else:
                self._cache[nombre, "palabra"] = None
        return self._cache[nombre, "palabra"]

    def guardar_palabra(self, nombre, frases):
        """Reemplaza la palabra secreta (solo sus huellas: ni audio ni texto)."""
        os.makedirs(os.path.join(self.carpeta, nombre), exist_ok=True)
        np.savez(os.path.join(self.carpeta, nombre, "palabra.npz"), *frases)
        self._cache[nombre, "palabra"] = list(frases)

    def distancia_palabra(self, nombre, frase):
        """Diferencia entre la frase dicha y la palabra secreta (la grabacion mas parecida), o None."""
        grabaciones = self.palabra(nombre)
        return None if grabaciones is None else min(distancia_frases(frase, g) for g in grabaciones)

    def palabra_mas_parecida(self, frase):
        """(nombre, distancia) del perfil cuya palabra secreta mas se parece a la frase."""
        nombres = os.listdir(self.carpeta) if os.path.isdir(self.carpeta) else []
        distancias = [(self.distancia_palabra(n, frase), n) for n in nombres]
        return min(((d, n) for d, n in distancias if d is not None), default=(None, None))[::-1]


def coherencia_palabra(frases):
    """Diferencia maxima entre las grabaciones del registro (deben parecerse entre si)."""
    return max(distancia_frases(a, b) for a, b in itertools.combinations(frases, 2))


def voz_de_frase(resultados):
    """Persona que reconoce el modelo de voz mientras se dice una frase.

    resultados: (nombre, probabilidad) de cada analisis del modelo durante la frase.
    Devuelve (persona o None, motivo).
    """
    seguras = [n for n, p in resultados if p >= UMBRAL_VOZ and n not in NO_PERSONAS]
    if len(set(seguras)) > 1:
        return None, "El modelo oye la voz de varias personas: " + ", ".join(sorted(set(seguras))) + "."
    if seguras and len(seguras) < VECES_VOZ:
        return None, (f"El modelo de voz reconoció a {seguras[0]} solo {len(seguras)} vez con "
                      f"{UMBRAL_VOZ:.0%} (hacen falta {VECES_VOZ}): habla más claro o un poco más largo.")
    if len(seguras) < VECES_VOZ:
        personas = [(p, n) for n, p in resultados if n not in NO_PERSONAS]
        mejor = f" (lo más alto: {max(personas)[1]} {max(personas)[0]:.0%})" if personas else ""
        return None, f"El modelo de voz no te reconoce con {UMBRAL_VOZ:.0%}{mejor}."
    return seguras[0], f"{seguras[0]}, reconocido {len(seguras)} veces con {UMBRAL_VOZ:.0%} o más."


def decidir_palabra(persona, perfiles, frase):
    """Modo voz: la persona que reconoce el modelo debe haber dicho SU palabra secreta.

    persona: resultado de voz_de_frase (o None). frase: huella de lo que se dijo.
    Devuelve (persona o None, motivo, cuenta_como_fallo). Si ni la voz ni la palabra son de
    nadie (otra gente hablando, ruido) no cuenta como intento fallido.
    """
    duena, distancia = perfiles.palabra_mas_parecida(frase)
    if duena is None:
        return None, "Nadie tiene palabra secreta: usa «Registrar palabra».", False
    palabra_valida = distancia <= UMBRAL_PALABRA
    if persona is None:
        if palabra_valida:
            return None, f"Es la palabra de {duena}, pero el modelo no reconoce su voz.", True
        return None, "No reconozco ni la voz ni la palabra.", False
    propia = perfiles.distancia_palabra(persona, frase)
    if propia is None:
        return None, f"{persona} no tiene palabra secreta (usa «Registrar palabra»).", False
    if propia > UMBRAL_PALABRA:
        return None, f"No es la palabra secreta de {persona} (diferencia {propia:.2f}, máximo {UMBRAL_PALABRA}).", True
    if duena != persona:
        return None, f"La frase se parece más a la palabra de {duena} que a la de {persona}.", True
    return persona, f"Palabra secreta de {persona} correcta (diferencia {propia:.2f}).", False


def decidir_identidad(resultados, perfiles, tipo, calcular_huella, umbral, umbral_correccion=None):
    """Decide quien es la persona.

    resultados: lista de (nombre, probabilidad), uno por cada analisis (cara, foto completa...).
    umbral_correccion: el modelo de Teachable Machine confunde a veces a las personas (por ejemplo dice
    "burbano" para julian). Si el modelo ve a una persona pero dice otro nombre (o los analisis no
    coinciden), se acepta el perfil registrado mas parecido cuando su parecido llega a este valor.
    None = sin correccion: el modelo y el perfil deben decir la misma persona.
    Devuelve (dato, motivo): dato es el nombre de la persona, "otros" o "falso".
    """
    nombres = {nombre for nombre, _ in resultados}
    if "falso" in nombres:
        return "falso", "El modelo detecta una foto o una pantalla."
    clases = sorted(nombres & NO_PERSONAS)
    if clases:
        return "otros", f"Clase «{clases[0]}»."
    if min(p for _, p in resultados) < UMBRAL_CONFIANZA:
        return "otros", f"Confianza menor al {UMBRAL_CONFIANZA:.0%}."
    if len(nombres) > 1 and umbral_correccion is None:
        return "otros", "Los análisis no coinciden."
    huella = calcular_huella()
    otro, parecido = perfiles.mas_parecido(tipo, huella)
    if otro is None:
        return "otros", f"Nadie tiene perfil de {tipo} registrado (usa «Registrar {tipo}»)."
    if nombres == {otro}:  # el modelo y el perfil dicen la misma persona
        if parecido < umbral:
            return "otros", f"No coincide con el perfil de {otro} (parecido {parecido:.2f}, mínimo {umbral})."
        return otro, f"Coincide con el perfil de {otro} (parecido {parecido:.2f})."
    modelo = " / ".join(sorted(nombres))
    if umbral_correccion is not None and parecido >= umbral_correccion:
        return otro, (f"El modelo dice {modelo}, pero coincide claramente con el perfil de {otro} "
                      f"(parecido {parecido:.2f}, mínimo {umbral_correccion}).")
    if len(nombres) == 1 and perfiles.similitud(modelo, tipo, huella) is None:
        return "otros", f"{modelo} no tiene perfil de {tipo} registrado (usa «Registrar {tipo}»)."
    return "otros", f"El modelo dice {modelo}, pero se parece más al perfil de {otro} ({parecido:.2f})."


# ---------------- Fases de un intento de acceso ----------------

class Verificador:
    def __init__(self, perfiles):
        self.perfiles = perfiles
        self.fallos = 0
        self.bloqueado_hasta = 0
        self.ultimo_acceso = (None, 0)  # (perfil, hasta cuando se muestra el mensaje de acceso)
        self.modo = "imagen"
        self.reiniciar()

    def iniciar(self, modo):
        """Empieza a verificar con el modo 'imagen' (rostro), 'pose' o 'voz'."""
        self.modo = modo
        self.reiniciar()

    def reiniciar(self):
        """Empieza un intento nuevo (los fallos acumulados se mantienen)."""
        self.fase = "identidad"
        self.persona = None
        self.reto = None
        self.inicio = None        # cuando empezo el intento
        self.limite = 0           # hasta cuando hay tiempo en la fase actual
        self.pausa_hasta = 0      # tiempo mostrando el resultado del intento
        self.candidato = None     # (valor, desde) para medir la constancia
        self.movimientos = []     # movimientos de la cabeza pedidos, al azar
        self.muestras = []        # ultimas posiciones de la cara mirando de frente
        self.referencia = None    # posicion de la cara mirando de frente (horizontal, vertical)
        self._esperar()

    def _esperar(self):
        """Titulo mientras no hay nadie delante de la camara (o nadie hablando)."""
        self.titulo = "Di tu palabra secreta" if self.modo == "voz" else "Ponte frente a la cámara"
        self.color = NEGRO

    def esperar_frase(self, ahora, hablando):
        """Modo voz: mientras no termina ninguna frase."""
        if self.fase == "resultado":
            if ahora < self.pausa_hasta:
                return
            self.reiniciar()
        if hablando:
            self.titulo, self.color = "Escuchando...", AZUL
        else:
            self._esperar()

    def frase(self, ahora, persona, motivo, cuenta_como_fallo):
        """Modo voz: termino una frase. persona = perfil cuya voz y palabra coinciden, o None."""
        if self.fase == "resultado" and ahora < self.pausa_hasta:
            return
        if persona:
            self.persona = persona
            self._conceder(ahora)
        elif cuenta_como_fallo:
            self._fallar(ahora, motivo)
        else:  # ruido u otra gente hablando: se avisa sin contar un intento
            self.titulo, self.color = motivo, ROJO
            self.fase, self.pausa_hasta = "resultado", ahora + 2

    @property
    def necesita(self):
        """Que dato necesita la fase actual: 'identidad', 'gesto' u 'orientacion' (de la cara)."""
        if self.fase == "gesto":
            return "gesto"
        if self.fase in ("cabeza", "centrar", "movimiento"):
            return "orientacion"
        return "identidad"

    def desplazamiento(self, orientacion):
        """Movimiento de la cara respecto a la referencia (1 = llega a la meta)."""
        return desplazamiento_cara(*(np.array(orientacion) - self.referencia))

    def _constante(self, valor, ahora, segundos):
        """True si 'valor' se ha mantenido igual durante 'segundos'."""
        if self.candidato is None or self.candidato[0] != valor:
            self.candidato = (valor, ahora)
        return ahora - self.candidato[1] >= segundos

    def _cambiar_fase(self, fase, ahora, segundos, reto=None):
        self.fase, self.reto, self.limite, self.candidato = fase, reto, ahora + segundos, None

    def _fallar(self, ahora, motivo):
        self.fallos += 1
        self.titulo, self.color = f"Intento fallido ({self.fallos}/{MAX_INTENTOS}): {motivo}", ROJO
        self.fase, self.pausa_hasta = "resultado", ahora + 2.5
        if self.fallos >= MAX_INTENTOS:
            self.fallos = 0
            self.bloqueado_hasta = self.pausa_hasta = ahora + SEGUNDOS_BLOQUEO
            self.titulo = "Límite de intentos fallidos"

    def _conceder(self, ahora):
        self.fallos = 0
        self.titulo, self.color = f"ACCESO CONCEDIDO: {self.persona}", VERDE
        self.fase, self.pausa_hasta = "resultado", ahora + 4
        self.ultimo_acceso = (self.persona, self.pausa_hasta)

    def paso(self, ahora, dato, confianza=None):
        """Avanza con el dato del cuadro actual de la camara:
        identidad   -> nombre de la persona, "otros", "falso" o None (nadie delante);
                       confianza es el porcentaje mas bajo de los analisis
        gesto       -> nombre del gesto o None
        orientacion -> (horizontal, vertical) de orientacion_cara() o None
        """
        if self.fase == "resultado":
            if ahora < self.pausa_hasta:
                return
            self.reiniciar()

        restante = max(self.limite - ahora, 0)
        if self.fase == "identidad":
            self._paso_identidad(ahora, dato, confianza)

        elif self.fase == "gesto":
            self.titulo, self.color = f"Reto: {self.reto} ({restante:.0f} s)", AZUL
            if self._constante(dato, ahora, SEGUNDOS_GESTO) and dato == self.reto:
                self._cambiar_fase("cabeza", ahora, TIEMPO_RETO, random.choice(["izquierda", "derecha"]))
            elif restante <= 0:
                self._fallar(ahora, "no hiciste el gesto pedido.")

        elif self.fase == "cabeza":
            self.titulo, self.color = f"Reto: gira la cabeza a la {self.reto} ({restante:.0f} s)", AZUL
            lado = None if dato is None else (
                "derecha" if dato[0] > UMBRAL_GIRO else "izquierda" if dato[0] < -UMBRAL_GIRO else "centro")
            if self._constante(lado, ahora, 0.3) and lado == self.reto:
                self._conceder(ahora)
            elif restante <= 0:
                self._fallar(ahora, "no giraste la cabeza.")

        elif self.fase == "centrar":
            self.titulo, self.color = f"Rostro reconocido: {self.persona}\nMira de frente un momento...", VERDE
            de_frente = mira_de_frente(dato)
            if de_frente:
                self.muestras = (self.muestras + [dato])[-5:]
            if self._constante(de_frente, ahora, 0.5) and de_frente:
                self.referencia = np.median(self.muestras, axis=0)
                self._cambiar_fase("movimiento", ahora, TIEMPO_RETO, self.movimientos[0])
            elif restante <= 0:
                self.reiniciar()  # no miro de frente: se vuelve a empezar sin contar un fallo

        elif self.fase == "movimiento":
            numero = NUM_MOVIMIENTOS - len(self.movimientos) + 1
            self.titulo, self.color = (f"{numero}/{NUM_MOVIMIENTOS}: {MOVIMIENTOS[self.reto]} "
                                       f"({restante:.0f} s)"), AZUL
            direccion = None if dato is None else direccion_cara(*(np.array(dato) - self.referencia))
            if self._constante(direccion, ahora, 0.3) and direccion == self.reto:
                self.movimientos.pop(0)
                if self.movimientos:
                    self._cambiar_fase("movimiento", ahora, TIEMPO_RETO, self.movimientos[0])
                else:
                    self._conceder(ahora)
            elif restante <= 0:
                self._fallar(ahora, f"no moviste la cabeza hacia {self.reto}.")

    def _paso_identidad(self, ahora, dato, confianza):
        if dato is None:  # nadie delante: no cuenta como intento
            self.inicio, self.candidato = None, None
            self._esperar()
            return
        if self.inicio is None:
            self.inicio = ahora
        estable = self._constante(dato, ahora, SEGUNDOS_CONSTANCIA)

        if dato == "falso":
            self.titulo, self.color = "¡Posible foto o pantalla!", ROJO
            if estable:
                return self._fallar(ahora, "se detectó una foto o pantalla.")
        elif dato == "otros":
            self.titulo, self.color = "Desconocido (otros)", ROJO
        elif estable:
            self.persona = dato
            if self.modo == "imagen":
                self.movimientos = random.sample(list(MOVIMIENTOS), NUM_MOVIMIENTOS)
                return self._cambiar_fase("centrar", ahora, TIEMPO_RETO)
            return self._cambiar_fase("gesto", ahora, TIEMPO_RETO, random.choice(GESTOS))
        else:  # cuenta el tiempo seguido sin bajar del porcentaje minimo (si baja, vuelve a 0)
            segundos = ahora - self.candidato[1]
            porcentaje = f" {confianza:.0%} (mínimo {UMBRAL_CONFIANZA:.0%})" if confianza is not None else ""
            self.titulo, self.color = (f"{dato}{porcentaje}\n"
                                       f"Manteniendo: {segundos:.1f} / {SEGUNDOS_CONSTANCIA:g} s"), AZUL

        if ahora - self.inicio > TIEMPO_IDENTIDAD:
            self._fallar(ahora, "no se pudo confirmar la identidad.")
