#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import threading
import logging
from datetime import datetime
from collections import defaultdict

import numpy as np
import pygame

import rclpy
from rclpy.node import Node
from img_check.srv import ImgCheck, MoveServo

from gpiozero import Button, LED
#=============================================
# GPIO (Raspberry Pi) do przycisków i diod LED
#=============================================

# Przyciski 
BUTTON_WHITE_PIN = 4
BUTTON_BLACK_PIN = 26

# LED-y 
LED_WHITE_GREEN_PIN  = 17
LED_WHITE_YELLOW_PIN = 27
LED_WHITE_RED_PIN    = 23
LED_WHITE_TURN_PIN   = 24

LED_BLACK_GREEN_PIN  = 16
LED_BLACK_YELLOW_PIN = 12
LED_BLACK_TURN_PIN   = 13
LED_BLACK_RED_PIN    = 25


#=============================================
# Wymiary planszy i oznaczenia figur
#=============================================
BOARD_N = 10

#legenda:
# player == 1 => czarne
# player == 2 => białe

EMPTY = 0
BLACK_MAN = 1
WHITE_MAN = 2
BLACK_KING = 11
WHITE_KING = 22

#plansza startowa
def board_start() -> np.ndarray:

    b = np.zeros((10, 10), dtype=int)

    # czarne (1) – rzędy 0 - 3
    for i in range(4):
        if i % 2 == 1:
            for j in range(0, 9, 2):
                b[i, j] = BLACK_MAN
        else:
            for j in range(1, 10, 2):
                b[i, j] = BLACK_MAN

    # białe (2) – rzędy 6 - 9
    for i in range(6, 10):
        if i % 2 == 1:
            for j in range(0, 9, 2):
                b[i, j] = WHITE_MAN
        else:
            for j in range(1, 10, 2):
                b[i, j] = WHITE_MAN

    return b


#tablica zajętości pól - mapowanie na 1 (czarne), 2 (białe) lub 0 (puste)
def occupancy_board(b: np.ndarray) -> np.ndarray:
    occ = np.zeros_like(b)
    occ[(b == BLACK_MAN) | (b == BLACK_KING)] = 1
    occ[(b == WHITE_MAN) | (b == WHITE_KING)] = 2
    return occ

#Promocja po zakończeniu ruchu - czarny pionek musi być na rzędzie 9, a biały na rzędzie 0
def check_promotions(b: np.ndarray) -> np.ndarray:
    nb = b.copy()
    for c in range(10):
        if nb[9, c] == BLACK_MAN:
            nb[9, c] = BLACK_KING
        if nb[0, c] == WHITE_MAN:
            nb[0, c] = WHITE_KING
    return nb

#zliczanie figur obu graczy z podziałem na zwykłe pionki i damki
def count_pieces(b: np.ndarray):
    p1 = int(np.sum(b == BLACK_MAN))
    p1k = int(np.sum(b == BLACK_KING))
    p2 = int(np.sum(b == WHITE_MAN))
    p2k = int(np.sum(b == WHITE_KING))
    return p1, p1k, p2, p2k

#słownik na bazie wartości zwracanych przez count_pieces(b)
def material_summary(b: np.ndarray):
    p1, p1k, p2, p2k = count_pieces(b)
    return {'p1_pawns': p1, 'p1_kings': p1k, 'p2_pawns': p2, 'p2_kings': p2k}


def position_key(b: np.ndarray, player_to_move: int):
    return (b.tobytes(), player_to_move)
#==============================================
#Obsługa końcówek remisowych
#==============================================

#końcówka z 16 ruchami bez bicia 
def is_endgame_16(summary):
    A = {'p': summary['p1_pawns'], 'k': summary['p1_kings']}
    B = {'p': summary['p2_pawns'], 'k': summary['p2_kings']}

    def c3k1k(X, Y): return X['k'] == 3 and X['p'] == 0 and Y['k'] == 1 and Y['p'] == 0
    def c2k1p1k(X, Y): return X['k'] == 2 and X['p'] == 1 and Y['k'] == 1 and Y['p'] == 0
    def c1k2p1k(X, Y): return X['k'] == 1 and X['p'] == 2 and Y['k'] == 1 and Y['p'] == 0

    return (
        c3k1k(A, B) or c3k1k(B, A) or
        c2k1p1k(A, B) or c2k1p1k(B, A) or
        c1k2p1k(A, B) or c1k2p1k(B, A)
    )

#końcówka z 5 ruchami bez bicia 
def is_endgame_5(summary):
    A = {'p': summary['p1_pawns'], 'k': summary['p1_kings']}
    B = {'p': summary['p2_pawns'], 'k': summary['p2_kings']}

    def c2k1k(X, Y): return X['k'] == 2 and X['p'] == 0 and Y['k'] == 1 and Y['p'] == 0
    def c1k1p1k(X, Y): return X['k'] == 1 and X['p'] == 1 and Y['k'] == 1 and Y['p'] == 0
    def c1k1k(X, Y): return X['k'] == 1 and X['p'] == 0 and Y['k'] == 1 and Y['p'] == 0

    return c2k1k(A, B) or c2k1k(B, A) or c1k1p1k(A, B) or c1k1p1k(B, A) or c1k1k(A, B)


# legalne ścieżki ruchu
DIRS_DIAG = [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def inside(r, c) -> bool:
    return 0 <= r < BOARD_N and 0 <= c < BOARD_N


def is_opponent(piece_val: int, player: int) -> bool:
    if player == 1:  # ruch czarnych, przeciwnik białe
        return piece_val in (WHITE_MAN, WHITE_KING)
    else:
        return piece_val in (BLACK_MAN, BLACK_KING)


def is_own(piece_val: int, player: int) -> bool:
    if player == 1:
        return piece_val in (BLACK_MAN, BLACK_KING)
    else:
        return piece_val in (WHITE_MAN, WHITE_KING)

#Maksymalna liczba bić dla bierki na polu (r, c) - bierka nie promuje w czasie ruchu
#zgodnie z zasadami warcabów polskich
def get_max_captures_from(b: np.ndarray, r: int, c: int, player: int) -> int:

    piece = b[r, c]
    if not is_own(piece, player):
        return 0

    opponent_vals = (WHITE_MAN, WHITE_KING) if player == 1 else (BLACK_MAN, BLACK_KING)

    def dfs_man(bb: np.ndarray, rr: int, cc: int, count: int) -> int:
        best = count
        for dr, dc in DIRS_DIAG:
            mr, mc = rr + dr, cc + dc
            lr, lc = rr + 2 * dr, cc + 2 * dc
            if inside(lr, lc) and inside(mr, mc):
                if bb[mr, mc] in opponent_vals and bb[lr, lc] == EMPTY:
                    nb = bb.copy()
                    nb[rr, cc] = EMPTY
                    nb[mr, mc] = EMPTY
                    nb[lr, lc] = piece  # dalej zwykły pionek
                    best = max(best, dfs_man(nb, lr, lc, count + 1))
        return best

    def dfs_king(bb: np.ndarray, rr: int, cc: int, count: int) -> int:
        best = count
        for dr, dc in DIRS_DIAG:
            i, j = rr + dr, cc + dc
            # przeskakujemy puste
            while inside(i, j) and bb[i, j] == EMPTY:
                i += dr
                j += dc
            # pierwsza napotkana bierka na przekątnej
            if inside(i, j) and bb[i, j] in opponent_vals:
                lr, lc = i + dr, j + dc
                # wszystkie możliwe lądowania za przeciwnikiem
                while inside(lr, lc) and bb[lr, lc] == EMPTY:
                    nb = bb.copy()
                    nb[rr, cc] = EMPTY
                    nb[i, j] = EMPTY
                    nb[lr, lc] = piece  # damka
                    best = max(best, dfs_king(nb, lr, lc, count + 1))
                    lr += dr
                    lc += dc
        return best

    if piece in (BLACK_MAN, WHITE_MAN):
        return dfs_man(b, r, c, 0)
    else:
        return dfs_king(b, r, c, 0)

#Zwracana jest liczba ścieżek bicia dla bierki na (r, c), ale tylko
# te ścieżki o maksymalnej długości
def get_capture_paths_from(b: np.ndarray, r: int, c: int, player: int):

    piece = b[r, c]
    if not is_own(piece, player):
        return []

    opponent_vals = (WHITE_MAN, WHITE_KING) if player == 1 else (BLACK_MAN, BLACK_KING)
    paths = []

    def dfs_man(bb: np.ndarray, rr: int, cc: int, path):
        found = False
        for dr, dc in DIRS_DIAG:
            mr, mc = rr + dr, cc + dc
            lr, lc = rr + 2 * dr, cc + 2 * dc
            if inside(lr, lc) and inside(mr, mc):
                if bb[mr, mc] in opponent_vals and bb[lr, lc] == EMPTY:
                    nb = bb.copy()
                    nb[rr, cc] = EMPTY
                    nb[mr, mc] = EMPTY
                    nb[lr, lc] = piece  # dalej zwykły pionek (bez promocji “w locie”)
                    dfs_man(nb, lr, lc, path + [(lr, lc)])
                    found = True
        if not found:
            paths.append(path)

    def dfs_king(bb: np.ndarray, rr: int, cc: int, path):
        found = False
        for dr, dc in DIRS_DIAG:
            i, j = rr + dr, cc + dc
            while inside(i, j) and bb[i, j] == EMPTY:
                i += dr
                j += dc
            if inside(i, j) and bb[i, j] in opponent_vals:
                lr, lc = i + dr, j + dc
                while inside(lr, lc) and bb[lr, lc] == EMPTY:
                    nb = bb.copy()
                    nb[rr, cc] = EMPTY
                    nb[i, j] = EMPTY
                    nb[lr, lc] = piece  # king
                    dfs_king(nb, lr, lc, path + [(lr, lc)])
                    found = True
                    lr += dr
                    lc += dc
        if not found:
            paths.append(path)

    if piece in (BLACK_MAN, WHITE_MAN):
        dfs_man(b, r, c, [(r, c)])
    else:
        dfs_king(b, r, c, [(r, c)])

    if not paths:
        return []
    mx = max(len(p) - 1 for p in paths)
    return [p for p in paths if (len(p) - 1) == mx and mx > 0]


#Globalnie dla gracza: zwraca (global_max, best_paths)
#gdzie best_paths to WSZYSTKIE ścieżki z maks liczbą bić.
def get_all_max_capture_paths(b: np.ndarray, player: int):

    global_max = 0
    best_paths = []

    for r in range(10):
        for c in range(10):
            if not is_own(b[r, c], player):
                continue
            paths = get_capture_paths_from(b, r, c, player)
            if not paths:
                continue
            local_max = max(len(p) - 1 for p in paths)
            if local_max > global_max:
                global_max = local_max
                best_paths = []
            if local_max == global_max and global_max > 0:
                for p in paths:
                    if len(p) - 1 == global_max:
                        best_paths.append(p)

    return global_max, best_paths

  
#Funkcja do obsługi legalnych posunięć
#- jeśli istnieje bicie: zwraca tylko ścieżki o największej liczbie bić
#- jeśli nie ma bicia: zwraca wszystkie legalne zwykłe ruchy
def get_all_legal_move_paths(b: np.ndarray, player: int):

    global_max, best_paths = get_all_max_capture_paths(b, player)
    if global_max > 0:
        return best_paths, global_max

    paths = []

    # zwykłe ruchy pionów
    forward = 1 if player == 1 else -1
    man_val = BLACK_MAN if player == 1 else WHITE_MAN
    king_val = BLACK_KING if player == 1 else WHITE_KING

    for r in range(10):
        for c in range(10):
            v = b[r, c]
            if v == man_val:
                for dc in (-1, 1):
                    nr, nc = r + forward, c + dc
                    if inside(nr, nc) and b[nr, nc] == EMPTY:
                        paths.append([(r, c), (nr, nc)])

            elif v == king_val:
                for dr, dc in DIRS_DIAG:
                    nr, nc = r + dr, c + dc
                    while inside(nr, nc) and b[nr, nc] == EMPTY:
                        paths.append([(r, c), (nr, nc)])
                        nr += dr
                        nc += dc

    return paths, 0


# plansza po ruchu dla danej ścieżki
def segment_is_capture(bb: np.ndarray, a, d, player: int) -> bool:
    (r1, c1), (r2, c2) = a, d
    piece = bb[r1, c1]
    dr = r2 - r1
    dc = c2 - c1

    if piece in (BLACK_MAN, WHITE_MAN):
        return abs(dr) == 2 and abs(dc) == 2

    # king
    if abs(dr) != abs(dc):
        return False
    step_r = 1 if dr > 0 else -1
    step_c = 1 if dc > 0 else -1
    enemies = 0
    for k in range(1, abs(dr)):
        rr = r1 + step_r * k
        cc = c1 + step_c * k
        if bb[rr, cc] != EMPTY:
            if is_opponent(bb[rr, cc], player):
                enemies += 1
            else:
                return False  # własna bierka blokuje
    return enemies == 1


# Wykonuje pełny ruch zapisany jako ścieżka, aktualizuje planszę,
# usuwa zbite bierki i zwraca nowy stan oraz informację o biciu.
def apply_move_path(b: np.ndarray, path, player: int):
    nb = b.copy()
    start_r, start_c = path[0]
    piece = nb[start_r, start_c]
    was_capture = False

    nb[start_r, start_c] = EMPTY
    cur_r, cur_c = start_r, start_c

    for idx in range(1, len(path)):
        nxt_r, nxt_c = path[idx]

        if segment_is_capture(nb, (cur_r, cur_c), (nxt_r, nxt_c), player):
            was_capture = True
            # usuń zbitą bierkę
            dr = 1 if nxt_r > cur_r else -1
            dc = 1 if nxt_c > cur_c else -1
            rr, cc = cur_r + dr, cur_c + dc
            while (rr, cc) != (nxt_r, nxt_c):
                if nb[rr, cc] != EMPTY:
                    nb[rr, cc] = EMPTY
                    break
                rr += dr
                cc += dc

        cur_r, cur_c = nxt_r, nxt_c

    nb[cur_r, cur_c] = piece
    nb = check_promotions(nb)
    return nb, was_capture


# Notacja 1–50 
def coords_to_sq(r, c) -> int:
    # tylko ciemne pola (r+c)%2==1
    if (r + c) % 2 == 0:
        return -1
    order = (c // 2) if (r % 2 == 1) else ((c - 1) // 2)
    return r * 5 + order + 1

# Zamienia ścieżkę ruchu na zapis notacyjny warcabów, rozróżniając ruch zwykły i bicie.
#zwraca zapis typu np. 23-28 albo 14x23x32
def path_to_notation(b_before: np.ndarray, path, player: int) -> str:
    if len(path) < 2:
        return ""
    cap = segment_is_capture(b_before, path[0], path[1], player)
    sep = "x" if cap else "-"
    nums = [coords_to_sq(r, c) for (r, c) in path]
    return sep.join(str(n) for n in nums)


# ==============================
# Parametry dla Pygame
# ==============================
LIGHT_BROWN = (240, 217, 181)
DARK_BROWN  = (181, 136, 99)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 200, 0)
RED   = (200, 0, 0)

TILE_SIZE = 60
BOARD_SIZE = TILE_SIZE * 10

#Renderowanie planszy w oknie Pygame
def draw_board(screen, b: np.ndarray):
    for row in range(10):
        for col in range(10):
            tile_color = LIGHT_BROWN if (row + col) % 2 == 0 else DARK_BROWN
            tile = pygame.Rect(col * TILE_SIZE, row * TILE_SIZE, TILE_SIZE, TILE_SIZE)
            pygame.draw.rect(screen, tile_color, tile)

            value = int(b[row, col])
            center_x = col * TILE_SIZE + TILE_SIZE // 2
            center_y = row * TILE_SIZE + TILE_SIZE // 2
            radius = TILE_SIZE // 2 - 6
            piece_rect = pygame.Rect(col * TILE_SIZE + 10, row * TILE_SIZE + 10, TILE_SIZE - 20, TILE_SIZE - 20)

            if value == BLACK_MAN:
                pygame.draw.circle(screen, BLACK, (center_x, center_y), radius)
            elif value == WHITE_MAN:
                pygame.draw.circle(screen, WHITE, (center_x, center_y), radius)
            elif value == BLACK_KING:
                pygame.draw.rect(screen, BLACK, piece_rect)
            elif value == WHITE_KING:
                pygame.draw.rect(screen, WHITE, piece_rect)

# Wyświetlanie wyniku gry oraz przycisków umożliwiających wyjście lub reset partii
def show_game_result(screen, result_text):
    font = pygame.font.SysFont(None, 72)
    small_font = pygame.font.SysFont(None, 40)

    text_surface = font.render(result_text, True, BLACK)
    text_rect = text_surface.get_rect(center=(BOARD_SIZE // 2, BOARD_SIZE // 2 - 50))
    screen.blit(text_surface, text_rect)

    exit_rect = pygame.Rect(BOARD_SIZE // 2 - 100, BOARD_SIZE // 2 + 20, 200, 50)
    pygame.draw.rect(screen, RED, exit_rect)
    exit_text = small_font.render("Wyjdź z gry", True, WHITE)
    screen.blit(exit_text, exit_text.get_rect(center=exit_rect.center))

    reset_rect = pygame.Rect(BOARD_SIZE // 2 - 100, BOARD_SIZE // 2 + 80, 200, 50)
    pygame.draw.rect(screen, GREEN, reset_rect)
    reset_text = small_font.render("Resetuj grę", True, WHITE)
    screen.blit(reset_text, reset_text.get_rect(center=reset_rect.center))

    pygame.display.flip()
    return exit_rect, reset_rect

# Krótka pauza bez blokowania obsługi zdarzeń Pygame.
# seconds - czas oczekiwania w sekundach
# clock - zegar Pygame do ograniczania szybkości pętli
def pump_sleep(seconds: float, clock: pygame.time.Clock):
    end = time.time() + seconds
    while time.time() < end:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise SystemExit
        clock.tick(30)


# ==============================
# ROS2 + logika gry
# ==============================
class WarcabyMaster(Node):
    def __init__(self):
        super().__init__('warcaby_master')

        self.cli_matrix = self.create_client(ImgCheck, 'img_check_service')
        self.cli_servo  = self.create_client(MoveServo, 'service_move_servo')

        # GPIO init
        self.button_white = Button(BUTTON_WHITE_PIN, bounce_time=0.2)
        self.button_black = Button(BUTTON_BLACK_PIN, bounce_time=0.2)

        self.led_white_green  = LED(LED_WHITE_GREEN_PIN)
        self.led_white_yellow = LED(LED_WHITE_YELLOW_PIN)
        self.led_white_red    = LED(LED_WHITE_RED_PIN)
        self.led_white_turn   = LED(LED_WHITE_TURN_PIN)

        self.led_black_green  = LED(LED_BLACK_GREEN_PIN)
        self.led_black_yellow = LED(LED_BLACK_YELLOW_PIN)
        self.led_black_turn   = LED(LED_BLACK_TURN_PIN)
        self.led_black_red    = LED(LED_BLACK_RED_PIN)

        self._all_leds_off()

        # eventy przycisków
        self.ev_white = threading.Event()
        self.ev_black = threading.Event()
        self.button_white.when_pressed = lambda: self.ev_white.set()
        self.button_black.when_pressed = lambda: self.ev_black.set()

        # log file
        log_dir = os.path.expanduser("~/warcaby_logs")
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(log_dir, f"warcaby_game_{ts}.log")

        self.logger = logging.getLogger("warcaby_master")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()

        sh = logging.StreamHandler(sys.stdout)
        fh = logging.FileHandler(self.log_path, encoding="utf-8")

        fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
        sh.setFormatter(fmt)
        fh.setFormatter(fmt)

        self.logger.addHandler(sh)
        self.logger.addHandler(fh)

        self.logger.info(f"Log zapisany do: {self.log_path}")

    #Wyłączenie wszystkich diod LED
    def _all_leds_off(self):
        for led in (
            self.led_white_green, self.led_white_yellow, self.led_white_red, self.led_white_turn,
            self.led_black_green, self.led_black_yellow, self.led_black_red, self.led_black_turn
        ):
            try:
                led.off()
            except Exception:
                pass

    #Włączenie diody odpowiedniego gracza
    def _set_turn_leds(self, player: int):
        # player 2 = białe 
        #player 1 = czarne
        self.led_white_turn.off()
        self.led_black_turn.off()
        if player == 2:
            self.led_white_turn.on()
        else:
            self.led_black_turn.on()
    
    #Włącza lub wyłącza żółtą diodę sygnalizującą niepoprawny ruch gracza
    def _set_illegal_led(self, player: int, on: bool):
        if player == 2:
            (self.led_white_yellow.on() if on else self.led_white_yellow.off())
        else:
            (self.led_black_yellow.on() if on else self.led_black_yellow.off())

    #Ustawienie diod sygnalizujących koniec partii zgodnie z jej rozstrzygnięciem
    def _set_gameover_leds(self, result_text: str):
        # gasi tury
        self.led_white_turn.off()
        self.led_black_turn.off()
        self.led_white_yellow.off()
        self.led_black_yellow.off()

        if result_text == "REMIS":
            self.led_white_red.on()
            self.led_black_red.on()
        elif result_text.lower().startswith("białe"):
            self.led_white_green.on()
            self.led_black_red.on()
        elif result_text.lower().startswith("czarne"):
            self.led_black_green.on()
            self.led_white_red.on()

    #cCzekanie aż oba serwisy ROS2 będą dostępne
    def wait_for_services(self):
        self.logger.info("Czekam na serwisy ROS2: img_check_service i service_move_servo ...")
        while rclpy.ok():
            ok1 = self.cli_matrix.wait_for_service(timeout_sec=0.5)
            ok2 = self.cli_servo.wait_for_service(timeout_sec=0.5)
            if ok1 and ok2:
                self.logger.info("Serwisy dostępne.")
                return True
        return False

    #wysłanie żądania do serwisu odpowiedzialnego za stan planszy
    def request_matrix(self):
        req = ImgCheck.Request()
        future = self.cli_matrix.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        return future.result()

    #wysłanie żądania do serwisu odpowiedzialnego ruch serwomechanizmu
    def move_servo(self, angle: int):
        req = MoveServo.Request()
        req.angle = int(angle)
        future = self.cli_servo.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        return future.result()

    #Pobranie z serwisu aktualnego stanu planszy i rzutowanie go na macierz 10x10
    def capture_img_board(self) -> np.ndarray | None:
        resp = self.request_matrix()
        if resp is None:
            self.logger.error("Brak odpowiedzi z img_check_service.")
            return None
        mat = getattr(resp, "matrix", None)
        if mat is None or len(mat) != 100:
            self.logger.error(f"Zły format macierzy: {type(mat)} len={0 if mat is None else len(mat)}")
            return None
        arr = np.array(mat, dtype=int).reshape((10, 10))
        return arr

    #Ustawia serwo, zapisuje komunikaty z serwisu i odczekuje wskazany czas.
    def set_servo_and_wait(self, angle: int, seconds: float, clock: pygame.time.Clock):
        resp = self.move_servo(angle)
        if resp is not None:
            out = getattr(resp, "output", "")
            err = getattr(resp, "error", "")
            if out:
                self.logger.info(f"MoveServo stdout: {out.strip()}")
            if err:
                self.logger.warning(f"MoveServo stderr: {err.strip()}")
        pump_sleep(seconds, clock)

    def run(self):
        pygame.init()
        screen = pygame.display.set_mode((BOARD_SIZE, BOARD_SIZE))
        pygame.display.set_caption("Warcaby – master node")
        clock = pygame.time.Clock()
        font = pygame.font.SysFont(None, 26)

        # Stan gry
        board = board_start()
        moves_history = []  # zapis partii

        # Remisy / liczniki 
        turn_number = 1
        player = 2  # białe zaczynają
        game_over = False
        result_text = ""

        position_counts = defaultdict(int)
        position_counts[position_key(board, player)] += 1

        kings_only_25_counter = 0
        endgame16_active = False
        endgame16_moves = {1: 0, 2: 0}
        endgame5_active = False
        endgame5_moves = {1: 0, 2: 0}

        # 1) Start: servo 60°, zdjęcie, weryfikacja startu, potem 145° i ruch białych
        self._all_leds_off()
        self._set_turn_leds(0)  # na start nic
        self.logger.info("START: ustawiam servo na 60°, czekam 2s, robię zdjęcie startowe.")
        self.set_servo_and_wait(60, 2.0, clock)

        img_board = self.capture_img_board()
        if img_board is None:
            self.logger.warning("Nie udało się pobrać img_board na starcie.")
        else:
            expected = occupancy_board(board)
            if np.array_equal(img_board, expected):
                self.logger.info("Startowa pozycja OK (img_board == expected).")
            else:
                diff = int(np.sum(img_board != expected))
                self.logger.warning(f"Startowa pozycja NIEZGODNA! Różniących pól: {diff}")
                # sygnalizacja: zapal żółte obu (żeby było widać, że jest problem)
                self.led_white_yellow.on()
                self.led_black_yellow.on()

        self.logger.info("Ustawiam servo na 145° (odsłaniam planszę).")
        self.set_servo_and_wait(145, 0.3, clock)

        player = 2  # białe
        self._set_turn_leds(player)
        self.logger.info("Ruch białych – czekam na przycisk białych (GPIO4).")

        illegal_retry = {1: False, 2: False}

        while True:
            # eventy okna
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise SystemExit

            screen.fill((0, 0, 0))
            draw_board(screen, board)

            # tura + status
            status = f"Tura: {'BIAŁE' if player == 2 else 'CZARNE'} | Turn#: {turn_number}"
            if illegal_retry[player]:
                status += " | POPRZEDNI RUCH ODRZUCONY (żółta)"
            text = font.render(status, True, (20, 20, 20))
            # jasny pasek pod tekst
            pygame.draw.rect(screen, (230, 230, 230), pygame.Rect(0, 0, BOARD_SIZE, 28))
            screen.blit(text, (8, 6))

            pygame.display.flip()
            clock.tick(30)

            rclpy.spin_once(self, timeout_sec=0.0)

            if game_over:
                exit_rect, reset_rect = show_game_result(screen, result_text)
                # obsługa kliknięcia reset/exit
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise SystemExit
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        if exit_rect.collidepoint(event.pos):
                            raise SystemExit
                        if reset_rect.collidepoint(event.pos):
                            self.logger.info("RESET gry (pygame).")
                            # reset stanu + LED
                            self._all_leds_off()
                            self.led_white_yellow.off()
                            self.led_black_yellow.off()

                            board = board_start()
                            moves_history.clear()

                            turn_number = 1
                            player = 2
                            game_over = False
                            result_text = ""

                            position_counts = defaultdict(int)
                            position_counts[position_key(board, player)] += 1
                            kings_only_25_counter = 0
                            endgame16_active = False
                            endgame16_moves = {1: 0, 2: 0}
                            endgame5_active = False
                            endgame5_moves = {1: 0, 2: 0}
                            illegal_retry = {1: False, 2: False}

                            # Procedura startowa jak na początku:
                            self.logger.info("RESET: servo 60°, 2s, zdjęcie, potem 145° i białe na ruchu.")
                            self.set_servo_and_wait(60, 2.0, clock)
                            img_board = self.capture_img_board()
                            if img_board is not None:
                                expected = occupancy_board(board)
                                if np.array_equal(img_board, expected):
                                    self.logger.info("RESET: startowa pozycja OK.")
                                else:
                                    diff = int(np.sum(img_board != expected))
                                    self.logger.warning(f"RESET: pozycja niezgodna, diff={diff}")
                                    self.led_white_yellow.on()
                                    self.led_black_yellow.on()
                            self.set_servo_and_wait(145, 0.3, clock)

                            self._set_turn_leds(player)
                            self.logger.info("Ruch białych – czekam na przycisk białych (GPIO4).")

                continue

            # czekanie na przycisk aktualnego gracza
            pressed = False
            if player == 2 and self.ev_white.is_set():
                self.ev_white.clear()
                pressed = True
            elif player == 1 and self.ev_black.is_set():
                self.ev_black.clear()
                pressed = True

            if not pressed:
                continue

            # 2/3) Gracz sygnalizuje ruch -> servo 60, 2s, zdjęcie, walidacja legalności
            self.logger.info(f"Przycisk {'BIAŁE' if player==2 else 'CZARNE'}: weryfikuję ruch.")
            self.set_servo_and_wait(60, 2.0, clock)

            img_board = self.capture_img_board()
            # zawsze odsłonia planszę po zdjęciu (nawet jeśli ruch odrzucony)
            self.set_servo_and_wait(145, 0.1, clock)

            if img_board is None:
                self.logger.error("Nie mam img_board – nie mogę zweryfikować ruchu. Odrzucam i proszę o ponowną próbę.")
                illegal_retry[player] = True
                self._set_illegal_led(player, True)
                continue

            legal_paths, global_max = get_all_legal_move_paths(board, player)

            self.logger.info(
                f"Legalne ścieżki: {len(legal_paths)} | global_max_bić={global_max} | "
                f"gracz={'BIAŁE' if player==2 else 'CZARNE'}"
            )
            # log listy ścieżek 
            for p in legal_paths[:80]:  # limit, żeby nie było za dużo logów przy damkach
                note = path_to_notation(board, p, player)
                self.logger.info(f"  PATH: {p} | note={note} | bić={len(p)-1 if global_max>0 else 0}")

            candidates = []
            for pth in legal_paths:
                nb, was_cap = apply_move_path(board, pth, player)
                candidates.append((pth, nb, was_cap))

            matches = []
            for pth, nb, was_cap in candidates:
                if np.array_equal(occupancy_board(nb), img_board):
                    matches.append((pth, nb, was_cap))

            if len(matches) == 0:
                # ODRZUCENIE
                illegal_retry[player] = True
                self._set_illegal_led(player, True)

                # powód odrzucenia: pokaż najbliższą kandydaturę
                best = None
                best_diff = 10**9
                for pth, nb, _ in candidates:
                    diff = int(np.sum(occupancy_board(nb) != img_board))
                    if diff < best_diff:
                        best_diff = diff
                        best = (pth, nb)
                if best is not None:
                    self.logger.warning(
                        f"RUCH ODRZUCONY: img_board nie pasuje do żadnej legalnej planszy. "
                        f"Najbliższa kandydatura diff={best_diff}, path={best[0]}"
                    )
                else:
                    self.logger.warning("RUCH ODRZUCONY: brak kandydatów (to raczej nie powinno się zdarzyć).")

                continue

            # ruch zaakceptowany
            if len(matches) > 1:
                self.logger.warning(f"UWAGA: znaleziono {len(matches)} pasujących kandydatów (ambiwalencja). Biorę pierwszy.")
            chosen_path, new_board, was_capture = matches[0]

            # gaś żółtą jeśli była
            illegal_retry[player] = False
            self._set_illegal_led(player, False)

            moving_piece = int(board[chosen_path[0][0], chosen_path[0][1]])
            notation = path_to_notation(board, chosen_path, player)

            self.logger.info(
                f"RUCH OK: {'BIAŁE' if player==2 else 'CZARNE'} | piece={moving_piece} | "
                f"path={chosen_path} | notation={notation} | capture={was_capture}"
            )
            moves_history.append({
                "turn": turn_number,
                "player": player,
                "path": chosen_path,
                "notation": notation,
                "was_capture": was_capture
            })

            board = new_board

            # ==============================
            # KONIEC TURY: win/draw 
            # ==============================
            # 25 ruchów: tylko damkami, bez bicia, bez ruchu pionem
            if (moving_piece in (BLACK_KING, WHITE_KING)) and (not was_capture):
                kings_only_25_counter += 1
            else:
                kings_only_25_counter = 0

            summary = material_summary(board)

            if is_endgame_16(summary):
                if not endgame16_active:
                    endgame16_active = True
                    endgame16_moves = {1: 0, 2: 0}
                endgame16_moves[player] += 1
            else:
                endgame16_active = False
                endgame16_moves = {1: 0, 2: 0}

            if is_endgame_5(summary):
                if not endgame5_active:
                    endgame5_active = True
                    endgame5_moves = {1: 0, 2: 0}
                endgame5_moves[player] += 1
            else:
                endgame5_active = False
                endgame5_moves = {1: 0, 2: 0}

            next_player = 1 if player == 2 else 2
            key = position_key(board, next_player)
            position_counts[key] += 1

            # wygrana: brak figur
            p1, p1k, p2, p2k = count_pieces(board)
            if (p1 + p1k) == 0:
                result_text = "Białe wygrały"
                game_over = True
            elif (p2 + p2k) == 0:
                result_text = "Czarne wygrały"
                game_over = True
            else:
                # brak legalnych ruchów przeciwnika
                opp_paths, opp_max = get_all_legal_move_paths(board, next_player)
                if len(opp_paths) == 0:
                    result_text = "Czarne wygrały" if player == 1 else "Białe wygrały"
                    game_over = True
                else:
                    # remisy
                    if position_counts[key] >= 3:
                        result_text = "REMIS"
                        game_over = True
                    elif kings_only_25_counter >= 25:
                        result_text = "REMIS"
                        game_over = True
                    elif endgame16_active and (endgame16_moves[1] >= 16) and (endgame16_moves[2] >= 16):
                        result_text = "REMIS"
                        game_over = True
                    elif endgame5_active and (endgame5_moves[1] >= 5) and (endgame5_moves[2] >= 5):
                        result_text = "REMIS"
                        game_over = True

            #aktualizacja LEDów i tury
            if game_over:
                self.logger.info(f"KONIEC GRY: {result_text}")
                self._set_gameover_leds(result_text)
          
                self.logger.info("Zapis partii (ścieżki):")
                for mrec in moves_history:
                    self.logger.info(
                        f"  #{mrec['turn']} {'BIAŁE' if mrec['player']==2 else 'CZARNE'}: "
                        f"{mrec['notation']} | {mrec['path']}"
                    )
                continue

            #następna tura
            turn_number += 1
            player = next_player
            self._set_turn_leds(player)

            self.logger.info(f"Następna tura: {'BIAŁE' if player==2 else 'CZARNE'} (turn_number={turn_number})")


def main(args=None):
    rclpy.init(args=args)
    node = WarcabyMaster()
    try:
        if not node.wait_for_services():
            return
        node.run()
    except SystemExit:
        pass
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node._all_leds_off()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()
        try:
            pygame.quit()
        except Exception:
            pass


if __name__ == "__main__":
    main()
