from flask import Flask, render_template
from flask_socketio import SocketIO
import time
import random
from brain import Agent
from brain import BallAgent
import math

# OBBIETTIVO AGGIORNATO: la palla si mmuove in maniera autonoma cercando di evitare il paddle controllato dalla DQN

# --- PARAMETRI DI VISUALIZZAZIONE ---
VISUALIZATION_MODE = True
TARGET_FPS = 480 # FPS target quando la visualizzazione è attiva

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# --- INIZIALIZZAZIONE NN ---
agent = Agent(5, 3)
ball_agent = BallAgent(6, 7) # la palla ha 2 perché le uniche informazioni di cui ha bisogno sono le posizioni dei due paddle
# come per il paddle le azioni che può decidere di fare sono 3 (0 = su, 1 = fermo, 2 = giù)

# Parametri Apprendimento
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.999999

# --- PARAMETRI DI GIOCO ---
WIDTH = 600
HEIGHT = 400
BALL_SIZE = 10
PADDLE_WIDTH = 10
PADDLE_HEIGHT = 60
PADDLE_SPEED = 8      # MODIFICA 1: Aumentata velocità paddle (era 3, ora 8)
MAX_BALL_VY = 8       # Limite velocità verticale palla per evitare bug
BALL_ACCELERATION = 0.5 # Quanto forte la palla può sterzare

# --- STATO DEL GIOCO ---
game_state = {
    # Posizioni Iniziali
    'ballX': WIDTH // 2, 
    'ballY': HEIGHT // 2,
    'ballVX': 4,
    'ballVY': 2,
    'paddle1Y': (HEIGHT - PADDLE_HEIGHT) // 2,
    'paddle2Y': (HEIGHT - PADDLE_HEIGHT) // 2,
    'score1': 0,
    'score2': 0
}

# Statistiche
episode_count = 0
frame_count = 0

# mi serve per ricevere informazioni riguardo allo stato attuale del gioco -> informazioni che mi serviranno per PADDLE AI 
def get_state_array_paddle():
    return [
        game_state['ballX'] / WIDTH,
        game_state['ballY'] / HEIGHT,
        game_state['ballVX'] / 10,
        game_state['ballVY'] / 10,
        game_state['paddle1Y'] / HEIGHT
    ]

# mi serve per ricevere informazioni riguardo allo stato attuale del gioco -> informazioni che mi serviranno per BALL AI 
def get_state_array_ball():
    return [
        game_state['paddle1Y'] / HEIGHT,
        game_state['paddle2Y'] / HEIGHT,
        game_state['ballY'] / HEIGHT,    # Dove sono io (FONDAMENTALE)
        game_state['ballVY'] / 10,      # Come mi sto muovendo
        game_state['ballX'] / WIDTH,     # Ora sa quanto manca all'impatto
        game_state['ballVX'] / 10
    ]

# funzione che resetta i valori di default del gioco
def reset_game():
    global episode_count
    episode_count += 1
    
    game_state['ballX'] = WIDTH // 2
    game_state['ballY'] = HEIGHT // 2
    game_state['ballVX'] = 4 if random.choice([True, False]) else -4
    game_state['ballVY'] = random.randint(-3, 3)
    if game_state['ballVY'] == 0: 
        game_state['ballVY'] = 1

# funzione che mi permette di definire l'angolo di rimbalzo in base a dove la palla colpisce il paddle
def calculate_bounce_angle(paddle_y, ball_y):
    paddle_center = paddle_y + PADDLE_HEIGHT / 2
    hit_offset = ball_y - paddle_center
    
    # Normalizza -1 a +1
    hit_factor = hit_offset / (PADDLE_HEIGHT / 2)
    
    # Angolo da -60° a +60°
    bounce_angle = hit_factor * 60  # gradi
    
    return bounce_angle

# funzione definisce la fisica del gioco
def physics(action, turn):
    # la fisica in questo caso non si aggiorna in maniera lineare ma in base alle scelte dell'AI
    # l spostamento della palla funziona in questa maniera:
    # 1. asse X: la palla si muove in maniera costante verso sinistra o destra in base alla sua velocità orizzontale
    # 2. asse Y: la palla può variare la propria velocità all'interno di un range cercando di evitare i paddle 

    # 1. Aggiorna Posizione X Palla
    game_state['ballX'] += game_state['ballVX']

    actual_state = get_state_array_ball()
    if turn % 2 != 0:
        # 2. Aggiorna Posizione Y Palla
        action_ball = ball_agent.get_action(actual_state, epsilon)
    else :
        action_ball = ball_agent.predict_action(actual_state)

    # Mappatura azioni palla (0-6) su accelerazione verticale
    # 0=Forte Su, 1=Medio Su, 2=Piano Su, 3=Nulla, 4=Piano Giu, 5=Medio Giu, 6=Forte Giu
    accel = 0
    if action_ball == 0: accel = -BALL_ACCELERATION * 2.0 # Forte SU
    elif action_ball == 1: accel = -BALL_ACCELERATION * 1.0
    elif action_ball == 2: accel = -BALL_ACCELERATION * 0.5
    elif action_ball == 3: accel = 0
    elif action_ball == 4: accel = BALL_ACCELERATION * 0.5
    elif action_ball == 5: accel = BALL_ACCELERATION * 1.0
    elif action_ball == 6: accel = BALL_ACCELERATION * 2.0 # Forte GIU

    # Applica accelerazione alla velocità verticale
    game_state['ballVY'] += accel

    # Limita la velocità massima verticale (altrimenti diventa un proiettile incontrollabile)
    game_state['ballVY'] = max(-MAX_BALL_VY, min(MAX_BALL_VY, game_state['ballVY']))

    # Applica velocità alla posizione
    game_state['ballY'] += game_state['ballVY']

    new_state_ball = get_state_array_ball()

    if turn % 2 != 0:
        # a questo punto ho bisogno di calcolare la ricompensa per la palla in base alla sua nuova posizione
        reward_ball, done = calculate_reward_ball(action_ball, new_state_ball)
        if done:
            ball_agent.remember(get_state_array_ball(), action_ball, reward_ball, new_state_ball, True)
        else :
            ball_agent.remember(get_state_array_ball(), action_ball, reward_ball, new_state_ball, False)
        
        # se ho memorizzato abbastanza per fare un batch allora procedo con l'addestramento
        if len(ball_agent.memory) >= ball_agent.batch_size:
            ball_agent.train_step()

    # 2. Rimbalzi Pareti
    if game_state['ballY'] <= 0:
        game_state['ballY'] = 0
        game_state['ballVY'] = abs(game_state['ballVY'])
    elif game_state['ballY'] >= HEIGHT - BALL_SIZE:
        game_state['ballY'] = HEIGHT - BALL_SIZE
        game_state['ballVY'] = -abs(game_state['ballVY'])

    # 3. Rimbalzo Paddle 1 (Sinistra - AI DQN)
    if (game_state['ballX'] <= PADDLE_WIDTH and 
        game_state['ballY'] + BALL_SIZE >= game_state['paddle1Y'] and 
        game_state['ballY'] <= game_state['paddle1Y'] + PADDLE_HEIGHT):
            angle = calculate_bounce_angle(game_state['paddle1Y'], game_state['ballY'])
        
            # Converti in velocità
            speed = math.sqrt(game_state['ballVX']**2 + game_state['ballVY']**2)
            game_state['ballVX'] = speed * math.cos(math.radians(angle))
            game_state['ballVY'] = speed * math.sin(math.radians(angle))
            # game_state['ballVX'] = 1.2 * abs(game_state['ballVX'])

    # 4. Rimbalzo Paddle 2 (Destra - Simple AI)
    if (game_state['ballX'] >= WIDTH - PADDLE_WIDTH - BALL_SIZE and 
        game_state['ballY'] + BALL_SIZE >= game_state['paddle2Y'] and 
        game_state['ballY'] <= game_state['paddle2Y'] + PADDLE_HEIGHT):
            angle = calculate_bounce_angle(game_state['paddle2Y'], game_state['ballY'])
            
            # Converti in velocità
            speed = math.sqrt(game_state['ballVX']**2 + game_state['ballVY']**2)
            game_state['ballVX'] = - speed * math.cos(math.radians(angle))
            game_state['ballVY'] = speed * math.sin(math.radians(angle))
            #game_state['ballVX'] = -1.2 * abs(game_state['ballVX'])

    # 5. AI Semplice per Paddle 2
    center_paddle2 = game_state['paddle2Y'] + PADDLE_HEIGHT / 2
    if game_state['ballY'] > center_paddle2:
        game_state['paddle2Y'] += 4
    elif game_state['ballY'] < center_paddle2:
        game_state['paddle2Y'] -= 4

    # 6. AI DQN per Paddle 1
    if action == 0: 
        game_state['paddle1Y'] -= 4
    elif action == 2: 
        game_state['paddle1Y'] += 4

    # 7. Limiti Paddle
    game_state['paddle2Y'] = max(0, min(HEIGHT - PADDLE_HEIGHT, game_state['paddle2Y']))
    game_state['paddle1Y'] = max(0, min(HEIGHT - PADDLE_HEIGHT, game_state['paddle1Y']))

def calculate_reward():
    """
    Reward function AVANZATA per Paddle vs Ball AI intelligente
    La palla ora cerca di schivare, quindi serve:
    - Anticipazione
    - Controllo dello spazio
    - Forzare la palla in posizioni svantaggiose
    """
    reward = 0
    done = False
    
    # ===== CALCOLI PRELIMINARI =====
    paddle_center = game_state['paddle1Y'] + PADDLE_HEIGHT / 2
    ball_center = game_state['ballY'] + BALL_SIZE / 2
    distance_to_ball = abs(paddle_center - ball_center)
    
    ball_in_my_half = game_state['ballX'] < WIDTH / 2
    ball_coming_to_me = game_state['ballVX'] < 0
    
    # ===== PREDIZIONE: Dove SARÀ la palla (non dove È) =====
    # La Ball AI si muove, devo anticipare!
    frames_to_arrival = abs(game_state['ballX'] - PADDLE_WIDTH) / abs(game_state['ballVX']) if game_state['ballVX'] != 0 else 999
    predicted_ball_y = game_state['ballY'] + (game_state['ballVY'] * frames_to_arrival)
    
    # Gestisci rimbalzi sui muri nella predizione
    while predicted_ball_y < 0 or predicted_ball_y > HEIGHT:
        if predicted_ball_y < 0:
            predicted_ball_y = -predicted_ball_y
        if predicted_ball_y > HEIGHT:
            predicted_ball_y = 2 * HEIGHT - predicted_ball_y
    
    predicted_ball_center = predicted_ball_y + BALL_SIZE / 2
    distance_to_predicted = abs(paddle_center - predicted_ball_center)
    
    # ===== 1. EVENTI TERMINALI =====
    
    if game_state['ballX'] > WIDTH:
        game_state['score1'] += 1
        reward = 150  # AUMENTATO! Battere una Ball AI intelligente vale di più
        done = True
        reset_game()
        return reward, done
    
    elif game_state['ballX'] < 0:
        game_state['score2'] += 1
        reward = -150  # Penalità maggiore per farsi schivare
        done = True
        reset_game()
        return reward, done
    
    # ===== 2. HIT DETECTION (Reward Maggiorato) =====
    
    hit_paddle = (game_state['ballX'] <= PADDLE_WIDTH and 
                  game_state['ballY'] + BALL_SIZE >= game_state['paddle1Y'] and 
                  game_state['ballY'] <= game_state['paddle1Y'] + PADDLE_HEIGHT and
                  ball_coming_to_me)
    
    if hit_paddle:
        # Reward BASE maggiorato (la palla cercava di schivarti!)
        base_hit_reward = 30  # Era 20
        
        # BONUS per precisione
        if distance_to_ball < 5:
            precision_bonus = 15  # Era 10 - Centro perfetto
        elif distance_to_ball < 10:
            precision_bonus = 8   # Era 5
        elif distance_to_ball < 15:
            precision_bonus = 4   # Era 2
        else:
            precision_bonus = 1
        
        # BONUS SPECIALE: Colpo su palla che stava schivando
        # Se la palla era lontana dal tuo centro = stava cercando di schivarti
        if distance_to_ball > 10:
            reward_evasion_blocked = 10  # Hai intercettato uno schivamento!
        else:
            reward_evasion_blocked = 0
        
        reward = base_hit_reward + precision_bonus + reward_evasion_blocked
        return reward, done
    
    # ===== 3. CONTROLLO DELLO SPAZIO (Nuovo!) =====
    # La Ball AI cerca di stare al centro verticale
    # Se TU stai al centro, la forzi verso zone svantaggiose
    
    ball_y_normalized = ball_center / HEIGHT  # 0 a 1
    paddle_y_normalized = paddle_center / HEIGHT
    
    # Se occupi lo spazio dove la palla vorrebbe stare (35-65% = centro)
    ball_wants_center = 0.35 < ball_y_normalized < 0.65
    paddle_in_center = 0.35 < paddle_y_normalized < 0.65
    
    if ball_in_my_half and ball_wants_center and paddle_in_center:
        # Stai BLOCCANDO la zona preferita della palla!
        reward += 3  # Reward per controllo spazio
    
    # ===== 4. FORZARE PALLA VERSO ZONE SVANTAGGIOSE =====
    # Se la palla è costretta verso i bordi (fascia -8 per la Ball AI)
    ball_in_danger_zone = ball_y_normalized < 0.20 or ball_y_normalized > 0.80
    
    if ball_in_my_half and ball_in_danger_zone:
        # La palla è in una zona che ODIA (prende -8 per frame)
        # Se sei posizionato per colpirla = ottimo!
        if distance_to_ball < PADDLE_HEIGHT / 2:
            reward += 4  # La stai "cacciando" in una zona cattiva per lei
        else:
            reward += 2  # È in zona cattiva ma potresti perderla
    
    # ===== 5. ANTICIPAZIONE INTELLIGENTE =====
    
    if ball_in_my_half and ball_coming_to_me:
        
        # Usa PREDIZIONE invece di posizione attuale
        # La Ball AI si muove per schivarti, devi anticipare!
        
        if frames_to_arrival < 20:  # Palla vicina
            # Reward basato su distanza dalla posizione PREDETTA
            if distance_to_predicted < 8:
                reward += 5  # AUMENTATO (era +3) - Anticipazione perfetta!
            elif distance_to_predicted < 15:
                reward += 3  # Era +1.5
            elif distance_to_predicted < 25:
                reward += 1  # Era +0.5
            else:
                reward -= 3  # Era -2 - Troppo lontano dalla predizione
            
            # Bonus extra per anticipazione EARLY
            if frames_to_arrival > 10 and distance_to_predicted < 10:
                reward += 3  # Hai già anticipato in anticipo!
        
        else:  # Palla lontana
            # Tracking normale sulla posizione attuale
            if distance_to_ball < 10:
                reward += 2
            elif distance_to_ball < 20:
                reward += 1
    
    # ===== 6. STRATEGIA DI CACCIA (Nuovo!) =====
    # Se la palla sta cercando di allontanarsi da te, inseguila!
    
    if ball_in_my_half and not ball_coming_to_me:
        # Palla si sta allontanando (appena colpita o schivando)
        
        # Se sei ancora vicino = mantieni pressione
        if distance_to_ball < 15:
            reward += 2  # Mantieni pressione anche dopo colpo
        
        # Altrimenti torna al centro per preparare difesa
        distance_from_center = abs(paddle_center - HEIGHT / 2)
        if distance_from_center < 15:
            reward += 1.5  # Era +1
        elif distance_from_center > 30:
            reward -= 1  # Era -0.5
    
    # ===== 7. POSIZIONE DIFENSIVA DINAMICA =====
    
    if not ball_in_my_half:
        # Palla lontana - posizione difensiva
        
        # Ma non solo "centro campo" - usa predizione!
        # Dove sarà la palla quando tornerà?
        future_frames = 30  # Guarda 30 frame avanti
        future_ball_y = game_state['ballY'] + (game_state['ballVY'] * future_frames)
        
        # Correggi per rimbalzi
        while future_ball_y < 0 or future_ball_y > HEIGHT:
            if future_ball_y < 0:
                future_ball_y = -future_ball_y
            if future_ball_y > HEIGHT:
                future_ball_y = 2 * HEIGHT - future_ball_y
        
        future_ball_center = future_ball_y + BALL_SIZE / 2
        distance_to_future_position = abs(paddle_center - future_ball_center)
        
        # Posizionati dove sarà la palla
        if distance_to_future_position < 20:
            reward += 3  # Anticipazione lunga!
        elif distance_to_future_position < 40:
            reward += 1.5
        else:
            # Fallback: centro campo
            distance_from_center = abs(paddle_center - HEIGHT / 2)
            if distance_from_center < 10:
                reward += 1
    
    # ===== 8. PENALITÀ BORDI (Aumentata) =====
    # Essere ai bordi è più rischioso vs Ball AI
    if game_state['paddle1Y'] <= 5:  # Era 3
        reward -= 3  # Era -2
    elif game_state['paddle1Y'] >= HEIGHT - PADDLE_HEIGHT - 5:
        reward -= 3
    
    # ===== 9. COPERTURA VERTICALE (Nuovo!) =====
    # Reward per coprire ampia zona verticale
    # Se la palla può muoversi solo in zone dove puoi raggiungerla = buono
    
    max_reachable_up = max(0, paddle_center - PADDLE_HEIGHT)
    max_reachable_down = min(HEIGHT, paddle_center + PADDLE_HEIGHT)
    coverage = (max_reachable_down - max_reachable_up) / HEIGHT
    
    if coverage > 0.4:  # Copri >40% del campo
        reward += 1
    
    # ===== 10. ANTI-PATTERN RIPETITIVO (Nuovo!) =====
    # La Ball AI imparerà i tuoi pattern - devi essere imprevedibile
    # Questo richiederebbe memoria, per ora placeholder
    # TODO: Traccia ultime 10 posizioni e penalizza se troppo prevedibile
    
    return reward, done

def calculate_reward_ball(action_ball, new_state_ball):
    """
    Reward function con penalità SEPARATE per soffitto e pavimento
    """
    reward_ball = 0
    done = False
    
    # ===== CALCOLI PRELIMINARI =====
    ball_center_y = game_state['ballY'] + BALL_SIZE / 2
    y_normalized = ball_center_y / HEIGHT
    
    distance_from_top = game_state['ballY']
    distance_from_bottom = HEIGHT - (game_state['ballY'] + BALL_SIZE)
    
    # ===== SISTEMA FASCE CON GESTIONE ESPLICITA TOP/BOTTOM =====
    
    BORDER_THRESHOLD = HEIGHT * 0.20  # 20% dall'alto/basso
    INTERMEDIATE_THRESHOLD = HEIGHT * 0.35  # 35% dall'alto/basso
    
    # Determina zona
    if distance_from_top < BORDER_THRESHOLD:
        # ===== ZONA SOFFITTO =====
        zone_name = "TOP"
        
        # Penalità esponenziale: più vicino = peggio
        proximity_ratio = distance_from_top / BORDER_THRESHOLD  # 0 (muro) a 1 (limite)
        zone_reward = -50 * (1 + proximity_ratio)**2
        
        
    elif distance_from_bottom < BORDER_THRESHOLD:
        # ===== ZONA PAVIMENTO =====
        zone_name = "BOTTOM"
        
        # Stessa penalità esponenziale
        proximity_ratio = distance_from_bottom / BORDER_THRESHOLD
        zone_reward = -50 * (1 + proximity_ratio)**2
        
        
    elif distance_from_top < INTERMEDIATE_THRESHOLD or distance_from_bottom < INTERMEDIATE_THRESHOLD:
        # ===== ZONA INTERMEDIA =====
        zone_name = "INTERMEDIATE"
        zone_reward = 0
        
    else:
        # ===== ZONA CENTRO =====
        zone_name = "CENTER"
        zone_reward = +5
    
    reward_ball += zone_reward
    
# ===== 1. EVENTI TERMINALI (GOL con PREMIO PRECISIONE) =====
    if game_state['ballX'] < 0 or game_state['ballX'] > WIDTH:
        
        # A. Calcolo il centro della palla
        ball_y_center = game_state['ballY'] + BALL_SIZE / 2
        center_field = HEIGHT / 2
        
        # B. Calcolo la distanza dal centro (valore assoluto)
        dist_from_center = abs(ball_y_center - center_field)
        
        # C. Normalizzo da 0.0 (Centro Perfetto) a 1.0 (Muro)
        max_dist = HEIGHT / 2
        precision_factor = 1.0 - (dist_from_center / max_dist)
        # Ora precision_factor è 1.0 se siamo al centro, 0.0 se siamo sul muro
        
        # D. Definizione Punteggi
        REWARD_BASE_GOL = 25.0   # Punti minimi per aver segnato
        REWARD_MAX_BONUS = 300.0 # Punti extra solo per la precisione
        
        # E. Calcolo Bonus Esponenziale
        # Usiamo la potenza alla terza (^3) per rendere il centro "esclusivo".
        # Esempio: 
        # - Al centro (1.0): 1.0^3 = 1.0 -> Bonus 250 -> Totale 300
        # - A metà (0.5)   : 0.5^3 = 0.125 -> Bonus 31 -> Totale 81 (Drastico calo!)
        bonus = REWARD_MAX_BONUS * (precision_factor ** 3)
        
        reward_ball = REWARD_BASE_GOL + bonus
    
    # ===== 2. HIT DETECTION =====
    hit_by_paddle1 = (game_state['ballX'] <= PADDLE_WIDTH and 
                      game_state['ballY'] + BALL_SIZE >= game_state['paddle1Y'] and 
                      game_state['ballY'] <= game_state['paddle1Y'] + PADDLE_HEIGHT and
                      game_state['ballVX'] < 0)
    
    hit_by_paddle2 = (game_state['ballX'] >= WIDTH - PADDLE_WIDTH - BALL_SIZE and 
                      game_state['ballY'] + BALL_SIZE >= game_state['paddle2Y'] and 
                      game_state['ballY'] <= game_state['paddle2Y'] + PADDLE_HEIGHT and
                      game_state['ballVX'] > 0)
    
    if hit_by_paddle1 or hit_by_paddle2:
        reward_ball = -100
        return reward_ball, done
    
    # ===== 3. REWARD MOVIMENTO (CRITICO!) =====
    if 'prev_ball_y' in game_state:
        y_change = abs(game_state['ballY'] - game_state['prev_ball_y'])
        
        if y_change > 3:
            reward_ball += 5
        elif y_change > 1:
            reward_ball += 2
        elif y_change < 0.5:
            reward_ball -= 5  # FERMO = MALE
    
    game_state['prev_ball_y'] = game_state['ballY']
    
    # ===== 4. REWARD VELOCITÀ VERTICALE =====
    if abs(game_state['ballVY']) > 3:
        reward_ball += 3
    elif abs(game_state['ballVY']) < 1.5:
        reward_ball -= 8
    
    # ===== 5. PENALITÀ TEMPO AI BORDI (TOP E BOTTOM SEPARATI) =====
    if 'frames_at_top' not in game_state:
        game_state['frames_at_top'] = 0
    if 'frames_at_bottom' not in game_state:
        game_state['frames_at_bottom'] = 0
    
    # Reset contatori
    if zone_name != "TOP":
        game_state['frames_at_top'] = 0
    if zone_name != "BOTTOM":
        game_state['frames_at_bottom'] = 0
    
    # Conta frame ai bordi
    if zone_name == "TOP":
        game_state['frames_at_top'] += 1
        time_penalty = -3 * game_state['frames_at_top']  # -3, -6, -9...
        reward_ball += time_penalty
        
        if game_state['frames_at_top'] > 20:
            reward_ball -= 100  # BASTA SOFFITTO!
    
    elif zone_name == "BOTTOM":
        game_state['frames_at_bottom'] += 1
        time_penalty = -3 * game_state['frames_at_bottom']
        reward_ball += time_penalty
        
        if game_state['frames_at_bottom'] > 20:
            reward_ball -= 100  # BASTA PAVIMENTO!
    
    # ===== 6. MOVIMENTO VERSO CENTRO (ESPLICITO PER TOP/BOTTOM) =====
    center_y = HEIGHT / 2
    
    if zone_name == "TOP":
        # Al soffitto, deve andare GIÙ (velocità positiva)
        if game_state['ballVY'] > 1.0:
            reward_ball += 15  # GRANDE BONUS per allontanarsi dal soffitto!
        elif game_state['ballVY'] < -1.0:
            reward_ball -= 15  # PENALITÀ per andare più verso il soffitto!
    
    elif zone_name == "BOTTOM":
        # Al pavimento, deve andare SU (velocità negativa)
        if game_state['ballVY'] < -1.0:
            reward_ball += 15  # GRANDE BONUS per allontanarsi dal pavimento!
        elif game_state['ballVY'] > 1.0:
            reward_ball -= 15  # PENALITÀ per andare più verso il pavimento!
    
    # ===== 7. PENALITÀ AZIONE CHE SPINGE VERSO BORDO =====
    # action_ball: 0,1,2 = SU | 3 = FERMO | 4,5,6 = GIÙ
    
    if zone_name == "TOP" and action_ball in [0, 1, 2]:
        # Sei al soffitto e premi SU → MALISSIMO!
        reward_ball -= 20
    
    elif zone_name == "BOTTOM" and action_ball in [4, 5, 6]:
        # Sei al pavimento e premi GIÙ → MALISSIMO!
        reward_ball -= 20
    
    # ===== 8. REWARD STRATEGICO (evita paddle) =====
    if game_state['ballX'] < WIDTH / 2:
        relevant_dist = abs((game_state['paddle1Y'] + PADDLE_HEIGHT/2) - ball_center_y)
    else:
        relevant_dist = abs((game_state['paddle2Y'] + PADDLE_HEIGHT/2) - ball_center_y)
    
    if relevant_dist > PADDLE_HEIGHT:
        reward_ball += 2
    elif relevant_dist < PADDLE_HEIGHT / 2:
        reward_ball -= 2
    
    return reward_ball, done

    
# ===== GAME LOOP (Background Task) =====
def game_loop():
    global epsilon, frame_count
    
    state = get_state_array_paddle()
    fps_counter = 0
    fps_start = time.time()

    turn = 0 # mi serve per capire di chi è il turno di imparare
    
    while True:
        frame_start = time.time()
        frame_count += 1
        fps_counter += 1
        
        if (frame_count % 10000 == 0):
            turn+=1

        if turn % 2 == 0:
            # è il turno del paddle AI di imparare
            # azione che AI ha scelto di compiere a partire da questo stato 
            action = agent.get_action(state, epsilon)
        else:
            # è il turno della palla AI di imparare
            action = agent.predict_action(state)

        # azione per essere compiuta ha bisogno che venga applicata la fisica
        physics(action, turn)
        
        # dallo stato iniziale se eseguo un azione arrivo a un new_state
        new_state = get_state_array_paddle()

        # calcolo le ricomepense che ho ricevuto nel passaggio al nuovo stato
        reward_value, done = calculate_reward()

        if turn % 2 == 0:
            # imparo che dallo stato iniziale se svolgo una determinata azione arrivo ad uno stato nuovo e ricevo una certa ricompensa
            # imparo anche se questa azione ha portato alla fine dell'episodio
            agent.remember(state, action, reward_value, new_state, done)
        
            # se ho memorizzato abbastanza informazioni da permettermi di fare un batch allora utilizzo queste informazioni per apprendere
            if len(agent.memory) >= agent.batch_size:
                agent.train_step()

        # visto che deve ripartire da capo il loop setto lo state = new_state
        state = new_state
        
        game_state['training_who'] = 'PADDLE' if turn % 2 == 0 else 'BALL'
        
        # serve per decrementare in maniera lineare il valore di epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
        

        # === EMIT TO BROWSER ===
        socketio.emit('update_packet', game_state, namespace='/')
        
        # === FPS STATS ===
        if time.time() - fps_start >= 1.0:
            print(f"📊 FPS: {fps_counter} | Episode: {episode_count} | "
                  f"Score: {game_state['score1']}-{game_state['score2']} | ε: {epsilon:.3f}")
            fps_counter = 0
            fps_start = time.time()
        
        # === FRAME RATE CONTROL ===
        if VISUALIZATION_MODE:
            elapsed = time.time() - frame_start
            target_time = 1.0 / TARGET_FPS
            
            if elapsed < target_time:
                socketio.sleep(target_time - elapsed)  # ← USA socketio.sleep()!

# ===== ROUTES =====
@app.route('/')
def index():
    return render_template('index.html')

# ===== START SERVER =====
if __name__ == '__main__':
    print("\n" + "="*50)
    print("🏓 PONG AI - DQN Training")
    print("="*50)
    print(f"🌐 Server: http://localhost:5001")
    print(f"📺 Visualization: {VISUALIZATION_MODE}")
    print("="*50 + "\n")
    
    # ===== AVVIA GAME LOOP COME BACKGROUND TASK =====
    socketio.start_background_task(game_loop)  # ← QUESTO È FONDAMENTALE!
    
    # ===== AVVIA SERVER =====
    socketio.run(app, debug=False, port=5001, host='0.0.0.0')