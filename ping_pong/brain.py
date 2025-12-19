import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim): # serve per definire la struttura della rete
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128) # primo layer
        self.fc2 = nn.Linear(128, 64) # secondo layer
        self.fc3 = nn.Linear(64, output_dim) # layer di output

    def forward(self, x): # definisce il flusso dei dati attraverso la rete
        x = F.relu(self.fc1(x)) # i dati che do in input passano per il primo layer e sono soggetti alla funzione di attivazione
        x = F.relu(self.fc2(x)) # idem per i dati che escono dal secondo layer
        x = self.fc3(x) # output dei layer
        return x

class Agent:
    def __init__ (self, input_dim, action_dim):
        self.brain = DQN(input_dim, action_dim) # creo il cervello del modello
        self.optimizer = optim.Adam(self.brain.parameters(), lr=0.001) # va a imparare andando a minimizzare la loss
        self.memory = deque(maxlen=10000) # memoria per l'esperienza 
        self.batch_size = 64 #quanti campioni devo fare prima di andare a vedere se ho fatto giusto
        self.gamma = 0.99 # fattore di sconto per le ricompense future
    
    def get_action(self, state, epsilon):
        if random.random() < epsilon: # esplorazione
            return random.randint(0, 2) # azione casuale
        else: # sfruttamento
            state = torch.FloatTensor(state).unsqueeze(0) # converto lo stato in tensore
            with torch.no_grad():
                q_values = self.brain(state) # passo lo stato attraverso la rete
            return q_values.argmax().item() # ritorno l'azione con il valore Q più alto
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # memorizzo l'esperienza
        # ogni volta che faccio una nuova esperienza vado a salvare queste 5 informaizioni in memoria
        # il done mi permette di sapere se l'episodio avrà ripercussioni future o meno
        # in questo caso facciamo che l'AI non vada a pianificare le sue mosse quindi non avrà ripercussioni future
    
    def train_step(self):
        # controllo se ho abbastanza ricordi per fare un batch -> per imparare qualcosa
        if len(self.memory) < self.batch_size:
            return

        # ho bisogno di prendere un batch di campioni in maniera casuale in modo tale che non ci siano correlazioni dovute alla successione temporale 
        batch = random.sample(self.memory, self.batch_size)
        # python fortissimo che mi riesce a dividere le tuple in 5 liste separate per categoria
        states, actions, rewards, next_states, dones = zip(*batch)

        # Converti tutto in tensori PyTorch -> da quel che ho capito facilita i calcoli
        states = torch.tensor(states).float()     
        actions = torch.tensor(actions).long().unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # CALCOLO DELL'ERRORE (LOSS)
        # per ogni stato che è stato preso ed è nel batch vado a vedere quale è stata la azione scelta dall'AI e come la valutava
        curr_Q = self.brain(states).gather(1, actions)
        # per ognuno degli stati appartenenti all batch vado a vedere lo stato futuro e prendo il valore più alto tra le tre azioni future
        next_Q = self.brain(next_states).max(1)[0].unsqueeze(1) # questa quantità che vado a calcolare mi rappresenta la bontà dello stato futuro
        expected_Q = rewards + (self.gamma * next_Q * (1 - dones)) # bellman equation
        # sono a X = 50, compio un azione arrivo a X = 55, prendo azione massima in questa situazione e calcolo expected Q 
        # anche la prima volta che sono passato per X = 55 ho preso il valore massimo ma quel valore massimo era un valore stupido perché l'avrei potuto prendere casualmente 
        # con il movimento casuale mi sarei potuto muovere li ma non sarebbe stato il max -> torno indietro e vedo: "sono passato per X = 55 e la cosa migliore sarebbe state compiere questa azione (max)"

        # la differenza tra valore che avrebbe avuto se avessi saputo dove mi avrebbe portato - azione
        loss = F.mse_loss(curr_Q, expected_Q.detach())
        # errore è semplicemtne differenza tra valore che era stato dato all'azione su dallo stato X = 50 con il valore della medesima azione nel medesimo stato sapendo che se vado a X = 55 c'è una mossa che mi da un sacco di punti

        # Aggiorna i pesi (Backpropagation)
        self.optimizer.zero_grad() # azzero i gradienti in modo da non mischiare i calcoli con i precedenti
        loss.backward() # torno indietro nella rete per vedere quale neurone ha contribuito di più all'errore
        self.optimizer.step() # aggiorno i pesi di quel neurone per minimizzare l'errore

    def predict_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0) # converto lo stato in tensore
        with torch.no_grad():
            q_values = self.brain(state) # passo lo stato attraverso la rete
        return q_values.argmax().item() # ritorno l'azione con il valore Q più alto

class BallAgent:
    def __init__ (self, input_dim, action_dim):
        self.brain = DQN(input_dim, action_dim) # creo il cervello del modello
        self.optimizer = optim.Adam(self.brain.parameters(), lr=0.001) # va a imparare andando a minimizzare la loss
        self.memory = deque(maxlen=10000) # memoria per l'esperienza 
        self.batch_size = 64 #quanti campioni devo fare prima di andare a vedere se ho fatto giusto
        self.gamma = 0.99 # fattore di sconto per le ricompense future
    
    def get_action(self, state, epsilon):
        if random.random() < epsilon: # esplorazione
            return random.randint(0, 6) # azione casuale
        else: # sfruttamento
            state = torch.FloatTensor(state).unsqueeze(0) # converto lo stato in tensore
            with torch.no_grad():
                q_values = self.brain(state) # passo lo stato attraverso la rete
            return q_values.argmax().item() # ritorno l'azione con il valore Q più alto
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # memorizzo l'esperienza
    
    def train_step(self):
        # controllo se ho abbastanza ricordi per fare un batch -> per imparare qualcosa
        if len(self.memory) < self.batch_size:
            return

        # ho bisogno di prendere un batch di campioni in maniera casuale in modo tale che non ci siano correlazioni dovute alla successione temporale 
        batch = random.sample(self.memory, self.batch_size)
        # python fortissimo che mi riesce a dividere le tuple in 5 liste separate per categoria
        states, actions, rewards, next_states, dones = zip(*batch)

        # Converti tutto in tensori PyTorch -> da quel che ho capito facilita i calcoli
        states = torch.tensor(states).float()     
        actions = torch.tensor(actions).long().unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # CALCOLO DELL'ERROE (LOSS)
        # per ogni stato che è stato preso ed è nel batch vado a vedere quale è stata la azione scelta dall'AI e come la valutava
        curr_Q = self.brain(states).gather(1, actions)
        # per ognuno degli stati appartenenti all batch vado a vedere lo stato futuro e prendo il valore più alto tra le tre azioni future
        next_Q = self.brain(next_states).max(1)[0].unsqueeze(1) # questa quantità che vado a calcolare mi rappresenta la bontà dello stato futuro
        expected_Q = rewards + (self.gamma * next_Q * (1 - dones)) # bellman equation
        # sono a X = 50, compio un azione arrivo a X = 55, prendo azione massima in questa situazione e calcolo expected Q 
        # anche la prima volta che sono passato per X = 55 ho preso il valore massimo ma quel valore massimo era un valore stupido perché l'avrei potuto prendere casualmente 
        # con il movimento casuale mi sarei potuto muovere li ma non sarebbe stato il max -> torno indietro e vedo: "sono passato per X = 55 e la cosa migliore sarebbe state compiere questa azione (max)"
    
    def predict_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0) # converto lo stato in tensore
        with torch.no_grad():
            q_values = self.brain(state) # passo lo stato attraverso la rete
        return q_values.argmax().item() # ritorno l'azione con il valore Q più alto

