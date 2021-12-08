from textloader import  string2code, id2lettre,code2string
import math
from icecream import ic
import torch

#  TODO:  Ce fichier contient les différentes fonction de génération



 

def generate(rnn, emb, decoder, eos, start="hello how", maxlen=200):
    """  Fonction de génération (l'embedding et le decodeur doivent être des fonctions du rnn). Initialise le réseau avec start
     (ou à 0 si start est vide) et génère une séquence de longueur maximale 200 ou qui s'arrête quand eos est généré.
        * rnn : le réseau
        * emb : la couche d'embedding
        * decoder : le décodeur
        * eos : ID du token end of sequence
        * start : début de la phrase
        * maxlen : longueur maximale
    """
    #  TODO:  Implémentez la génération à partir du RNN, et d'une fonction decoder qui renvoie les logits (logarithme de probabilité à une constante près, i.e. ce qui vient avant le softmax) des différentes sorties possibles
    y_seq=[]
    str=string2code(start)
    y_seq.append(str)
    str = str.unsqueeze(1)
    with torch.no_grad():
        decoded,h=rnn.forward(str)
        H=h[-1].unsqueeze(0)
        char = torch.argmax(torch.softmax(decoded[-1],-1),-1)
    y_seq.append(char)
    #generate other terms: 
    while char != eos : 
        char = emb(char)
        h,_,_=rnn.one_step(char,H)
        char = decoder(h)
        char = torch.argmax(torch.softmax(char,dim=-1),dim=-1)
        y_seq.append(char)
        if len(y_seq) == maxlen : 
            y_seq.append(eos)
            break
        
    return code2string(y_seq)



def generate_beam(rnn, emb, decoder, eos, k, start="", maxlen=200):
    """
        Génere une séquence en beam-search : à chaque itération, on explore pour chaque candidat les k symboles les plus probables; puis seuls les k meilleurs candidats de l'ensemble des séquences générées sont conservés (au sens de la vraisemblance) pour l'itération suivante.
        * rnn : le réseau
        * emb : la couche d'embedding
        * decoder : le décodeur
        * eos : ID du token end of sequence
        * k : le paramètre du beam-search
        * start : début de la phrase
        * maxlen : longueur maximale
    """
    #  TODO:  Implémentez le beam Search


# p_nucleus
def p_nucleus(decoder, alpha: float):
    """Renvoie une fonction qui calcule la distribution de probabilité sur les sorties

    Args:
        * decoder: renvoie les logits étant donné l'état du RNN
        * alpha (float): masse de probabilité à couvrir
    """
    def compute(h):
        """Calcule la distribution de probabilité sur les sorties

        Args:
           * h (torch.Tensor): L'état à décoder
        """
        #  TODO:  Implémentez le Nucleus sampling ici (pour un état s)
    return compute
