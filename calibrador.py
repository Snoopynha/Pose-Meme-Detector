import cv2
import mediapipe as mp
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

calibrando = False
tempo_inicio = 0
contagem_duracao = 5

print("--- CALIBRADOR ---")
print("Este código gera coordenadas que funcionam em qualquer lugar da tela.")
print("1. Faça a pose.")
print("2. Pressione 'c' para contar.")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    # Espelhar antes de processar para a lógica ficar intuitiva
    image = cv2.flip(image, 1)
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if calibrando:
            tempo_atual = time.time()
            tempo_decorrido = tempo_atual - tempo_inicio
            tempo_restante = int(contagem_duracao - tempo_decorrido) + 1

            if tempo_restante > 0:
                cv2.putText(image, f"Capturando: {tempo_restante}", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            else:
                print("\n" + "="*50)
                print("DADOS RELATIVOS (Ignoram a posição na tela):")
                print("="*50)
                
                if results.multi_handedness:
                    for idx, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
                        label_mao = handedness.classification[0].label 
                        pontos = hand_landmarks.landmark
                        
                        # Pega o pulso como referência
                        pulso_x = pontos[0].x
                        pulso_y = pontos[0].y
                        
                        print(f"--- MÃO {idx + 1}: {label_mao.upper()} ---")
                        
                        lista_relativa = []
                        
                        # Calcula a posição de cada ponto ignorado a posição do pulso
                        for lm in pontos:
                            rel_x = lm.x - pulso_x
                            rel_y = lm.y - pulso_y
                            lista_relativa.append((round(rel_x, 3), round(rel_y, 3)))
                            
                        print(f"Lista Relativa (Copie isso):")
                        print(lista_relativa)
                        print("-" * 20)

                    print("="*50 + "\n")
                    calibrando = False 

    if not calibrando:
        cv2.putText(image, "Aperte 'c' para calibrar", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Calibrador Relativo', image)

    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c') and not calibrando:
        calibrando = True
        tempo_inicio = time.time()

cap.release()
cv2.destroyAllWindows()