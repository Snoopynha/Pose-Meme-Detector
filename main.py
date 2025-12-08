import cv2 # Permissões da Webcam
import mediapipe as mp
import random
import tkinter as tk # Pega o tamanho da tela

# --- Configurações Iniciais ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

WINDOW_NAME = 'Nu Metal Detector'

# --- Funções Auxiliares ---
# Função para centralizar a janela na tela
def center_window(window_name, frame_width, frame_height):
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()
    
    # Calcula a posição x, y
    x = int((screen_width - frame_width) / 2)
    y = int((screen_height - frame_height) / 2)
    
    cv2.namedWindow(window_name)
    cv2.moveWindow(window_name, x, y)

# Função para desenhar a interface
def draw_ui(frame, text):
    h, w, _ = frame.shape
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    color_text = (255, 255, 255)
    
    # Descobre o tamanho do texto para centralizar
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    
    text_x = (w - text_w) // 2
    text_y = h - 20
    
    # Desenhar um retângulo preto semitransparente atrás do texto
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 60), (w, h), (0, 0, 0), -1)
    alpha = 0.6 # Transparência (0.0 a 1.0)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    # Escreve o texto por cima
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, color_text, thickness, cv2.LINE_AA)

# Função pra checar a posição das mãos
def is_hand_open(hand_landmarks):
    try:
        landmarks = hand_landmarks.landmark

        is_index_open = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
        is_middle_open = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
        is_ring_open = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.RING_FINGER_PIP].y
        is_pinky_open = landmarks[mp_hands.HandLandmark.PINKY_TIP].y < landmarks[mp_hands.HandLandmark.PINKY_PIP].y
        
        return is_index_open and is_middle_open and is_ring_open and is_pinky_open
    except Exception as e:
        return False

# --- Configuração das Imagens ---
# Upload das imagens
IMAGE_FILENAMES = [
    'assets/numetal.jpg',  
    'assets/calma.jpg',  
    'assets/avril.jpg',  
    'assets/serj.jpg',  
    'assets/davi.jpg'   
]

# Definindo o tamanho das imagens
MEME_WIDTH = 200
MEME_HEIGHT = 150 
# Carregando as imagens que foram processadas
loaded_images = [] 

print("Carregando imagens...")
for path in IMAGE_FILENAMES:
    try:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            # Apenas avisa, mas não para o código
            print(f"Aviso: Arquivo {path} não encontrado. Continuando...")
            continue
        
        # Redimensiona a imagem
        img_resized = cv2.resize(img, (MEME_WIDTH, MEME_HEIGHT), interpolation=cv2.INTER_AREA)

        # Processa a transparência
        if img_resized.shape[2] == 4: 
            b, g, r, alpha = cv2.split(img_resized)
            img_bgr = cv2.merge((b, g, r))
            alpha_mask = alpha / 255.0
            loaded_images.append( (img_bgr, alpha_mask) ) 
        else: # Se não tem imagem
            loaded_images.append( (img_resized, None) ) 
        
        print(f"Sucesso ao carregar e processar: {path}")

    except Exception as e:
        print(f"ERRO ao carregar '{path}': {e}")
        print("Verifique o nome e se o arquivo existe. Pulando este arquivo.")

if not loaded_images: # Verifica se a lista ta vazia
    print("NENHUMA IMAGEM FOI CARREGADA. SUBSTITUTA SENDO CRIADA.")
    placeholder = cv2.merge([
        cv2.zeros((MEME_HEIGHT, MEME_WIDTH), dtype='uint8'),
        cv2.zeros((MEME_HEIGHT, MEME_WIDTH), dtype='uint8'),
        cv2.ones((MEME_HEIGHT, MEME_WIDTH), dtype='uint8') * 255
    ])
    loaded_images.append((placeholder, None))

print(f"--- {len(loaded_images)} imagens carregadas. Iniciando webcam. ---")

# --- LOOP Principal ---
# Inicializa a Webcam
cap = cv2.VideoCapture(0)

# Controle de estado
window_centered = False
pose_detected_previously = False 
current_image_to_display = None # Guarda a imagem aleatória escolhida

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1) # Inverte para modo selfie

    if not window_centered:
        h, w, _ = frame.shape
        center_window(WINDOW_NAME, w, h)
        window_centered = True
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

    open_hands_count = 0
    pose_found_this_frame = False # Para checar a pose (no frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                bgr_frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS)

            if is_hand_open(hand_landmarks):
                open_hands_count += 1
        
        # Checa se a pose foi detectada (no frame congelado)
        if len(results.multi_hand_landmarks) == 2 and open_hands_count == 2:
            pose_found_this_frame = True
    

    # Lógica de exibição de imagem
    if pose_found_this_frame:
        # Se a pose foi encontrada AGORA e NÃO estava sendo mostrada antes
        if not pose_detected_previously:
            pose_detected_previously = True # Ativa a trava
            # ESCOLHE UMA NOVA IMAGEM ALEATÓRIA da lista
            current_image_to_display = random.choice(loaded_images) 
        
        if current_image_to_display:
            # Pega a imagem e a máscara
            img_bgr, img_alpha = current_image_to_display
            
            # Centraliza a imagem do meme na tela
            frame_h, frame_w, _ = bgr_frame.shape
            x_offset = int((frame_w - MEME_WIDTH) / 2)
            y_offset = 50 

            # Verifica limites
            if y_offset + MEME_HEIGHT < frame_h and x_offset + MEME_WIDTH < frame_w:
                roi = bgr_frame[y_offset : y_offset + MEME_HEIGHT, x_offset : x_offset + MEME_WIDTH]

                if img_alpha is not None: 
                    for c in range(0, 3):
                        bgr_frame[y_offset : y_offset + MEME_HEIGHT, x_offset : x_offset + MEME_WIDTH, c] = \
                            roi[:, :, c] * (1 - img_alpha) + \
                            img_bgr[:, :, c] * img_alpha
                else:
                    bgr_frame[y_offset : y_offset + MEME_HEIGHT, x_offset : x_offset + MEME_WIDTH] = img_bgr

            draw_ui(bgr_frame, "POSE DETECTADA! (ROCK ON!)")

    else: # Se a pose não foi encontrada neste frame
        pose_detected_previously = False
        current_image_to_display = None # Limpa a imagem
        # Texto de instrução
        draw_ui(bgr_frame, "Mostre as duas maos abertas | 'q' para sair")

    cv2.imshow(WINDOW_NAME, bgr_frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

print("Fechando...")
cap.release()
cv2.destroyAllWindows()
hands.close()