import cv2
import mediapipe as mp

# --- CONFIGURAÇÕES ---
TOLERANCIA_MATCH = 0.07
NOME_JANELA = "Detector de Memes (NU METAL)" 

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# --- DADOS CALIBRADOS ---
POSES_DB = {
    "calma" : {
        "imagem": "teste/calma.jpg",
        "mao_1": [(0.0, 0.0), (0.054, -0.019), (0.095, -0.079), (0.126, -0.128), (0.159, -0.147), (0.064, -0.174), (0.079, -0.244), (0.085, -0.29), (0.09, -0.33), (0.035, -0.185), (0.043, -0.267), (0.048, -0.32), (0.05, -0.365), (0.006, -0.18), (0.009, -0.256), (0.014, -0.304), (0.019, -0.345), (-0.023, -0.162), (-0.028, -0.221), (-0.028, -0.26), (-0.026, -0.297)],
        "mao_2": [(0.0, 0.0), (-0.052, -0.02), (-0.095, -0.078), (-0.122, -0.129), (-0.153, -0.149), (-0.065, -0.171), (-0.08, -0.24), (-0.084, -0.284), (-0.085, -0.325), (-0.035, -0.183), (-0.044, -0.263), (-0.046, -0.313), (-0.046, -0.356), (-0.007, -0.179), (-0.01, -0.253), (-0.012, -0.299), (-0.014, -0.339), (0.023, -0.162), (0.023, -0.218), (0.021, -0.256), (0.018, -0.293)]
    },
    "comuna": {
        "imagem": "teste/comuna.webp", 
        "mao_1": [(0.0, 0.0), (0.058, -0.001), (0.118, -0.033), (0.168, -0.057), (0.215, -0.069), (0.068, -0.064), (0.082, -0.101), (0.086, -0.13), (0.089, -0.163), (0.023, -0.079), (0.021, -0.122), (0.016, -0.156), (0.007, -0.191), (-0.016, -0.083), (-0.026, -0.128), (-0.036, -0.158), (-0.045, -0.191), (-0.05, -0.076), (-0.069, -0.112), (-0.087, -0.131), (-0.105, -0.152)],
        "mao_2": [(0.0, 0.0), (-0.061, 0.009), (-0.121, -0.022), (-0.169, -0.041), (-0.219, -0.045), (-0.084, -0.078), (-0.1, -0.12), (-0.114, -0.139), (-0.127, -0.16), (-0.037, -0.095), (-0.044, -0.143), (-0.047, -0.168), (-0.05, -0.195), (0.008, -0.094), (0.011, -0.143), (0.018, -0.164), (0.021, -0.187), (0.048, -0.078), (0.069, -0.114), (0.089, -0.126), (0.104, -0.14)]
    },
    "noopy": {
        "imagem": "teste/noopy.jpg",
        "mao_1": [(0.0, 0.0), (-0.049, -0.034), (-0.071, -0.108), (-0.037, -0.169), (-0.002, -0.204), (-0.074, -0.221), (-0.098, -0.31), (-0.112, -0.368), (-0.121, -0.417), (-0.032, -0.232), (-0.038, -0.345), (-0.038, -0.413), (-0.035, -0.468), (0.008, -0.213), (0.017, -0.272), (0.008, -0.211), (0.002, -0.162), (0.042, -0.178), (0.04, -0.198), (0.027, -0.147), (0.018, -0.107)],
        "mao_2": [(0.0, 0.0), (0.05, -0.028), (0.074, -0.093), (0.044, -0.136), (0.002, -0.162), (0.079, -0.209), (0.106, -0.288), (0.12, -0.339), (0.128, -0.386), (0.037, -0.226), (0.048, -0.33), (0.05, -0.397), (0.051, -0.454), (-0.002, -0.212), (-0.009, -0.27), (-0.012, -0.206), (-0.013, -0.154), (-0.037, -0.179), (-0.044, -0.209), (-0.037, -0.163), (-0.031, -0.121)]
    },
    "moon": {
        "imagem": "teste/moon.jpg",
        "mao_1": [(0.0, 0.0), (-0.029, -0.016), (-0.035, -0.038), (-0.038, -0.064), (-0.037, -0.083), (0.004, -0.053), (-0.006, -0.09), (-0.018, -0.113), (-0.026, -0.128), (0.024, -0.068), (0.011, -0.106), (-0.004, -0.135), (-0.016, -0.151), (0.043, -0.081), (0.035, -0.123), (0.021, -0.154), (0.009, -0.171), (0.06, -0.094), (0.06, -0.134), (0.049, -0.164), (0.034, -0.181)],
        "mao_2": [(0.0, 0.0), (0.017, -0.078), (0.013, -0.142), (0.016, -0.194), (0.015, -0.234), (-0.061, -0.149), (-0.037, -0.245), (-0.006, -0.295), (0.021, -0.326), (-0.059, -0.14), (-0.034, -0.254), (0.001, -0.304), (0.029, -0.334), (-0.044, -0.134), (-0.02, -0.244), (0.013, -0.293), (0.04, -0.321), (-0.018, -0.128), (0.005, -0.212), (0.027, -0.253), (0.046, -0.279)]
    }
}

# --- FUNÇÕES ---

def get_relative_coordinates(hand_landmarks):
    points = hand_landmarks.landmark
    wrist = points[0]
    relative_list = []
    for lm in points:
        rel_x = lm.x - wrist.x
        rel_y = lm.y - wrist.y
        relative_list.append((rel_x, rel_y))
    return relative_list

def compare_hands(live_coords, target_coords, tol):
    total_error = 0
    for i in range(21):
        diff_x = abs(live_coords[i][0] - target_coords[i][0])
        diff_y = abs(live_coords[i][1] - target_coords[i][1])
        total_error += (diff_x + diff_y)
    return (total_error / 21) < tol

def overlay_image(background, overlay_path):
    try:
        overlay = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)
        if overlay is None:
            return background
            
        h_back, w_back, _ = background.shape
        # Redimensiona para altura de 300px
        scale_factor = 100 / overlay.shape[0]
        new_w = int(overlay.shape[1] * scale_factor)
        new_h = int(overlay.shape[0] * scale_factor)
        overlay = cv2.resize(overlay, (new_w, new_h))

        y_offset = (h_back - new_h) // 2
        x_offset = (w_back - new_w) // 2
        
        if y_offset < 0: y_offset = 0
        if x_offset < 0: x_offset = 0

        if overlay.shape[2] == 4:
            alpha_s = overlay[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s

            for c in range(0, 3):
                roi = background[y_offset:y_offset+new_h, x_offset:x_offset+new_w, c]
                if roi.shape[0] == new_h and roi.shape[1] == new_w:
                    background[y_offset:y_offset+new_h, x_offset:x_offset+new_w, c] = \
                        (alpha_s * overlay[:, :, c] + alpha_l * roi)
        else:
            background[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = overlay

    except Exception as e:
        print(f"Erro no overlay: {e}")
        
    return background

# --- LOOP PRINCIPAL ---

cap = cv2.VideoCapture(0)
# Cria a janela uma única vez antes do loop
cv2.namedWindow(NOME_JANELA)

print("Iniciando... Faça as poses!")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    current_match = None 

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        if len(results.multi_hand_landmarks) == 2:
            h1_live = get_relative_coordinates(results.multi_hand_landmarks[0])
            h2_live = get_relative_coordinates(results.multi_hand_landmarks[1])
            
            for nome_pose, dados in POSES_DB.items():
                target_h1 = dados["mao_1"]
                target_h2 = dados["mao_2"]
                
                # Checa normal e invertido
                match_a = compare_hands(h1_live, target_h1, TOLERANCIA_MATCH) and \
                          compare_hands(h2_live, target_h2, TOLERANCIA_MATCH)
                match_b = compare_hands(h1_live, target_h2, TOLERANCIA_MATCH) and \
                          compare_hands(h2_live, target_h1, TOLERANCIA_MATCH)
                
                if match_a or match_b:
                    current_match = nome_pose
                    break 
    
    if current_match:
        image_path = POSES_DB[current_match]["imagem"]
        frame = overlay_image(frame, image_path)
        cv2.putText(frame, f"POSE: {current_match.upper()}", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Esperando pose...", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow(NOME_JANELA, frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()