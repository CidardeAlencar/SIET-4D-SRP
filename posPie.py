import cv2
import mediapipe as mp
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore
import time

# Inicializa MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Inicializa Firebase
cred = credentials.Certificate("siet-4d-firebase-adminsdk-ofsoh-2d346ffcfd.json")
firebase_admin.initialize_app(cred)
db = firestore.client()
firebase_updated = False

# Función para calcular la distancia entre dos puntos
def calculate_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

# Función para calcular el punto medio entre dos puntos
def midpoint(point1, point2):
    return [(point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2]

# Función para calcular el ángulo entre tres puntos
def calculate_angle(a, b, c):
    a = np.array(a)  # Primer punto
    b = np.array(b)  # Segundo punto
    c = np.array(c)  # Tercer punto
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# Función para actualizar firebase
def update_firebase():
    evaluations_ref = db.collection('evaluations')

    # Recupera todos los documentos
    docs = evaluations_ref.get()

    # Ordena los documentos por el último valor del arreglo 'timestamp'
    sorted_docs = sorted(docs, key=lambda x: x.to_dict().get('timestamp', [])[-1] if x.to_dict().get('timestamp') else None, reverse=True)

    if sorted_docs:
        # Toma el ID del documento con el último valor de timestamp
        doc_id = sorted_docs[0].id

        doc_ref = evaluations_ref.document(doc_id)
        doc = doc_ref.get()

        if doc.exists:
            # Obtén el documento actual
            data = doc.to_dict()

            # Verifica si 'pospie' ya existe y es una lista
            pospie = data.get('pospie', [])
            if not isinstance(pospie, list):
                pospie = []

            if pospie:
                # Cambia el valor del último elemento del arreglo
                pospie[-1] = '10'
            else:
                # Si el arreglo está vacío, añade el valor '10'
                pospie.append('10')

            # Actualiza el documento con el nuevo arreglo
            doc_ref.update({'pospie': pospie})
        else:
            # Si el documento no existe, crea un nuevo campo 'pospie' como lista con el primer elemento
            doc_ref.set({'pospie': ['10']})


# Captura de video
cap = cv2.VideoCapture(0)
cv2.namedWindow('Estimacion de posicion', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Estimacion de posicion', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Inicializa el contador de tiempo
start_time = time.time()
countdown_duration = 10  # 10 segundos

while cap.isOpened():
    ret, frame = cap.read()
    # Calcula el tiempo transcurrido
    elapsed_time = time.time() - start_time
    countdown = countdown_duration - int(elapsed_time)

    if countdown > 0:
        # Muestra el contador en la pantalla
        cv2.putText(frame, f'La evaluacion se termina en {countdown}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        break
    
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    
    if results.pose_landmarks:
        #para que se muestre todos los puntos descomentar
        #mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,mp_drawing.DrawingSpec(color=(128,0,250),thickness=2,circle_radius=3),mp_drawing.DrawingSpec(color=(255,255,255),thickness=2))
        
        landmarks = results.pose_landmarks.landmark
        
        # Obtener coordenadas
        #Coordenadas brazo izquierdo
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        #Coordenadas brazo derecho
        elbowR = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        shoulderR = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        hipR = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        #Coordenadas pies
        ankle_left = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        ankle_right = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
        hip_left = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        hip_right = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        
        # Convertir coordenadas a píxeles
        #brazo izquierdo
        shoulder_pixel = tuple(np.multiply(shoulder, [640, 480]).astype(int))
        elbow_pixel = tuple(np.multiply(elbow, [640, 480]).astype(int))
        wrist_pixel = tuple(np.multiply(wrist, [640, 480]).astype(int))
        #brazo derecho
        shoulderR_pixel = tuple(np.multiply(shoulderR, [640, 480]).astype(int))
        elbowR_pixel = tuple(np.multiply(elbowR, [640, 480]).astype(int))
        hipR_pixel = tuple(np.multiply(hipR, [640, 480]).astype(int))
        #Coordenadas pies
        h, w, _ = frame.shape
        ankle_left_pixel = tuple(np.multiply(ankle_left, [w, h]).astype(int))
        ankle_right_pixel = tuple(np.multiply(ankle_right, [w, h]).astype(int))
        hip_left_pixel = tuple(np.multiply(hip_left, [w, h]).astype(int))
        hip_right_pixel = tuple(np.multiply(hip_right, [w, h]).astype(int))

        # Calcular el punto medio entre las caderas (centro de masa aproximado)
        mid_hip_pixel = midpoint(hip_left_pixel, hip_right_pixel)
        mid_hip_pixel = tuple(np.multiply(mid_hip_pixel, [1, 1]).astype(int))
        
        # Verificar la simetría de los tobillos respecto al centro de masa
        left_distance = calculate_distance(ankle_left_pixel, mid_hip_pixel)
        right_distance = calculate_distance(ankle_right_pixel, mid_hip_pixel)
        
        tolerance = 0.1 * calculate_distance(ankle_left_pixel, ankle_right_pixel)  # 10% de la distancia entre tobillos
        balanced = abs(left_distance - right_distance) < tolerance
            
        #Resaltar coordenadas
        #brazo izquierdo
        cv2.circle(frame,shoulder_pixel,6,(0,255,255),4)
        cv2.circle(frame,elbow_pixel,6,(128,0,250),4)
        cv2.circle(frame,wrist_pixel,6,(255,191,0),4)
        #brazo derecho
        cv2.circle(frame,shoulderR_pixel,6,(0,255,255),4)
        cv2.circle(frame,elbowR_pixel,6,(128,0,250),4)
        cv2.circle(frame,hipR_pixel,6,(255,191,0),4)
        #Coordenadas pies
        cv2.circle(frame, mid_hip_pixel, 6, (0, 255, 255), 4)
        cv2.circle(frame, ankle_left_pixel, 6, (0, 255, 255), 4)
        cv2.circle(frame, ankle_right_pixel, 6, (0, 255, 255), 4)
        
        # Calcular ángulo
        #brazo izquierdo
        angle = calculate_angle(shoulder, elbow, wrist)
        #brazo derecho
        angle2 = calculate_angle(elbowR, shoulderR, hipR)

        # Determinar el color del área del ángulo
        #brazo izquierdo
        if 35 <= angle <= 70:
            color = (0, 255, 0)  # Verde
        else:
            color = (0, 0, 255)  # Rojo
        #brazo derecho
        if 25 <= angle2 <= 45:
            color2 = (0, 255, 0)  # Verde
        else:
            color2 = (0, 0, 255)  # Rojo
        
        # Dibujar líneas entre los puntos
        #brazo izquierdo
        cv2.line(frame, shoulder_pixel, elbow_pixel, (255, 255, 255), 10)
        cv2.line(frame, elbow_pixel, wrist_pixel, (255, 255, 255), 10)
        #cv2.line(frame, wrist_pixel, shoulder_pixel, (255, 255, 255), 5)
        #brazo derecho
        cv2.line(frame, elbowR_pixel, shoulderR_pixel, (255, 255, 255), 10)
        cv2.line(frame, shoulderR_pixel, hipR_pixel, (255, 255, 255), 10)
        #Coordenadas pies
        cv2.line(frame, ankle_left_pixel, mid_hip_pixel, (0, 255, 0) if balanced else (0, 0, 255), 2)
        cv2.line(frame, ankle_right_pixel, mid_hip_pixel, (0, 255, 0) if balanced else (0, 0, 255), 2)
        
        # Dibujar el área del ángulo
        overlay = frame.copy()
        
        #brazo izquierdo
        poly_points = np.array([shoulder_pixel, elbow_pixel, wrist_pixel])
        cv2.fillPoly(frame, [poly_points], color)
        #brazo derecho
        poly_points2 = np.array([elbowR_pixel,shoulderR_pixel, hipR_pixel])
        cv2.fillPoly(frame, [poly_points2], color2)

        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        
        # Mostrar ángulo en la imagen
        #brazo izquierdo
        cv2.putText(frame, str(int(angle)), elbow_pixel, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        #brazo derecho
        cv2.putText(frame, str(int(angle2)), shoulderR_pixel, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        #Coordenadas pies
        cv2.putText(frame, 'Correcta' if balanced and color == (0, 255, 0) and color2 == (0, 255, 0) else 'Incorrecta', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if balanced and color == (0, 255, 0) and color2 == (0, 255, 0) else (0, 0, 255), 2, cv2.LINE_AA)

        # Capturar pantalla si todas las condiciones son verdes
        if color == (0, 255, 0) and color2 == (0, 255, 0) and balanced and not firebase_updated:
            update_firebase()
            firebase_updated = True
        elif color != (0, 255, 0) or color2 != (0, 255, 0) or not balanced:
            firebase_updated = False
        
    cv2.imshow('Estimacion de posicion', frame)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
