import cv2
from ultralytics import YOLO
import numpy as np
from typing import List, Tuple, Dict, Optional      #questo mi serve perchè avere la possibilità di dare un input in più alla struttura, riduce i costi computazionali (sapere già se sarà una tupla o meno, lo facilità nel ragionamento)

class Landmarks_detector():
    def __init__(self,
                # static_image_mode: bool=False,                        '''Non c’è distinzione tra immagine statica e video → YOLO lavora frame-per-frame'''
                max_num_faces: int=1,                                   #questo rappresenta il numero massimo di volti da rilevare, che posso tracciare all'interno del video
                # refine_landmarks: bool=True,                          #questo mi permetterà di andare a rifinire meglio la precisione attorno al contorno occhi e labbra
                # min_detection_confidence: float=0.5,                    #queste due righe sono le due soglie di confidenza per rilevare un nuovo volto oppure a continuare a tracciare un volto già rilevato
                # min_tracking_confidence: float=0.5,                   '''Non esiste “tracking confidence”: YOLO non tiene memoria tra i frame, ogni predizione è indipendente'''
                model_path: str=r"C:\Users\matti\Desktop\corsi_di_formazione\2025_02_BitCamp_python_AI\esercizi\esercizi_reti_neurali\riconoscimento_facciale\face_yolov8s.pt",                       #è il file di pytorch con i pesi del modello YOLO che gli carico
                conf: float=0.5,                                        #rappresenta la soglia di confidenza, se la predizione è più bassa di 0.5 scarta l'immagine perchè non la considera come un volto (serve per dare più robustezza nel caso in cui ci siano immagini dove NON c'è un singolo volto)
                device: Optional[str] = None):                          #rappresente il tipo di device che si riesce a utilizzare, cambia lui in automatico se CPU o GPU (se uso GPU, scrivere: "cuda:0")
        
        self.yo_model = YOLO(model_path)                                                            #è il modello di rete neurale per i landmarks facciali          
        self.yo_conf = conf        
        self.yo_max_num_faces = max_num_faces
        self.device = device
        #mappo i punti del modello
        self.kp_map = {
            'left_eye': 0,
            'right_eye': 1,
            'nose': 2,
            'mouth_left': 3,
            'mouth_right': 4
        }
        self.define_lms_index()#definire gli indici dei landmark

    #Non devo più fornire array di indici pixel come con mediapipe → con YOLO bastano le etichette (0–4).
    #define_lms_index serve solo a organizzare semanticamente i 5 keypoints, non a “dire a YOLO cosa cercare”.
    # YOLO conosce i 5 punti, non devo guidarlo con indici extra
    def define_lms_index(self):
        self.left_eye_indices = ['left_eye']
        self.right_eye_indices = ['right_eye']
        self.nose_indices = ['nose']
        self.mouth_indices = ['mouth_left', 'mouth_right']
    
    #creo la funzione per rilevare i landmarks delle immagini
    def detect_lms(self,image:np.ndarray)->Optional[List]:  #con la freccia vado a precisare che ... (riguardare registrazione)
        #carico l'immagine
        # img_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB) '''questa conversione in yolo non serve perchè lavora anche in BGR'''
        #processo l'immagine
        results = self.yo_model(image,conf=self.yo_conf,max_det=self.yo_max_num_faces,verbose=False)
        if not results or len(results) == 0:
            return None
        r0 = results[0]
        if r0.boxes is None or len(r0.boxes) == 0:
            return None
        return [r0]

    #funzione eper estrarre gli elementi facciali
    def extract_keypoints_dict(self,r0)->Dict:   #preciso che mi deve ritornare un 'dict' //// (...,img_shape:Tuple[int,int])
        # height, width = img_shape[:2]
        facial_elements = {
            'left_eye': [],
            'right_eye': [],
            'nose': [],
            'mouth': []  #qui gli unirò la parte destra e sinistra della bocca
        }

        kps = getattr(r0,'keypoints', None)
        if kps is None or kps.xy is None:
            return facial_elements
        
        # xy: [num_faces, K, 2]
        xy = kps.xy  # tensor o ndarray
        if hasattr(xy, "cpu"):
            xy = xy.cpu().numpy()

        # Prendiamo SOLO il primo volto per coerenza con la tua logica
        face_kps = xy[0]  # shape [K,2]

        # Mappa 5-punti -> elementi
        def get_point(name):
            idx = self.kp_map.get(name, None)
            if idx is None or idx >= face_kps.shape[0]:
                return None
            x_f, y_f = face_kps[idx]
            # height, width = self.last_img_shape[:2]  # salva frame.shape quando processi
            # Le coordinate dei keypoints YOLO sono già in pixel (di solito),
            # Se sembra normalizzato, scala ai pixel
            # if 0.0 <= x_f <= 1.0 and 0.0 <= y_f <= 1.0:
            #     x_f *= width
            #     y_f *= height
            return (int(x_f), int(y_f))

        le = get_point('left_eye')
        re = get_point('right_eye')
        nn = get_point('nose')
        ml = get_point('mouth_left')
        mr = get_point('mouth_right')

        if le: facial_elements['left_eye'].append(le)
        if re: facial_elements['right_eye'].append(re)
        if nn: facial_elements['nose'].append(nn)
        # Per “bocca” mettiamo i due angoli se presenti
        for p in [ml, mr]:
            if p: facial_elements['mouth'].append(p)

        return facial_elements

    def extract_facial_elements(self, landmarks, img_shape: Tuple[int, int]) -> Dict:
        """
        PRIMA: leggevamo 468 punti dal FaceMesh e selezionavamo per indice.
        ORA: 'landmarks' è in realtà il results[0] di YOLO.
        """
        r0 = landmarks
        return self.extract_keypoints_dict(r0)

    #definisco una funzione per definire gli elementi facciali
    def draw_facial_elements(self,image:np.ndarray,facial_elements:Dict) -> np.ndarray:
        annotated_image = image.copy()
        #definisco i colori per i diversi elementi
        colors = {
            'left_eye': (0,0,255),
            'right_eye': (0,0,255),
            'nose': (255,0,0),
            'mouth': (0,255,0)
        }

        for element,points in facial_elements.items():
            color1 = colors.get(element,(255,255,255))
            for p in points:
                cv2.circle(annotated_image,p,1,color1,-1)  
                #1 è il raggio del cerchio in pixel
                #-1 significata che il cerchietto è pieno
            # Con soli 5 punti non ha senso chiudere polilinee complesse;
            # per la bocca uniamo i due angoli se entrambi presenti.
            if element == 'mouth' and len(points) == 2:
                pts = np.array(points, np.int32)
                cv2.polylines(annotated_image, [pts], False, color1, 1)

        return annotated_image
    
    #ora prendo i cerchi delle regioni facciali
    def get_facial_region_center(self,facial_elements:Dict) -> Dict:
        centers = {}
        for element, points in facial_elements.items():
            if points:
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                centers[element] = (int(np.mean(xs)), int(np.mean(ys)))
            else:
                centers[element] = None
        return centers
    
    def process_frame(self,frame:np.ndarray) -> Tuple[np.ndarray, Dict, Dict]:  #deve restituire due dict perchè uno è ... (riguardare registrazione)
        lms_list = self.detect_lms(frame)
        #inizializzo un dict vuoto per i lineamenti e i centri
        facial_elements = {}
        centers = {}
        #faccio una copia del frame
        annotated_frame = frame.copy()
        # Disegniamo almeno i bounding box per feedback visivo
        if lms_list:
            r0 = lms_list[0]

            # 1) Disegno box (utile anche se non ho keypoints)
            if r0.boxes is not None and len(r0.boxes) > 0:
                annotated_frame = r0.plot()

            # 2) Provo ad estrarre i keypoints
            facial_elements = self.extract_facial_elements(r0, frame.shape)
            centers = self.get_facial_region_center(facial_elements)

            # 3) Disegno overlay dei (pochi) landmarks sopra al frame annotato dai box
            annotated_frame = self.draw_facial_elements(annotated_frame, facial_elements)

            # 4) Etichette testuali
            for element, center in centers.items():
                if center:
                    cv2.putText(annotated_frame, element.replace('_',' ').title(),
                                (center[0]-30, center[1]-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        return annotated_frame, facial_elements, centers
    
def test_facial_lms():
    detector = Landmarks_detector() #è l'oggetto di lendmarks detector
    cap = cv2.VideoCapture(0) #di solito nei pc è di default '0'
    if not cap.isOpened():
        print('Error: can\'t open the webcam')
        return
    print('Webcam opened. Press "Q" to quit')

    while True:
        ret,frame = cap.read()  #'ret' rappresenta i return delle funzioni
        if not ret:
            print('Error: can\'t read the frame')
            break
        annotated_frame, facial_elements, centers = detector.process_frame(frame)
        info = f'Delected elements: {len([k for k,v in facial_elements.items() if v])}'
        cv2.putText(annotated_frame,info,(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
        cv2.imshow('Facial landmarks detection',annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): #questo è il comando per uscire
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    test_facial_lms()