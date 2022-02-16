import cv2
import argparse
import numpy as np
from matplotlib import pyplot as plt

#tambahkan letak alamat file hasil training (.cfg, .wight, .names)
ap = argparse.ArgumentParser()
ap.add_argument('-c', '--config', 
                help = 'path to yolo config file', default='E:\Backkup mbak ira\YOLO\yolo_vip1.cfg')
ap.add_argument('-w', '--weights', 
                help = 'path to yolo pre-trained weights', default='E:\Backkup mbak ira\YOLO\yolo_5000.weights')
ap.add_argument('-cl', '--classes', 
                help = 'path to text file containing class names',default='E:\Backkup mbak ira\YOLO\obj (2).names')
args = ap.parse_args()


# Mendapat nama output layer, output untuk YOLOv3 yaitu ['yolo_16', 'yolo_23']
def getOutputsNames(net):
    layersNames = net.getLayerNames()
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Menggambar bounding box di sekeliling object dan menunjukkan nama class yang bersangkutan
def draw_pred(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    #kondisi yang menunjukkan posisi dosen atau mahasiswa
    if((x > 150 and x < 250) and y < 75 ):
        label1 = "Dosen"
        label = "{}: {:.2f}% ({},{})".format(label1, confidence * 100, x, y)
        print("[INFO] {}".format(label))
        cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    else:
        label2 = "Mahasiswa"
        label = "{}: {:.2f}% ({},{})".format(label2, confidence * 100, x, y)
        print("[INFO] {}".format(label))
        cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

# Mendeklarasikan judul program python  
window_title= "People Detector"   
cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)


# Memuat nama class
classes = None
with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]
print(classes)

#Menghasilkan warna untuk berbagai class secara acak
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

#Konfigurasi file dan memuat file dengan ekstensi .weights 
net = cv2.dnn.readNet(args.weights,args.config)


#Mendefinisikan variabel image untuk pembacaan gambar
image = cv2.imread('00000003575000000 001.jpg')
image = cv2.resize(image, (620, 480)) 
    
blob = cv2.dnn.blobFromImage(image, 1.0/255.0, (416,416), [0,0,0], True, crop=False)
Width = image.shape[1]
Height = image.shape[0]
net.setInput(blob)
    
outs = net.forward(getOutputsNames(net))
    
class_ids = []
confidences = []
boxes = []
conf_threshold = 0.5
nms_threshold = 0.4


#Mengaplikasikan algoritma pada bounding box
while cv2.waitKey(1) < 0:
    
    for out in outs: 
        for detection in out:
            
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_pred(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))

    #menampilkan hasil pengolahan citra menggunakan library open cv
    cv2.imshow(window_title, image)

    plt.subplot(1,1,1)
    #menampilkan hasil pengolahan citra dengan plot
    plt.imshow(image)
    plt.show()
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cv2.destroyAllWindows()
