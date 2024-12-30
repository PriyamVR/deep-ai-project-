আপনার প্রজেক্টের জন্য পুরো গাইডলাইন ধাপে ধাপে নিচে দেওয়া হলো। এটি Google Colab-এ **YOLOv3** দিয়ে **Weapon Detection**, **Motion Detection**, এবং **Twilio Call Alert** চালানোর জন্য উপযোগী করে তৈরি করা হয়েছে।

---

## **ধাপ ১: প্রজেক্টের প্রয়োজনীয় ফাইল সংগ্রহ এবং প্রস্তুতি**

আপনার প্রজেক্টের জন্য প্রয়োজনীয় ফাইলগুলো:
1. **YOLOv3 Configuration Files:**
   - `yolov3_custom_train.cfg`
   - `yolov3_custom_train_final.weights`
   - `yolo.names`
2. **Python কোড ফাইল:**
   - `motion_detection.py` (মোশন ডিটেকশন লজিকের জন্য)
   - `yolo_detection.py` (YOLO মডেলের জন্য)
   - `twilio_call.py` (Twilio কলিং ফাংশনের জন্য)
3. **ডেটাসেট:**
   - ছবি এবং লেবেল ফাইল (ইমেজ ফাইল এবং `.txt` লেবেল ফাইল)।

---

## **ধাপ ২: Google Colab-এ প্রজেক্ট ফাইল আপলোড করা**

### **2.1: প্রজেক্ট ফাইল আপলোড করুন**
আপনার প্রজেক্ট ফাইলগুলো একটি `.zip` ফাইল হিসেবে সংরক্ষণ করুন এবং Colab-এ আপলোড করুন:
```python
from google.colab import files
uploaded = files.upload()  # .zip ফাইলটি আপলোড করুন
```

### **2.2: `.zip` ফাইল আনজিপ করুন**
```bash
!unzip your_project.zip -d /content/project
```

---

## **ধাপ ৩: প্রয়োজনীয় লাইব্রেরি ইনস্টল করা**

Google Colab-এ আপনার প্রজেক্টের জন্য প্রয়োজনীয় সমস্ত লাইব্রেরি ইনস্টল করুন:
```python
!pip install opencv-python-headless numpy twilio
```

---

## **ধাপ ৪: YOLOv3 সেটআপ করা**

### **4.1: Darknet ইনস্টল করুন**
YOLOv3 Darknet ফ্রেমওয়ার্ক ইন্সটল করুন:
```bash
!git clone https://github.com/AlexeyAB/darknet.git
%cd darknet
!make
```

### **4.2: প্রয়োজনীয় ফোল্ডার তৈরি করুন**
```bash
!mkdir -p data/obj
!mkdir backup
```

### **4.3: `.data` এবং `.names` ফাইল তৈরি করুন**
```bash
# .data ফাইল তৈরি
!echo "classes= 1" > data/obj.data
!echo "train  = data/train.txt" >> data/obj.data
!echo "valid  = data/test.txt" >> data/obj.data
!echo "names = data/obj.names" >> data/obj.data
!echo "backup = backup/" >> data/obj.data

# .names ফাইল তৈরি
!echo "weapon" > data/obj.names
```

---

## **ধাপ ৫: YOLOv3 মডেল ট্রেন বা ইনফারেন্স (ডিটেকশন)**

### **5.1: YOLOv3 মডেল ট্রেনিং ডেটা প্রস্তুত করুন**
আপনার ইমেজ এবং লেবেল ফাইল `data/obj` ফোল্ডারে আপলোড করুন। ট্রেনিং ডেটার পাথ লিস্ট তৈরি করুন:
```bash
!find $(pwd)/data/obj -name "*.jpg" > data/train.txt
```

### **5.2: YOLOv3 মডেল ডিটেকশন চালান**
Python কোড দিয়ে YOLOv3 মডেল চালান:
```python
import cv2
import numpy as np

# Load YOLO model
net = cv2.dnn.readNet("yolov3_custom_train_final.weights", "yolov3_custom_train.cfg")
classes = []
with open("yolo.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load image
image = cv2.imread("/content/project/test_image.jpg")
height, width, _ = image.shape

# Prepare input blob
blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)

# Run inference
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
outs = net.forward(output_layers)

# Process detections
class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x, center_y, w, h = (
                int(detection[0] * width),
                int(detection[1] * height),
                int(detection[2] * width),
                int(detection[3] * height),
            )
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Draw bounding boxes
for i in range(len(boxes)):
    x, y, w, h = boxes[i]
    label = str(classes[class_ids[i]])
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display image
from google.colab.patches import cv2_imshow
cv2_imshow(image)
```

---

## **ধাপ ৬: মোশন ডিটেকশন যুক্ত করা**

Python স্ক্রিপ্টের মাধ্যমে **মোশন ডিটেকশন** যোগ করুন:
```python
# Initialize camera
cap = cv2.VideoCapture(0)
ret, frame1 = cap.read()
ret, frame2 = cap.read()

while True:
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 5000:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("feed", frame1)
    frame1 = frame2
    ret, frame2 = cap.read()

    if cv2.waitKey(10) == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

---

## **ধাপ ৭: Twilio Call Alert যুক্ত করা**

Twilio কল ফাংশনটি নিম্নরূপ যুক্ত করুন:
```python
from twilio.rest import Client

# Twilio credentials
account_sid = "your_account_sid"
auth_token = "your_auth_token"

def make_call():
    client = Client(account_sid, auth_token)
    call = client.calls.create(
        url='http://demo.twilio.com/docs/voice.xml',
        to='+919876543456',  # ভেরিফায়েড নম্বর
        from_='+19548803109'  # Twilio Virtual Number
    )
    print("Call initiated:", call.sid)

# Call this function when motion or weapon is detected
make_call()
```

---

## **ধাপ ৮: চূড়ান্ত সংমিশ্রণ**

1. **মোশন ডিটেকশন** চালানোর সময় কোনো মোশন ডিটেক্ট হলে YOLOv3 চালান।
2. **Twilio Call Alert** চালান যখন কোনো অস্ত্র শনাক্ত হয়।

---

### যদি কোনো ধাপ বা ফাংশনে সমস্যা হয়, জানাতে দ্বিধা করবেন না। 😊