নিশ্চয়ই, আমি আপনাকে শুরু থেকে শেষ পর্যন্ত সব ধাপ ডিটেইলসে বলে দিচ্ছি। এগুলো অনুসরণ করলে আপনার প্রজেক্ট সফলভাবে শেষ করতে পারবেন। যেহেতু আপনার GPU নেই, তাই CPU-ভিত্তিক সেটআপ এবং ট্রেনিং এর কথা মাথায় রেখেই এই গাইডলাইন দেওয়া হচ্ছে।

---

## **YOLOv3 ট্রেনিং এবং ইনফারেন্স (CPU ব্যবহার করে)**

### **ধাপ ১: প্যাকেজ ইনস্টলেশন এবং পরিবেশ প্রস্তুতি**
1. **প্রয়োজনীয় লাইব্রেরি ইনস্টল করুন:**
   ```python
   !pip install ultralytics opencv-python-headless numpy
   ```

2. **Google Colab-এ ফাইল আপলোড করা:**
   ```python
   from google.colab import files
   uploaded = files.upload()
   ```

3. **ZIP ফাইল আনজিপ করুন:**
   ```python
   import zipfile
   zip_path = "group3.zip"  # আপনার ZIP ফাইলের নাম
   extract_path = "/content/project"
   with zipfile.ZipFile(zip_path, 'r') as zip_ref:
       zip_ref.extractall(extract_path)
   ```

4. **ফাইল কাঠামো যাচাই করুন:**
   আপনার `Images`, `Labels`, এবং `yolo.yaml` ফাইল আছে কিনা তা নিশ্চিত করুন।

---

### **ধাপ ২: ডেটাসেট এবং yolo.yaml আপডেট**
1. **yolo.yaml আপডেট করুন:**
   ```yaml
   train: /content/project/group3/Train/Images
   val: /content/project/group3/Val/Images

   nc: 2
   names: ['gun_single', 'gun_double']
   ```

2. **ফাইল আপডেট ভেরিফাই করুন:**
   ```python
   with open("/content/project/group3/yolo.yaml", "r") as file:
       print(file.read())
   ```

---

### **ধাপ ৩: ডেটাসেট যাচাই**
1. **ইমেজ এবং লেবেল যাচাই করুন:**
   ```python
   import os

   train_images = len([f for f in os.listdir("/content/project/group3/Train/Images") if f.endswith(".jpg")])
   train_labels = len([f for f in os.listdir("/content/project/group3/Train/Labels") if f.endswith(".txt")])
   print(f"Train images: {train_images}")
   print(f"Train labels: {train_labels}")
   ```

2. **মিসিং লেবেল ফাইল চেক করুন:**
   ```python
   image_files = [f.split(".")[0] for f in os.listdir("/content/project/group3/Train/Images") if f.endswith(".jpg")]
   label_files = [f.split(".")[0] for f in os.listdir("/content/project/group3/Train/Labels") if f.endswith(".txt")]

   missing_labels = [img for img in image_files if img not in label_files]
   print("Missing label files:", missing_labels)
   ```

---

### **ধাপ ৪: YOLOv3 মডেল ট্রেনিং**
1. **ট্রেনিং শুরু করুন:**
   ```python
   from ultralytics import YOLO

   model = YOLO("yolov3.pt")  # YOLOv3 প্রি-ট্রেনড ওজন ব্যবহার করুন
   model.train(data="/content/project/group3/yolo.yaml", epochs=10, imgsz=320, batch=2)
   ```

2. **সম্ভাব্য সমস্যাগুলো মোকাবিলা করুন:**
   - যদি কোনো ফাইল মিসিং থাকে, সেগুলো ঠিক করুন।
   - মডেলের `epochs`, `batch` এবং `imgsz` কমিয়ে দিন যদি CPU ব্যবহার করেন।

---

### **ধাপ ৫: মডেল সেভ এবং টেস্ট**
1. **ট্রেনড মডেল সেভ করুন:**
   ```python
   model.save("/content/project/group3/trained_model.pt")
   ```

2. **ইনফারেন্স বা টেস্ট চালান:**
   ```python
   results = model.predict(source="/content/project/group3/Val/Images", save=True)
   ```

3. **ফলাফল দেখুন:**
   ```python
   from IPython.display import Image, display
   display(Image(filename='/content/runs/predict/exp/image1.jpg'))  # আপনার আউটপুট ইমেজ দেখাবে
   ```

---

### **ধাপ ৬: মোশন ডিটেকশন (মোশন স্ক্রিপ্ট যোগ করা)**
1. **মোশন ডিটেকশন কোড যুক্ত করুন (motion.py):**
   ```python
   import cv2

   cap = cv2.VideoCapture(0)
   ret, frame1 = cap.read()
   ret, frame2 = cap.read()

   while cap.isOpened():
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
           cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)

       cv2.imshow("feed", frame1)
       frame1 = frame2
       ret, frame2 = cap.read()

       if cv2.waitKey(10) == 27:
           break

   cap.release()
   cv2.destroyAllWindows()
   ```

---

### **ধাপ ৭: Twilio কল অ্যালার্ট (twilio.py)**

1. **Twilio অ্যাকাউন্ট সেটআপ করুন এবং নিচের কোডটি ব্যবহার করুন:**
   ```python
   from twilio.rest import Client

   # Twilio অ্যাকাউন্ট ক্রেডেনশিয়াল
   account_sid = "your_account_sid"
   auth_token = "your_auth_token"
   client = Client(account_sid, auth_token)

   def make_call():
       call = client.calls.create(
           url='http://demo.twilio.com/docs/voice.xml',
           to='+1234567890',  # আপনার ফোন নাম্বার
           from_='+0987654321'  # Twilio থেকে প্রাপ্ত নাম্বার
       )
       print("Call initiated:", call.sid)
   ```

2. **ডিটেকশন এবং মোশন ডিটেকশন একসাথে লিঙ্ক করুন।**

---

### **ধাপ ৮: প্রজেক্ট ফাইনালাইজ**
1. **ফাইল এবং ফলাফল সংরক্ষণ করুন।**
2. **আপনার ট্রেনড মডেল এবং কোড গিটহাবে আপলোড করুন।**
3. **একটি README.md ফাইল তৈরি করুন যেখানে কোডের ব্যাখ্যা থাকবে।**

---

এটি করলে আপনার পুরো প্রজেক্ট সম্পূর্ণ হয়ে যাবে। যেকোনো সমস্যায় আমি আছি সাহায্যের জন্য! 😊