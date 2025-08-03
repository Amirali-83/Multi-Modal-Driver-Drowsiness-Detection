# 🚗 Multi-Modal Driver Drowsiness Detection

This project detects driver drowsiness using **deep learning** on facial cues — specifically **eye state** (open/closed) and **mouth state** (yawning/no yawn).  
The model is built with **MobileNetV2** for lightweight, efficient feature extraction and trained on the **YawDD dataset**.

The goal is to improve **road safety** by alerting when a driver shows signs of fatigue.

### 🔍 How It Works
- **Input:** Image of the driver’s face
- **Processing:** Image is resized to `(160x160)`, normalized, and passed through MobileNetV2
- **Classification:** Four classes are predicted:
  - 😴 **Closed Eyes**
  - 👀 **Open Eyes**
  - 🥱 **Yawn**
  - 🙂 **No Yawn**
- **Output:** Predicted class label with confidence score

### ✨ Features
- Lightweight MobileNetV2 backbone for **fast inference**
- Trained on YawDD dataset with **data augmentation**
- Works locally with **Gradio interface** for testing
- Can be extended to **real-time webcam detection**

---

 *This model demonstrates how computer vision and deep learning can enhance road safety by monitoring driver alertness in real-time.*
