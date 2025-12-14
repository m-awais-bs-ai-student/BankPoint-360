# BankPoint 360: AI-Powered ATM & CDM System

[![Language](https://img.shields.io/badge/language-Python_3.x-blue.svg)](https://www.python.org/)
[![AI Engine](https://img.shields.io/badge/AI-DeepFace-orange.svg)](https://github.com/serengil/deepface)
[![Vision](https://img.shields.io/badge/Vision-OpenCV-green.svg)](https://opencv.org/)
[![Data](https://img.shields.io/badge/Data-Pandas-150458.svg)](https://pandas.pydata.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 🏦 Overview
**BankPoint 360** is a comprehensive banking management system that bridges the gap between traditional banking software and modern AI security. Unlike standard console applications, this project implements **real-time biometrics** and **computer vision** to simulate a physical ATM/CDM environment.

It features a **High-Security Withdrawal System** requiring Face Verification + OTP (Email), and a **Smart Cash Deposit Machine** that uses image processing to recognize currency denominations via a webcam.

## 🚀 Key Features

### 🔐 Security & Auth
* **Multi-Factor Withdrawal:** Withdrawals are protected by a 2-step verification process:
    1.  **Biometric Scan:** Real-time Face Recognition using `DeepFace` (ArcFace model).
    2.  **OTP Verification:** Generates and emails a One-Time Password via SMTP.
* **Secure Login:** PIN-based authentication with input masking (compatible with Windows & Linux).
* **Data Hashing:** SHA-256 hashing capability for PIN security.

### 💵 Smart CDM (Cash Deposit Machine)
* **Visual Currency Detection:** Uses **OpenCV** and **ORB Feature Matching** to identify physical currency notes (e.g., 500, 1000, 5000 denominations) via webcam.
* **Real-Time Validation:** Algorithms checks for feature match quality to distinguish between valid notes and clutter.
* **Session Tracking:** Calculates total deposit in real-time during the scanning session.

### 👨‍💼 Admin Dashboard
* **User Management:** Create accounts, Block/Unblock users, and Delete records.
* **Manual Override:** Admin capability to deposit funds manually.
* **Audit Logs:** View detailed transaction histories for any user.
* **Search Engine:** Find accounts by Name, ID, or Email.

### 📂 Core Banking
* **Persistence:** All user data and transaction logs are stored in CSV files (`accounts.csv`, `transactions.csv`) using **Pandas**.
* **Transaction History:** Detailed log of all Deposits, Withdrawals, and Transfers.

## 🛠️ Tech Stack

* **Language:** Python
* **Computer Vision:** OpenCV (`cv2`)
* **AI/Biometrics:** DeepFace (ArcFace / RetinaFace)
* **Data Handling:** Pandas (CSV)
* **Communication:** SMTP (Email Automation)
* **Security:** Hashlib, Getpass/Msvcrt

## ⚙️ Installation & Setup

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-username/BankPoint-360.git](https://github.com/your-username/BankPoint-360.git)
    cd BankPoint-360
    ```

2.  **Install Dependencies:**
    This project requires several heavy libraries for AI processing.
    ```bash
    pip install pandas opencv-python deepface tf-keras
    ```
    *(Note: DeepFace may require TensorFlow/Keras backend)*

3.  **Configure Paths:**
    Open `main.py` and update the `DB_FOLDER_PATH` to a valid directory on your machine where face images will be stored.
    ```python
    DB_FOLDER_PATH = r"C:\Your\Path\To\user_database"
    ```

4.  **Configure SMTP (Optional):**
    To enable OTP emails, update the `send_otp_via_smtp` function with your own SMTP credentials (e.g., Gmail App Password or Brevo).

5.  **Run the Application:**
    ```bash
    python main.py
    ```

## 📸 Usage Guide

1.  **Admin Setup:**
    * Login as Admin (Default User: `admin`, Pass: `123`).
    * Create a new User Account.
    * **Face Registration:** The system will automatically launch the webcam to register the user's face for future security checks.
2.  **User Actions:**
    * Login with the new Account ID and PIN.
    * **Deposit:** Select "CDM" and show a currency note to the camera to test detection.
    * **Withdraw:** Select "Withdraw". The system will scan your face. If it matches the registered image, it will email you an OTP to complete the transaction.

## ⚠️ Disclaimer
This project is for **educational purposes only**. The security implementations (while functional) are designed for demonstration and should not be used for real financial systems without professional auditing.

## 📜 License
This project is open-source and available under the MIT License.
