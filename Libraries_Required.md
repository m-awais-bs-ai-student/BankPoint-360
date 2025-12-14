# Technical Dependencies & Libraries

This document outlines the system requirements and third-party libraries necessary to run **BankPoint 360**. It provides a breakdown of each library's specific role within the codebase.

## 🛠️ Third-Party Libraries (Installation Required)
These libraries must be installed via `pip` for the project to function.

### 1. DeepFace
* **Package Name:** `deepface` (and `tf-keras`)
* **Usage in Code:**
    * **Biometric Verification:** Used in the `withdraw_amount()` function to enforce high-security access.
    * **Face Matching:** Specifically utilizes the `DeepFace.find()` method with the **ArcFace** model and **RetinaFace** backend to compare live webcam input against the `user_database` folder.
    * **Distance Metric:** Uses `'cosine'` similarity to determine if the face is a match (Threshold: `0.55`).

### 2. OpenCV (Computer Vision)
* **Package Name:** `opencv-python`
* **Usage in Code:**
    * **CDM Module:** Powers the Cash Deposit Machine. It captures video frames (`cv2.VideoCapture`) and processes them using **ORB** (Oriented FAST and Rotated BRIEF) feature matching to detect specific currency notes (500, 1000, 5000).
    * **Image Processing:** Applies Gaussian Blur and CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance webcam images for better detection accuracy.
    * **Face Capture:** Used in `live_face_registration()` to detect faces using Haar Cascades (`haarcascade_frontalface_default.xml`) and save them to disk.

### 3. Pandas
* **Package Name:** `pandas`
* **Usage in Code:**
    * **Database Management:** Acts as the primary database engine. It loads `accounts.csv` and `transactions.csv` into DataFrames for fast querying.
    * **Data Persistence:** Handles all row insertions (new accounts), updates (balance changes), and deletions (removing accounts).
    * **History Tracking:** Used to filter and display transaction history for specific `account_ids`.

### 4. NumPy
* **Package Name:** `numpy`
* **Usage in Code:**
    * **Image Arrays:** Serves as the fundamental data structure for OpenCV; every video frame captured in the **CDM Module** is stored and processed as a NumPy array (`ndarray`).
    * **Numerical Backend:** Provides the high-performance array calculations required by both `pandas` (for dataframes) and `deepface` (for vector embeddings).

---

## 🐍 Python Standard Libraries (Built-in)
These libraries are included with Python and do not require separate installation, but play critical roles in the system logic.

### 1. smtplib & email
* **Role:** Email Automation
* **Usage:** Manages the SMTP connection to the email server (Brevo/Gmail) to dispatch One-Time Passwords (OTPs) for user verification during withdrawals.

### 2. hashlib
* **Role:** Cryptography
* **Usage:** Generates SHA-256 hashes of user PINs (`hash_pin` function) to ensure sensitive credentials are not stored in plain text (where applicable).

### 3. msvcrt (Windows Only)
* **Role:** System Input
* **Usage:** Enables the `get_hidden_pin()` function to mask user input with asterisks (`****`) in the console, providing a secure ATM-like typing experience on Windows.

### 4. re (Regular Expressions)
* **Role:** Validation
* **Usage:**
    * **Input Sanitization:** Validates complex email formats.
    * **String Parsing:** Extracts numerical denominations (e.g., "500") from image filenames during the note detection process.

### 5. os & time
* **Role:** System Operations
* **Usage:**
    * **File I/O:** Manages file paths for face images and checks if CSV databases exist.
    * **Stabilization:** `time.sleep()` is used to create stabilization delays during camera initialization and note scanning loops.

---

## 📥 Installation Command

To install all required third-party dependencies at once, run the following command in your terminal:

```bash
pip install pandas opencv-python deepface tf-keras numpy
