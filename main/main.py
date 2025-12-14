# ==============================
# 💡 DEPENDENCIES 💡
# ==============================
import smtplib  # For OTP and Emails
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pandas as pd  # For CSV/DataFrames
import random
import time
import os
from datetime import datetime
import getpass
import hashlib  # For PIN Hashing
import re  # For Email Validation
import cv2  # For Camera/Face/CDM Operations
from deepface import DeepFace  # For Face Recognition/Verification

# -------------------------------------------------------------
# ⚠️ WINDOWS ONLY DEPENDENCY FOR PIN MASKING ⚠️
# Used for masking PIN input
try:
    import msvcrt

    IS_WINDOWS = True
except ImportError:
    IS_WINDOWS = False
# -------------------------------------------------------------

# =============================================================
# ⚙️ GLOBAL CONFIGURATION ⚙️
# =============================================================
ACCOUNTS_FILE = "accounts.csv"
TRANSACTIONS_FILE = "transactions.csv"

# 🌟 FACE/NOTE RECOGNITION CONFIGURATION 🌟
# IMPORTANT: Update this path to your actual database location!
DB_FOLDER_PATH = r"C:\Users\muham\PycharmProjects\PythonProject\CDM_Working\user_database"
db_folder = DB_FOLDER_PATH  # Alias for CDM/Face functions

# DeepFace Configuration (used for high-security withdrawal check)
MODEL_NAME = "ArcFace"
DETECTOR_BACKEND = "retinaface"
METRIC = "cosine"
# Face recognition distance threshold (higher value = more forgiving match)
STRICT_THRESHOLD = 0.55

# --- CDM CASH DETECTION CONFIGURATION (From chota_hasnain.txt) ---
PATH = r'C:\Users\muham\PycharmProjects\PythonProject\CDM_Working\CDM_Working\Notes'
MIN_MATCH_THRESHOLD = 18
ROI_X = 200
ROI_Y = 100
ROI_WIDTH = 400
ROI_HEIGHT = 200
MIN_DENOMINATION = 500

# -------------------------------------------------------------
# 💾 GLOBAL DATABASES & MANAGEMENT 💾
# -------------------------------------------------------------
accounts_df = pd.DataFrame()
transactions_df = pd.DataFrame()

def initialize_csv_files():
    """Initializes the required CSV files if they don't exist."""
    global accounts_df
    if not os.path.exists(ACCOUNTS_FILE):
        df = pd.DataFrame(columns=[
            "account_id", "full_name", "father_name", "cnic", "contact_no", "email_address",
            "dob", "user_name", "address", "password", "balance", "status", "creation_date",
            "face_image_path"  # <-- ADDED COLUMN for face image path
        ], dtype=str)

        df.to_csv(ACCOUNTS_FILE, index=False)
        print("[System] Initial EMPTY 'accounts.csv' created.")

    if not os.path.exists(TRANSACTIONS_FILE):
        df = pd.DataFrame(columns=["account_id", "type", "amount", "date"], dtype=str)
        df.to_csv(TRANSACTIONS_FILE, index=False)


def load_databases():
    """Loads accounts and transactions data into global DataFrames."""
    global accounts_df, transactions_df
    try:
        # Load all columns as string first, then convert balance
        accounts_df = pd.read_csv(ACCOUNTS_FILE, dtype=str)
        transactions_df = pd.read_csv(TRANSACTIONS_FILE, dtype=str)
        if not accounts_df.empty:
            # Ensure balance is float for calculations
            accounts_df["balance"] = pd.to_numeric(accounts_df["balance"], errors='coerce').fillna(0.0).astype(float)
        print("[System] Databases loaded successfully.")
    except Exception as e:
        # Re-initialize if load fails (e.g., first run)
        print(f"[Warning] Could not load databases ({e}). Re-initializing.")
        initialize_csv_files()
        load_databases()  # Recursive call to load now that they are initialized


def save_accounts():
    """Saves the global accounts DataFrame to the CSV file."""
    accounts_df.to_csv(ACCOUNTS_FILE, index=False)


def save_transactions():
    """Saves the global transactions DataFrame to the CSV file."""
    transactions_df.to_csv(TRANSACTIONS_FILE, index=False)

# =============================================================
# 🔑 SECURITY & UTILITY FUNCTIONS 🔑
# =============================================================

def hash_pin(pin):
    """Hashes the PIN using SHA-256."""
    return hashlib.sha256(pin.encode()).hexdigest()


def get_hidden_pin(prompt="Enter your PIN: "):
    """Handles PIN input with masking (Windows compatible)."""
    if IS_WINDOWS:
        print(prompt, end="", flush=True)
        pin = ""
        while True:
            char = msvcrt.getch()
            if char == b'\r':  # Enter key
                print()
                break
            elif char == b'\x08':  # Backspace key
                if pin:
                    pin = pin[:-1]
                    print("\b \b", end="", flush=True)
            elif char.isdigit() and len(pin) < 4:
                pin += char.decode()
                print("*", end="", flush=True)
        return pin
    else:
        # Use getpass for non-Windows masking
        return getpass.getpass(prompt)


def validate_pin(pin):
    """Checks if PIN is 4 digits."""
    return pin.isdigit() and len(pin) == 4


def validate_name(name):
    """Validates name: must contain only letters and spaces."""
    if name.replace(" ", "").isalpha():
        return True
    else:
        print("Error: Name must contain only letters.")
        return False


def validate_cnic(cnic):
    """Validates CNIC: must be 13 digits and not already in accounts.csv."""

    # --- Check if CNIC already exists ---
    try:
        # Only check if file exists
        if os.path.exists(ACCOUNTS_FILE):
            # Read the file and check for CNIC
            with open(ACCOUNTS_FILE, "r") as f:
                for line in f:
                    if cnic in line:
                        print("Account already exists.")
                        return "menu"   # go back to main menu
    except Exception as e:
        print(f"CNIC validation warning: {e}")

    # --- Validate CNIC format ---
    if cnic.isdigit() and len(cnic) == 13:
        return True
    else:
        print("Error: CNIC must be 13 numeric digits.")
        return False



def validate_contact(contact):
    """Validates Contact: must be 11 numeric digits."""
    if contact.isdigit() and len(contact) == 11:
        return True
    else:
        print("Error: Contact number must be 11 digits.")
        return False


def validate_dob(dob):
    """Validates Date of Birth format."""
    try:
        datetime.strptime(dob, "%Y-%m-%d")
        return True
    except:
        print("Error: Date of Birth must be in YYYY-MM-DD format and valid.")
        return False


def validate_address(address):
    """Validates address length."""
    if 0 < len(address) <= 50:
        return True
    else:
        print("Error: Address should not exceed 50 characters.")
        return False


def validate_email(email):
    """Validate email format with explicit checks."""
    try:
        # Check if email is not empty
        if not email or not isinstance(email, str):
            print("Error: Email cannot be empty")
            return False

        # Check for only allowed characters
        allowed_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789@.-_")
        if not all(c in allowed_chars for c in email):
            print("Error: Email can only contain letters, numbers, @, ., - and _")
            return False

        # Check for exactly one @ symbol
        if email.count('@') != 1:
            print("Error: Email must contain exactly one @ symbol")
            return False

        # Split into local part and domain
        try:
            local_part, domain = email.split('@')
        except ValueError:
            return False

        # Check local part
        if not local_part:
            print("Error: Email local part cannot be empty")
            return False

        if len(local_part) > 64:
            print("Error: Email local part too long (max 64 characters)")
            return False

        # Check domain
        if not domain:
            print("Error: Email domain cannot be empty")
            return False

        if '.' not in domain:
            print("Error: Email domain must contain a dot ('.')")
            return False

        # Check domain parts
        domain_parts = domain.split('.')
        if len(domain_parts) < 2:
            print("Error: Invalid domain format")
            return False

        for part in domain_parts:
            if not part:
                print("Error: Domain part cannot be empty (e.g., '.com' or 'google.' are invalid)")
                return False

        return True

    except Exception as e:
        print(f"Error validating email: {str(e)}")
        return False


def verify_pin(account_id, entered_pin):
    """
    Verifies PIN against stored password WITHOUT automatically hashing.
    """
    global accounts_df
    try:
        user_row = accounts_df[accounts_df["account_id"] == account_id]
        if user_row.empty:
            return None, None, None

        user_data = user_row.iloc[0]
        stored_password = user_data["password"]

        # 1. Check if the stored password is a hash (SHA-256 is 64 chars)
        if len(stored_password) == 64 and all(c in '0123456789abcdef' for c in stored_password.lower()):
            # Stored password is hashed: Compare input hash to stored hash
            entered_pin_hashed = hash_pin(entered_pin)
            if stored_password == entered_pin_hashed:
                if user_data["status"] != "active":
                    return None, None, None
                return user_data["account_id"], user_data["full_name"], user_data["email_address"]
            else:
                return None, None, None

        # 2. Stored password is CLEAR TEXT
        elif stored_password == entered_pin:
            # Match found on clear PIN - BUT DON'T HASH AND SAVE IT
            if user_data["status"] != "active":
                return None, None, None
            return user_data["account_id"], user_data["full_name"], user_data["email_address"]

        return None, None, None
    except Exception as e:
        print(f"Error during verification: {e}")
        return None, None, None

def send_otp_via_smtp(receiver_email):
    """
    Sends a one-time password via SMTP using the provided credentials.
    """
    otp = str(random.randint(100000, 999999))

    # 🔑 USER-PROVIDED CREDENTIALS 🔑
    smtp_server = "smtp-relay.brevo.com"
    smtp_port = 587
    smtp_username = "9d6b3d001@smtp-brevo.com"
    smtp_password = "bskkrBXPovPEcp8"
    sender_email = "bahrianbank@gmail.com"

    try:
        msg = MIMEMultipart()
        msg["Subject"] = "OTP for Bahrian Bank Transaction"
        msg["From"] = sender_email
        msg["To"] = receiver_email.strip()
        text = f"Your One-Time Password (OTP) is: {otp}\nThis OTP will expire in 1 minute."
        msg.attach(MIMEText(text, "plain"))

        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()

        # 🚀 ACTIVE LOGIN AND SEND 🚀
        server.login(smtp_username, smtp_password)
        server.sendmail(sender_email, receiver_email, msg.as_string())

        server.quit()
        print(f"\n✅ OTP sent successfully to {receiver_email}. Please check your inbox.")
        return otp

    except Exception as e:
        print("\n❌ [SMTP Error] Failed to send OTP.")
        print(f"   Reason: Check network connection or verify the SMTP credentials provided.")
        print(f"   Original Error: {e}")
        # Return OTP for simulation/testing purposes if sending fails
        return otp

    # =============================================================


# 📸 FACE RECOGNITION FUNCTIONS 📸
# =============================================================

def live_face_registration(user_id, user_name, other_data=None):
    # Ensure the database folder exists
    if not os.path.exists(db_folder):
        os.makedirs(db_folder)

    """
    Captures face from camera, saves image, and logs details.
    :param user_id: The ID you already collected
    :param user_name: The Name you already collected
    :param other_data: Any extra info (email, phone, etc.)
    :return: The full file path of the saved image if successful, otherwise None.
    """

    # 1. Initialize Camera
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    print(f"\nStarting Face Capture for: {user_name} (ID: {user_id})")
    print("Please face the camera. Press 's' to Capture and Save.")

    captured = False
    image_save_path = None # <-- Will store the path if successful

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(1)    #1 for stabalization
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Using 1.1, 4 from chota_hasnain.txt
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # Draw Green Box
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Signup - Capture Face", frame)
        key = cv2.waitKey(1) & 0xFF

        # 2. Capture Logic
        if key == ord('s'):
            if len(faces) > 0:
                # Crop the face
                (x, y, w, h) = faces[0]
                face_roi = frame[y:y + h, x:x + w]

                # --- A. SAVE IMAGE TO DATABASE FOLDER ---
                # Naming convention: ID_Name.jpg
                filename = f"{user_id}_{user_name.replace(' ', '_')}.jpg"
                save_path = os.path.join(db_folder, filename)
                cv2.imwrite(save_path, face_roi)
                print(f"📸 Image Saved: {save_path}")

                # --- B. SAVE DATA TO CSV --- (Path saved by caller)
                print("✅ Face registration image captured.")

                captured = True
                image_save_path = save_path # <-- Store the path
                break  # Stop loop after success
            else:
                print("❌ No face detected! Try again.")

        elif key == ord('q'):
            print("Capture cancelled.")
            break

    cap.release()
    cv2.destroyAllWindows()
    # Return the path if captured, otherwise None
    return image_save_path if captured else None


def perform_face_recognition(verified_user_name):
    """Handles face authentication for transactions using DeepFace."""
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # Initialize face cascade for drawing the box
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    print("\n" + "=" * 60)
    print("       STEP 1: BIOMETRIC AUTHENTICATION")
    print("=" * 60)

    is_matched = False
    max_face_attempts = 3
    face_attempts = 0

    # Print instruction only ONCE before the camera loop starts
    print("Please look at the camera and press 's' to capture and search.")

    while face_attempts < max_face_attempts:
        ret, frame = cap.read()
        if not ret:
            time.sleep(1)
            continue

        # Continuous face detection and drawing for user focus
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # Draw Green Box around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display camera frame
        cv2.imshow("Webcam - Face Authentication", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            face_attempts += 1
            try:
                # Actual DeepFace Search call (uses a more robust detector like retinaface)
                dfs = DeepFace.find(
                    img_path=frame, db_path=DB_FOLDER_PATH, model_name=MODEL_NAME,
                    detector_backend=DETECTOR_BACKEND, distance_metric=METRIC,
                    enforce_detection=True, silent=True
                )

                # DeepFace returns a list of DataFrames for matches
                if len(dfs) > 0 and not dfs[0].empty:
                    best_match = dfs[0].iloc[0]
                    distance = best_match['distance']

                    # Normalizing names for comparison (e.g., "Uzair Faraz" -> "uzair_faraz")
                    clean_verified_user_name = verified_user_name.lower().replace(" ", "_")
                    matched_file_name = os.path.basename(best_match['identity']).lower()

                    # Match check: Distance must be below threshold AND the matched file name must contain the user's name
                    if distance <= STRICT_THRESHOLD and clean_verified_user_name in matched_file_name:
                        print(f"\n✅ FACE CONFIRMED MATCH for {verified_user_name} (Distance: {distance:.4f})")
                        is_matched = True
                        break
                    else:
                        print(
                            f"\n❌ FACE MATCH REJECTED: Low accuracy or wrong user detected. (Distance: {distance:.4f})")
                else:
                    print(f"\n❌ Face NOT found in database.")

            except Exception as e:
                print(f"Error/No face detected: {e}")

            if not is_matched and face_attempts < max_face_attempts:
                print(f"Retrying capture (Attempt {face_attempts}/{max_face_attempts}). Press 's' again.")


        elif key == ord('q'):
            print("Face authentication cancelled by user.")
            break

    cap.release()
    cv2.destroyAllWindows()
    return is_matched


# =============================================================
# 💵 CDM DEPOSIT MODULE (FULL NOTE DETECTION) 💵
# =============================================================

# --- CDM UTILITIES ---
def enhance_image(img):
    """Enhance image for feature detection."""
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(blur)


def is_valid_note(detected_name, match_count, all_matches):
    """Checks if the note detection is valid based on thresholds and rules."""
    if detected_name is None or match_count < MIN_MATCH_THRESHOLD:
        return False, 0

    # Extract denomination from name
    numbers = re.findall(r'\d+', detected_name)
    if not numbers:
        return False, 0

    denomination = int(numbers[0])

    # Check 1: Denomination must be >= 500
    if denomination < MIN_DENOMINATION:
        return False, 0

    # Check 2: Must have clear lead over other matches
    if len(all_matches) > 1:
        sorted_matches = sorted(all_matches, key=lambda x: x[0], reverse=True)
        top_match = sorted_matches[0]
        second_match = sorted_matches[1] if len(sorted_matches) > 1 else (0, "")

        if top_match[0] - second_match[0] < 8:  # Arbitrary confidence gap
            return False, denomination

        if second_match[0] > 0 and top_match[0] / second_match[0] < 2.0:
            return False, denomination

    # For very high denomination notes (5000), need even more matches
    if denomination == 5000 and match_count < 25:
        return False, denomination

    return True, denomination


def load_reference_images():
    """Loads reference note images for ORB feature matching."""
    orb = cv2.ORB_create(nfeatures=1500)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    class_names = []
    ref_descriptors = []

    reference_images = ['500.jpg', '500_bs.jpg', '1000.jpg', '1000_bs.jpg', '5000.jpg', '5000_bs.jpg']

    print("Loading reference images...")
    for img_name in reference_images:
        img_path = os.path.join(PATH, img_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"  ✗ Could not load: {img_name} from {PATH}")
            continue

        img = cv2.resize(img, (ROI_WIDTH, ROI_HEIGHT))
        enhanced = enhance_image(img)
        kp, des = orb.detectAndCompute(enhanced, None)

        if des is not None:
            class_names.append(img_name.replace('.jpg', ''))
            ref_descriptors.append(des)
            print(f"  ✓ Loaded: {img_name}")

    print(f"\nLoaded {len(class_names)} images")
    return orb, bf, class_names, ref_descriptors


# --- Deposit Function ---
def deposit(acc_id):
    """
    Handles the cash deposit process using the detailed CDM note detection logic.
    """
    global accounts_df, transactions_df

    try:
        # Start physical cash deposit process
        print("\n" + "=" * 60)
        print("CASH DEPOSIT PROCESS")
        print("=" * 60)
        print("Please scan your notes through the camera.")
        print("Press 's' to scan each note")
        print("Press 'q' when all notes have been scanned")
        input("\nPress any key when ready to start scanning...")

        # Load reference images
        orb, bf, class_names, ref_descriptors = load_reference_images()

        # Scanning process
        print("\nReady to scan notes. Place one note at a time in front of camera.")
        print("Camera will open shortly...")

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Cannot open camera. Please deposit manually.")
            try:
                amount = float(input("Enter total deposit amount manually: "))
            except:
                print("Invalid manual amount.")
                return
            scanned_amount = amount
        else:
            scanned_amount = 0
            scanned_notes = []
            last_detections = []
            MAX_HISTORY = 3

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Get ROI
                roi = frame[ROI_Y:ROI_Y + ROI_HEIGHT, ROI_X:ROI_X + ROI_WIDTH]
                roi_resized = cv2.resize(roi, (ROI_WIDTH, ROI_HEIGHT))
                roi_enhanced = enhance_image(roi_resized)

                # Draw ROI box
                color = (0, 0, 255)  # Default Red
                status_text = "NO NOTE"

                # Detect features
                kp2, des2 = orb.detectAndCompute(roi_enhanced, None)
                detected_note = None
                best_matches = 0
                all_matches = []

                if des2 is not None:
                    for i, des1 in enumerate(ref_descriptors):
                        if des1 is None: continue

                        # Match features using k-Nearest Neighbors
                        matches = bf.knnMatch(des1, des2, k=2)
                        good = []

                        # Ratio Test
                        for matches_pair in matches:
                            if len(matches_pair) < 2:
                                continue
                            m, n = matches_pair
                            if m.distance < 0.75 * n.distance:
                                good.append([m])

                        match_count = len(good)
                        all_matches.append((match_count, class_names[i]))

                        if match_count > best_matches:
                            best_matches = match_count
                            detected_note = class_names[i]

                # Validate the detection
                is_valid, denomination = is_valid_note(detected_note, best_matches, all_matches)

                # Addition: Status and Color based on validation
                current_status = ""

                if is_valid:
                    # Valid high denomination note
                    if "_bs" in detected_note:
                        side = " (Back)"
                        note_name = detected_note.replace("_bs", "")
                    else:
                        side = " (Front)"
                        note_name = detected_note

                    current_status = f"DETECTED: {note_name}{side}"
                    color = (0, 255, 0)  # Green

                elif detected_note and best_matches >= MIN_MATCH_THRESHOLD:
                    # Detected but denomination too low or low confidence
                    numbers = re.findall(r'\d+', detected_note)
                    if numbers:
                        denom = int(numbers[0])
                        if denom < MIN_DENOMINATION:
                            current_status = f"SMALL NOTE: {denom}"
                            color = (0, 165, 255)  # Orange
                        else:
                            current_status = "LOW CONFIDENCE"
                            color = (255, 255, 0)  # Yellow
                    else:
                        current_status = "UNKNOWN"
                        color = (255, 0, 0)  # Blue
                else:
                    current_status = "NO NOTE"
                    color = (0, 0, 255)  # Red

                # Stabilization logic
                last_detections.append(current_status)
                if len(last_detections) > MAX_HISTORY:
                    last_detections.pop(0)

                # Only show "DETECTED" if it appears in most recent frames
                if "DETECTED" in current_status and len(last_detections) == MAX_HISTORY:
                    detected_count = last_detections.count(current_status)
                    if detected_count < 2:  # Need at least 2 out of 3 frames
                        # Downgrade to "LOW CONFIDENCE"
                        current_status = "LOW CONFIDENCE"
                        color = (255, 255, 0)  # Yellow

                status_text = current_status

                # Draw on frame
                cv2.rectangle(frame, (ROI_X, ROI_Y),
                              (ROI_X + ROI_WIDTH, ROI_Y + ROI_HEIGHT),
                              color, 2)
                cv2.putText(frame, status_text, (ROI_X, ROI_Y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                # Show instructions
                cv2.putText(frame, "Press 's' to SCAN | 'q' to FINISH", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Show scanned amount
                cv2.putText(frame, f"Session Total: Rs. {scanned_amount:,}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Show match info for debugging
                if all_matches and len(all_matches) > 0:
                    sorted_matches = sorted(all_matches, key=lambda x: x[0], reverse=True)
                    y_offset = ROI_Y + ROI_HEIGHT + 60

                    for i, (count, name) in enumerate(sorted_matches[:2]):
                        if count > 0:
                            text = f"{name}: {count}"
                            cv2.putText(frame, text, (ROI_X, y_offset),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)
                            y_offset += 20

                cv2.imshow('Cash Deposit - Press s to scan', frame)

                key = cv2.waitKey(1) & 0xFF

                if key == ord('s'):
                    # Re-check validity just before capture
                    is_valid, denomination = is_valid_note(detected_note, best_matches, all_matches)

                    if is_valid and denomination > 0:
                        scanned_amount += denomination
                        scanned_notes.append(denomination)
                        print(f"\n✓ NOTE CAPTURED: Rs. {denomination}")
                        print(f"  Session total: Rs. {scanned_amount:,}")
                        time.sleep(0.5)
                    else:
                        print("\n✗ Cannot capture - No valid note detected!")
                        time.sleep(0.5)

                elif key == ord('q'):
                    print("\nClosing camera...")
                    break

            cap.release()
            cv2.destroyAllWindows()

            if scanned_amount == 0:
                print("\nNo notes were scanned. Deposit cancelled!")
                return

            print(f"\nTotal deposit amount: Rs. {scanned_amount:,}")
            amount = scanned_amount

        # Update account balance with scanned amount
        idx = accounts_df[accounts_df["account_id"] == acc_id].index[0]
        accounts_df.loc[idx, "balance"] += amount

        # Add transaction record
        transactions_df = pd.concat([transactions_df, pd.DataFrame([{
            "account_id": acc_id,
            "type": "Deposit",
            "amount": amount,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }])], ignore_index=True)

        save_accounts()
        save_transactions()
        print(f"\n✅ Deposit Successful! Amount: Rs. {amount:,}")

        # Show updated balance
        bal = accounts_df.loc[accounts_df["account_id"] == acc_id, "balance"].iloc[0]
        print(f"Updated Balance: Rs. {bal}")

    except Exception as e:
        print(f"Error during deposit: {e}")


# =============================================================
# 👤 USER ACCOUNT & TRANSACTION FUNCTIONS 👤
# =============================================================

def welcome_message(name, acc_id):
    """Displays a welcome message."""
    print("\n" + "=" * 50)
    print(f"Welcome back, {name} (Account ID: {acc_id})!")
    print("=" * 50)


def user_login():
    """Handles Account ID + PIN login."""
    global accounts_df
    max_attempts = 3
    attempts = 0
    while attempts < max_attempts:
        print("\n--- User Login ---")
        acc_id = input("Enter Account ID: ").strip()
        if acc_id not in accounts_df["account_id"].values:
            print("Account not found.")
            attempts += 1
            remaining = max_attempts - attempts
            if remaining > 0:
                print(f"Attempts remaining: {remaining}\n")
            else:
                print("Maximum attempts exceeded. Access denied.\n")
            continue

        # PIN input is masked for login security
        pin_input = get_hidden_pin("Enter 4-digit PIN: ").strip()
        if not validate_pin(pin_input):
            print("PIN must be 4 digits.")
            attempts += 1
            remaining = max_attempts - attempts
            if remaining > 0:
                print(f"Attempts remaining: {remaining}\n")
            else:
                print("Maximum attempts exceeded. Access denied.\n")
            continue

        acc_id, full_name, email_address = verify_pin(acc_id, pin_input)

        if acc_id:
            welcome_message(full_name, acc_id)
            return acc_id
        else:
            attempts += 1
            remaining = max_attempts - attempts
            if remaining > 0:
                print(f"Incorrect Account ID or PIN. Attempts remaining: {remaining}\n")
            else:
                print("Maximum attempts exceeded. Access denied.\n")
                return None
    return None


def withdraw_amount(acc_id):
    """ Handles withdrawal. **Requires Face Recognition + OTP Verification**. """
    global accounts_df, transactions_df
    user_row_df = accounts_df[accounts_df["account_id"] == acc_id]
    user_data = user_row_df.iloc[0]
    full_name = user_data["full_name"]
    email_address = user_data["email_address"].strip()
    status = user_data["status"]

    if status != "active":
        print("Blocked accounts cannot withdraw money!")
        return

    print("\n--- INITIATING HIGH-SECURITY WITHDRAWAL CHECK ---")

    # 1. Face Recognition
    if not perform_face_recognition(full_name):
        print("\nBIOMETRIC AUTHENTICATION FAILED. Withdrawal denied.")
        return

    # 2. OTP Verification
    print("\nSTEP 2: OTP Verification")
    otp = send_otp_via_smtp(email_address)
    if not otp:
        print("Failed to generate or send OTP. Withdrawal denied.")
        return

    user_otp = input("Enter the OTP you received: ")
    if user_otp != otp:
        print("Incorrect OTP. Withdrawal denied.")
        return

    print("High-security verification passed. Proceeding with withdrawal.")

    # --- CORE WITHDRAWAL LOGIC ---
    try:
        amount = float(input("Enter amount to withdraw: $"))
    except ValueError:
        print("Invalid amount.")
        return

    idx = accounts_df[accounts_df["account_id"] == acc_id].index[0]
    current_balance = accounts_df.loc[idx, "balance"]

    if current_balance >= amount and amount > 0:
        new_balance = current_balance - amount
        accounts_df.loc[idx, "balance"] = new_balance

        transaction_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_transaction = {"account_id": acc_id, "type": "Withdrawal", "amount": str(amount), "date": transaction_date}
        global transactions_df
        transactions_df = pd.concat([transactions_df, pd.DataFrame([new_transaction])], ignore_index=True)

        save_accounts()
        save_transactions()
        print(f"Successfully withdrawn ${amount:.2f}. Remaining Balance: ${new_balance:.2f}\n")
    else:
        print(f"Insufficient balance or invalid amount.\n")


def check_balance(acc_id):
    bal = accounts_df.loc[accounts_df["account_id"] == acc_id, "balance"].iloc[0]
    print(f"\nYour current balance is: ${bal:.2f}")


def change_pin(acc_id):
    """
    Change PIN function - STORE AS CLEAR TEXT as requested.
    """
    global accounts_df
    print("\n--- CHANGE PIN (PIN INPUT IS VISIBLE) ---")

    # 1. Verify current PIN (UNMASKED)
    current_pin_input = input("Enter Current 4-digit PIN: ").strip()
    if not validate_pin(current_pin_input):
        print("Invalid PIN format. Must be 4 digits.")
        return

    # Use simplified verification (not the verify_pin that hashes)
    user_row = accounts_df[accounts_df["account_id"] == acc_id]
    if user_row.empty:
        print("Account not found!")
        return

    user_data = user_row.iloc[0]
    stored_password = user_data["password"]

    # Check if stored is hash or clear
    if len(stored_password) == 64 and all(c in '0123456789abcdef' for c in stored_password.lower()):
        # It's a hash - compare hashed input
        entered_pin_hashed = hash_pin(current_pin_input)
        if stored_password != entered_pin_hashed:
            print("Incorrect current PIN. Change request denied.")
            return
    elif stored_password != current_pin_input:
        # It's clear text - compare directly
        print("Incorrect current PIN. Change request denied.")
        return

    # 2. Get and validate new PIN TWICE (UNMASKED)
    while True:
        new_pin = input("Enter New 4-digit PIN: ").strip()
        if not validate_pin(new_pin):
            print("New PIN must be 4 digits.")
            continue

        confirm_pin = input("Confirm New 4-digit PIN: ").strip()

        if new_pin == confirm_pin:
            break
        print("New PIN and confirmation PIN do not match. Try again.")

    # 3. Store the CLEAR TEXT PIN
    idx = accounts_df[accounts_df["account_id"] == acc_id].index[0]
    accounts_df.loc[idx, "password"] = new_pin
    save_accounts()
    print("Your PIN has been changed successfully. (PIN is stored in clear text)")

def user_transaction_history(acc_id):
    global transactions_df
    user_trans = transactions_df[transactions_df["account_id"] == acc_id]
    if user_trans.empty:
        print("No transactions found!")
        return

    print(f"\nTransaction History for Account ID: {acc_id}")
    print("{:<15} | {:<10} | {:<20}".format("Type", "Amount", "Date & Time"))
    print("-" * 50)
    for _, row in user_trans.iterrows():
        print("{:<15} | {:<10} | {:<20}".format(row['type'], row['amount'], row['date']))
    print("..................................................")


def user_menu(acc_id):
    """The main menu for the authenticated user."""
    while True:
        print("\n--- User Menu ---")
        print("1. Check Balance")
        print("2. Deposit Amount (CDM)")
        print("3. Withdraw Amount (High Security)")
        print("4. Change PIN")
        print("5. Display Transaction History")
        print("6. Logout")
        choice = input("Enter choice (1-6): ").strip()

        if choice == "1":
            check_balance(acc_id)
        elif choice == "2":
            deposit(acc_id) # Detailed CDM logic
        elif choice == "3":
            withdraw_amount(acc_id) # Face + OTP logic
        elif choice == "4":
            change_pin(acc_id)
        elif choice == "5":
            user_transaction_history(acc_id)
        elif choice == "6":
            break
        else:
            print("Wrong input! Enter a number between 1 and 6.")


# =============================================================
# 💻 ADMIN FUNCTIONALITIES 💻
# =============================================================
ADMIN_USER = "admin"
ADMIN_PASS = "123" # Admin password is NOT hashed in accounts.csv for simplified login in this script.

def admin_login():
    """Handles admin login."""
    print("\n--- Admin Login ---")
    u = input("Enter admin username: ").strip()
    p = input("Enter admin password: ").strip()
    if u != ADMIN_USER or p != ADMIN_PASS:
        print("Invalid admin login!")
        return False
    print("Admin Login Successful!")
    return True


def admin_create_account():
    global accounts_df
    print("\n--- Admin Create Account ---")

    # Input validation loops
    while True:
        full_name = input("Enter Full Name: ").strip()
        if validate_name(full_name): break
    while True:
        father_name = input("Enter Father Name: ").strip()
        if validate_name(father_name): break
    while True:
        cnic = input("Enter CNIC: ").strip()
        result = validate_cnic(cnic)
        if result == True: break
        elif result == "menu": return # Jump to main menu function
    while True:
        contact_no = input("Enter Contact Number: ").strip()
        if validate_contact(contact_no): break
    while True:
        email_address = input("Enter Email Address: ").strip().lower()
        if validate_email(email_address): break
    while True:
        dob = input("Enter Date of Birth (YYYY-MM-DD): ").strip()
        if validate_dob(dob): break
    while True:
        user_name = input("Enter User Name: ").strip()
        if validate_name(user_name): break
    while True:
        address = input("Enter Address: ").strip()
        if validate_address(address): break
    while True:
        # PIN input is visible here
        password = input("Set 4-digit PIN: ").strip()
        if validate_pin(password): break
        print("PIN must be 4 digits.")
    while True:
        try:
            balance = float(input("Set Initial Balance: "))
            if balance >= 0: break
            print("Balance cannot be negative.")
        except ValueError:
            print("Invalid balance amount.")


    # Account ID generation
    if not accounts_df.empty:
        try:
            numeric_ids = pd.to_numeric(accounts_df["account_id"], errors='coerce')
            max_id = numeric_ids.max()
            # If accounts exist, increment max_id, otherwise start at 1001
            account_id = str(int(max_id) + 1) if pd.notna(max_id) and max_id > 0 else "1001"
        except:
            account_id = "1001"
    else:
        # If accounts_df is empty (first account creation), start at 1001
        account_id = "1001"

    creation_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Storing CLEAR PIN (as requested)
    new_account = {
        "account_id": account_id, "full_name": full_name, "father_name": father_name,
        "cnic": cnic, "contact_no": contact_no, "email_address": email_address,
        "dob": dob, "user_name": user_name, "address": address,
        "password": password, # Clear PIN (integer format string)
        "balance": balance, "status": "active", "creation_date": creation_date,
        "face_image_path": ""  # <-- Initialized new field
    }

    accounts_df = pd.concat([accounts_df, pd.DataFrame([new_account])], ignore_index=True)
    save_accounts()
    print(f"\nAccount created successfully! Account ID: {account_id}, Initial Balance: {balance}")

    # Call face registration after admin account creation
    print("\n--- Face Registration Required (Admin) ---")
    # live_face_registration now returns the saved path or None
    image_path = live_face_registration(account_id, full_name, other_data=f"Admin Created - CNIC: {cnic}")

    # Update accounts_df with the path if available
    if image_path:
        # Find the row index we just created
        idx = accounts_df[accounts_df["account_id"] == account_id].index[0]
        accounts_df.loc[idx, "face_image_path"] = image_path
        save_accounts()
        print("\n✅ Account creation completed successfully with face registration! (Path saved to accounts.csv)")
    else:
        print("\n⚠️ Account created but face registration was not completed. (Path not saved)")


def admin_view_all_accounts():
    """Displays all accounts."""
    if accounts_df.empty:
        print("No accounts available!")
        return

    for _, row in accounts_df.iterrows():
        print(f"\nAccount ID: {row['account_id']} | Name: {row['full_name']} | Father: {row['father_name']} |")
        print(f"DOB: {row['dob']} | Address: {row['address']}")

        # Ensure balance is formatted correctly in case it's a number (it is a float after loading)
        balance_str = f"{row['balance']:.2f}" if isinstance(row['balance'], (int, float)) else str(row['balance'])

        print(
            f"Contact No: {row['contact_no']} | CNIC: {row['cnic']} | Email: {row['email_address']} | Username: {row['user_name']} | PIN: {row['password']} | Balance: {balance_str} | Status: {row['status']}")

        # FIX: Explicitly check if the value is a non-empty string before passing to os.path.basename
        image_path = row.get('face_image_path')

        # Check if the path exists, is a string, and is not just an empty string
        if image_path and isinstance(image_path, str) and image_path.strip():
            print(f"Face Image: {os.path.basename(image_path)}")
        else:
            print("Face Image: Not Registered")
        print("..................................................")
    # The function returns naturally, allowing the admin_menu loop to continue.


def admin_search_account():
    """Searches for an account by name, ID, or email."""
    if accounts_df.empty:
        print("No accounts available!")
        return

    search = input("Search by Name, Account ID, or Email: ").strip()
    user = accounts_df[(accounts_df["full_name"] == search) | (accounts_df["account_id"] == search) | (
            accounts_df["email_address"] == search.lower())]

    if user.empty:
        print("Account not found!")
        return

    row = user.iloc[0]
    print(f"\n--- Account Found ---")
    print(f"Account ID: {row['account_id']} | Name: {row['full_name']} | Father: {row['father_name']} |")
    print(f"DOB: {row['dob']} | Address: {row['address']}")

    # Ensure balance is formatted correctly
    balance_str = f"{row['balance']:.2f}" if isinstance(row['balance'], (int, float)) else str(row['balance'])

    print(
        f"Contact No: {row['contact_no']} | CNIC: {row['cnic']} | Email: {row['email_address']} | Username: {row['user_name']} | PIN: {row['password']} | Balance: {balance_str} | Status: {row['status']}")

    # Check if the path is valid before using os.path.basename
    image_path = row.get('face_image_path')
    if image_path and isinstance(image_path, str) and image_path.strip():
        print(f"Face Image: {os.path.basename(image_path)}")
    else:
        print("Face Image: Not Registered")
    print("---------------------")


def admin_manage_account():
    """Allows admin to block, unblock, or delete an account."""
    global accounts_df
    acc_id = input("Enter Account ID to manage: ").strip()
    if acc_id not in accounts_df["account_id"].values:
        print("Account not found!")
        return

    user_row_df = accounts_df[accounts_df["account_id"] == acc_id]
    current_status = user_row_df.iloc[0]["status"]

    print(f"\n--- Manage Account {acc_id} (Current Status: {current_status}) ---")
    print("1. Block/Unblock Account")
    print("2. Delete Account")

    choice = input("Enter choice (1 or 2): ").strip()

    if choice == "1":
        if current_status == "active":
            accounts_df.loc[accounts_df["account_id"] == acc_id, "status"] = "blocked"
            print("Account blocked!")
        else:
            accounts_df.loc[accounts_df["account_id"] == acc_id, "status"] = "active"
            print("Account unblocked!")
    elif choice == "2":
        # --- LOGIC: Delete face image file before deleting account ---
        user_row = user_row_df.iloc[0]
        image_path = user_row.get("face_image_path")

        if image_path and isinstance(image_path, str) and os.path.exists(image_path):
            try:
                os.remove(image_path)
                print(f"User image deleted from database: {os.path.basename(image_path)}")
            except Exception as e:
                print(f"Warning: Could not delete image file {image_path}. Error: {e}")
        # --- END LOGIC ---

        # Delete account row from DataFrame
        accounts_df = accounts_df[accounts_df["account_id"] != acc_id]
        print("Account deleted!")
    else:
        print("Invalid choice.")

    save_accounts()


def admin_add_amount():
    """Adds a manual deposit to a user account."""
    global accounts_df, transactions_df
    acc_id = input("Enter Account ID: ").strip()
    if acc_id not in accounts_df["account_id"].values:
        print("Account not found!")
        return

    try:
        amount = float(input("Enter amount to add: "))
    except:
        print("Invalid amount!")
        return

    accounts_df.loc[accounts_df["account_id"] == acc_id, "balance"] += amount

    transactions_df = pd.concat([transactions_df, pd.DataFrame([{
        "account_id": acc_id,
        "type": "Admin Deposit",
        "amount": amount,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }])], ignore_index=True)

    save_accounts()
    save_transactions()
    print("Amount added successfully!")


def admin_change_user_pin():
    """
    Allows admin to reset a user's PIN - STORE AS CLEAR TEXT.
    """
    global accounts_df
    acc_id = input("Enter Account ID: ").strip()
    if acc_id not in accounts_df["account_id"].values:
        print("Account not found!")
        return

    # Get and validate new PIN TWICE (UNMASKED)
    while True:
        new_pin = input("Enter New 4-digit PIN: ").strip()
        if not validate_pin(new_pin):
            print("New PIN must be 4 digits.")
            continue

        confirm_pin = input("Confirm New 4-digit PIN: ").strip()

        if new_pin == confirm_pin:
            break
        print("New PIN and confirmation PIN do not match. Try again.")

    # Store as clear text (as requested)
    accounts_df.loc[accounts_df["account_id"] == acc_id, "password"] = new_pin
    save_accounts()
    print("PIN changed successfully and is stored in clear text.")

def admin_view_user_transactions():
    """Allows admin to view the transaction history for a specific user."""
    global transactions_df
    acc_id = input("Enter Account ID to view transactions: ").strip()
    if acc_id not in accounts_df["account_id"].values:
        print("Account not found!")
        return

    user_trans = transactions_df[transactions_df["account_id"] == acc_id]

    if user_trans.empty:
        print(f"No transactions found for Account ID: {acc_id}")
        return

    print(f"\nTransaction History for Account ID: {acc_id}")
    print("{:<15} | {:<10} | {:<20}".format("Type", "Amount", "Date & Time"))
    print("-" * 50)
    for _, row in user_trans.iterrows():
        print("{:<15} | {:<10} | {:<20}".format(row['type'], row['amount'], row['date']))
    print("..................................................")


def admin_menu():
    """The main menu for the authenticated admin."""
    while True:
        print("\n--- Admin Menu ---")
        print("1. Create New Account")
        print("2. View All Accounts")
        print("3. Search Account")
        print("4. Manage Account (Block/Delete)")
        print("5. Add Amount (Manual Deposit)")
        print("6. Change User PIN (Visible Input)")
        print("7. View User Transactions")
        print("8. Logout") # Option 8 is now Logout
        ch = input("Enter choice (1-8): ").strip() # Menu choices updated

        if ch == "1":
            admin_create_account()
        elif ch == "2":
            admin_view_all_accounts()
        elif ch == "3":
            admin_search_account()
        elif ch == "4":
            admin_manage_account()
        elif ch == "5":
            admin_add_amount()
        elif ch == "6":
            admin_change_user_pin()
        elif ch == "7":
            admin_view_user_transactions()
        elif ch == "8": # Logout
            break
        else:
            print("Wrong input! Enter number between 1-8.") # Error message updated


# =============================================================
# Main Program
# =============================================================
def main():
    print("\n===================================")
    print("      Welcome to BankPoint 360      ")
    print(" Automated ATM & CDM Management System ")
    print("===================================")
    initialize_csv_files()
    load_databases()

    while True:
        print("\n===== Main Menu =====")
        # 1. Create Account (Public User) REMOVED
        print("1. User Login (PIN is masked)")
        print("2. Admin Login")
        print("3. Exit")
        choice = input("Enter choice (1-3): ").strip() # Menu updated to 1-3

        if choice == "1":
            acc_id = user_login()
            if acc_id:
                user_menu(acc_id)
        elif choice == "2":
            if admin_login():
                admin_menu()
        elif choice == "3":
            print("Exiting BankPoint 360...")
            break
        else:
            print("Wrong input! Enter number between 1-3.") # Error message updated

if __name__ == "__main__":
    # Ensure the database folder exists on startup
    if not os.path.exists(DB_FOLDER_PATH):
        os.makedirs(DB_FOLDER_PATH)
        print(f"[System] Created database folder: {DB_FOLDER_PATH}")
    main()
