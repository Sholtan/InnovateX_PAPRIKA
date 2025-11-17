import os
import platform
import subprocess
from ultralytics import YOLO
import cv2


def load_model(model_path="finetuned_selected.pt"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    print("Model loaded.")
    return model


def open_with_default_viewer(path = 'detected/'):
    system = platform.system()

    try:
        if system == "Linux":
            subprocess.run(
                ["xdg-open", path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            print(f"[INFO] Annotated image saved at: {path}")
            print("[INFO] Please open it manually.")
    except Exception as e:
        print(f"[WARNING] Could not auto-open image viewer: {e}")
        print(f"[INFO] Annotated image saved at: {path}")


def detect_and_show(model, image_path):
    if not os.path.exists(image_path):
        print(f"[ERROR] File not found: {image_path}")
        return

    try:
        results = model(image_path)
    except Exception as e:
        print(f"[ERROR] Failed to run detection: {e}")
        return

    if len(results) == 0:
        print("[INFO] No results returned by the model.")
        return

    result = results[0]

    # Get annotated image (BGR numpy array with boxes & labels)
    annotated = result.plot()

    # Save annotated image next to original
    base, ext = os.path.splitext(image_path)
    out_path = f"detected/{base}_detected{ext}"
    try:
        cv2.imwrite(out_path, annotated)
        print(f"[INFO] Annotated image saved to: {out_path}")
    except Exception as e:
        print(f"[ERROR] Could not save annotated image: {e}")
        return

    # Open with system viewer (no Qt from Python)
    open_with_default_viewer(out_path)

    # Print detections to terminal
    names = model.names
    print("\nDetected objects:")
    if result.boxes is None or len(result.boxes) == 0:
        print("  No objects detected.")
    else:
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            xyxy = box.xyxy[0].tolist()
            cls_name = names.get(cls_id, f"class_{cls_id}")
            print(f"  {cls_name} (conf={conf:.2f}) at {xyxy}")


def main():
    model = load_model("YOLOvn_finetuned_onselected.pt")

    while True:
        print("\n=== Menu ===")
        print("1 - Detect")
        print("2 - Exit")
        choice = input("Choose an option: ").strip()



        if choice == "1":
            image_path = input("Enter image filename (path): ").strip()

            image_path += ".jpg"
            
            try:
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"File not found: {image_path}")
                detect_and_show(model, image_path)
            except FileNotFoundError as e:
                print(f"[ERROR] {e}")
            except Exception as e:
                print(f"[ERROR] Unexpected error: {e}")

        elif choice == "2":
            print("Exiting.")
            break

        else:
            print("Invalid option, please choose 1 or 2.")


if __name__ == "__main__":
    main()
