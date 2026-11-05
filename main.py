import os
import torch
import scipy.io.wavfile
import shutil

from PIL import Image
from tkinter import Tk, filedialog
from transformers import pipeline, VitsModel, VitsTokenizer


IMAGE_MODEL_NAME = "Salesforce/blip-image-captioning-base"
VOICE_MODEL_NAME = "facebook/mms-tts-eng"


INPUT_FOLDER = "input"
OUTPUT_FOLDER = "output"
DESCRIPTION_FILE = "descriptions.txt"

if not os.path.exists(INPUT_FOLDER):
    os.makedirs(INPUT_FOLDER)

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)


print("Loading AI models...")
print("Please wait...")

try:
    image_model = pipeline(
        "image-to-text",
        model=IMAGE_MODEL_NAME,
        device=-1
    )

    voice_tokenizer = VitsTokenizer.from_pretrained(
        VOICE_MODEL_NAME
    )

    voice_model = VitsModel.from_pretrained(
        VOICE_MODEL_NAME
    )

    print("AI models loaded successfully!")

except Exception as error:
    print("Error loading AI models:", error)
    exit()

last_description = ""


def generate_voice(description, audio_name):

    if description == "":
        print("There is no description to read.")
        return

    if not audio_name.lower().endswith(".wav"):
        audio_name = audio_name + ".wav"

    audio_path = os.path.join(OUTPUT_FOLDER, audio_name)

    try:
        inputs = voice_tokenizer(
            description,
            return_tensors="pt"
        )

        with torch.no_grad():
            output = voice_model(**inputs)

        waveform = output.waveform[0].cpu().numpy()

        scipy.io.wavfile.write(
            audio_path,
            rate=voice_model.config.sampling_rate,
            data=waveform.astype("float32")
        )

        print("\nAudio saved to:", audio_path)

    except Exception as error:
        print("Error generating speech:", error)


def analyze_image(image_path):

    global last_description

    if not os.path.exists(image_path):
        print("Error: File does not exist.")
        return

    extension = os.path.splitext(image_path)[1].lower()

    if extension not in [".jpg", ".jpeg", ".png"]:
        print("Error: Unsupported image format.")
        return

    try:
        image = Image.open(image_path).convert("RGB")

    except Exception as error:
        print("Error opening image:", error)
        return

    print("\nAnalyzing image...")

    try:
        result = image_model(image)

        if not result or not result[0].get("generated_text"):
            print("AI could not create an image description.")
            return

        description = result[0]["generated_text"].strip()

        if description == "":
            print("AI could not create an image description.")
            return

        last_description = description

        print("\nImage description:")
        print(description)

        print("\nImage model:", IMAGE_MODEL_NAME)

        with open(
            DESCRIPTION_FILE,
            "a",
            encoding="utf-8"
        ) as file:

            file.write("\nImage: " + os.path.basename(image_path))
            file.write("\nDescription: " + description)
            file.write("\n" + "-" * 50 + "\n")

        audio_name = input(
            "\nEnter audio file name: "
        )

        if audio_name == "":
            audio_name = os.path.splitext(
                os.path.basename(image_path)
            )[0]

        print("\nGenerating speech...")

        print("Voice model:", VOICE_MODEL_NAME)

        generate_voice(description, audio_name)

    except Exception as error:
        print("Error analyzing image:", error)


def select_images():

    root = None

    try:
        root = Tk()
        root.withdraw()

        root.attributes("-topmost", True)
        root.update()

        image_paths = filedialog.askopenfilenames(
            parent=root,
            title="Select photographs",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png"),
                ("JPG files", "*.jpg"),
                ("JPEG files", "*.jpeg"),
                ("PNG files", "*.png")
            ]
        )

        root.attributes("-topmost", False)

        if not image_paths:
            print("No images selected.")
            return

        for image_path in image_paths:

            print("\n" + "=" * 45)
            print("Selected image:", os.path.basename(image_path))

            file_name = os.path.basename(image_path)
            input_path = os.path.join(INPUT_FOLDER, file_name)

            # Копирование фотографии в папку input
            shutil.copy2(image_path, input_path)

            print("Image copied to:", input_path)

            # Анализ фотографии из папки input
            analyze_image(input_path)

    except Exception as error:
        print("Error selecting images:", error)

    finally:
        if root is not None:
            root.destroy()


def repeat_voice():

    global last_description

    if last_description == "":
        print("There is no previous description.")
        return

    print("\nPrevious description:")
    print(last_description)

    audio_name = input(
        "\nEnter new audio file name: "
    )

    if audio_name == "":
        audio_name = "repeated_description"

    print("\nGenerating speech again...")

    generate_voice(last_description, audio_name)


def main():

    while True:

        print("\n")
        print("=" * 45)
        print("AI VISION & VOICE ASSISTANT")
        print("=" * 45)

        print("1. Select and analyze photographs")
        print("2. Repeat the last voice description")
        print("3. Exit")

        choice = input("\nChoose an option: ")

        if choice == "1":

            select_images()

        elif choice == "2":

            repeat_voice()

        elif choice == "3":

            print("\nTask completed successfully!")
            break

        else:

            print("Invalid choice. Try again.")


if __name__ == "__main__":
    main()

