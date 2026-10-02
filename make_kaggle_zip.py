from pathlib import Path
import zipfile

SOURCE = Path(r"C:\Users\alzah\smoking-project\behavior_dataset_balanced")
OUTPUT = Path(r"C:\Users\alzah\smoking-project\behavior_dataset_balanced_kaggle.zip")

with zipfile.ZipFile(
    OUTPUT,
    "w",
    compression=zipfile.ZIP_STORED,
    allowZip64=True
) as zf:

    for file in SOURCE.rglob("*"):
        if file.is_file():
            # Relative path using forward slashes
            relative_path = file.relative_to(SOURCE).as_posix()

            # Keep the main folder inside the ZIP
            arcname = f"{SOURCE.name}/{relative_path}"

            zf.write(file, arcname)

print("ZIP created:")
print(OUTPUT)