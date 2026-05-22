import argparse
import os
from pathlib import Path
from PIL import Image


def resize_images(source_dir, dest_dir, ratio):
    """
    Resizes all images in source_dir by a given ratio and saves them to dest_dir.
    """
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)

    # Validation: Ensure source directory actually exists
    if not source_path.exists():
        print(f"Error: Source directory '{source_dir}' does not exist.")
        return

    dest_path.mkdir(parents=True, exist_ok=True)

    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
    count = 0

    print(f"Starting downsizing by a ratio of {ratio}...")
    print(f"Source: {source_path.resolve()}")
    print(f"Destination: {dest_path.resolve()}\n" + "-" * 40)

    for file_path in source_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in valid_extensions:
            try:
                with Image.open(file_path) as img:
                    new_width = max(1, int(img.width / ratio))
                    new_height = max(1, int(img.height / ratio))

                    resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

                    target_file = dest_path / file_path.name
                    resized_img.save(target_file, format="JPEG", quality=100, subsampling=0)

                    print(f"Resized: {file_path.name} -> {new_width}x{new_height}")
                    count += 1

            except Exception as e:
                print(f"Error processing {file_path.name}: {e}")

    print("-" * 40)
    print(f"Finished! Successfully downsized {count} images.")


if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description="Downsize all images in a directory by a specific scaling ratio."
    )

    # Positional or optional arguments
    parser.add_argument(
        "source",
        type=str,
        help="Path to the source directory containing original images."
    )
    parser.add_argument(
        "destination",
        type=str,
        help="Path to the destination directory where downsized images will be saved."
    )
    parser.add_argument(
        "-r", "--ratio",
        type=float,
        default=0.5,
        help="Scaling ratio as a float (e.g., 0.5 for half-size, 0.25 for quarter-size). Default is 0.5."
    )

    args = parser.parse_args()

    # Pass the parsed arguments into the function
    resize_images(args.source, args.destination, args.ratio)