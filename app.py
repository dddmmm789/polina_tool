import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans
import logging
from flask import Flask, render_template, request, send_file
import pandas as pd

app = Flask(__name__)

# Setup logging to capture detailed logs in a file
logging.basicConfig(
    filename='embroidery_tool_web.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load DMC colors from the CSV file
def load_dmc_colors():
    dmc_colors_path = "/Users/danm/Documents/Embroided/DMC_colors_github.csv"
    dmc_df = pd.read_csv(dmc_colors_path)
    dmc_colors = []

    # Using 'Floss#', 'Description', 'Red', 'Green', 'Blue' columns from the CSV
    for _, row in dmc_df.iterrows():
        dmc_colors.append({
            'name': row['Description'],
            'code': row['Floss#'],
            'rgb': (row['Red'], row['Green'], row['Blue'])
        })

    return dmc_colors

# Function to calculate Euclidean distance between two RGB values
def euclidean_distance(rgb1, rgb2):
    return np.sqrt(sum((p - q) ** 2 for p, q in zip(rgb1, rgb2)))

# Function to find the closest DMC color for a given RGB
def find_closest_dmc_color(rgb, dmc_colors):
    closest_dmc = None
    min_distance = float('inf')

    for dmc in dmc_colors:
        distance = euclidean_distance(rgb, dmc['rgb'])
        if distance < min_distance:
            min_distance = distance
            closest_dmc = dmc

    return closest_dmc

# Function to generate the DMC output image
def generate_dmc_output_image(cluster_centers):
    print("Starting DMC color output generation...")

    # Load the DMC colors
    dmc_colors = load_dmc_colors()

    # Create an image to display the DMC color output
    chart_width = 800
    chart_height = len(cluster_centers) * 80  # Adjust height to fit DMC and RGB info
    chart_image = Image.new("RGB", (chart_width, chart_height), (255, 255, 255))
    draw = ImageDraw.Draw(chart_image)

    # Load a font (or use default if unavailable)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        print("Arial font not found, using default font.")
        font = ImageFont.load_default()

    y_position = 10
    for i, rgb in enumerate(cluster_centers):
        try:
            # Find the closest DMC color for the given RGB
            closest_dmc = find_closest_dmc_color(tuple(rgb), dmc_colors)

            # Draw the color rectangle
            draw.rectangle([(10, y_position), (100, y_position + 50)], fill=tuple(rgb))

            # Draw the RGB info
            rgb_text = f"RGB: {tuple(rgb)}"
            draw.text((120, y_position + 10), rgb_text, font=font, fill=(0, 0, 0))

            # Draw the DMC info
            dmc_text = f"DMC {closest_dmc['code']} - {closest_dmc['name']} (DMC RGB: {closest_dmc['rgb']})"
            draw.text((120, y_position + 30), dmc_text, font=font, fill=(0, 0, 0))

            y_position += 80  # Move to the next row for the next color

        except Exception as e:
            print(f"Error drawing color: {e}")
            return None

    # Save the DMC color output image
    dmc_output_path = os.path.join("static", "dmc_colors_output.jpg")
    print(f"Attempting to save DMC color output image to: {dmc_output_path}")

    try:
        chart_image.save(dmc_output_path)
        print("DMC color output image successfully saved.")
    except Exception as e:
        print(f"Error saving DMC color output image: {e}")
        return None

    return dmc_output_path

# Function to load the image
def load_image(image_path):
    logging.info(f"Loading image from path: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        logging.error(f"Error loading image from {image_path}. Please check the file path.")
        return None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb

# Function to process the image and reduce colors
def preprocess_image(image, max_colors=10):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    image = image.convert("RGB")
    image_array = np.array(image)
    pixels = image_array.reshape((-1, 3))

    kmeans = KMeans(n_clusters=max_colors, random_state=42).fit(pixels)
    cluster_centers = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_

    processed_pixels = cluster_centers[labels].reshape(image_array.shape)
    processed_image = Image.fromarray(processed_pixels.astype('uint8'))

    processed_image_path = os.path.join("static", "processed_image.png")
    processed_image.save(processed_image_path)

    return processed_image_path, cluster_centers

# Route to upload and process the image
@app.route("/", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"

        # Save the uploaded file
        image_path = os.path.join("uploads", file.filename)
        file.save(image_path)

        # Get the number of colors from the form
        max_colors = int(request.form.get("colors"))

        # Process the image
        image = load_image(image_path)
        if image is None:
            return "Error loading image"

        processed_image_path, unique_colors = preprocess_image(image, max_colors)

        # Generate the DMC color output image
        dmc_output_path = generate_dmc_output_image(unique_colors)

        if dmc_output_path is None:
            return "Error generating DMC color output."

        # Render the result page
        return render_template("result.html", processed_image_path=processed_image_path, dmc_output_path=dmc_output_path)

    return render_template("upload.html")

# Route to serve static images and files
@app.route("/static/<filename>")
def send_image(filename):
    return send_file(os.path.join("static", filename), as_attachment=False)

@app.route("/download/<filename>")
def download_file(filename):
    return send_file(os.path.join("static", filename), as_attachment=True)

if __name__ == "__main__":
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    if not os.path.exists("static"):
        os.makedirs("static")
    app.run(debug=True, host="0.0.0.0", port=5001)
