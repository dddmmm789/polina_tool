# Palina’s Embroidery Studio

Welcome to Palina’s Embroidery Studio, a web application designed to help you convert images into beautiful embroidery patterns using DMC colors. This tool utilizes image processing and machine learning techniques to create color palettes that match your images, making embroidery easy and fun!
WHile the input is any photograph.The output is a simplified embroidery pattern and a list of DMC threads to purchase. 
You can try it live on: https://polina-tool.onrender.com/ but it is the free account of render.com so be patient.

## Features

- **Upload Images**: Users can upload images to be processed.
- **Color Reduction**: The application reduces the number of colors in the image to a specified number using KMeans clustering.
- **DMC Color Matching**: Automatically matches the reduced colors to the closest DMC embroidery thread colors.
- **Output Generation**: Generates an output image displaying the original colors alongside their closest DMC matches.
- **Download Options**: Provides options to download the processed images and the DMC color chart.

## Technologies Used

- Python
- Flask (web framework)
- OpenCV (image processing)
- NumPy (numerical operations)
- PIL (Pillow for image handling)
- scikit-learn (KMeans for color clustering)
- Pandas (data manipulation)
- HTML/CSS (for front-end)

## Installation

To run this application locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/palinas-embroidery-studio.git
   cd palinas-embroidery-studio
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   python app.py
   ```

5. **Open your browser** and go to `http://localhost:5001` to access the application.

## Usage

1. **Upload an Image**: Use the upload feature to select an image file from your device.
2. **Select Number of Colors**: Input the desired number of colors for the output.
3. **Process the Image**: Submit the form to generate the processed image and DMC color matches.
4. **View Results**: The results page will display the processed image alongside the DMC color chart.
5. **Download**: You can download the processed images and DMC color output.

## Logging

Detailed logs of the application's activity are stored in `embroidery_tool_web.log`. This includes error messages and information logs that can help in debugging.

## Contributing

Contributions are welcome! If you would like to contribute, please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License.

---

Feel free to modify any sections or add additional information specific to your project. Let me know if you need any further assistance!
