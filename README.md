# Potato Disease Classification Web Application

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Flask](https://img.shields.io/badge/Flask-2.x-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

A web-based application that uses deep learning to classify diseases in potato plants from leaf images. The application can identify various potato diseases using a trained CNN model, providing farmers and agricultural experts with a quick and efficient way to diagnose plant health issues.

## Features

- 🚀 Real-time disease classification
- 📸 Support for both JPG and PNG image formats
- 🎯 High accuracy disease prediction
- 💻 User-friendly web interface
- 🔄 Automatic image format conversion
- 🛡️ Secure file handling

## Project Structure

```
potato-disease-classification/
│
├── artifacts/
│   ├── models/
│   │   └── final_model.keras
│   └── preprocessing/
│       └── class_mapping.pkl
│
├── static/
│   └── css/
│       └── style.css
│
├── templates/
│   └── index.html
│
├── uploads/
│
├── app.py
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/potato-disease-classification.git
cd potato-disease-classification
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Requirements

```txt
Flask==2.0.1
tensorflow==2.x
Pillow==8.3.1
Werkzeug==2.0.1
numpy==1.19.5
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Upload an image of a potato plant leaf through the web interface.

4. The application will process the image and display the predicted disease classification.

## Model Information

The application uses a Convolutional Neural Network (CNN) trained on a dataset of potato plant leaves. The model can classify the following conditions:
- Early Blight
- Late Blight
- Healthy

## Features in Detail

### Image Processing
- Automatic conversion of PNG images to JPG format
- Handles transparent backgrounds in PNG images
- Resizes images to 128x128 pixels for model input
- Maintains image quality during processing

### Security Features
- Secure filename handling
- Temporary file cleanup
- Error handling and validation
- Input sanitization

## API Endpoints

- `GET /`: Serves the main page
- `POST /predict`: Accepts image uploads and returns predictions

## Contributing

1. Fork the repository
2. Create a new branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## Future Improvements

- [ ] Add support for multiple crop types
- [ ] Implement batch processing of images
- [ ] Add disease treatment recommendations
- [ ] Improve model accuracy
- [ ] Add user authentication
- [ ] Implement result history tracking

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

Created by [@yourusername](https://github.com/yourusername) - feel free to contact me!

## Acknowledgments

- Dataset source: [Add source here]
- Thanks to contributors and maintainers
- Special thanks to the agricultural experts who helped in validation

---
**Note**: Make sure to replace placeholder values (like [@yourusername]) with your actual information before publishing.
