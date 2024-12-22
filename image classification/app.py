if __name__ == '__main__':
    from flask import Flask, request, render_template, jsonify
    from PIL import Image
    import requests
    from io import BytesIO
    import torch
    from transformers import BlipProcessor, BlipForConditionalGeneration

    # Initialize Flask app
    app = Flask(__name__)

    # Load the pre-trained Hugging Face model and processor
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/upload', methods=['POST'])
    def upload_image():
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        try:
            # Open the image
            image = Image.open(file.stream)

            # Process the image with BLIP model
            inputs = processor(images=image, return_tensors="pt")
            out = model.generate(**inputs)
            description = processor.decode(out[0], skip_special_tokens=True)

            return jsonify({"description": description})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    app.run(debug=True)
