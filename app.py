import insightface
import os
import onnxruntime
import cv2
import gfpgan
import tempfile
import time
import gradio as gr


class Predictor:
    def __init__(self):
        self.setup()

    def setup(self):
        os.makedirs('models', exist_ok=True)
        os.chdir('models')
        if not os.path.exists('GFPGANv1.4.pth'):
            os.system(
                'wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'
            )
        if not os.path.exists('inswapper_128.onnx'):
            os.system(
                'wget https://huggingface.co/ashleykleynhans/inswapper/resolve/main/inswapper_128.onnx'
            )
        os.chdir('..')

        """Load the model into memory to make running multiple predictions efficient"""
        self.face_swapper = insightface.model_zoo.get_model('models/inswapper_128.onnx',
                                                            providers=onnxruntime.get_available_providers())
        self.face_enhancer = gfpgan.GFPGANer(model_path='models/GFPGANv1.4.pth', upscale=1)
        self.face_analyser = insightface.app.FaceAnalysis(name='buffalo_l')
        self.face_analyser.prepare(ctx_id=0, det_size=(640, 640))

    def get_face(self, img_data):
        analysed = self.face_analyser.get(img_data)
        try:
            largest = max(analysed, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
            return largest
        except:
            print("No face found")
            return None

    def predict(self, input_image, swap_image):
        """Run a single prediction on the model"""
        try:
            frame = cv2.imread(input_image.name)
            face = self.get_face(frame)
            source_face = self.get_face(cv2.imread(swap_image.name))
            try:
                print(frame.shape, face.shape, source_face.shape)
            except:
                print("printing shapes failed.")
            result = self.face_swapper.get(frame, face, source_face, paste_back=True)

            _, _, result = self.face_enhancer.enhance(
                result,
                paste_back=True
            )
            out_path = tempfile.mkdtemp() + f"/{str(int(time.time()))}.jpg"
            cv2.imwrite(out_path, result)
            return out_path
        except Exception as e:
            print(f"{e}")
            return None


# Instantiate the Predictor class
predictor = Predictor()
title = "Swap Faces Using Our Model!!!"

# Create Gradio Interface
iface = gr.Interface(
    fn=predictor.predict,
    inputs=[
        gr.inputs.Image(type="file", label="Target Image"),
        gr.inputs.Image(type="file", label="Swap Image")
    ],
    outputs=gr.outputs.Image(type="file", label="Result"),
    title=title,
    examples=[["input.jpg", "swap img.jpg"]])
# Launch the Gradio Interface
iface.launch()
