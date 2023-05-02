# -*- coding: utf-8 -*-
"""
Created on Tue May  2 01:47:02 2023

@author: 99sma
"""
from tkinter import filedialog

import tkinter as tk
from PIL import ImageTk, Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch


class CaptionGenerator:
    def __init__(self, model_path):
        self.model_path = model_path
        self.feature_extractor = ViTImageProcessor.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VisionEncoderDecoderModel.from_pretrained(model_path).to(device)

        self.max_length = 16
        self.num_beams = 4
        self.gen_kwargs = {"max_length": self.max_length, "num_beams": self.num_beams
        }

    def generate_captions(self, image_path):
        i_image = Image.open(image_path)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")

        pixel_values = self.feature_extractor(images=i_image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.model.device)

        output_ids = self.model.generate(pixel_values, **self.gen_kwargs)

        preds = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        preds = [pred.strip() for pred in preds]
        return preds


class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.title("Image Captioning")
        self.master.geometry("800x600")
        self.pack(fill=tk.BOTH, expand=True)

        self.image_path = ""
        self.captions = []

        self.caption_generator = CaptionGenerator(model_path="C:/Users/99sma/.spyder-py3/image_captioning_model")

        self.create_widgets()

    def create_widgets(self):
        # Image insertion button
        self.insert_image_button = tk.Button(self, text="Insert Image", command=self.open_image)
        self.insert_image_button.pack(pady=20)

        # Image display
        self.image_label = tk.Label(self)
        self.image_label.pack(pady=20)

        # Caption display
        self.caption_label = tk.Label(self, text="Click 'Generate Captions' to generate captions for an image.")
        self.caption_label.pack(pady=10)

        # Generate Captions button
        self.generate_button = tk.Button(self, text="Generate Captions", command=self.generate_captions)
        self.generate_button.pack(pady=10)

        # Quit button
        self.quit_button = tk.Button(self, text="Quit", command=self.master.destroy)
        self.quit_button.pack(pady=10)

    def generate_captions(self):
       if self.image_path:
           self.captions = self.caption_generator.generate_captions(self.image_path)
       else:
           self.image_path = tk.filedialog.askopenfilename(initialdir=".", title="Select Image",
                                                        filetypes=[("Image Files", "*.png *.jpg *.jpeg")])
           if self.image_path:
               self.captions = self.caption_generator.generate_captions(self.image_path)

       # Update caption label
       self.caption_label.configure(text="\n".join(self.captions))

       # Update image label
       image = Image.open(self.image_path)
       image = image.resize((400, 400))
       image = ImageTk.PhotoImage(image)
       self.image_label.configure(image=image)
       self.image_label.image = image


    def open_image(self):
        self.image_path = tk.filedialog.askopenfilename(initialdir=".", title="Select Image",
                                                        filetypes=[("Image Files", "*.png *.jpg *.jpeg")])
        self.generate_captions()



if __name__ == "__main__":
    root = tk.Tk()
    app = Application(master=root)
    app.mainloop()
