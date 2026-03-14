# image-classification
sorts images into different classes using a ResNet with 18 layers.

# Setup instructions

Assuming you have python installed, run:

`python -m venv env source`

`env/bin/activate`, or on Windows: `env\Scripts\activate`

`pip install -r requirements.txt`

Then, simply run the command `python3 .\output_to_image` from the root folder. The images in /output should give you a hint as to what gesture you need to do in order for the model to recognize each class (most are with left hand only). Attempt in a well lit room.
