import tensorflow as tf

# Load the fixed model
model = tf.keras.models.load_model("fixed_model.h5", compile=False)

print("Model loaded successfully ✅")
print("TensorFlow version:", tf.__version__)
