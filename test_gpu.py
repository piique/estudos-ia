import tensorflow as tf

# Check for available GPUs
gpus = tf.config.list_physical_devices('GPU')

print(f"TensorFlow version: {tf.__version__}")

if gpus:
  print(f"Num GPUs Available: {len(gpus)}")
  try:
    # Print details for the first GPU found
    print(f"GPU device found: {gpus[0]}")
    # Further details
    details = tf.config.experimental.get_device_details(gpus[0])
    print(f"GPU Name: {details.get('device_name', 'N/A')}")
  except RuntimeError as e:
    print(e)
else:
  print("No GPU devices found. TensorFlow is using the CPU.")
