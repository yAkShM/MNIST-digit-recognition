import numpy as np
import struct
import matplotlib.pyplot as plt

def convert_mnist_to_npy(image_path, label_path):
    # 1. Processing Images
    with open(image_path, 'rb') as f:
        # The first 16 bytes are magic number, num_images, rows, cols
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols)
        np.save('mnist_images.npy', images)
        print(f"Saved {num_images} images to mnist_images.npy")

    # 2. Processing Labels
    with open(label_path, 'rb') as f:
        # The first 8 bytes are magic number and num_items
        magic, num_items = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        np.save('mnist_labels.npy', labels)
        print(f"Saved {num_items} labels to mnist_labels.npy")


# UPDATE THESE STRINGS with your actual local filenames:
convert_mnist_to_npy(r'C:\Users\HP-PC\Downloads\MNIST_main\t10k-images.idx3-ubyte', r'C:\Users\HP-PC\Downloads\MNIST_main\t10k-labels.idx1-ubyte')

training = np.load('mnist_images.npy')
training_label = np.load('mnist_labels.npy')

normalized = training/225

plt.Figure(figsize=(10,5))
plt.subplot(121)
plt.imshow(normalized[10])
plt.show()

flattened_data = normalized.reshape(10000, -1)
print(flattened_data.shape)

#INITIALIZE

W = np.random.randn(784, 10) * 0.01
b = np.zeros((1, 10))
learning_rate = 0.01
epochs = 50 # Number of times to go through all 10,000 images

print("Starting training...")

for epoch in range(epochs):
    total_loss = 0
    
    # This loop goes through all 10,000 images in your flattened_data
    for i in range(len(flattened_data)):
        
        # 1. DATA SELECTION (Using 'i' instead of a fixed 0)

        x = flattened_data[i:i+1]            # Current image pixels
        actual_digit = training_label[i]     # Current actual label
        
        # Create y_true (The Target)
        y_true = np.zeros((1, 10))
        y_true[0, actual_digit] = 1

        # 2. FORWARD PASS
        z = np.dot(x, W) + b
        a = np.maximum(0, z)                 # ReLU Activation

        # 3. LOSS CALCULATION
        error = a - y_true
        loss_val = np.mean(np.square(error))
        total_loss += loss_val

        # 4. BACKPROPAGATION (Gradients)
        dZ = np.array(error, copy=True)
        dZ[z <= 0] = 0                       # ReLU Derivative
        
        dw = np.dot(x.T, dZ)
        db = np.sum(dZ, axis=0, keepdims=True)

        # 5. PARAMETER UPDATE
        W = W - (learning_rate * dw)
        b = b - (learning_rate * db)

    # Calculate average loss for the entire epoch
    avg_loss = total_loss / len(flattened_data)
    print(f"Epoch {epoch+1}/{epochs} completed. Average Loss: {avg_loss:.6f}")

print("Model Trained!")

# 1. Pick a test image (e.g., the 500th image)
test_index = 100
x_test = flattened_data[test_index : test_index + 1]
actual_label = training_label[test_index]

# 2. RUN THE FORWARD PASS (No training here!)
z_test = np.dot(x_test, W) + b
a_test = np.maximum(0, z_test)

# 3. GET THE PREDICTION
# np.argmax finds the index of the highest value in the array
prediction = np.argmax(a_test)

# 4. VISUALIZE AND COMPARE
plt.imshow(training[test_index], cmap='gray')
plt.title(f"Actual: {actual_label} | Model Predicted: {prediction}")
plt.show()

if prediction == actual_label:
    print("Correct! The weights successfully recognized the pattern.")
else:
    print("Incorrect. The model needs more training epochs or a better learning rate.")


np.save('trained_weights.npy', W)
np.save('trained_bias.npy', b)
print("Model weights saved.")

correct_guesses = 0

for i in range(len(flattened_data)):
    # 1. Forward Pass Only
    x_test = flattened_data[i:i+1]
    z_test = np.dot(x_test, W) + b
    a_test = np.maximum(0, z_test) # ReLU
    
    # 2. Compare Prediction to Reality
    prediction = np.argmax(a_test)
    actual = training_label[i]
    
    if prediction == actual:
        correct_guesses += 1

# 3. Final Score
accuracy = (correct_guesses / len(flattened_data)) * 100
print(f"Final Model Accuracy: {accuracy:.2f}%")
