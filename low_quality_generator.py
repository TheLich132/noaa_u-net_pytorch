import cv2
import numpy as np
import random
import os

# Function to add Gaussian noise to an image
def add_gaussian_noise(image):
    """
    Adds Gaussian noise to the image to simulate signal noise.
    
    Parameters:
    - image: numpy array representing the image.
    - mean: Mean value of the Gaussian noise distribution.

    Returns:
    - Noisy image with Gaussian noise added.
    """
    # Random mean value
    mean = random.uniform(0, 2)

    # Random std deviation
    std = random.uniform(0, 2)

    # Generate Gaussian noise
    noise = np.random.normal(mean, std, image.shape[:2]).astype(np.uint8)
    # Stack noise for each channel (R, G, B)
    noise = np.stack([noise] * image.shape[2], axis=-1)
    # Add noise to the image
    noisy_image = cv2.add(image, noise)
    return noisy_image

# Function to apply Gaussian blur to an image
def apply_gaussian_blur(image, kernel_size=(5, 5)):
    """
    Applies Gaussian blur to smooth the image and simulate low quality.
    
    Parameters:
    - image: numpy array representing the image.
    - kernel_size: Size of the Gaussian kernel (odd values).
    - sigma: Standard deviation for the Gaussian kernel.
    
    Returns:
    - Blurred image.
    """
    # Random sigma value
    sigma = random.uniform(1.0, 1.1)
    blurred_image = cv2.GaussianBlur(image, kernel_size, sigma)
    return blurred_image

# Function to reduce the resolution of an image
def reduce_resolution(image, scale=0.5):
    """
    Reduces the resolution of the image and then upscales it back to original size.
    
    Parameters:
    - image: numpy array representing the image.
    - scale: Scale factor for downscaling the resolution.
    
    Returns:
    - Low-resolution image.
    """
    # Get original dimensions
    height, width = image.shape[:2]
    # Resize to a smaller size (downscale)
    small = cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_LINEAR)
    # Upscale back to original size
    low_res_image = cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)
    return low_res_image

# Function to add JPEG compression artifacts to an image
def add_jpeg_compression(image):
    """
    Compresses the image using JPEG to introduce compression artifacts.
    
    Parameters:
    - image: numpy array representing the image.
    - quality: JPEG quality (lower values mean more artifacts).
    
    Returns:
    - Compressed image with JPEG artifacts.
    """
    # Random quality value
    quality = random.randint(75, 100)
    # Set JPEG quality parameter
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    # Compress and then decompress the image
    _, encimg = cv2.imencode('.jpg', image, encode_param)
    compressed_image = cv2.imdecode(encimg, 1)
    return compressed_image

# Function to add simulated satellite streaks to the image
def add_satellite_streaks(image):
    """
    Adds simulated satellite streaks with a noise-like appearance to a NOAA APT image 
    to imitate signal interference, similar to Gaussian or TV noise.
    
    Parameters:
    - image: numpy array representing the NOAA APT image (grayscale).
    
    Returns:
    - Image with simulated satellite streaks that look like static noise.
    """
    # Get image dimensions
    height, width = image.shape[:2]

    # Create a copy of the image
    output_image = image.copy()

    # Number of streaks to generate
    num_streaks = random.randint(1, 15)

    # Generate streaks with a TV noise effect
    for _ in range(num_streaks):
        # Random streak thickness and position
        streak_thickness = random.randint(1, 10)
        y_pos = random.randint(0, height - 1)

        # Loop over the width of the image and add noise along the streak
        for x in range(0, width):
            # Randomize intensity to simulate static noise
            noise_intensity = random.gauss(128, 50)  # Gaussian noise around mid-gray
            noise_intensity = max(0, min(255, int(noise_intensity)))  # Clamp to valid range
            
            # Draw pixel-sized noise within the streak's thickness
            for dy in range(streak_thickness):
                y_offset = y_pos + dy
                if y_offset < height:
                    output_image[y_offset, x] = noise_intensity

    return output_image

# Function to apply all low-quality effects to an image
def generate_low_quality_images(image):
    """
    Applies a series of transformations to degrade the image quality.
    
    Parameters:
    - image: numpy array representing the original image.
    
    Returns:
    - Final image with reduced quality.
    """
    # Add Gaussian noise
    noisy_image = add_gaussian_noise(image)
    # Apply Gaussian blur
    blurred_image = apply_gaussian_blur(noisy_image)
    # Reduce resolution
    low_res_image = reduce_resolution(blurred_image, scale=1.0)
    # Add JPEG compression artifacts
    jpeg_compressed_image = add_jpeg_compression(low_res_image)
    print(jpeg_compressed_image.shape)
    # Add satellite streaks
    final_image = add_satellite_streaks(jpeg_compressed_image)
    return final_image  

# Directory paths for input (high-quality images) and output (low-quality images)
input_dir = "/var/home/jakub/Obrazy/apt_training_set/high_quality"
output_dir = "/var/home/jakub/Obrazy/apt_training_set/low_quality"

# Ensure output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Process each image in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Construct the full path to the input image
        image_path = os.path.join(input_dir, filename)
        # Read the image
        image = cv2.imread(image_path)
        # Apply quality degradation transformations
        low_quality_image = generate_low_quality_images(image)
        # Construct the path for the output image
        output_path = os.path.join(output_dir, f"low_{filename}")
        # Save the low-quality image
        cv2.imwrite(output_path, low_quality_image)
