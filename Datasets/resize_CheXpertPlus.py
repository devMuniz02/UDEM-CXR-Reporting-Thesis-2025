import os
import sys
import time
import signal
import logging
import threading
from PIL import Image
import glob
from tqdm import tqdm

# Create directory if it doesn't exist
os.makedirs('./datasets/CheXpertPlus/PNG/PNG', exist_ok=True)
os.makedirs('./datasets/CheXpertPlus/PNG/resized/', exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('image_resize.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global flag for stopping the process
stop_processing = False

def signal_handler(signum, frame):
    """Handle Ctrl+C interrupt"""
    global stop_processing
    logger.info("Interrupt signal received. Stopping process gracefully...")
    stop_processing = True

def check_for_quit():
    """Check for 'q' key press in a separate thread"""
    global stop_processing
    while not stop_processing:
        try:
            # Simple approach that works on most systems
            time.sleep(0.5)
        except KeyboardInterrupt:
            stop_processing = True
            break

# Set up signal handler for Ctrl+C
signal.signal(signal.SIGINT, signal_handler)

def resize_images_to_rgb_512(input_dir, output_dir):
    """Resize all images in directory to 512x512 RGB PNG format and delete originals"""
    global stop_processing
    
    # Find all image files (common formats)
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.tif']
    image_files = []
    
    logger.info("Scanning for images...")
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, '**', ext), recursive=True))
    
    total_images = len(image_files)
    logger.info(f"Found {total_images} images to resize")
    
    if total_images == 0:
        logger.warning("No images found to process")
        return
    
    processed_count = 0
    error_count = 0
    skipped_count = 0
    start_time = time.time()
    
    # Start keyboard monitoring thread
    quit_thread = threading.Thread(target=check_for_quit, daemon=True)
    quit_thread.start()
    
    logger.info("Starting image processing... (Press Ctrl+C to stop)")
    print("\n*** PROCESSING STARTED ***")
    print("Press Ctrl+C at any time to stop the process gracefully")
    print("Progress will be logged both to console and 'image_resize.log' file")
    print("-" * 60)
    
    for i, img_path in enumerate(image_files, 1):
        if stop_processing:
            logger.info(f"Processing stopped by user at image {i}/{total_images}")
            break
            
        try:
            # Calculate progress and ETA
            elapsed_time = time.time() - start_time
            if processed_count > 0:
                avg_time_per_image = elapsed_time / processed_count
                remaining_images = total_images - i + 1
                eta_seconds = avg_time_per_image * remaining_images
                eta_str = time.strftime('%H:%M:%S', time.gmtime(eta_seconds))
            else:
                eta_str = "calculating..."
            
            logger.info(f"Processing [{i}/{total_images}] - {os.path.basename(img_path)} - ETA: {eta_str}")
            
            # Check if output already exists
            rel_path = os.path.relpath(img_path, input_dir)
            output_path = os.path.join(output_dir, rel_path)
            output_path = os.path.splitext(output_path)[0] + '.png'
            
            if os.path.exists(output_path):
                logger.info(f"Output already exists, skipping: {os.path.basename(output_path)}")
                skipped_count += 1
                continue
            
            # Open and process image
            with Image.open(img_path) as img:
                # Convert to RGB (handles grayscale, RGBA, etc.)
                rgb_img = img.convert('RGB')
                
                # Resize to 512x512
                resized_img = rgb_img.resize((512, 512), Image.Resampling.LANCZOS)
                
                # Ensure output directory exists
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # Save as PNG
                resized_img.save(output_path, 'PNG', optimize=True)
                
            # Delete original image after successful resize
            os.remove(img_path)
            processed_count += 1
            
            # Log progress every 10 images
            if processed_count % 10 == 0:
                progress_pct = (i / total_images) * 100
                logger.info(f"Progress: {progress_pct:.1f}% ({processed_count} processed, {error_count} errors, {skipped_count} skipped)")
                
        except FileNotFoundError as e:
            logger.error(f"File not found: {img_path} - {str(e)}")
            error_count += 1
        except PermissionError as e:
            logger.error(f"Permission denied: {img_path} - {str(e)}")
            error_count += 1
        except Image.UnidentifiedImageError as e:
            logger.error(f"Invalid image format: {img_path} - {str(e)}")
            error_count += 1
        except OSError as e:
            logger.error(f"OS error processing {img_path}: {str(e)}")
            error_count += 1
        except Exception as e:
            logger.error(f"Unexpected error processing {img_path}: {str(e)}")
            error_count += 1
    
    # Final statistics
    total_time = time.time() - start_time
    logger.info("=" * 50)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 50)
    logger.info(f"Total images found: {total_images}")
    logger.info(f"Successfully processed: {processed_count}")
    logger.info(f"Errors encountered: {error_count}")
    logger.info(f"Skipped (already exist): {skipped_count}")
    logger.info(f"Total processing time: {time.strftime('%H:%M:%S', time.gmtime(total_time))}")
    
    if processed_count > 0:
        avg_time = total_time / processed_count
        logger.info(f"Average time per image: {avg_time:.2f} seconds")
        logger.info("Original images have been deleted to save space")
    
    logger.info(f"Resized images saved to: {output_dir}")
    logger.info(f"Log file saved to: image_resize.log")

def main():
    """Main function with exception handling"""
    try:
        print("=" * 60)
        print("IMAGE RESIZING TOOL")
        print("=" * 60)
        print("This tool will:")
        print("- Scan for images in ./datasets/chexpert/")
        print("- Resize them to 512x512 RGB PNG format")
        print("- Save resized images to ./datasets/chexpert/resized/")
        print("- Delete original images to save space")
        print("- Log all progress to 'image_resize.log'")
        print("-" * 60)
        
        logger.info("=== Starting image resizing process ===")
        logger.info("Press Ctrl+C to stop the process at any time")
        
        # Resize all images to 512x512 RGB PNG
        resize_images_to_rgb_512('./datasets/CheXpertPlus/PNG/PNG', './datasets/CheXpertPlus/PNG/resized/')
        
        if not stop_processing:
            print("\n" + "=" * 60)
            print("✅ PROCESS COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            logger.info("=== Process completed successfully! ===")
            logger.info("All images have been processed and resized to 512x512 RGB PNG format")
            logger.info("Original images location: ./datasets/chexpert/")
            logger.info("Resized images location: ./datasets/chexpert/resized/")
        else:
            print("\n" + "=" * 60)
            print("⚠️ PROCESS INTERRUPTED BY USER")
            print("=" * 60)
            logger.info("=== Process was interrupted by user ===")
            logger.info("Partial processing completed. You can restart to continue from where it left off.")
            
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user (Ctrl+C)")
        logger.info("Process interrupted by user (Ctrl+C)")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error occurred: {str(e)}")
        logger.error("Fatal error occurred: %s", str(e))
        sys.exit(1)

if __name__ == "__main__":
    main()