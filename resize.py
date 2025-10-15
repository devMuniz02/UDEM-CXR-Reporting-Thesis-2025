import os
from PIL import Image
from multiprocessing import Pool, cpu_count
import time

# --- CONFIGURACIÓN ---
source_images_dir = r"/mnt/e/MIMIC/p12"
output_dir = r"/mnt/e/MIMIC/matched_images_and_masks_mimic_512"
TARGET_SIZE = (512, 512)
NUM_PROCESSES = cpu_count()
 
# --- FIN DE CONFIGURACIÓN ---

def process_image(args):
    image_path, output_path, target_size = args
    
    if os.path.exists(output_path):
        return "skipped"
    
    try:
        with Image.open(image_path) as img:
            if img.mode in ('RGBA', 'LA', 'P'):
                img = img.convert('RGB')
            
            img_resized = img.resize(target_size, Image.LANCZOS)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            if not output_path.lower().endswith('.png'):
                output_path = output_path.rsplit('.', 1)[0] + '.png'
            
            img_resized.save(output_path, 'PNG', optimize=True)
        return "success"
    
    except Exception as e:
        print(f"❌ Error: {os.path.basename(image_path)} - {e}")
        return "error"

def get_all_image_paths():
    """Solo buscar imágenes, no todos los archivos"""
    image_paths = []
    found_count = 0
    
    print("🔍 Buscando imágenes...")
    
    for root, dirs, files in os.walk(source_images_dir):
        for filename in files:
            if filename.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp")):
                image_path = os.path.join(root, filename)
                relative_path = os.path.relpath(image_path, source_images_dir)
                
                if not relative_path.lower().endswith('.png'):
                    relative_path = relative_path.rsplit('.', 1)[0] + '.png'
                
                output_path = os.path.join(output_dir, relative_path)
                image_paths.append((image_path, output_path, TARGET_SIZE))
                
                found_count += 1
                if found_count % 1000 == 0:
                    print(f"📊 Imágenes encontradas: {found_count:,}...")
    
    return image_paths

def resize_all_images_parallel():
    print("🔄 Buscando imágenes...")
    start_time = time.time()
    
    all_images = get_all_image_paths()
    total_images = len(all_images)
    
    scan_time = time.time() - start_time
    print(f"✅ Encontradas {total_images:,} imágenes en {scan_time:.2f} segundos")
    
    if total_images == 0:
        print("❌ No se encontraron imágenes. Verifica la ruta:")
        print(f"   Directorio fuente: {source_images_dir}")
        return
    
    print(f"⚡ Usando {NUM_PROCESSES} procesos paralelos")
    print("⏳ Iniciando redimensionado...\n")
    
    copied_count = 0
    skipped_count = 0
    error_count = 0
    
    # Procesamiento paralelo
    with Pool(processes=NUM_PROCESSES) as pool:
        results = pool.imap_unordered(process_image, all_images, chunksize=50)
        
        for i, result in enumerate(results, 1):
            if result == "success":
                copied_count += 1
            elif result == "skipped":
                skipped_count += 1
            elif result == "error":
                error_count += 1
            
            if i % 100 == 0 or i == total_images:
                elapsed = time.time() - start_time
                images_per_sec = i / elapsed if elapsed > 0 else 0
                
                print(f"📊 {i:,}/{total_images:,} ({i/total_images*100:.1f}%) - "
                      f"✅{copied_count:,} ⏭️{skipped_count:,} ❌{error_count:,} - "
                      f"🚀{images_per_sec:.1f} img/s")
    
    total_time = time.time() - start_time
    
    print("\n" + "="*60)
    print("✅ PROCESO COMPLETADO")
    print(f"Total: {total_images:,} | Nuevas: {copied_count:,}")
    print(f"Saltadas: {skipped_count:,} | Errores: {error_count:,}")
    print(f"Tiempo: {total_time:.1f}s | Velocidad: {total_images/total_time:.1f} img/s")
    print("="*60)

if __name__ == "__main__":
    resize_all_images_parallel()