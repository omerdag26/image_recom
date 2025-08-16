# populate_db.py
import argparse
from database import ImageDatabase
from loader import image_generator
from tqdm import tqdm

# This function populates the database with images found in the specified directory by iterating over the generator.
def populate_database(image_root, db_path='images.db'):
    # Create the database instance
    db = ImageDatabase(db_path)
    # Count for progress tracking
    count = 0
    # Estimated for loop
    estimated = 560000
    # Iterate over the image generator and add images to the database
    print(f"Populating database {db_path} with images from {image_root}...")
    for filepath, w, h in tqdm(image_generator(image_root),total=estimated):
        db.add_image(count, filepath, w, h)
        count += 1
    # Close the database connection
    db.close()
    # Print the final count of images added
    print(f"Done: inserted {count} images into {db_path}.")

# Main function to handle command line arguments and start the population process
if __name__ == '__main__':
    # Set up argument parsing for command line usage
    parser = argparse.ArgumentParser(
        description='Populate the SQLite DB from a directory of images.')
    # Image root for the images
    parser.add_argument('image_root',
                        help='Root folder (etc. /Volumes/BigDataA/data/image_data/)')
    # Database file name, defaulting to 'images.db'
    parser.add_argument('--db', default='images.db',
                        help='DB file name (default: images.db)')
    # Parse the arguments
    args = parser.parse_args()
    # Create the database by populating it with images
    populate_database(args.image_root, args.db)