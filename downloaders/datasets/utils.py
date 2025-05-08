#!/usr/bin/env python3
import base64
import io
import logging
import os
import pandas as pd
import sys

from pathlib import Path
from PIL import Image
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

N_ROWS = 4000

class ImageProcessor:
    def __init__(self, temp_dir: str, output_dir: str):
        """Initialize the image processor.
        
        Args:
            temp_dir: Directory containing the parquet files
            output_dir: Directory where to save the extracted images
        """
        self.temp_dir = Path(temp_dir)
        self.output_dir = Path(output_dir)
        
    def _get_image_bytes(self, row: pd.Series, idx: int) -> Optional[bytes]:
        """Extract image bytes from a dataframe row.
        
        Args:
            row: Pandas Series containing the row data
            idx: Row index for logging purposes
            
        Returns:
            bytes: Image data if successfully extracted, None otherwise
        """
        try:
            # Handle different image data structures
            if isinstance(row.get('image'), dict) and 'bytes' in row['image']:
                img_bytes = row['image']['bytes']
            elif 'image.bytes' in row:
                img_bytes = row['image.bytes']
            else:
                logger.warning(f"No image data found in row {idx}")
                return None
                
            if pd.isna(img_bytes):
                logger.warning(f"Image data is NA in row {idx}")
                return None
                
            # Debug image data
            logger.debug(f"Row {idx} image data type: {type(img_bytes)}")
            logger.debug(f"Row {idx} image data length: {len(img_bytes) if isinstance(img_bytes, (bytes, str)) else 'N/A'}")
            
            # Handle string (possibly base64) vs bytes
            if isinstance(img_bytes, str):
                try:
                    return base64.b64decode(img_bytes)
                except:
                    logger.warning(f"Row {idx} - Failed base64 decode, using direct bytes")
                    return img_bytes.encode() if isinstance(img_bytes, str) else img_bytes
            return img_bytes
            
        except Exception as e:
            logger.error(f"Error extracting image bytes from row {idx}: {str(e)}")
            return None
            
    def _save_image(self, img_data: bytes, output_path: Path) -> bool:
        """Save image data to a file.
        
        Args:
            img_data: Raw image bytes
            output_path: Path where to save the image
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            img = Image.open(io.BytesIO(img_data))
            img.save(output_path)
            logger.info(f"Saved {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving image to {output_path}: {str(e)}")
            return False
            
    def process_parquet_files(self, prefix: str, image_prefix: str) -> None:
        """Process parquet files and extract images.
        
        Args:
            prefix: Prefix to filter parquet files (e.g., 'train')
            image_prefix: Prefix for saved image files (e.g., 'chart' or 'receipt')
        """
        for parquet_file in self.temp_dir.glob('*.parquet'):
            if not parquet_file.name.startswith(prefix):
                continue
                
            logger.info(f"Processing {parquet_file}...")
            df = pd.read_parquet(parquet_file)
            logger.info(f"Columns: {df.columns}")
            
            # Limit to N_ROWS rows
            df = df.head(N_ROWS)
            
            # Process each row
            for idx, row in df.iterrows():
                img_data = self._get_image_bytes(row, idx)
                if img_data is None:
                    continue
                    
                output_path = self.output_dir / f'{image_prefix}_{idx}.jpg'
                self._save_image(img_data, output_path)

def process_chartqa_parquet(temp_dir: str, output_dir: str) -> None:
    """Process ChartQA parquet files and extract images."""
    processor = ImageProcessor(temp_dir, output_dir)
    processor.process_parquet_files('train', 'chart')

def process_cord_parquet(temp_dir: str, output_dir: str) -> None:
    """Process CORD parquet files and extract images."""
    processor = ImageProcessor(temp_dir, output_dir)
    processor.process_parquet_files('train', 'receipt')

if __name__ == '__main__':
    if len(sys.argv) != 4:
        logger.error("Usage: utils.py <dataset_type> <temp_dir> <output_dir>")
        sys.exit(1)
        
    dataset_type = sys.argv[1]
    temp_dir = sys.argv[2]
    output_dir = sys.argv[3]
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    processors = {
        'chartqa': process_chartqa_parquet,
        'cord': process_cord_parquet
    }
    
    if dataset_type not in processors:
        logger.error(f"Unknown dataset type: {dataset_type}")
        sys.exit(1)
        
    processors[dataset_type](temp_dir, output_dir) 