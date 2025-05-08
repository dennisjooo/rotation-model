"""
MIDV-500 Dataset Downloader

Based on the work by Fatih Cagatay Akyon (https://github.com/fcakyon/midv500)
Modified and enhanced with progress bars and error handling.

Original paper:
[1] Bulatov, K., Matalov, D., Arlazarov, V.V.: MIDV-500: a dataset for identity document analysis 
and recognition on mobile devices in video stream. Компьютерная оптика 44(5), 818-824 (2020)
"""

from __future__ import annotations
import concurrent.futures
import ftplib
import logging
import os
import zipfile
from pathlib import Path
from tqdm.auto import tqdm
from typing import List, Optional
from urllib.parse import urlparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FTPDownloader:
    """Enhanced FTP downloader with progress tracking and error handling.
    
    Attributes:
        pbar: Progress bar for tracking download progress
        last: Last downloaded chunk size for progress calculation
    """
    
    def __init__(self, total_size: int) -> None:
        """Initialize the downloader with a progress bar.
        
        Args:
            total_size: Total size of the file to download in bytes
        """
        self.pbar = tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading")
        self.last = 0
        
    def update(self, block: bytes) -> None:
        """Update progress bar with new data block.
        
        Args:
            block: New data block received from FTP
        """
        current = len(block)
        self.pbar.update(current - self.last)
        self.last = current
        
    def finish(self) -> None:
        """Close the progress bar."""
        self.pbar.close()

def get_file_size(ftp: ftplib.FTP, path: str) -> int:
    """Get file size from FTP server with error handling.
    
    Args:
        ftp: FTP connection object
        path: Path to the file on FTP server
        
    Returns:
        File size in bytes, or 0 if size cannot be determined
    """
    try:
        ftp.voidcmd('TYPE I')
        size = ftp.size(path)
        return size if size is not None else 0
    except Exception as e:
        logger.warning(f"Could not get size for {path}: {str(e)}")
        return 0

def estimate_total_size(links: List[str]) -> int:
    """Estimate total download size with improved error handling.
    
    Args:
        links: List of FTP URLs to check
        
    Returns:
        Total size in bytes of all files
    """
    total_size = 0
    try:
        ftp = ftplib.FTP('smartengines.com', timeout=30)
        ftp.login()
        
        print("Estimating total download size...")
        with tqdm(total=len(links), desc="Checking files", unit="file") as pbar:
            for link in links:
                try:
                    parsed = urlparse(link)
                    size = get_file_size(ftp, parsed.path)
                    total_size += size
                except Exception as e:
                    logger.warning(f"Error checking size for {link}: {str(e)}")
                finally:
                    pbar.update(1)
        
        ftp.quit()
    except Exception as e:
        logger.error(f"FTP connection error: {str(e)}")
        return 0
    
    return total_size

def download(url: str, dst: str | Path, retries: int = 3) -> Optional[str]:
    """Download file from FTP server with retries and error handling.
    
    Args:
        url: FTP URL to download
        dst: Destination directory
        retries: Number of retry attempts
        
    Returns:
        Path to downloaded file, or None if download failed
    """
    parsed = urlparse(url)
    filename = os.path.basename(parsed.path)
    dst_path = os.path.join(dst, filename)
    
    os.makedirs(dst, exist_ok=True)
    
    for attempt in range(retries):
        try:
            ftp = ftplib.FTP('smartengines.com', timeout=30)
            ftp.login()
            
            size = get_file_size(ftp, parsed.path)
            downloader = FTPDownloader(size)
            
            with open(dst_path, 'wb') as f:
                def callback(data: bytes) -> None:
                    f.write(data)
                    downloader.update(data)
                ftp.retrbinary(f'RETR {parsed.path}', callback)
            
            downloader.finish()
            ftp.quit()
            return dst_path
            
        except Exception as e:
            logger.error(f"Download attempt {attempt + 1} failed: {str(e)}")
            if attempt == retries - 1:
                return None
            continue

def unzip(zip_path: str | Path, dst: str | Path) -> None:
    """Extract zip file with progress tracking and error handling.
    
    Args:
        zip_path: Path to zip file
        dst: Destination directory for extracted files
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            total_size = sum(info.file_size for info in zip_ref.filelist)
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Extracting") as pbar:
                for member in zip_ref.filelist:
                    try:
                        zip_ref.extract(member, dst)
                        pbar.update(member.file_size)
                    except Exception as e:
                        logger.error(f"Error extracting {member.filename}: {str(e)}")
    except Exception as e:
        logger.error(f"Error opening zip file {zip_path}: {str(e)}")
        raise

def process_single_file(link: str, download_dir: str | Path) -> bool:
    """Process a single dataset file - download, unzip, and cleanup.
    
    Args:
        link: FTP URL to download
        download_dir: Directory to download and extract the dataset
        
    Returns:
        bool: True if processing was successful, False otherwise
    """
    print("\n" + "-" * 70)
    print(f"Processing: {os.path.basename(link)}")
    try:
        # Download zip file
        zip_path = download(link, download_dir)
        if zip_path is None:
            logger.error(f"Failed to download {link}")
            return False
        
        # Unzip file
        unzip(zip_path, download_dir)
        
        # Remove zip file
        os.remove(zip_path)
        print(f"Cleaned up {os.path.basename(zip_path)}")
        return True
        
    except Exception as e:
        logger.error(f"Error processing {link}: {str(e)}")
        return False

def download_dataset(download_dir: str | Path, skip_confirmation: bool = False, max_workers: int = 4) -> None:
    """Download and extract MIDV-500 dataset using parallel processing.
    
    Args:
        download_dir: Directory to download and extract the dataset
        skip_confirmation: Whether to skip the confirmation prompt
        max_workers: Maximum number of concurrent downloads
    """
    # Estimate total size
    total_size = estimate_total_size(dataset_links)
    print(f"\nTotal download size: {total_size / (1024**3):.1f} GB")
    print(f"Estimated extracted size: {(total_size * 2) / (1024**3):.1f} GB")
    
    # Ask for confirmation unless skipped
    if not skip_confirmation:
        response = input("\nDo you want to proceed with the download? [y/N] ")
        if response.lower() != 'y':
            print("Download cancelled.")
            return
    
    # Add overall progress bar for all files
    with tqdm(total=len(dataset_links), desc="Overall Progress", unit="file") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all download tasks
            future_to_link = {
                executor.submit(process_single_file, link, download_dir): link 
                for link in dataset_links
            }
            
            # Process completed downloads
            for future in concurrent.futures.as_completed(future_to_link):
                link = future_to_link[future]
                try:
                    success = future.result()
                    if success:
                        pbar.update(1)
                except Exception as e:
                    logger.error(f"Unexpected error processing {link}: {str(e)}")

# List of selected MIDV-500 dataset files (most common document types)
dataset_links = [
    # IDs from major countries
    "ftp://smartengines.com/midv-500/dataset/04_aut_id.zip",      # Austrian ID
    "ftp://smartengines.com/midv-500/dataset/09_chn_id.zip",      # Chinese ID
    "ftp://smartengines.com/midv-500/dataset/14_deu_id_new.zip",  # German ID (new)
    "ftp://smartengines.com/midv-500/dataset/20_esp_id_new.zip",  # Spanish ID (new)
    "ftp://smartengines.com/midv-500/dataset/25_fin_id.zip",      # Finnish ID
    "ftp://smartengines.com/midv-500/dataset/29_ita_id.zip",      # Italian ID
    "ftp://smartengines.com/midv-500/dataset/32_lva_id.zip",      # Latvian ID
    "ftp://smartengines.com/midv-500/dataset/35_pol_id.zip",      # Polish ID
    "ftp://smartengines.com/midv-500/dataset/41_srb_id.zip",      # Serbian ID
    "ftp://smartengines.com/midv-500/dataset/43_svk_id.zip",      # Slovak ID
    "ftp://smartengines.com/midv-500/dataset/47_usa_id.zip",      # USA ID
    "ftp://smartengines.com/midv-500/dataset/48_usa_id_new.zip",  # USA ID (new)
    "ftp://smartengines.com/midv-500/dataset/49_rus_internalpassport.zip",  # Russian internal passport
    
    # International Passports
    "ftp://smartengines.com/midv-500/dataset/06_bra_passport.zip",     # Brazilian passport
    "ftp://smartengines.com/midv-500/dataset/16_deu_passport_new.zip", # German passport (new)
    "ftp://smartengines.com/midv-500/dataset/27_hrv_passport.zip",     # Croatian passport
    "ftp://smartengines.com/midv-500/dataset/45_ukr_passport.zip",     # Ukrainian passport
    
    # Driver's Licenses
    "ftp://smartengines.com/midv-500/dataset/02_aut_drvlic_new.zip",  # Austrian driver license (new)
    "ftp://smartengines.com/midv-500/dataset/12_deu_drvlic_new.zip",  # German driver license (new)
    "ftp://smartengines.com/midv-500/dataset/23_fin_drvlic.zip",      # Finnish driver license
    "ftp://smartengines.com/midv-500/dataset/31_jpn_drvlic.zip",      # Japanese driver license
]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='Download MIDV-500 dataset with enhanced progress tracking',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--dir', default="data/midv500", help='Directory to download the dataset')
    parser.add_argument('--yes', '-y', action='store_true', help='Skip confirmation and proceed with download')
    parser.add_argument('--workers', type=int, default=4, help='Number of concurrent downloads')
    args = parser.parse_args()
    
    download_dataset(args.dir, args.yes, args.workers) 