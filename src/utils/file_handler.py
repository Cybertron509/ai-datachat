"""File handling utilities for loading and processing various file formats"""
import pandas as pd
from pathlib import Path
import json
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class FileHandler:
    """Handle file operations for data loading"""
    
    SUPPORTED_FORMATS = ['.csv', '.xlsx', '.xls', '.json']
    
    @staticmethod
    def load_file(file_path: Path) -> pd.DataFrame:
        """Load data from various file formats"""
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()
        
        try:
            if suffix == '.csv':
                return pd.read_csv(file_path)
            elif suffix in ['.xlsx', '.xls']:
                return pd.read_excel(file_path)
            elif suffix == '.json':
                return pd.read_json(file_path)
            else:
                raise ValueError(f"Unsupported file format: {suffix}")
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {str(e)}")
            raise
    
    @staticmethod
    def get_file_info(file_path: Path) -> Dict[str, Any]:
        """Get file metadata"""
        file_path = Path(file_path)
        return {
            "name": file_path.name,
            "format": file_path.suffix[1:].upper(),
            "size_mb": round(file_path.stat().st_size / (1024 * 1024), 2)
        }