"""
Storage utilities for saving and loading data in Parquet format.

Author: @BorcherdingL
Date: 4/9/2025
"""

import os
import logging

class ParquetStorage:
    """"Utility class for saving and loading data in Parquet format."""
    
    @staticmethod
    def save_to_parquet(data, file_path):
        """Save data to a Parquet file."""
        try:
            import pandas as pd
            import pyarrow as pa
            import pyarrow.parquet as pq
            
            # Convert to DataFrame if it's a dictionary
            if isinstance(data, dict):
                df = pd.DataFrame([data])
            elif isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = data
                
            # Save to Parquet
            pq.write_table(pa.Table.from_pandas(df), file_path)
            logging.info(f"Data saved to {file_path}")
            return True
        except Exception as e:
            logging.error(f"Error saving to Parquet: {e}")
            return False
            
    @staticmethod
    def load_from_parquet(file_path):
        """Load data from a Parquet file."""
        try:
            import pandas as pd
            if not os.path.exists(file_path):
                return None
                
            import pyarrow.parquet as pq
            table = pq.read_table(file_path)
            df = table.to_pandas()
            return df
        except Exception as e:
            logging.error(f"Error loading from Parquet: {e}")
            return None
            
    @staticmethod
    def append_to_parquet(data, file_path):
        """Append data to an existing Parquet file or create a new one."""
        try:
            import pandas as pd
            # Load existing data if available
            if os.path.exists(file_path):
                existing_df = ParquetStorage.load_from_parquet(file_path)
                if existing_df is not None:
                    # Convert new data to DataFrame
                    if isinstance(data, dict):
                        new_df = pd.DataFrame([data])
                    elif isinstance(data, list):
                        new_df = pd.DataFrame(data)
                    else:
                        new_df = data
                        
                    # Combine and save
                    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                    return ParquetStorage.save_to_parquet(combined_df, file_path)
            
            # If file doesn't exist or loading failed, create new file
            return ParquetStorage.save_to_parquet(data, file_path)
        except Exception as e:
            logging.error(f"Error appending to Parquet: {e}")
            return False