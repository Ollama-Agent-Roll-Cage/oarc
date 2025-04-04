from typing import Union, List, Dict, Tuple, Literal, Optional, Any
import logging
import json
from dataclasses import dataclass, field
import time
import asyncio
from functools import lru_cache
import threading


class OllamaGPUMonitor:
    """
    A class to monitor Ollama processes and GPU usage.
    Combines Ollama process information with GPU status for API integration.
    """

    def __init__(
        self, 
        ollama_client=None, 
        default_memory_unit: str = "MB",
        cache_duration: int = 5,
        enable_background_refresh: bool = False,
        background_refresh_interval: int = 10,
        retry_attempts: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize the OllamaGPUMonitor.

        Args:
            ollama_client: An instance of ollama.Client or ollama.AsyncClient (optional)
            default_memory_unit: Default unit for memory measurements (default: "MB")
            cache_duration: Duration in seconds to cache results (default: 5)
            enable_background_refresh: Whether to enable background data refresh (default: False)
            background_refresh_interval: Interval in seconds for background refresh (default: 10)
            retry_attempts: Number of retry attempts for failed operations (default: 3)
            retry_delay: Delay in seconds between retry attempts (default: 1.0)
        """
        self.default_memory_unit = default_memory_unit
        self.ollama_client = ollama_client
        self.logger = logging.getLogger(__name__)
        self.cache_duration = cache_duration
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        
        # Cache variables
        self._gpu_cache = {"timestamp": 0, "data": []}
        self._ollama_cache = {"timestamp": 0, "data": []}
        self._combined_cache = {"timestamp": 0, "data": []}
        
        # Background refresh settings
        self.enable_background_refresh = enable_background_refresh
        self.background_refresh_interval = background_refresh_interval
        self._refresh_thread = None
        self._stop_event = threading.Event()
        
        # Initialize client
        self._setup_client()
        
        # Start background refresh if enabled
        if self.enable_background_refresh:
            self._start_background_refresh()

    def __del__(self):
        """Cleanup when object is deleted."""
        self._stop_background_refresh()

    def _setup_client(self):
        """Set up Ollama client if not provided."""
        if self.ollama_client is None:
            try:
                import ollama
                self.ollama_client = ollama.Client()
                self.logger.info("Initialized Ollama synchronous client")
            except ImportError:
                self.logger.error("Ollama Python library not found. Please install with 'pip install ollama'")
                raise ImportError("Ollama Python library not found. Please install with 'pip install ollama'")
            except Exception as e:
                self.logger.error(f"Failed to initialize Ollama client: {str(e)}")
                raise

    def _start_background_refresh(self):
        """Start the background refresh thread."""
        if self._refresh_thread is None or not self._refresh_thread.is_alive():
            self._stop_event.clear()
            self._refresh_thread = threading.Thread(target=self._background_refresh_worker, daemon=True)
            self._refresh_thread.start()
            self.logger.info("Started background refresh thread")

    def _stop_background_refresh(self):
        """Stop the background refresh thread."""
        if self._refresh_thread and self._refresh_thread.is_alive():
            self._stop_event.set()
            self._refresh_thread.join(timeout=2.0)
            self.logger.info("Stopped background refresh thread")

    def _background_refresh_worker(self):
        """Worker function for background data refresh."""
        while not self._stop_event.is_set():
            try:
                # Refresh caches
                self.get_gpu_status(force_refresh=True)
                self.get_ollama_processes(force_refresh=True)
                self.get_model_gpu_usage(force_refresh=True)
                self.logger.debug("Background refresh completed")
            except Exception as e:
                self.logger.error(f"Error in background refresh: {str(e)}")
            
            # Sleep until next refresh
            self._stop_event.wait(self.background_refresh_interval)

    def _should_refresh_cache(self, cache: Dict, force_refresh: bool = False) -> bool:
        """
        Determine if cache should be refreshed.
        
        Args:
            cache: Cache dictionary with timestamp and data
            force_refresh: Whether to force a refresh regardless of timestamp
            
        Returns:
            bool: True if cache should be refreshed
        """
        if force_refresh:
            return True
        return time.time() - cache["timestamp"] > self.cache_duration

    def _with_retry(self, func, *args, **kwargs):
        """
        Execute a function with retry logic.
        
        Args:
            func: Function to execute
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            Any: Result of the function
        """
        last_exception = None
        for attempt in range(self.retry_attempts):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                self.logger.warning(f"Attempt {attempt+1}/{self.retry_attempts} failed: {str(e)}")
                time.sleep(self.retry_delay)
        
        self.logger.error(f"All {self.retry_attempts} retry attempts failed")
        if last_exception:
            raise last_exception
        return None

    async def _async_with_retry(self, func, *args, **kwargs):
        """
        Execute an async function with retry logic.
        
        Args:
            func: Async function to execute
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            Any: Result of the function
        """
        last_exception = None
        for attempt in range(self.retry_attempts):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                self.logger.warning(f"Async attempt {attempt+1}/{self.retry_attempts} failed: {str(e)}")
                await asyncio.sleep(self.retry_delay)
        
        self.logger.error(f"All {self.retry_attempts} async retry attempts failed")
        if last_exception:
            raise last_exception
        return None

    @staticmethod
    def convert_byte_unit(
        value: float,
        src_unit: Literal["b", "B", "KB", "MB", "GB", "TB"],
        target_unit: Literal["b", "B", "KB", "MB", "GB", "TB"],
    ) -> float:
        """
        Convert value in source unit to target unit.
        First converts source unit to Bytes, then to target unit.

        Args:
            value: The value to convert
            src_unit: Source unit ("b", "B", "KB", "MB", "GB", "TB")
            target_unit: Target unit ("b", "B", "KB", "MB", "GB", "TB")

        Returns:
            float: The converted value in target_unit

        Raises:
            ValueError: If source or target unit is not valid
        """
        # Handle edge cases
        if value is None:
            return 0.0
            
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"Value must be a number, got {type(value)}")
            
        # Convert to bytes first
        if src_unit in ["b", "bit"]:
            value = value / 8
        elif src_unit in ["B", "Byte"]:
            pass
        elif src_unit == "KB":
            value = value * 1024
        elif src_unit == "MB":
            value = value * 1024**2
        elif src_unit == "GB":
            value = value * (1024**3)
        elif src_unit == "TB":
            value = value * (1024**4)
        else:
            raise ValueError(f"Source unit '{src_unit}' is not valid")
        
        # Convert from bytes to target unit
        if target_unit in ["b", "bit"]:
            target_value = value * 8
        elif target_unit in ["B", "Byte"]:
            target_value = value
        elif target_unit == "KB":
            target_value = value / 1024
        elif target_unit == "MB":
            target_value = value / 1024**2
        elif target_unit == "GB":
            target_value = value / (1024**3)
        elif target_unit == "TB":
            target_value = value / (1024**4)
        else:
            raise ValueError(f"Target unit '{target_unit}' is not valid")
            
        return target_value

    def _get_gpu_status_impl(self, unit: str = None) -> List[Dict]:
        """
        Internal implementation to get GPU status information.

        Args:
            unit: Memory unit to use

        Returns:
            List[Dict]: List of dictionaries containing GPU information
        """
        if unit is None:
            unit = self.default_memory_unit
            
        try:
            import pynvml
            infos = []
            
            # Initialize pynvml
            pynvml.nvmlInit()
            
            # Get GPU count
            device_count = pynvml.nvmlDeviceGetCount()
            
            # Get information for each GPU
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_name = pynvml.nvmlDeviceGetName(handle)
                
                # Get additional information if available
                try:
                    temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                except:
                    temperature = None
                    
                try:
                    power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert from mW to W
                except:
                    power_usage = None
                    
                try:
                    power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0  # Convert from mW to W
                except:
                    power_limit = None
                
                gpu_info = {
                    "gpu_id": i,
                    "gpu_name": gpu_name,
                    "total_memory": self.convert_byte_unit(
                        info.total, src_unit="B", target_unit=unit
                    ),
                    "used_memory": self.convert_byte_unit(
                        info.used, src_unit="B", target_unit=unit
                    ),
                    "used_memory_ratio": info.used / info.total if info.total > 0 else 0,
                    "gpu_utilization": utilization.gpu,
                    "memory_utilization": utilization.memory,
                    "free_memory_ratio": info.free / info.total if info.total > 0 else 0,
                    "free_memory": self.convert_byte_unit(
                        info.free, src_unit="B", target_unit=unit
                    ),
                    "temperature": temperature,
                    "power_usage": power_usage,
                    "power_limit": power_limit,
                    "unit": unit
                }
                infos.append(gpu_info)
                
            # Release pynvml
            pynvml.nvmlShutdown()
            return infos
            
        except ImportError:
            self.logger.error("pynvml module not found. Please install with 'pip install nvidia-ml-py'")
            return []
        except Exception as e:
            self.logger.error(f"Failed to get GPU status: {str(e)}")
            self.logger.exception(e)
            return []

    def get_gpu_status(self, unit: str = None, force_refresh: bool = False) -> List[Dict]:
        """
        Get GPU status information with caching.

        Args:
            unit: Memory unit to use (default: uses instance default_memory_unit)
            force_refresh: Whether to force a refresh regardless of cache

        Returns:
            List[Dict]: List of dictionaries containing GPU information
        """
        if unit is None:
            unit = self.default_memory_unit
            
        # Check if we should use cached data
        if not self._should_refresh_cache(self._gpu_cache, force_refresh) and unit == self._gpu_cache.get("unit", self.default_memory_unit):
            return self._gpu_cache["data"]
            
        # Get fresh data with retry
        data = self._with_retry(self._get_gpu_status_impl, unit)
        
        # Update cache
        self._gpu_cache = {
            "timestamp": time.time(),
            "data": data,
            "unit": unit
        }
        
        return data

    def _get_ollama_processes_impl(self) -> List[Dict]:
        """
        Internal implementation to get information about running Ollama processes.

        Returns:
            List[Dict]: List of dictionaries containing Ollama process information
        """
        try:
            processes = self.ollama_client.ps()
            
            # Ensure consistent structure and populate any missing fields
            standardized_processes = []
            for process in processes:
                standardized_process = {
                    "name": process.get("name", "Unknown"),
                    "id": process.get("id", "Unknown"),
                    "size": process.get("size", "Unknown"),
                    "processor": process.get("processor", "Unknown"),
                    "until": process.get("until", "Unknown"),
                }
                standardized_processes.append(standardized_process)
                
            return standardized_processes
        except Exception as e:
            self.logger.error(f"Failed to get Ollama processes: {str(e)}")
            return []

    def get_ollama_processes(self, force_refresh: bool = False) -> List[Dict]:
        """
        Get information about running Ollama processes with caching.

        Args:
            force_refresh: Whether to force a refresh regardless of cache

        Returns:
            List[Dict]: List of dictionaries containing Ollama process information
        """
        # Check if we should use cached data
        if not self._should_refresh_cache(self._ollama_cache, force_refresh):
            return self._ollama_cache["data"]
            
        # Get fresh data with retry
        data = self._with_retry(self._get_ollama_processes_impl)
        
        # Update cache
        self._ollama_cache = {
            "timestamp": time.time(),
            "data": data
        }
        
        return data

    def _get_model_gpu_usage_impl(self) -> List[Dict]:
        """
        Internal implementation to get combined information about Ollama models and their GPU usage.

        Returns:
            List[Dict]: List of dictionaries with model and GPU usage information
        """
        try:
            processes = self.get_ollama_processes()
            gpu_stats = self.get_gpu_status()
            
            # Handle edge case where no GPUs are available
            if not gpu_stats:
                return [{
                    "model_name": process.get("name", "Unknown"),
                    "model_id": process.get("id", "Unknown"),
                    "size": process.get("size", "Unknown"),
                    "processor": process.get("processor", "Unknown"),
                    "until": process.get("until", "Unknown"),
                    "gpu_info": None
                } for process in processes]
            
            # Map GPU utilization to models
            result = []
            
            for process in processes:
                model_info = {
                    "model_name": process.get("name", "Unknown"),
                    "model_id": process.get("id", "Unknown"),
                    "size": process.get("size", "Unknown"),
                    "processor": process.get("processor", "Unknown"),
                    "until": process.get("until", "Unknown"),
                    "gpu_info": None
                }
                
                # Try to match with GPU information
                if "GPU" in process.get("processor", ""):
                    gpu_percentage = process.get("processor", "0% GPU")
                    try:
                        gpu_percentage = int(gpu_percentage.split("%")[0])
                    except (ValueError, IndexError):
                        gpu_percentage = 0
                    
                    # Try to match based on GPU utilization    
                    best_match = None
                    best_match_diff = float('inf')
                    
                    for gpu in gpu_stats:
                        diff = abs(gpu["gpu_utilization"] - gpu_percentage)
                        if diff < best_match_diff:
                            best_match = gpu
                            best_match_diff = diff
                    
                    # Use best match if close enough, otherwise use most utilized GPU
                    if best_match_diff < 30:  # Threshold for matching
                        model_info["gpu_info"] = best_match
                    else:
                        # Find GPU with highest utilization
                        most_utilized_gpu = max(gpu_stats, key=lambda g: g["gpu_utilization"])
                        model_info["gpu_info"] = most_utilized_gpu
                        model_info["gpu_match_quality"] = "low"
                        
                result.append(model_info)
                
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to get model GPU usage: {str(e)}")
            return []

    def get_model_gpu_usage(self, force_refresh: bool = False) -> List[Dict]:
        """
        Get combined information about Ollama models and their GPU usage with caching.

        Args:
            force_refresh: Whether to force a refresh regardless of cache

        Returns:
            List[Dict]: List of dictionaries with model and GPU usage information
        """
        # Check if we should use cached data
        if not self._should_refresh_cache(self._combined_cache, force_refresh):
            return self._combined_cache["data"]
            
        # Get fresh data with retry
        data = self._with_retry(self._get_model_gpu_usage_impl)
        
        # Update cache
        self._combined_cache = {
            "timestamp": time.time(),
            "data": data
        }
        
        return data

    def format_status(self, include_gpu: bool = True, include_timestamp: bool = True) -> str:
        """
        Format the status of Ollama processes and GPU usage as a string.

        Args:
            include_gpu: Whether to include GPU information (default: True)
            include_timestamp: Whether to include timestamp (default: True)

        Returns:
            str: Formatted status string
        """
        try:
            processes = self.get_ollama_processes()
            
            result = ""
            if include_timestamp:
                result += f"STATUS AS OF: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                
            if not processes:
                result += "NO OLLAMA PROCESSES RUNNING.\n"
            else:
                result += "OLLAMA PROCESSES:\n"
                result += "{:<25} {:<15} {:<10} {:<15} {:<20}\n".format(
                    "NAME", "ID", "SIZE", "PROCESSOR", "UNTIL"
                )
                result += "─" * 85 + "\n"
                
                for process in processes:
                    result += "{:<25} {:<15} {:<10} {:<15} {:<20}\n".format(
                        process.get("name", "Unknown"),
                        process.get("id", "Unknown")[:12],
                        process.get("size", "Unknown"),
                        process.get("processor", "Unknown"),
                        process.get("until", "Unknown")
                    )
                
            if include_gpu:
                gpu_stats = self.get_gpu_status()
                if gpu_stats:
                    result += "\nGPU STATUS:\n"
                    result += "─" * 50 + "\n"
                    for i, gpu in enumerate(gpu_stats):
                        result += f"GPU {i}: {gpu['gpu_name']}\n"
                        result += f"  Memory: {gpu['used_memory']:.2f}/{gpu['total_memory']:.2f} {gpu.get('unit', self.default_memory_unit)}"
                        result += f" ({gpu['used_memory_ratio']*100:.1f}%)\n"
                        result += f"  Utilization: {gpu['gpu_utilization']}%\n"
                        
                        if gpu.get('temperature') is not None:
                            result += f"  Temperature: {gpu['temperature']}°C\n"
                            
                        if gpu.get('power_usage') is not None and gpu.get('power_limit') is not None:
                            result += f"  Power: {gpu['power_usage']:.2f}W / {gpu['power_limit']:.2f}W"
                            result += f" ({gpu['power_usage']/gpu['power_limit']*100:.1f}%)\n"
                        
                        result += "\n"
                
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to format status: {str(e)}")
            return f"Error formatting status: {str(e)}"

    async def _async_get_ollama_processes_impl(self) -> List[Dict]:
        """
        Internal implementation to get information about running Ollama processes asynchronously.

        Returns:
            List[Dict]: List of dictionaries containing Ollama process information
        """
        try:
            # Check if client is AsyncClient
            if not hasattr(self.ollama_client, 'ps') or not callable(getattr(self.ollama_client, 'ps')):
                import ollama
                self.ollama_client = ollama.AsyncClient()
                
            processes = await self.ollama_client.ps()
            
            # Ensure consistent structure and populate any missing fields
            standardized_processes = []
            for process in processes:
                standardized_process = {
                    "name": process.get("name", "Unknown"),
                    "id": process.get("id", "Unknown"),
                    "size": process.get("size", "Unknown"),
                    "processor": process.get("processor", "Unknown"),
                    "until": process.get("until", "Unknown"),
                }
                standardized_processes.append(standardized_process)
                
            return standardized_processes
        except Exception as e:
            self.logger.error(f"Failed to get Ollama processes asynchronously: {str(e)}")
            return []

    async def async_get_ollama_processes(self, force_refresh: bool = False) -> List[Dict]:
        """
        Get information about running Ollama processes asynchronously with caching.

        Args:
            force_refresh: Whether to force a refresh regardless of cache

        Returns:
            List[Dict]: List of dictionaries containing Ollama process information
        """
        # Check if we should use cached data
        if not self._should_refresh_cache(self._ollama_cache, force_refresh):
            return self._ollama_cache["data"]
            
        # Get fresh data with retry
        data = await self._async_with_retry(self._async_get_ollama_processes_impl)
        
        # Update cache
        self._ollama_cache = {
            "timestamp": time.time(),
            "data": data
        }
        
        return data

    async def async_get_model_gpu_usage(self, force_refresh: bool = False) -> List[Dict]:
        """
        Get combined information about Ollama models and their GPU usage asynchronously with caching.

        Args:
            force_refresh: Whether to force a refresh regardless of cache

        Returns:
            List[Dict]: List of dictionaries with model and GPU usage information
        """
        try:
            # Check if we should use cached data
            if not self._should_refresh_cache(self._combined_cache, force_refresh):
                return self._combined_cache["data"]
                
            processes = await self.async_get_ollama_processes(force_refresh)
            gpu_stats = self.get_gpu_status(force_refresh=force_refresh)  # GPU stats still use synchronous API
            
            # Handle edge case where no GPUs are available
            if not gpu_stats:
                data = [{
                    "model_name": process.get("name", "Unknown"),
                    "model_id": process.get("id", "Unknown"),
                    "size": process.get("size", "Unknown"),
                    "processor": process.get("processor", "Unknown"),
                    "until": process.get("until", "Unknown"),
                    "gpu_info": None
                } for process in processes]
                
                # Update cache
                self._combined_cache = {
                    "timestamp": time.time(),
                    "data": data
                }
                
                return data
            
            # Map GPU utilization to models
            result = []
            
            for process in processes:
                model_info = {
                    "model_name": process.get("name", "Unknown"),
                    "model_id": process.get("id", "Unknown"),
                    "size": process.get("size", "Unknown"),
                    "processor": process.get("processor", "Unknown"),
                    "until": process.get("until", "Unknown"),
                    "gpu_info": None
                }
                
                # Try to match with GPU information
                if "GPU" in process.get("processor", ""):
                    gpu_percentage = process.get("processor", "0% GPU")
                    try:
                        gpu_percentage = int(gpu_percentage.split("%")[0])
                    except (ValueError, IndexError):
                        gpu_percentage = 0
                    
                    # Try to match based on GPU utilization    
                    best_match = None
                    best_match_diff = float('inf')
                    
                    for gpu in gpu_stats:
                        diff = abs(gpu["gpu_utilization"] - gpu_percentage)
                        if diff < best_match_diff:
                            best_match = gpu
                            best_match_diff = diff
                    
                    # Use best match if close enough, otherwise use most utilized GPU
                    if best_match_diff < 30:  # Threshold for matching
                        model_info["gpu_info"] = best_match
                    else:
                        # Find GPU with highest utilization
                        most_utilized_gpu = max(gpu_stats, key=lambda g: g["gpu_utilization"])
                        model_info["gpu_info"] = most_utilized_gpu
                        model_info["gpu_match_quality"] = "low"
                        
                result.append(model_info)
            
            # Update cache
            self._combined_cache = {
                "timestamp": time.time(),
                "data": result
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to get model GPU usage asynchronously: {str(e)}")
            return []

    async def async_format_status(self, include_gpu: bool = True, include_timestamp: bool = True) -> str:
        """
        Format the status of Ollama processes and GPU usage as a string asynchronously.

        Args:
            include_gpu: Whether to include GPU information (default: True)
            include_timestamp: Whether to include timestamp (default: True)

        Returns:
            str: Formatted status string
        """
        try:
            processes = await self.async_get_ollama_processes()
            
            result = ""
            if include_timestamp:
                result += f"STATUS AS OF: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                
            if not processes:
                result += "NO OLLAMA PROCESSES RUNNING.\n"
            else:
                result += "OLLAMA PROCESSES:\n"
                result += "{:<25} {:<15} {:<10} {:<15} {:<20}\n".format(
                    "NAME", "ID", "SIZE", "PROCESSOR", "UNTIL"
                )
                result += "─" * 85 + "\n"
                
                for process in processes:
                    result += "{:<25} {:<15} {:<10} {:<15} {:<20}\n".format(
                        process.get("name", "Unknown"),
                        process.get("id", "Unknown")[:12],
                        process.get("size", "Unknown"),
                        process.get("processor", "Unknown"),
                        process.get("until", "Unknown")
                    )
                
            if include_gpu:
                gpu_stats = self.get_gpu_status()
                if gpu_stats:
                    result += "\nGPU STATUS:\n"
                    result += "─" * 50 + "\n"
                    for i, gpu in enumerate(gpu_stats):
                        result += f"GPU {i}: {gpu['gpu_name']}\n"
                        result += f"  Memory: {gpu['used_memory']:.2f}/{gpu['total_memory']:.2f} {gpu.get('unit', self.default_memory_unit)}"
                        result += f" ({gpu['used_memory_ratio']*100:.1f}%)\n"
                        result += f"  Utilization: {gpu['gpu_utilization']}%\n"
                        
                        if gpu.get('temperature') is not None:
                            result += f"  Temperature: {gpu['temperature']}°C\n"
                            
                        if gpu.get('power_usage') is not None and gpu.get('power_limit') is not None:
                            result += f"  Power: {gpu['power_usage']:.2f}W / {gpu['power_limit']:.2f}W"
                            result += f" ({gpu['power_usage']/gpu['power_limit']*100:.1f}%)\n"
                        
                        result += "\n"
                
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to format status asynchronously: {str(e)}")
            return f"Error formatting status: {str(e)}"

    def to_dict(self) -> Dict:
        """
        Export all monitoring data as a dictionary.

        Returns:
            Dict: Dictionary containing all monitoring data
        """
        return {
            "timestamp": time.time(),
            "ollama_processes": self.get_ollama_processes(),
            "gpu_status": self.get_gpu_status(),
            "model_gpu_usage": self.get_model_gpu_usage()
        }

    async def async_to_dict(self) -> Dict:
        """
        Export all monitoring data as a dictionary asynchronously.

        Returns:
            Dict: Dictionary containing all monitoring data
        """
        return {
            "timestamp": time.time(),
            "ollama_processes": await self.async_get_ollama_processes(),
            "gpu_status": self.get_gpu_status(),
            "model_gpu_usage": await self.async_get_model_gpu_usage()
        }

    def to_json(self, indent: int = 2) -> str:
        """
        Export all monitoring data as a JSON string.

        Args:
            indent: JSON indentation level

        Returns:
            str: JSON string representation of all monitoring data
        """
        try:
            return json.dumps(self.to_dict(), indent=indent)
        except Exception as e:
            self.logger.error(f"Failed to convert to JSON: {str(e)}")
            return json.dumps({"error": str(e)}, indent=indent)

    async def async_to_json(self, indent: int = 2) -> str:
        """
        Export all monitoring data as a JSON string asynchronously.

        Args:
            indent: JSON indentation level

        Returns:
            str: JSON string representation of all monitoring data
        """
        data = await self.async_to_dict()
        return json.dumps(data, indent=indent)