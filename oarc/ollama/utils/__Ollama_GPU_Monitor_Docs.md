# OllamaGPUMonitor Documentation

The `OllamaGPUMonitor` class provides a comprehensive solution for monitoring Ollama processes and GPU usage. It integrates Ollama's process management capabilities with NVIDIA GPU monitoring to provide a unified view of your Ollama models' resource usage.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Features](#features)
- [Class Reference](#class-reference)
  - [Initialization](#initialization)
  - [Core Methods](#core-methods)
  - [Asynchronous Methods](#asynchronous-methods)
  - [Utility Methods](#utility-methods)
- [Examples](#examples)
  - [Synchronous Usage](#synchronous-usage)
  - [Asynchronous Usage](#asynchronous-usage)
  - [API Integration](#api-integration)
- [Extending the Class](#extending-the-class)
- [Troubleshooting](#troubleshooting)

## Installation

To use the `OllamaGPUMonitor` class, you need to install the following dependencies:

```bash
pip install ollama nvidia-ml-py
```

Make sure that Ollama is installed and running on your system. The `nvidia-ml-py` package provides access to NVIDIA Management Library (NVML) which is used for GPU monitoring.

## Quick Start

Basic usage example:

```python
from ollama_gpu_monitor import OllamaGPUMonitor

# Create a monitor instance
monitor = OllamaGPUMonitor()

# Get formatted status information
print(monitor.format_status())

# Get detailed GPU information
gpu_stats = monitor.get_gpu_status(unit="GB")
print(gpu_stats)

# Get combined model and GPU usage
model_usage = monitor.get_model_gpu_usage()
print(model_usage)
```

## Features

- **Comprehensive GPU Monitoring**: Track memory usage, utilization, temperature, and power metrics
- **Ollama Process Management**: Get information about running Ollama models
- **Intelligent Mapping**: Match Ollama processes to their corresponding GPU resources
- **Caching System**: Efficient data retrieval with configurable time-based caching
- **Background Refresh**: Optional background thread for keeping data up-to-date
- **Asynchronous Support**: Full async API for non-blocking operations
- **Robust Error Handling**: Comprehensive retry mechanisms and error reporting
- **Multiple Output Formats**: Get data as Python dictionaries, formatted text, or JSON

## Class Reference

### Initialization

```python
OllamaGPUMonitor(
    ollama_client=None,
    default_memory_unit="MB",
    cache_duration=5,
    enable_background_refresh=False,
    background_refresh_interval=10,
    retry_attempts=3,
    retry_delay=1.0
)
```

#### Parameters:

- **ollama_client**: An instance of `ollama.Client` or `ollama.AsyncClient` (optional)
- **default_memory_unit**: Default unit for memory measurements (default: "MB")
- **cache_duration**: Duration in seconds to cache results (default: 5)
- **enable_background_refresh**: Whether to enable background data refresh (default: False)
- **background_refresh_interval**: Interval in seconds for background refresh (default: 10)
- **retry_attempts**: Number of retry attempts for failed operations (default: 3)
- **retry_delay**: Delay in seconds between retry attempts (default: 1.0)

### Core Methods

#### `get_gpu_status(unit=None, force_refresh=False)`

Get GPU status information with caching.

- **Parameters**:
  - **unit**: Memory unit to use (default: uses instance default_memory_unit)
  - **force_refresh**: Whether to force a refresh regardless of cache
- **Returns**: List of dictionaries containing GPU information

#### `get_ollama_processes(force_refresh=False)`

Get information about running Ollama processes with caching.

- **Parameters**:
  - **force_refresh**: Whether to force a refresh regardless of cache
- **Returns**: List of dictionaries containing Ollama process information

#### `get_model_gpu_usage(force_refresh=False)`

Get combined information about Ollama models and their GPU usage with caching.

- **Parameters**:
  - **force_refresh**: Whether to force a refresh regardless of cache
- **Returns**: List of dictionaries with model and GPU usage information

#### `format_status(include_gpu=True, include_timestamp=True)`

Format the status of Ollama processes and GPU usage as a string.

- **Parameters**:
  - **include_gpu**: Whether to include GPU information (default: True)
  - **include_timestamp**: Whether to include timestamp (default: True)
- **Returns**: Formatted status string

#### `to_dict()`

Export all monitoring data as a dictionary.

- **Returns**: Dictionary containing all monitoring data

#### `to_json(indent=2)`

Export all monitoring data as a JSON string.

- **Parameters**:
  - **indent**: JSON indentation level
- **Returns**: JSON string representation of all monitoring data

### Asynchronous Methods

#### `async_get_ollama_processes(force_refresh=False)`

Get information about running Ollama processes asynchronously with caching.

- **Parameters**:
  - **force_refresh**: Whether to force a refresh regardless of cache
- **Returns**: List of dictionaries containing Ollama process information

#### `async_get_model_gpu_usage(force_refresh=False)`

Get combined information about Ollama models and their GPU usage asynchronously with caching.

- **Parameters**:
  - **force_refresh**: Whether to force a refresh regardless of cache
- **Returns**: List of dictionaries with model and GPU usage information

#### `async_format_status(include_gpu=True, include_timestamp=True)`

Format the status of Ollama processes and GPU usage as a string asynchronously.

- **Parameters**:
  - **include_gpu**: Whether to include GPU information (default: True)
  - **include_timestamp**: Whether to include timestamp (default: True)
- **Returns**: Formatted status string

#### `async_to_dict()`

Export all monitoring data as a dictionary asynchronously.

- **Returns**: Dictionary containing all monitoring data

#### `async_to_json(indent=2)`

Export all monitoring data as a JSON string asynchronously.

- **Parameters**:
  - **indent**: JSON indentation level
- **Returns**: JSON string representation of all monitoring data

### Utility Methods

#### `convert_byte_unit(value, src_unit, target_unit)`

Convert value in source unit to target unit.

- **Parameters**:
  - **value**: The value to convert
  - **src_unit**: Source unit ("b", "B", "KB", "MB", "GB", "TB")
  - **target_unit**: Target unit ("b", "B", "KB", "MB", "GB", "TB")
- **Returns**: The converted value in target_unit
- **Raises**: ValueError if source or target unit is not valid

## Examples

### Synchronous Usage

```python
# Initialize the monitor with background refresh
monitor = OllamaGPUMonitor(
    default_memory_unit="GB",
    enable_background_refresh=True,
    background_refresh_interval=5,
    cache_duration=3
)

# Print formatted status
print(monitor.format_status())

# Get detailed GPU information with custom unit
gpu_info = monitor.get_gpu_status(unit="MB")

# Export to JSON for storage/analysis
json_data = monitor.to_json(indent=2)
with open("ollama_status.json", "w") as f:
    f.write(json_data)

# Clean up
monitor._stop_background_refresh()
```

### Asynchronous Usage

```python
import asyncio
import ollama

async def monitor_ollama():
    # Initialize with async client
    async_client = ollama.AsyncClient()
    monitor = OllamaGPUMonitor(
        ollama_client=async_client,
        default_memory_unit="GB"
    )
    
    # Get async formatted status
    status = await monitor.async_format_status()
    print(status)
    
    # Get async data
    processes = await monitor.async_get_ollama_processes()
    usage = await monitor.async_get_model_gpu_usage()
    
    # Print JSON representation
    json_data = await monitor.async_to_json(indent=2)
    print("\nAsync JSON Export:")
    print(json_data)

# Run the async function
asyncio.run(monitor_ollama())
```

### API Integration

```python
# FastAPI example
from fastapi import FastAPI, Depends
from fastapi.responses import JSONResponse
import ollama

app = FastAPI()
monitor = None

@app.on_event("startup")
async def startup_event():
    global monitor
    async_client = ollama.AsyncClient()
    monitor = OllamaGPUMonitor(
        ollama_client=async_client,
        enable_background_refresh=True,
        background_refresh_interval=10
    )

@app.on_event("shutdown")
def shutdown_event():
    global monitor
    if monitor:
        monitor._stop_background_refresh()

@app.get("/api/ollama/status")
async def get_status():
    global monitor
    status_json = await monitor.async_to_json()
    return JSONResponse(content=json.loads(status_json))

@app.get("/api/ollama/processes")
async def get_processes():
    global monitor
    processes = await monitor.async_get_ollama_processes()
    return JSONResponse(content={"processes": processes})

@app.get("/api/ollama/gpu")
async def get_gpu_info():
    global monitor
    gpu_info = monitor.get_gpu_status()
    return JSONResponse(content={"gpu_info": gpu_info})
```

## Extending the Class

The `OllamaGPUMonitor` class is designed to be extensible. Here are some ways you might want to extend it:

1. **Custom Metrics**: Add more GPU metrics or Ollama process information
2. **Additional Output Formats**: Add methods for exporting to different formats
3. **Database Integration**: Add methods to store monitoring data in a database
4. **Alert System**: Add thresholds and alerting functionality
5. **Web UI Integration**: Add methods for web-based monitoring dashboards

Example extension:

```python
class ExtendedOllamaMonitor(OllamaGPUMonitor):
    def __init__(self, *args, **kwargs):
        self.alert_threshold = kwargs.pop('alert_threshold', 0.9)
        super().__init__(*args, **kwargs)
    
    def check_alerts(self):
        """Check for any metrics exceeding thresholds."""
        alerts = []
        gpu_stats = self.get_gpu_status()
        
        for gpu in gpu_stats:
            if gpu['used_memory_ratio'] > self.alert_threshold:
                alerts.append({
                    'type': 'high_memory_usage',
                    'gpu_id': gpu['gpu_id'],
                    'value': gpu['used_memory_ratio'],
                    'threshold': self.alert_threshold
                })
        
        return alerts
```

## Troubleshooting

### Common Issues

1. **Missing Dependencies**
   - Make sure both `ollama` and `nvidia-ml-py` are installed
   - For async operations, ensure you have a recent version of Python with asyncio support

2. **Ollama Connection Issues**
   - Verify that Ollama is running (`ollama ps` in terminal)
   - Check that you're connecting to the correct Ollama API endpoint
   - Check for firewall or network issues

3. **GPU Monitoring Issues**
   - Ensure you have NVIDIA GPUs and the appropriate drivers installed
   - Run `nvidia-smi` in terminal to verify that your system can see the GPUs
   - Check that your user has permissions to access GPU metrics

4. **Background Thread Issues**
   - If you're experiencing crashes, try disabling the background refresh
   - Increase the refresh interval to reduce system load
   - Make sure to call `_stop_background_refresh()` when you're done using the monitor

### Debugging

The class uses Python's logging system. You can enable more detailed logging by configuring the logger:

```python
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Now create your monitor
monitor = OllamaGPUMonitor()
```

---

This documentation covers the basic and advanced usage of the `OllamaGPUMonitor` class. For further assistance, refer to the source code or open an issue in the repository.
