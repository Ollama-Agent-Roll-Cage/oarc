# OARC

A Python package for OARC functionality.

## Installation

```bash
# Clone the repository
git clone https://github.com/Leoleojames1/oarc.git
cd oarc

# Create & activate virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install the package with pip (editable mode)
pip install -e .

# Install additional dependencies
oarc setup
```

## Development

```bash
# Install development dependencies
oarc develop
```

### Building from source

```bash
# Build the oarc as a wheel from source
oarc build
```

## Running OARC

```bash
# Activate environment where OARC is installed
oarc
```

## Commands

- `oarc` - Run the main CLI tool
- `oarc setup` - Install all dependencies
- `oarc develop` - Setup developer mode
- `oarc build` = Build from source code