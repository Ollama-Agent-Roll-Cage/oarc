# test_oarc.py
from oarc import oarcAPI
import uvicorn

def main():
    # Initialize API
    api = oarcAPI()
    
    # Run FastAPI server
    uvicorn.run(
        api.app, 
        host="0.0.0.0", 
        port=2020,
        reload=True  # Enable auto-reload during development
    )

if __name__ == "__main__":
    main()