import uvicorn
from src.routes import create_app

# Create an instance of FastAPI
app = create_app()

app = create_app()


# This block allows you to run the application using Uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
