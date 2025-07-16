
## How to Build and Run with Docker

1. Build the Docker image:
   
   docker build --progress=plain -t hvac .

2. Run the server:
   
   docker run --rm -p 18080:18080 hvac

## API Endpoint

### POST /predict

Send a POST request to `http://localhost:18080/predict` with a multipart/form-data body containing the image file as the first part. The response will be a JSON with the prediction results.

Example using curl:

    curl -X POST -F 'file=@your_image.jpg' http://localhost:18080/predict

Replace `your_image.jpg` with the path to your image file.
