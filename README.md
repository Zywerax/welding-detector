# Welding Defect Detection System

## Project Overview

This project is a real-time welding quality inspection system built with FastAPI, designed to detect and classify welding defects using computer vision and deep learning. The system provides automated video analysis, frame-by-frame defect detection, and comprehensive classification of nine distinct welding defect types.

The application addresses the challenge of manual welding inspection, which is time-consuming, subjective, and prone to human error. By leveraging convolutional neural networks and motion detection algorithms, the system enables quality control teams to analyze welding recordings efficiently, identify defects with high precision, and maintain consistent inspection standards across production environments.

This project demonstrates production-grade backend architecture with a focus on scalable service design, comprehensive testing strategies, and real-time video processing capabilities suitable for industrial automation contexts.

## Architecture

The project follows a layered architecture pattern with clear separation of concerns:

```
app/
├── api/                    # Presentation Layer
│   ├── routes/            # API endpoint definitions
│   │   ├── camera.py      # Live camera operations
│   │   ├── recording.py   # Video recording management
│   │   ├── ml.py          # Binary classification (OK/NOK)
│   │   ├── defects.py     # Multi-class defect classification
│   │   └── labeling.py    # Training data annotation
│   └── models.py          # Pydantic request/response schemas
├── services/              # Business Logic Layer
│   ├── camera_service.py           # Camera capture and recording
│   ├── ml_classification_service.py # Binary weld quality model
│   ├── defect_classifier_service.py # 9-class defect detection
│   ├── video_analysis_service.py   # Frame-by-frame analysis
│   ├── motion_detection_service.py # Welding activity detection
│   ├── frame_extractor_service.py  # Video frame extraction
│   ├── image_enhancement_service.py # Preprocessing pipeline
│   ├── video_overlay_service.py    # Annotation rendering
│   ├── frame_overlay_service.py    # Real-time overlay
│   └── labeling_service.py         # Dataset management
├── config/                # Configuration Layer
│   └── settings.py        # Environment-based configuration
└── main.py               # Application entry point
```

### Design Principles

**Service Layer Pattern**: All business logic is encapsulated in service classes, making routes thin controllers that delegate to specialized services. This ensures testability and reusability.

**Singleton Pattern**: Services like `CameraService` and `FrameOverlayService` are implemented as singletons to manage shared resources (hardware camera, recording state) safely across concurrent requests.

**Dependency Injection**: FastAPI's dependency injection system is used throughout the application to provide service instances to routes, enabling easy mocking during testing.

**Async/Await Pattern**: Streaming endpoints use FastAPI's `StreamingResponse` with async generators for efficient real-time video streaming without blocking the event loop.

This architecture was chosen to support:
- Concurrent video analysis workflows
- Real-time camera streaming with minimal latency
- Background processing for video annotation
- Extensibility for adding new detection models
- Comprehensive unit testing with minimal coupling

## Technologies Used

### Core Framework
- **FastAPI 0.115.12** - Modern async web framework with automatic OpenAPI documentation
- **Uvicorn** - ASGI server for production deployment
- **Pydantic** - Data validation and settings management using Python type annotations

### Computer Vision & Machine Learning
- **PyTorch 2.5.1** - Deep learning framework for model training and inference
- **torchvision** - Computer vision utilities and pretrained models
- **timm** - PyTorch Image Models library (EfficientNet-B0 backbone)
- **OpenCV (cv2)** - Real-time video processing and frame manipulation
- **NumPy** - Numerical operations for image arrays

### Testing & Quality Assurance
- **pytest 8.4.2** - Testing framework with fixture support
- **pytest-cov** - Coverage measurement and reporting
- **pytest-asyncio** - Async test support for FastAPI endpoints
- **unittest.mock** - Mocking external dependencies (PyTorch, OpenCV)

### Additional Libraries
- **python-multipart** - File upload handling
- **Pillow (PIL)** - Image format conversion and manipulation
- **loguru** - Structured logging with rotation support

### Frontend Technologies
- **Vue 3** - Progressive JavaScript framework with Composition API
- **Vite 6.0** - Next generation frontend build tool
- **Tailwind CSS v4** - Utility-first CSS framework with modern features
- **Axios** - Promise-based HTTP client for browser

### DevOps & Deployment
- **Docker** - Frontend containerization (Dockerfile in app_frontend/)
- **Uvicorn** - Lightning-fast ASGI server for local backend

**Note**: Backend runs locally for USB camera access; frontend can be containerized.

## Installation and Setup

### Prerequisites
- Python 3.10 or higher
- CUDA-compatible GPU (optional, for training acceleration)
- Webcam or video capture device (for live camera features)

### Local Development Setup

1. **Clone the repository**
```bash
git clone https://github.com/MarB-tech/welding-detector.git
cd welding-detector
```

2. **Create virtual environment**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment**
Create a `.env` file or use default settings in `app/config/settings.py`:
```env
# Optional: Override defaults
ML_LEARNING_RATE_DEFAULT=0.001
ML_VALIDATION_SPLIT=0.2
DEFECT_LEARNING_RATE=0.001
```

5. **Run the application**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at: `http://localhost:8000`

Interactive API documentation: `http://localhost:8000/docs`

6. **Run tests**
```bash
# Run all tests with coverage
pytest --cov=app --cov-report=html --cov-report=term-missing

# Run specific test module
pytest tests/test_ml_classification_service.py -v

# Run tests with specific markers
pytest -m unit -v

# Open HTML coverage report
start htmlcov/index.html  # Windows
open htmlcov/index.html   # macOS
```

### Frontend Setup

The project includes a Vue 3 frontend application with real-time video streaming interface.

**Prerequisites:**
- Node.js 18+ and npm

**Local Development:**

1. **Navigate to frontend directory**
```bash
cd app_frontend
```

2. **Install dependencies**
```bash
npm install
```

3. **Run development server**
```bash
npm run dev
```

Frontend will be available at: `http://localhost:5173`

**Frontend Stack:**
- **Vue 3** - Progressive JavaScript framework with Composition API
- **Vite** - Fast build tool and development server
- **Tailwind CSS v4** - Utility-first CSS framework
- **Axios** - HTTP client for API communication

### Docker Deployment (Frontend Only)

**Important**: The backend must run locally because Docker containers cannot access USB camera devices (grabber). Only the frontend can be containerized.

**Architecture:**
- Backend: Runs locally with direct USB camera access
- Frontend: Runs in Docker container, communicates with local backend API

**Steps:**

1. **Start backend locally** (in one terminal)
```bash
# Activate virtual environment
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/macOS

# Run backend
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

2. **Build and run frontend in Docker** (in another terminal)
```bash
cd app_frontend

# Build Docker image
docker build -t welding-detector-frontend .

# Run container
docker run -p 5173:5173 welding-detector-frontend
```

3. **Access services**
- Backend API: `http://localhost:8000`
- API Documentation: `http://localhost:8000/docs`
- Frontend UI: `http://localhost:5173`

**Why this setup?**
- USB camera devices require direct hardware access
- Docker containers have limited access to host USB devices
- Running backend locally ensures reliable camera capture
- Frontend is stateless and can be safely containerized

## API Endpoints

### Camera Operations
- `GET /camera/frame` - Capture single JPEG frame
- `GET /camera/stream` - Real-time MJPEG video stream
- `GET /camera/health` - Camera status and configuration
- `PUT /camera/settings` - Update camera parameters (FPS, resolution, quality)

### Recording Management
- `POST /recording/start` - Begin video recording
- `POST /recording/stop` - Stop recording and save file
- `GET /recording/status` - Current recording state
- `GET /recording/list` - List all saved recordings
- `GET /recording/download/{filename}` - Download MP4 file
- `DELETE /recording/{filename}` - Remove recording
- `GET /recording/{filename}/frame/{index}` - Extract specific frame

### Video Analysis
- `POST /recording/{filename}/apply-overlay` - Generate annotated video with defects
- `GET /recording/{filename}/detect-motion` - Detect welding activity timestamps
- `POST /recording/{filename}/extract-frames` - Extract frames at intervals
- `GET /recording/{filename}/overlay-status` - Check annotation progress

### Machine Learning - Binary Classification
- `GET /ml/info` - Model status and configuration
- `GET /ml/stats` - Training dataset statistics
- `POST /ml/train` - Train binary OK/NOK classifier
- `POST /ml/predict` - Classify single frame

### Machine Learning - Defect Classification
- `GET /defects/info` - Defect model status
- `GET /defects/stats` - Defect dataset statistics
- `GET /defects/types` - List of 9 defect categories
- `POST /defects/train` - Train multi-class defect classifier
- `POST /defects/classify` - Identify specific defect type

### Labeling & Dataset Management
- `POST /labeling/capture` - Capture frame and assign label
- `GET /labeling/stats` - Dataset distribution per class
- `GET /labeling/samples/{label}` - List samples for specific label
- `DELETE /labeling/sample/{label}/{filename}` - Remove mislabeled sample

### Example Request/Response

**POST /ml/predict**
```bash
curl -X POST "http://localhost:8000/ml/predict" \
  -F "file=@welding_frame.jpg"
```

Response:
```json
{
  "prediction": "OK",
  "confidence": 0.9234,
  "probabilities": {
    "OK": 0.9234,
    "NOK": 0.0766
  },
  "model_version": "v1.0",
  "inference_time_ms": 42.3
}
```

**POST /defects/classify**
```bash
curl -X POST "http://localhost:8000/defects/classify" \
  -F "file=@defect_frame.jpg"
```

Response:
```json
{
  "defect_type": "porosity",
  "confidence": 0.8721,
  "top_3_predictions": [
    {"type": "porosity", "confidence": 0.8721},
    {"type": "spatter", "confidence": 0.0834},
    {"type": "crack", "confidence": 0.0245}
  ]
}
```

## Testing

### Test Architecture

The project maintains 68% code coverage across 166 unit tests, structured using pytest with comprehensive mocking strategies:

```
tests/
├── conftest.py                          # Shared fixtures and test configuration
├── test_main.py                         # Application startup and CORS
├── test_config.py                       # Settings validation
├── test_models.py                       # Pydantic schema validation
├── test_api_routes.py                   # Camera and streaming endpoints
├── test_api_routes_recording.py         # Recording API endpoints
├── test_ml_routes.py                    # ML training and inference routes
├── test_camera_service.py               # Camera capture logic
├── test_ml_classification_service.py    # Binary classification service
├── test_defect_classifier_service.py    # Multi-class defect detection
├── test_video_analysis_service.py       # Frame-by-frame analysis
├── test_motion_detection_service.py     # Motion detection algorithms
├── test_frame_extractor_service.py      # Frame extraction utilities
├── test_image_enhancement_service.py    # Preprocessing pipeline
├── test_video_overlay_service.py        # Background annotation processing
├── test_frame_overlay_service.py        # Real-time overlay rendering
└── test_labeling_service.py            # Dataset management
```

### Testing Strategy

**Mocking External Dependencies**: PyTorch models (`torch.load`, `timm.create_model`) and OpenCV operations (`cv2.VideoCapture`, `cv2.VideoWriter`) are mocked to ensure tests run without GPU requirements or physical cameras.

**Fixture-Based Isolation**: Each test uses isolated fixtures with temporary directories and fresh service instances, preventing state leakage between tests.

**Dependency Override Pattern**: FastAPI's dependency injection is overridden in tests to inject mock services, ensuring routes are tested with controlled service behavior.

**Real Tensor Operations**: Training tests use real PyTorch tensors with gradient tracking to validate backpropagation flow, while mocking optimizer and scheduler to avoid actual training overhead.

This approach was chosen to:
- Enable fast test execution (30 seconds for full suite)
- Run tests in CI/CD without specialized hardware
- Verify integration between layers without external dependencies
- Maintain high coverage of business logic and edge cases

### Running Tests

```bash
# Full test suite with coverage
pytest --cov=app --cov-report=html --cov-report=term-missing -v

# Test specific module
pytest tests/test_ml_classification_service.py -v

# Test with markers
pytest -m "unit" -v

# Generate coverage report
pytest --cov=app --cov-report=html
```

Coverage report available at: `htmlcov/index.html`

## Implementation Challenges

### Challenge 1: Concurrent Camera Access
**Problem**: Multiple HTTP requests accessing the same `cv2.VideoCapture` instance caused frame corruption and deadlocks.

**Solution**: Implemented singleton pattern with thread-safe frame buffering. The camera service maintains a background thread that continuously captures frames into a buffer, while HTTP requests read from the buffer without blocking camera operations.

**Trade-off**: Increased memory usage (~10MB for 30 FPS buffer) in exchange for guaranteed frame consistency and concurrent access support.

### Challenge 2: Background Video Processing
**Problem**: Video annotation with ML inference is CPU-intensive and would block API responses if executed synchronously.

**Solution**: Designed `VideoOverlayService` with background threading. The service maintains a process registry with status tracking, allowing clients to start annotation jobs and poll for completion via separate endpoints.

**Learning**: Background task management in FastAPI requires careful consideration of process lifecycle, error handling, and status persistence across API restarts.

### Challenge 3: Training Pipeline Testing
**Problem**: Testing ML training logic requires mocking complex PyTorch components (DataLoader, optimizer, loss functions) while maintaining realistic tensor operations for gradient validation.

**Solution**: Created hybrid mocks that return real tensors from forward passes but use mocked optimizers and schedulers. This validates backpropagation logic without actual model training overhead.

**Design Decision**: Chose mocking over integration tests with small models because training even lightweight models adds 5-10 seconds per test, making TDD impractical.

### Challenge 4: Motion Detection Accuracy
**Problem**: Frame differencing produces excessive false positives in videos with lighting changes or camera vibration.

**Solution**: Implemented multi-stage filtering: Gaussian blur preprocessing, adaptive thresholding, contour area filtering, and temporal smoothing across 5-frame windows. Added configurable sensitivity parameters exposed via API.

**Trade-off**: Conservative thresholds reduce false positives but may miss brief welding events. Chose to optimize for precision over recall based on industrial QC requirements.

### Challenge 5: Model Persistence and Versioning
**Problem**: Trained models need to persist across server restarts while supporting multiple model versions and rollback capabilities.

**Solution**: Store models in `models/` directory with atomic file writes. Maintain `training_info.json` with metadata (accuracy, timestamp, class distribution). Services load models lazily on first prediction request.

**Improvement Opportunity**: Current implementation lacks formal versioning. Future enhancement would add semantic versioning and migration strategies for model schema changes.

## Possible Improvements

### Performance Optimizations
- **Model Quantization**: Convert PyTorch models to ONNX or TorchScript for 2-3x inference speedup
- **Batch Inference**: Modify video analysis to process frames in batches rather than sequentially
- **GPU Memory Pooling**: Implement CUDA memory caching to reduce allocation overhead during training
- **Async Frame Extraction**: Use `asyncio` subprocess for parallel `ffmpeg` frame extraction

### Scalability Enhancements
- **Distributed Processing**: Integrate Celery or Redis Queue for multi-worker video processing
- **Model Serving**: Deploy separate ML inference service using TorchServe or Triton Inference Server
- **Database Integration**: Add PostgreSQL for recording metadata, analysis results, and audit trails
- **Object Storage**: Migrate video files to S3-compatible storage for horizontal scaling

### Feature Additions
- **Grad-CAM Visualization**: Return attention heatmaps showing which image regions influenced defect classification
- **Model Ensembling**: Combine predictions from multiple models to improve accuracy
- **Active Learning**: Flag low-confidence predictions for manual review and dataset augmentation
- **WebSocket Streaming**: Replace MJPEG with WebSocket-based streaming for lower latency
- **Authentication**: Add API key or OAuth2 authentication for production deployment

### Code Quality Improvements
- **Type Coverage**: Add `mypy` strict mode for complete type checking
- **API Versioning**: Implement `/v1/` prefixed routes with deprecation strategy
- **Error Monitoring**: Integrate Sentry or similar for production error tracking
- **Metrics Collection**: Add Prometheus metrics for inference latency, throughput, and model accuracy
- **Documentation**: Expand inline documentation and add architectural decision records (ADRs)

### Testing Enhancements
- **Integration Tests**: Add end-to-end tests with real video files and small trained models
- **Load Testing**: Use Locust to validate concurrent streaming performance
- **Contract Testing**: Add Pact or similar for frontend-backend contract validation
- **Mutation Testing**: Use `mutmut` to verify test suite effectiveness

## Screenshots

### API Interactive Documentation
![API Documentation](docs/screenshots/api-docs.png)
*FastAPI's automatic OpenAPI documentation showing all available endpoints, request schemas, and response models*

### Real-Time Camera Stream
![Camera Stream Interface](docs/screenshots/camera-stream.png)
*Live MJPEG stream from camera for monitoring welding process in real-time*

### Video Analysis Results
![Analysis Dashboard](docs/screenshots/analysis-results.png)
*Defect analysis summary showing detected defect types, confidence scores, and timeline visualization*

### Video Analysis with Overlay
![Annotated Video with Defect Overlay](docs/screenshots/video-overlay.png)
*Post-recording video analysis with ML-based defect detection bounding boxes and classification labels rendered frame-by-frame*

---

## Project Context

This system was developed as part of an engineering thesis project and has been validated on real industrial welding equipment. The solution addresses actual quality control challenges in welding manufacturing processes, providing automated defect detection capabilities that complement traditional manual inspection methods.

**Testing Environment**: The system has been tested and validated using a professional welding machine under real operating conditions, demonstrating practical applicability in industrial settings.

**Documentation**: This README was generated with AI assistance and thoroughly reviewed by the project author to ensure technical accuracy and completeness.

## License

This project is developed for academic research and engineering thesis purposes.

## Contact

For technical inquiries or collaboration opportunities, please contact via GitHub.

---

**Technical Note**: This project demonstrates production-ready software engineering practices including layered architecture, comprehensive testing (68% coverage, 166 tests), real-time video processing, and deep learning integration suitable for industrial automation applications.
