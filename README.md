# Digit Classifier: PyTorch CNN & FastAPI 

A robust, full-stack handwritten digit classifier built from scratch. This project demonstrates end-to-end model development, from designing and training a custom Convolutional Neural Network (CNN) in PyTorch to serving it via a FastAPI backend, complete with a minimalist live-inference frontend.

## Key Highlights
- **Custom CNN Architecture**: Built and trained from scratch (no pretrained weights).
- **High Performance**: Reached **99.14%** accuracy on the MNIST test set after just 8 epochs.
- **Production-Ready Serving**: Model hosted via a `/predict` endpoint using FastAPI.
- **Polished UI**: Pure HTML/CSS/JS frontend with a monochrome aesthetic and real-time confidence bar chart.

---

## 🧠 Model Architecture
To ensure complete transparency and explainability, the model uses a straightforward, highly effective custom CNN architecture rather than a complex off-the-shelf model.

```
Input: 1x28x28 (Grayscale image)

1. Conv2D: 1 -> 32 channels, 3x3 kernel, padding=1
2. ReLU Activation
3. MaxPool2D: 2x2
4. Conv2D: 32 -> 64 channels, 3x3 kernel, padding=1
5. ReLU Activation
6. MaxPool2D: 2x2
7. Flatten: 64 * 7 * 7 = 3136 features
8. Linear (Dense): 3136 -> 128
9. ReLU Activation
10. Linear (Dense): 128 -> 10 (Output Logits)
```

**Why this architecture?**
- **Two Convolutional Layers:** Provide enough depth to learn edges/curves (Layer 1) and combine them into higher-level structural features of the digits (Layer 2).
- **Max Pooling:** Reduces spatial dimensions (28x28 $\rightarrow$ 14x14 $\rightarrow$ 7x7), making the network translation-invariant and reducing computational cost.
- **Adam Optimizer:** Chosen over SGD for faster convergence due to adaptive learning rates, reaching 99%+ accuracy within fewer epochs.

---

## 📊 Training Results & Metrics

The model was trained for 8 epochs with a batch size of 64 and a learning rate of 0.001.
- **Final Train Loss**: `0.0087`
- **Final Test Loss**: `0.0336`
- **Final Test Accuracy**: `99.14%`

### Evaluation Analysis
*(See `confusion_matrix.png` and `misclassified_examples.png` in the repository)*

The confusion matrix shows exceptional performance across all digits. Most errors are edge cases where the hand-drawn digit is genuinely ambiguous (e.g., a `4` that looks like a `9`, or a sloppily drawn `7` mistaken for a `1`). 

By saving misclassified examples, we can observe that the model correctly learned the canonical shapes of the digits; it only struggles when human input heavily deviates from standard writing patterns.

---

## 🚀 How to Run Locally

### Prerequisites
- Python 3.9+
- Virtual Environment (recommended)

### Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd digit-classifier
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model (Optional - weights are included):**
   ```bash
   python train.py
   python evaluate.py # To generate metrics and plots
   ```

4. **Start the FastAPI backend:**
   ```bash
   uvicorn api:app --reload
   ```

5. **Test the UI:**
   Open your browser and navigate to [http://localhost:8000](http://localhost:8000). Draw a digit on the canvas to see live predictions!

---

## 🔮 Next Steps & Improvements
Given more time, here is how I would scale and improve this project:
1. **Data Augmentation**: Implement random rotations, scaling, and translations (`torchvision.transforms.RandomAffine`) during training. Users draw digits at different sizes and angles; augmentation would make the model more robust to live input.
2. **Dropout Regularization**: Add `nn.Dropout(0.5)` before the fully connected layers to prevent overfitting, which would allow us to train for more epochs safely.
3. **Batch Normalization**: Add `nn.BatchNorm2d` after the conv layers to stabilize and accelerate training.
4. **Export to ONNX/TensorRT**: For true production deployment, converting the model to ONNX would allow serving it more efficiently without loading the full PyTorch library in the backend, or even running it directly in the browser via ONNX Runtime Web.
