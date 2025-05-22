# neurips25_9562

## 📥 Clone the Repository

```bash
git clone https://anonymous.4open.science/r/neurips25_9562
cd neurips25_9562
```

## 🛠️ Environment and Dependencies

This code has been tested with:

- Python 3.8  
- CUDA 11.7  
- PyTorch 1.13.1

### Required Python Packages

Ensure the following packages are installed:

- `pyyaml`
- `scikit-image`
- `box`
- `lpips`
- `imageio`
- `matplotlib`
- `importlib`
- `numpy`
- `pandas`
- `scipy`
- `opencv-python`

You can install them using:

```bash
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

pip install pyyaml scikit-image box lpips imageio matplotlib importlib-metadata numpy pandas scipy opencv-python
```

---

## 📁 Dataset

The RFID spectrum dataset is available at:

> https://github.com/XPengZhao/NeRF2

After downloading, place the dataset into the following directory:

```bash
./data/
```

---

## 🧪 Training

1. Open `./configs/spectrum.yml`
2. Set the mode to training:

```yaml
mode: train
```

3. Run the training script:

```bash
python main.py
```

---

## 🔍 Testing

1. Open `./configs/spectrum.yml`
2. Set the mode to testing:

```yaml
mode: test
```

3. Run the inference script:

```bash
python main.py
```

