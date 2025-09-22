# 🧠 Advanced Deep Learning & NLP Projects

This repository is a **collection of deep learning and machine learning projects** covering a wide range of applications in **computer vision, natural language processing (NLP), and time-series analysis**.  
The works explore **state-of-the-art neural network architectures** such as CNNs, U-Nets, RNNs, Transformers, and Autoencoders, applied to challenging datasets including **electronic cryomicroscopy (Cryo-EM) images**, **peptide sequences**, **ECG signals**, and **face images**.

Each project is organized in its own folder with code, experiments, and (where possible) sample datasets or links to external resources.

---

## 📂 Projects Overview

### 1️⃣ CNN + MLP for Cryo-EM Image Classification
A hybrid **Convolutional Neural Network (CNN)** and **Multilayer Perceptron (MLP)** model for classifying **electronic cryomicroscopy (Cryo-EM) images**.  
- **Goal:** Identify structural patterns in Cryo-EM data to assist in molecular research.  
- **Techniques:** CNN for feature extraction, MLP for final classification.

---

### 2️⃣ U-Net Model for Cryo-EM Image Restoration
Implementation of a **U-Net architecture** to **restore and denoise Cryo-EM images**, improving clarity for downstream structural analysis.  
- **Goal:** Reduce noise introduced during electron microscopy imaging.  
- **Techniques:** Encoder–decoder U-Net with skip connections for high-fidelity reconstruction.

---

### 3️⃣ U-Net Semantic Segmentation of Cryo-EM Images
Semantic segmentation of Cryo-EM images using **U-Net** to identify and label molecular structures.  
- **Goal:** Automate the detection of biologically relevant regions in microscopy images.  
- **Outcome:** Produces pixel-level masks for precise molecular boundaries.

---

### 4️⃣ Peptide Classification with RNNs, LSTM & Transformers
Deep sequence models (**RNN**, **LSTM**, and **Transformer**) for classifying **antimicrobial peptides (AMPs)**.  
- **Goal:** Predict whether a given peptide belongs to the AMP family.  
- **Context:** AMPs (also known as **host defence peptides**) play a critical role in the innate immune response and can be used in novel drug discovery.  
- **Techniques:** Comparison of recurrent and attention-based architectures for biological sequence modeling.

---

### 5️⃣ Anomaly Detection in ECG Time Series
An **Autoencoder (AE)** combined with a **threshold-based** method to detect anomalies in **ECG5000 electrocardiogram signals**.  
- **Goal:** Identify abnormal heartbeats that may indicate arrhythmias.  
- **Approach:** Reconstruction error from AE is used to detect abnormal patterns.

---

### 6️⃣ Denoising Autoencoder for Image Reconstruction & Gray2RGB
A **Denoising Autoencoder (DAE)** capable of reconstructing images from noisy inputs and performing **grayscale-to-RGB colorization**.  
- **Applications:** Image enhancement, restoration, and color prediction.

---

### 7️⃣ Variational & Convolutional Autoencoders for Face Generation
Implementation of **Variational Autoencoders (VAE)** and **Convolutional Autoencoders (CAE)** for **face image generation**.  
- **Goal:** Explore latent space representations and generative modeling.  
- **Outcome:** Generates new, realistic face images from learned distributions.

---

## 🧩 Key Features & Contributions
- Demonstrates the use of **state-of-the-art architectures** across multiple domains:  
  - **Computer Vision:** CNNs, U-Nets, Autoencoders, VAEs, CAEs  
  - **Time-Series Analysis:** Autoencoders for ECG anomaly detection  
  - **NLP & Bioinformatics:** RNNs, LSTMs, Transformers for peptide classification  
- Includes **data preprocessing pipelines**, model training scripts, and evaluation metrics.  
- Serves as a **learning resource** for applying deep learning to real-world biomedical, biological, and signal-processing problems.
