# Edge AI & AI-IoT Assignment

## Overview
This repository contains the implementation and conceptual design for an **Edge AI Prototype** and an **AI-IoT Smart Agriculture Simulation**. The assignment demonstrates the application of **emerging AI trends**, including Edge AI, AI-IoT integration, and lightweight AI deployment using TensorFlow Lite.

---

## Objectives
- Build a **lightweight image classification model** suitable for Edge AI deployment.
- Convert the trained model to **TensorFlow Lite** for real-time, low-latency applications.
- Design an **AI-IoT smart agriculture simulation**, integrating sensor data with AI predictions.
- Understand **ethical and practical considerations** of deploying AI at the edge and in IoT systems.

---

## Part 1: Edge AI Prototype

### Dataset
- **CIFAR-10** (subset for lightweight demo)
- Classes selected: airplane, automobile, bird

### Implementation
- **Framework:** TensorFlow / Keras
- **Model:** Lightweight CNN (2 Conv2D + MaxPooling layers, Dense layers)
- **Training:** 3 epochs, batch size 64
- **Metrics:** Accuracy on test set
- **Edge Deployment:** Model converted to TensorFlow Lite

### How to Run
1. Install dependencies:
   ```bash
   pip install tensorflow numpy matplotlib
