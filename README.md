# California House Price Predictor 🏠

Predict median house values in California districts using a Random Forest Regressor and a Streamlit interactive interface.

## 🌟 Overview
This project provides a simple yet powerful web application to explore the California Housing dataset. Users can adjust various demographic and geographical parameters to see how they impact predicted property values.

## 🚀 Live Demo
*(Optional: Add a link to your hosted app, e.g., via Streamlit Community Cloud)*

## 🛠️ Features
- **Interactive Sidebar**: Adjust Median Income, House Age, Population, and more.
- **Real-time Prediction**: Instantly see the estimated value in USD.
- **Data Insights**:
  - Correlation Heatmap: Visualize relationship between features.
  - Feature Importance: See which variables (like Latitude or Income) influence price the most.

## 🧬 How It Works
- **Model**: Random Forest Regressor from `scikit-learn`.
- **Dataset**: [California Housing Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html) (fetched via `sklearn.datasets`).
- **Features**: 8 numerical features (MedInc, HouseAge, AveRooms, etc.).

## 📦 Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/house-price-prediction.git
   cd house-price-prediction
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model (Optional)**:
   If you want to re-train the model or don't have the `.pkl` file:
   ```bash
   python model.py
   ```

4. **Run the Streamlit App**:
   ```bash
   streamlit run app.py
   ```

## ⚠️ Important Note on File Size
The trained model file `house_price_model.pkl` is approximately **144MB**, which exceeds GitHub's standard 100MB file limit. To store this in your repository, you should use **Git Large File Storage (LFS)**:

```bash
git lfs install
git lfs track "*.pkl"
git add .gitattributes
```

---
*Built with ❤️ using Streamlit and Scikit-Learn.*
