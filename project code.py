#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import re
import nltk
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

import tensorflow as tf
from nltk.corpus import stopwords
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


# In[2]:


#EDA process
#load csv file
df_projection=pd.read_csv('indiana_projections.csv')
df_report=pd.read_csv('indiana_reports.csv.zip')
#merge them on uid column and we use inner join to keep both text and image in one row
merge_df=pd.merge(df_projection,df_report,on='uid',how='inner')
#clean the data
print(merge_df.isnull().sum().sum())
print(merge_df.isnull().sum().sort_values(ascending=1))
clean_df=merge_df.dropna(subset=['comparison','findings','indication','impression']) #removing the nulls using dropna()
clean_df=clean_df.reset_index(drop=True)                                             #reseting the index
print(f'orginal samples{len(merge_df)}')
print(f'cleaned samples{len(clean_df)}')
# Remove rows where text is just whitespace or too short to be a real report
clean_df=clean_df[clean_df['findings'].str.strip().str.len()>5]
clean_df.duplicated().sum()
clean_df.to_csv('cleaned_multimodel_dat.csv',index=False)
# Path Integrity Check
image_folder = r"C:\Users\SPOORTHI\Downloads\New folder\images_normalized"
clean_df['full_path'] = clean_df['filename'].apply(lambda x: os.path.join(image_folder, x.strip()))
exists = clean_df['full_path'].apply(lambda x: os.path.exists(x))
clean_df = clean_df[exists].reset_index(drop=True) 
print(f"Verified Dataset Size: {len(clean_df)}")


# In[3]:


#NLP
stop_words=set(stopwords.words('english'))
stop_words
def get_medical_cleaning_tools():
    """
    Customizes the stopword list for medical reports.
    Keeps negation words like 'no', 'not', 'without'.
    """
    stop_words = set(stopwords.words('english'))
    # Removing negation words from the stopword list
    # In radiology, "no pneumonia" is the opposite of "pneumonia"
    negations = {'no', 'not', 'none', 'neither', 'never', 'without', 'negative'}
    medical_stop_words = stop_words - negations
    return medical_stop_words
def clean_medical_text(text,medical_stop_words):
    text=str(text) #Ensure text is a string
    text=text.lower()   #converting all string to lower case
    # Removing Punctuation
    # This keeps alphanumeric characters and removes symbols like ! . , ; :
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    # Removing (Medical-Safe) Stop Words
    words = text.split()
    cleaned_words = [w for w in words if w not in medical_stop_words]
    # Join back and remove extra whitespaces
    return " ".join(cleaned_words).strip()   
med_stops = get_medical_cleaning_tools()

# Apply to your Master Table
# dataframe is called 'clean_df'
clean_df['findings_proc'] = clean_df['findings'].apply(lambda x: clean_medical_text(x, med_stops))
clean_df['impression_proc'] = clean_df['impression'].apply(lambda x: clean_medical_text(x, med_stops))

# 1. Initialize the Vectorizer
# max_features=1000 keeps the top 1000 most important words (prevents memory crashes)
tfidf_vec = TfidfVectorizer(max_features=1000)

# 2. Fit and Transform the Findings
# This creates a 'Sparse Matrix' of numbers
text_features = tfidf_vec.fit_transform(clean_df['findings_proc'])

# Convert to a regular array for fusion later
text_vectors = text_features.toarray()

print(f"Text Feature Shape: {text_vectors.shape}") # Should be (Number of samples, 1000)



# In[4]:


#CNN
# Load pre-trained model without the final classification layer
base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))

# Create a function to load and process images (TensorFlow compatible)
def load_and_preprocess(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    return preprocess_input(img)

# Build the "Fast Pipeline"
# This prepares images while the CPU/GPU is busy
path_ds = tf.data.Dataset.from_tensor_slices(clean_df['full_path'].values)
image_ds = path_ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
batch_ds = image_ds.batch(64).prefetch(tf.data.AUTOTUNE)

#Run Inf erence in Batches (The Speed Boost)
base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
print("Starting High-Speed Extraction...")
image_vectors = base_model.predict(batch_ds)

print(f"Extraction complete! Shape: {image_vectors.shape}")

# Save immediately so you NEVER have to run this again
np.save('image_features_final.npy', image_vectors)
print(f"Image Feature Shape: {image_vectors.shape}")


# In[23]:


#Grad-CAM
def make_gradcam_heatmap(img_path, model):
    # 1. Identify the last 4D (convolutional) layer
    last_conv_layer_name = None
    for layer in reversed(model.layers):
        if len(layer.output.shape) == 4: 
            last_conv_layer_name = layer.name
            break

    # 2. Create the Grad-CAM Model
    # We use model.input directly without brackets
    grad_model = tf.keras.models.Model(
        inputs=model.input, 
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    # 3. Preprocess the image
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # 4. Record gradients
    with tf.GradientTape() as tape:
        # Pass the img_array DIRECTLY (no brackets) to match 'keras_tensor_155'
        last_conv_layer_output, preds = grad_model(img_array)
        class_channel = preds[:, :] 

    # 5. Gradient Calculation
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 6. Compute and Normalize Heatmap
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)

    return heatmap.numpy()


# In[6]:


#fusion
#assuming image_vectors and text_vectors have the same number of rows
# We used hstack (horizontal stack) to fuse them
fused_features = np.hstack((image_vectors, text_vectors))

print(f"Fused Vector Shape: {fused_features.shape}") 

# Create 5 clusters (Patient Profiles)

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
clusters = kmeans.fit_predict(fused_features)
# Add cluster labels back to your dataframe for analysis
clean_df['patient_cluster'] =clusters

print("Cluster Distribution:")
print(clean_df['patient_cluster'].value_counts())


# In[7]:


# ML MODEL(RANDOM FOREST)
#Create a simple label: 1 if "normal" is NOT in the text, 0 if it is
y = clean_df['findings_proc'].apply(lambda x: 0 if 'normal' in x else 1)

# Split the fused data
X_train, X_test, y_train, y_test = train_test_split(fused_features, y, test_size=0.2, random_state=42)

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate
y_pred = rf_model.predict(X_test)


# In[27]:


def test_new_patient_pro(index_num):
    # 1. Retrieve Data
    test_image_path = clean_df['full_path'].iloc[index_num]
    test_report = clean_df['findings'].iloc[index_num]
    cluster_id = clean_df['patient_cluster'].iloc[index_num]

    # 2. Get Risk Probability
    feat = fused_features[index_num].reshape(1, -1)
    # [Normal Probability, Abnormal Probability]
    prob_score = rf_model.predict_proba(feat)[0][1] * 100 

    # 3. Set Cluster Names based on Word Cloud analysis
    cluster_names = {
        0: "Unremarkable / Baseline Findings", # High freq of 'normal', 'clear', 'limits'
        1: "Cardiac & Mediastinal Conditions",
        2: "Pulmonary Opacities / Effusions",
        3: "Degenerative / Osseous Findings",
        4: "Post-Surgical / Hardware"
    }
    assigned_name = cluster_names.get(cluster_id, "Mixed Clinical Findings")

    # 4. Generate Heatmap
    heatmap = make_gradcam_heatmap(test_image_path, base_model)

    # 5. Visualization
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Original Image
    original_img = tf.keras.preprocessing.image.load_img(test_image_path)
    ax[0].imshow(original_img)
    ax[0].set_title(f"Original X-Ray (Index {index_num})")
    ax[0].axis('off')

    # Right: Heatmap Overlay
    ax[1].imshow(original_img)
    # Overlay the heatmap with transparency
    ax[1].imshow(heatmap, cmap='jet', alpha=0.4, extent=(0, original_img.size[0], original_img.size[1], 0))
    ax[1].set_title("AI Attention Area (Grad-CAM)")
    ax[1].axis('off')

    plt.tight_layout()
    plt.show()

    feat = fused_features[index_num].reshape(1, -1)
    pred_label = rf_model.predict(feat)[0]
    cluster_num = clean_df['patient_cluster'].iloc[index_num]

    # 6. Clinical Printout
    print("-" * 50)
    print(f"AI DIAGNOSTIC REPORT")
    print("-" * 50)
    print(f"AI Assigned Cluster: {cluster_num}")
    print(f"Assigned Cluster   : {assigned_name}")
    print(f"AI Predicted the patient is : {'Abnormal' if pred_label == 1 else 'Normal'}")
    print(f"Abnormality Risk   : {prob_score:.2f}%")
    print(f"Original Findings  : {test_report[:250]}...") # Truncated for display
    print("-" * 50)

    if prob_score > 50:
        print("ACTION REQUIRED: Flagged for high-priority radiologist review.")
    else:
        print("ACTION REQUIRED:  No urgent findings detected.")

# Execute for a sample patient
test_new_patient_pro(4580)


# In[9]:


# elbow plot
distortions = []
K_range = range(1, 10)
for k in K_range:
    kmeds = KMeans(n_clusters=k, random_state=42,n_init=10).fit(fused_features)
    distortions.append(kmeds.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(K_range,distortions, 'bo-')
plt.title('Elbow Method for Optimal Clusters')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (Error)')
plt.show()


# In[10]:


#WORD CLOUD (Cluster Interpretation)
print("📊 Generating Word Clouds...")
def show_cluster_cloud(cluster_num):
    text = " ".join(clean_df[clean_df['patient_cluster'] == cluster_num]['findings_proc'])
    wordcloud = WordCloud(background_color='black',width=800,height=400).generate(text)
    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud,interpolation='bilinear')
    plt.title(f'Word Cloud for Cluster {cluster_num}')
    plt.axis('off')
    plt.show()

show_cluster_cloud(0) 


# In[11]:


#CLUSTER VISUALIZATION (t-SNE)
# 1. Standardize the data (This is the most important step for 'clumsy' clusters)
scaler = StandardScaler()
fused_scaled = scaler.fit_transform(fused_features)
print(" Generating t-SNE Plot (This may take a minute)...")

# 2. Use t-SNE  (t-SNE is better at separating overlapping groups)
# Perplexity 30-50 is usually perfect for this dataset size
tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
tsne_results = tsne.fit_transform(fused_scaled)

# 3. Plot the cleaner clusters
plt.figure(figsize=(10, 7))
scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], 
            c=clean_df['patient_cluster'], 
            cmap='Spectral', # Brighter colors for better separation
            alpha=0.6, 
            edgecolors='w', 
            s=40)

plt.title('Advanced Patient Profile Visualization (t-SNE)', fontsize=15)
plt.colorbar(scatter, label='Cluster ID')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()


# In[12]:


# --- 2. CONFUSION MATRIX HEATMAP (Model Performance) ---
print("📊 Generating Confusion Matrix...")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'Abnormal'], 
            yticklabels=['Normal', 'Abnormal'])
plt.title('Confusion Matrix: Prediction Results')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()


# In[28]:


from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score
# 1. Generate the detailed Classification Report
# This gives you Precision, Recall, and F1-Score for both Normal (0) and Abnormal (1)
report = classification_report(y_test, y_pred, target_names=['Normal', 'Abnormal'])
print("Detailed Classification Report:")
print(report)

# 2. Creating a clean Summary Table for your Evaluator
metrics_data = {
    "Metric": ["Accuracy", "Precision (Abnormal)", "Recall (Abnormal)", "F1-Score (Abnormal)"],
    "Score (%)": [
        round(accuracy_score(y_test, y_pred) * 100, 2),
        round(precision_score(y_test, y_pred) * 100, 2),
        round(recall_score(y_test, y_pred) * 100, 2),
        round(f1_score(y_test, y_pred) * 100, 2)
    ]
}

metrics_df = pd.DataFrame(metrics_data)
print("\n--- Performance Summary Table ---")
print(metrics_df.to_string(index=False))


# In[ ]:




