# **College uniform classification system**


 
## **Description**

The College Uniform Classification System is a deep learning-based application that not only classifies whether individuals are wearing a uniform, but now also detects the presence and position of students in images. The updated version supports multi-person detection, bounding box annotations, and gives feedback even when no student is present in the image.

It leverages Convolutional Neural Networks (CNNs), transfer learning, and object detection techniques to deliver high accuracy (up to 97%) even with a limited dataset.

----

## **Demeo Video**





https://github.com/user-attachments/assets/26761c41-fd23-438d-b51d-f9b7ffcac9cf


----------------

## **Key Features**

1. **🔍 Smart Student Detection + Classification**  
   - Detects students in an image and only classifies areas where students are present.  
   - Outputs:
     - 🟩 **Green Box** + *"Wearing uniform"* if uniform is detected.  
     - 🟥 **Red Box** + *"Not wearing uniform"* if no uniform is detected.  
     - ❌ *"No students present"* if no person is detected.

2. **🧠 Binary Image Classification**  
   - Classifies each detected person into:
     - **Wearing Uniform**
     - **Not Wearing Uniform**

3. **🗂️ Custom Dataset Creation**  
   - Collected data using Python web scraping.  
   - Prepared a custom dataset specifically for this task.

4. **🧪 Data Augmentation**  
   - Enhances model performance by generating diverse training samples through:
     - Flipping  
     - Rotation  
     - Zooming  
     - Shifting

5. **💻 User-Friendly Interface**  
   - Built using **Streamlit** for quick testing and simple interaction.

6. **📷 Flexible Input Options**  
   - Classify images by:
     - Uploading from local system
     - Live camera feed input


## **Feature Improvements**

🎥 Real-time Multi-Person Video Detection:
Adding support to classify multiple students in real-time video streams.
This will involve:

Real-time object detection (YOLO or Haar cascades)

Frame-wise analysis for uniform presence

Efficient video stream processing



------

## **contact**

For any inquiries or questions, please contact us at sana000dasaradha@gmail.com

