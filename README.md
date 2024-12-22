### Instructions for Testing and Training the Model

#### **TEST PRE-TRAINED MODEL**
- To test the pre-trained model, run the file named `04inference_classifier.py`.

---

#### **TRAIN YOUR MODEL**

1. **Delete Existing Files**  
   If the following files or folders exist, delete them:  
   - `data` (complete folder)  
   - `data.pickle`  
   - `model.p`

2. **Collect Data**  
   - Run the file `01collect_image.py` to collect data.  
   - Press `Q` to start capturing images.  
   - This script is designed to collect data for three letters: `A`, `B`, and `L`.  
   - To add more letters, modify the script as needed.

3. **Create Dataset**  
   - Run the file `02create_dataset.py`.  
   - Since the captured images are large, this step extracts the X-Y coordinates of fingers and stores the processed data in a file named `data.pickle`.

4. **Train the Classifier**  
   - Run the file `03train_classifier.py`.  
   - This script trains a `RandomForestClassifier` on the dataset stored in `data.pickle`.

5. **Test Your Model**  
   - Run the file `04inference_classifier.py` to test the trained model.  
   - Press `Q` to quit.

> **_DEPENDENCY_**
>1. opencv-python
>2. mediapipe
>3. scikit-learn