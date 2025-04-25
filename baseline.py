import argparse
from keras.layers import Input, multiply, Softmax, Dense, Embedding, Conv2D, MaxPool2D, Lambda, LSTM, TimeDistributed, Masking, Bidirectional
from keras.layers import Reshape, Flatten, Dropout, Concatenate, BatchNormalization, Add
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.models import Model, load_model
import keras.backend as K
from sklearn.model_selection import train_test_split, KFold
from data_helpers import Dataloader
from sklearn.metrics import classification_report, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import os, pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Activation, Multiply

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

# # Verify GPU is detected
# physical_devices = tf.config.list_physical_devices('GPU')
# print("TensorFlow is using GPU:", tf.config.list_logical_devices('GPU'))
# if len(physical_devices) > 0:
#     print(f"Using GPU: {physical_devices}")
#     # Configure memory growth
#     for device in physical_devices:
#         tf.config.experimental.set_memory_growth(device, True)
#     print("Memory growth enabled")
# else:
#     print("No GPU found! Using CPU instead.")

# # Enable mixed precision (works well with RTX cards)
# tf.keras.mixed_precision.set_global_policy('mixed_float16')

class MetricsPlotter(Callback):
    def __init__(self, val_data, val_mask):
        super().__init__()
        self.val_data = val_data
        self.val_mask = val_mask
        self.val_f1 = []
        self.train_loss = []
        self.val_loss = []

    def on_epoch_end(self, epoch, logs=None):
        x_val, y_val = self.val_data
        pred = self.model.predict(x_val)
        y_true = []
        y_pred = []
        for i in range(pred.shape[0]):
            for j in range(pred.shape[1]):
                if self.val_mask[i][j] == 1:
                    y_true.append(np.argmax(y_val[i][j]))
                    y_pred.append(np.argmax(pred[i][j]))
        f1 = f1_score(y_true, y_pred, average='weighted')
        self.val_f1.append(f1)
        self.train_loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        print(f"\nEpoch {epoch+1} — Val F1-score: {f1:.4f}")

    def on_train_end(self, logs=None):
        plt.figure()
        plt.plot(self.train_loss, label='Train Loss')
        plt.plot(self.val_loss, label='Val Loss')
        plt.plot(self.val_f1, label='Val F1-score')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title('Training Progress')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig("training_plot.png")
        plt.show()
        
class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=None, reduction='auto', name="focal_loss"):
        super().__init__(reduction=reduction, name=name)
        self.gamma = gamma
        self.alpha = alpha

    def __call__(self, y_true, y_pred, sample_weight=None):
        # Convert inputs to float32 to avoid type mismatches
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Use tf functions directly
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Calculate crossentropy
        cross_entropy = -y_true * tf.math.log(y_pred)
        
        # Calculate focal term
        loss = tf.math.pow(1 - y_pred, self.gamma) * cross_entropy
        
        # Apply class weights if available
        if self.alpha is not None:
            loss = self.alpha * loss
            
        return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))
    
    def get_config(self):
        config = {
            "gamma": self.gamma,
            "alpha": self.alpha
        }
        base_config = super().get_config()
        return {**base_config, **config}
    

def simple_attention(inputs):
    """Basic attention that avoids shape issues completely"""
    # Compute one attention score per timestep
    attention = Dense(1, activation='tanh')(inputs)  # Shape: (batch, timesteps, 1)
    
    # Ensure softmax is applied on the correct dimension
    attention_weights = Softmax(axis=1)(attention)  # Shape: (batch, timesteps, 1)
    
    # This will properly broadcast across the feature dimension
    attended_outputs = Lambda(lambda x: x[0] * x[1])([inputs, attention_weights])
    
    return attended_outputs

class bc_LSTM:

    def __init__(self, args):
        self.classification_mode = args.classify
        self.modality = args.modality
        MODEL_PATHS = {
            'sentiment': r'D:\DL\audio_model_files\audio_weights_sentiment.keras',
            'emotion': r'D:\DL\audio_model_files\audio_weights_emotion.keras'
        }
        self.PATH = MODEL_PATHS[args.classify.lower()]
        self.OUTPUT_PATH = "./data/pickles/{}_{}.pkl".format(args.modality,self.classification_mode.lower())
        
        # Capture optimization flags
        self.use_mixup = args.mixup if hasattr(args, 'mixup') else False
        self.use_augment = args.augment if hasattr(args, 'augment') else False  
        self.use_focal = args.focal if hasattr(args, 'focal') else False
        
        print("Model initiated for {} classification".format(self.classification_mode))
        if self.use_mixup:
            print("Using mixup augmentation")
        if self.use_augment:
            print("Using audio data augmentation")
        if self.use_focal:
            print("Using focal loss")

    def load_data(self,):
        print('Loading data')
        self.data = Dataloader(mode = self.classification_mode)

        if self.modality == "text":
            self.data.load_text_data()
        elif self.modality == "audio":
            self.data.load_audio_data()
        elif self.modality == "bimodal":
            self.data.load_bimodal_data()
        else:
            exit()

        self.train_x = self.data.train_dialogue_features
        self.val_x = self.data.val_dialogue_features
        self.test_x = self.data.test_dialogue_features

        self.train_y = self.data.train_dialogue_label
        self.val_y = self.data.val_dialogue_label
        self.test_y = self.data.test_dialogue_label

        self.train_mask = self.data.train_mask
        self.val_mask = self.data.val_mask
        self.test_mask = self.data.test_mask

        self.train_id = self.data.train_dialogue_ids.keys()
        self.val_id = self.data.val_dialogue_ids.keys()
        self.test_id = self.data.test_dialogue_ids.keys()

        self.sequence_length = self.train_x.shape[1]
        
        self.classes = self.train_y.shape[2]
            
    def augment_audio_data(self):
        """Apply simple augmentation to audio features"""
        print("Applying audio data augmentation...")
        
        # Get original data
        orig_train_x = self.train_x.copy()
        orig_train_y = self.train_y.copy()
        orig_train_mask = self.train_mask.copy()
        
        augmented_train_x = []
        augmented_train_y = []
        augmented_train_mask = []
        
        # Noise addition
        for i in range(len(orig_train_x)):
            # Original sample
            augmented_train_x.append(orig_train_x[i])
            augmented_train_y.append(orig_train_y[i])
            augmented_train_mask.append(orig_train_mask[i])
            
            # Only add noise to actual audio features (where mask is 1)
            if self.classification_mode.lower() == "emotion":
                # For emotion, add noise augmentation to minority classes
                sample_label = np.argmax(orig_train_y[i][orig_train_mask[i] == 1], axis=1)
                # Check if this sample has any minority class instances
                if np.any(np.isin(sample_label, [1, 4, 5, 6])):  # Assuming these are minority classes
                    noise_level = 0.05  # 5% noise
                    noise = np.random.normal(0, noise_level, orig_train_x[i].shape)
                    noise_sample = orig_train_x[i] + noise
                    augmented_train_x.append(noise_sample)
                    augmented_train_y.append(orig_train_y[i])
                    augmented_train_mask.append(orig_train_mask[i])
                    
                    # Add time-stretched version for more diversity
                    stretch_factor = 1.05  # 5% speed up
                    time_stretch = np.zeros_like(orig_train_x[i])
                    for j in range(orig_train_x[i].shape[0]):
                        if j < int(orig_train_x[i].shape[0]/stretch_factor):
                            # Simplified time stretching
                            orig_idx = int(j * stretch_factor)
                            if orig_idx < orig_train_x[i].shape[0]:
                                time_stretch[j] = orig_train_x[i][orig_idx]
                    
                    augmented_train_x.append(time_stretch)
                    augmented_train_y.append(orig_train_y[i])
                    augmented_train_mask.append(orig_train_mask[i])
            else:
                # For sentiment, apply augmentation to all samples
                # Add slight noise to all samples
                noise_level = 0.03  # 3% noise
                noise = np.random.normal(0, noise_level, orig_train_x[i].shape)
                noise_sample = orig_train_x[i] + noise
                augmented_train_x.append(noise_sample)
                augmented_train_y.append(orig_train_y[i])
                augmented_train_mask.append(orig_train_mask[i])
        
        # Convert to numpy arrays
        self.train_x = np.array(augmented_train_x)
        self.train_y = np.array(augmented_train_y)
        self.train_mask = np.array(augmented_train_mask)
        
        print(f"Data augmented: {len(orig_train_x)} -> {len(self.train_x)} samples")

    def calc_test_result(self, pred_label, test_label, test_mask):
        true_label=[]
        predicted_label=[]

        for i in range(pred_label.shape[0]):
            for j in range(pred_label.shape[1]):
                if test_mask[i,j]==1:
                    true_label.append(np.argmax(test_label[i,j]))
                    predicted_label.append(np.argmax(pred_label[i,j]))
        print("Confusion Matrix :")
        print(confusion_matrix(true_label, predicted_label))
        print("Classification Report :")
        print(classification_report(true_label, predicted_label, digits=4))
        
        accuracy = accuracy_score(true_label, predicted_label)
        precision, recall, f1, _ = precision_recall_fscore_support(true_label, predicted_label, average='weighted')
        
        print(f'Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'Weighted F1-Score: {f1:.4f}')
        
        # Save metrics
        metrics_dir = f"./data/results/{self.modality}_{self.classification_mode.lower()}"
        if not os.path.exists(metrics_dir):
            os.makedirs(metrics_dir)
            
        with open(f"{metrics_dir}/metrics.txt", "w") as f:
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1-Score: {f1:.4f}\n")
            f.write("\nConfusion Matrix:\n")
            f.write(str(confusion_matrix(true_label, predicted_label)))
            f.write("\n\nClassification Report:\n")
            f.write(classification_report(true_label, predicted_label, digits=4))
        
        return accuracy, precision, recall, f1

    def get_audio_model(self):
        # Import regularizer
        from tensorflow.keras.regularizers import l2
        
        # Modality specific hyperparameters
        self.epochs = 150
        self.batch_size = 32

        # Modality specific parameters
        self.embedding_dim = self.train_x.shape[2]

        print("Creating Regularized Audio Model...")
        
        inputs = Input(shape=(self.sequence_length, self.embedding_dim), dtype='float32')
        masked = Masking(mask_value=0)(inputs)
        
        # Add BatchNormalization for better convergence
        x = BatchNormalization()(masked)
        
        # First LSTM layer - slightly reduced complexity (448 instead of 512)
        lstm1 = Bidirectional(LSTM(448, activation='tanh', return_sequences=True, 
                                dropout=0.35, recurrent_dropout=0.25))(x)
        x = BatchNormalization()(lstm1)
        
        # Second LSTM layer - slightly reduced complexity
        lstm2 = Bidirectional(LSTM(448, activation='tanh', return_sequences=True, 
                                dropout=0.35, recurrent_dropout=0.25), name="utter")(x)
        x = BatchNormalization()(lstm2)
        
        # Add dense layers with L2 regularization
        x = TimeDistributed(Dense(384, activation='relu', 
                                kernel_regularizer=l2(0.0015)))(x)  # Reduced from 512 to 384 units
        x = Dropout(0.4)(x)  # Increased from 0.3 to 0.4
        x = BatchNormalization()(x)
        
        x = TimeDistributed(Dense(192, activation='relu', 
                                kernel_regularizer=l2(0.0015)))(x)  # Reduced from 256 to 192 units
        x = Dropout(0.4)(x)  # Increased from 0.3 to 0.4
        
        output = TimeDistributed(Dense(self.classes, activation='softmax'))(x)

        model = Model(inputs, output)
        return model

    def get_text_model(self):
        # Modality specific hyperparameters
        self.epochs = 100
        self.batch_size = 50

        # Modality specific parameters
        self.embedding_dim = self.data.W.shape[1]

        # For text model
        self.vocabulary_size = self.data.W.shape[0]
        self.filter_sizes = [3,4,5]
        self.num_filters = 512

        print("Creating Model...")

        sentence_length = self.train_x.shape[2]

        # Initializing sentence representation layers
        embedding = Embedding(input_dim=self.vocabulary_size, output_dim=self.embedding_dim, weights=[self.data.W], input_length=sentence_length, trainable=False)
        conv_0 = Conv2D(self.num_filters, kernel_size=(self.filter_sizes[0], self.embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')
        conv_1 = Conv2D(self.num_filters, kernel_size=(self.filter_sizes[1], self.embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')
        conv_2 = Conv2D(self.num_filters, kernel_size=(self.filter_sizes[2], self.embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')
        maxpool_0 = MaxPool2D(pool_size=(sentence_length - self.filter_sizes[0] + 1, 1), strides=(1,1), padding='valid')
        maxpool_1 = MaxPool2D(pool_size=(sentence_length - self.filter_sizes[1] + 1, 1), strides=(1,1), padding='valid')
        maxpool_2 = MaxPool2D(pool_size=(sentence_length - self.filter_sizes[2] + 1, 1), strides=(1,1), padding='valid')
        dense_func = Dense(100, activation='tanh', name="dense")
        dense_final = Dense(units=self.classes, activation='softmax')
        reshape_func = Reshape((sentence_length, self.embedding_dim, 1))

        def slicer(x, index):
            return x[:,K.constant(index, dtype='int32'),:]

        def slicer_output_shape(input_shape):
            shape = list(input_shape)
            assert len(shape) == 3  # batch, seq_len, sent_len
            new_shape = (shape[0], shape[2])
            return new_shape

        def reshaper(x):
            return K.expand_dims(x, axis=3)

        def flattener(x):
            x = K.reshape(x, [-1, x.shape[1]*x.shape[3]])
            return x

        def flattener_output_shape(input_shape):
            shape = list(input_shape)
            new_shape = (shape[0], 3*shape[3])
            return new_shape

        inputs = Input(shape=(self.sequence_length, sentence_length), dtype='int32')
        cnn_output = []
        for ind in range(self.sequence_length):
            
            local_input = Lambda(slicer, output_shape=slicer_output_shape, arguments={"index":ind})(inputs) # Batch, word_indices
            
            #cnn-sent
            emb_output = embedding(local_input)
            reshape = Lambda(reshaper)(emb_output)
            concatenated_tensor = Concatenate(axis=1)([maxpool_0(conv_0(reshape)), maxpool_1(conv_1(reshape)), maxpool_2(conv_2(reshape))])
            flatten = Lambda(flattener, output_shape=flattener_output_shape,)(concatenated_tensor)
            dense_output = dense_func(flatten)
            dropout = Dropout(0.5)(dense_output)
            cnn_output.append(dropout)

        def stack(x):
            return K.stack(x, axis=1)
        cnn_outputs = Lambda(stack)(cnn_output)

        masked = Masking(mask_value=0)(cnn_outputs)
        lstm = Bidirectional(LSTM(300, activation='relu', return_sequences=True, dropout=0.4))(masked)
        lstm = Bidirectional(LSTM(300, activation='relu', return_sequences=True, dropout=0.4), name="utter")(lstm)
        output = TimeDistributed(Dense(self.classes,activation='softmax'))(lstm)

        model = Model(inputs, output)
        return model

    def get_bimodal_model(self):
        # Modality specific hyperparameters
        self.epochs = 100
        self.batch_size = 10

        # Modality specific parameters
        self.embedding_dim = self.train_x.shape[2]

        print("Creating Model...")
        
        inputs = Input(shape=(self.sequence_length, self.embedding_dim), dtype='float32')
        masked = Masking(mask_value=0)(inputs)
        lstm = Bidirectional(LSTM(300, activation='tanh', return_sequences=True, dropout=0.4), name="utter")(masked)
        output = TimeDistributed(Dense(self.classes,activation='softmax'))(lstm)

        model = Model(inputs, output)
        return model


    def cross_validate(self, k=5):
        """Perform k-fold cross-validation"""
        print(f"Performing {k}-fold cross-validation...")
        
        # Combine train and validation data
        combined_x = np.concatenate([self.train_x, self.val_x], axis=0)
        combined_y = np.concatenate([self.train_y, self.val_y], axis=0)
        combined_mask = np.concatenate([self.train_mask, self.val_mask], axis=0)
        
        # Create k folds
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(combined_x)):
            print(f"\nTraining fold {fold+1}/{k}")
            
            # Split data
            fold_train_x = combined_x[train_idx]
            fold_train_y = combined_y[train_idx]
            fold_train_mask = combined_mask[train_idx]
            
            fold_val_x = combined_x[val_idx]
            fold_val_y = combined_y[val_idx]
            fold_val_mask = combined_mask[val_idx]
            
            # Create and train model
            model = self.get_audio_model()
            
            # Use focal loss if specified
            if self.use_focal:
                loss_function = FocalLoss(gamma=2.0)
            else:
                loss_function = 'categorical_crossentropy'
                
            model.compile(optimizer='adam', loss=loss_function)
            
            # Train with early stopping
            early_stopping = EarlyStopping(monitor='val_loss', patience=10)
            model.fit(
                fold_train_x, fold_train_y,
                epochs=50,
                batch_size=self.batch_size,
                sample_weight=fold_train_mask,
                shuffle=True,
                callbacks=[early_stopping],
                validation_data=(fold_val_x, fold_val_y, fold_val_mask)
            )
            
            # Evaluate on test set
            test_preds = model.predict(self.test_x)
            
            # Calculate metrics
            true_label = []
            predicted_label = []
            
            for i in range(test_preds.shape[0]):
                for j in range(test_preds.shape[1]):
                    if self.test_mask[i,j] == 1:
                        true_label.append(np.argmax(self.test_y[i,j]))
                        predicted_label.append(np.argmax(test_preds[i,j]))
            
            # Calculate F1 score for this fold
            f1 = f1_score(true_label, predicted_label, average='weighted')
            accuracy = accuracy_score(true_label, predicted_label)
            
            print(f"Fold {fold+1} results - F1: {f1:.4f}, Accuracy: {accuracy:.4f}")
            fold_results.append((f1, accuracy))
        
        # Calculate average results
        avg_f1 = sum([r[0] for r in fold_results]) / k
        avg_acc = sum([r[1] for r in fold_results]) / k
        
        print(f"\nCross-validation results (average of {k} folds):")
        print(f"F1-score: {avg_f1:.4f}")
        print(f"Accuracy: {avg_acc:.4f}")
        
        return fold_results

    def mixup_data(self, x, y, mask, alpha=0.2):
        """Applies mixup augmentation to the data"""
        batch_size = len(x)
        indices = np.random.permutation(batch_size)
        x_mixed = []
        y_mixed = []
        mask_mixed = []
        
        for i in range(batch_size):
            # Only mix samples with similar sequence lengths
            original_length = np.sum(mask[i])
            mixed_length = np.sum(mask[indices[i]])
            
            # Only mix if lengths are similar
            if abs(original_length - mixed_length) <= 3:
                lam = np.random.beta(alpha, alpha)
                x_mixed.append(lam * x[i] + (1 - lam) * x[indices[i]])
                y_mixed.append(lam * y[i] + (1 - lam) * y[indices[i]])
                # For mask, take the maximum to ensure we don't lose information
                mask_mixed.append(np.maximum(mask[i], mask[indices[i]]))
            else:
                # If lengths are too different, use original sample
                x_mixed.append(x[i])
                y_mixed.append(y[i])
                mask_mixed.append(mask[i])
        
        return np.array(x_mixed), np.array(y_mixed), np.array(mask_mixed)

    def train_model(self):
        checkpoint = ModelCheckpoint(self.PATH, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

        if self.modality == "audio":
            model = self.get_audio_model()
            optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
        elif self.modality == "text":
            model = self.get_text_model()
            optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
        elif self.modality == "bimodal":
            model = self.get_bimodal_model()
            optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
            
        # Apply mixup augmentation if specified
        if self.use_mixup:
            print("Applying mixup augmentation to training data...")
            self.train_x, self.train_y, self.train_mask = self.mixup_data(
                self.train_x, self.train_y, self.train_mask, alpha=0.2)

        # Add learning rate scheduler
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1)
        
        # Improved early stopping with monitoring of F1-score
        early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

        # Class weighting for both emotion and sentiment
        if self.classification_mode.lower() == "emotion":
            train_labels = np.argmax(self.train_y, axis=2)
            flat_labels = train_labels.flatten()
            flat_mask = self.train_mask.flatten()
            masked_labels = flat_labels[flat_mask == 1]

            classes = np.unique(masked_labels)
            weights = compute_class_weight(class_weight="balanced", classes=classes, y=masked_labels)
            # Scale the weights to avoid extreme values
            max_weight = max(weights)
            weights = [min(w, max_weight * 0.8) for w in weights]  # Cap weights
            class_weight_dict = {int(i): float(w) for i, w in zip(classes, weights)}

            print("Emotion class weights (adjusted):", class_weight_dict)

            weighted_mask = np.zeros_like(train_labels, dtype='float32')
            for i in range(train_labels.shape[0]):
                for j in range(train_labels.shape[1]):
                    if self.train_mask[i][j] == 1:
                        cls = train_labels[i][j]
                        weighted_mask[i][j] = class_weight_dict[cls]

            sample_weight = weighted_mask
        else:
            # For sentiment, apply milder class weighting
            train_labels = np.argmax(self.train_y, axis=2)
            flat_labels = train_labels.flatten()
            flat_mask = self.train_mask.flatten()
            masked_labels = flat_labels[flat_mask == 1]

            classes = np.unique(masked_labels)
            weights = compute_class_weight(class_weight="balanced", classes=classes, y=masked_labels)
            # Scale the weights for sentiment
            weights = [w * 0.6 for w in weights]  # Reduce weight intensity for sentiment
            class_weight_dict = {int(i): float(w) for i, w in zip(classes, weights)}
            print("Sentiment class weights:", class_weight_dict)

            weighted_mask = np.zeros_like(train_labels, dtype='float32')
            for i in range(train_labels.shape[0]):
                for j in range(train_labels.shape[1]):
                    if self.train_mask[i][j] == 1:
                        cls = train_labels[i][j]
                        weighted_mask[i][j] = class_weight_dict[cls]

            sample_weight = weighted_mask

        # Add metrics plotter and other callbacks
        callbacks = [early_stopping, checkpoint, reduce_lr]
        try:
            callbacks.append(MetricsPlotter(val_data=(self.val_x, self.val_y), val_mask=self.val_mask))
        except Exception as e:
            print(f"Could not add metrics plotter: {e}")

        # Choose loss function
        if self.use_focal:
            loss_function = FocalLoss(gamma=2.0)
            print("Using Focal Loss for training")
        else:
            loss_function = 'categorical_crossentropy'

        # Implement gradient clipping to prevent exploding gradients
        model.compile(
            optimizer=optimizer, 
            loss=loss_function, 
            metrics=['accuracy'], 
        )

        print(f"Starting training for {self.epochs} epochs")
        model.fit(
            self.train_x, self.train_y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            sample_weight=sample_weight,
            shuffle=True,
            callbacks=callbacks,
            validation_data=(self.val_x, self.val_y, self.val_mask)
        )

        self.test_model()


    def test_model(self):
        custom_objects = {'FocalLoss': FocalLoss}
        model = load_model(self.PATH, custom_objects=custom_objects)
        intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer("utter").output)

        intermediate_output_train = intermediate_layer_model.predict(self.train_x)
        intermediate_output_val = intermediate_layer_model.predict(self.val_x)
        intermediate_output_test = intermediate_layer_model.predict(self.test_x)

        train_emb, val_emb, test_emb = {}, {}, {}
        for idx, ID in enumerate(self.train_id):
            train_emb[ID] = intermediate_output_train[idx]
        for idx, ID in enumerate(self.val_id):
            val_emb[ID] = intermediate_output_val[idx]
        for idx, ID in enumerate(self.test_id):
            test_emb[ID] = intermediate_output_test[idx]
        pickle.dump([train_emb, val_emb, test_emb], open(self.OUTPUT_PATH, "wb"))

        predictions = model.predict(self.test_x)
        self.calc_test_result(predictions, self.test_y, self.test_mask)
        
        # Perform ensemble prediction if we have already saved multiple models
        ensemble_dir = f"./data/models/ensemble/{self.modality}_{self.classification_mode.lower()}"
        if os.path.exists(ensemble_dir):
            model_files = [f for f in os.listdir(ensemble_dir) if f.endswith('.keras')]
            if len(model_files) >= 3:  # Only do ensemble if we have at least 3 models
                print(f"\nPerforming ensemble prediction with {len(model_files)} models...")
                ensemble_predictions = []
                
                for model_file in model_files:
                    model_path = os.path.join(ensemble_dir, model_file)
                    try:
                        m = load_model(model_path, custom_objects={'FocalLoss': FocalLoss})
                        pred = m.predict(self.test_x)
                        ensemble_predictions.append(pred)
                    except Exception as e:
                        print(f"Error loading model {model_file}: {e}")
                
                if ensemble_predictions:
                    # Average the predictions
                    avg_pred = np.mean(ensemble_predictions, axis=0)
                    print("\nEnsemble Model Results:")
                    self.calc_test_result(avg_pred, self.test_y, self.test_mask)

    def resume_training(self):
        """Resume training from the last checkpoint"""
        print("Resuming training from checkpoint...")
        
        # Load the model with custom objects
        custom_objects = {'FocalLoss': FocalLoss}
        model = load_model(self.PATH, custom_objects=custom_objects)
        
        # Continue training with early stopping and reduced learning rate
        early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1)
        checkpoint = ModelCheckpoint(self.PATH, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
        
        # Add metrics plotter if available
        callbacks = [early_stopping, checkpoint, reduce_lr]
        try:
            callbacks.append(MetricsPlotter(val_data=(self.val_x, self.val_y), val_mask=self.val_mask))
        except Exception as e:
            print(f"Could not add metrics plotter: {e}")
        
        # Continue training
        model.fit(
            self.train_x, self.train_y,
            epochs=50,  # Additional epochs
            initial_epoch=20,  # Start from where you left off
            batch_size=self.batch_size,
            sample_weight=self.train_mask,
            shuffle=True,
            callbacks=callbacks,
            validation_data=(self.val_x, self.val_y, self.val_mask)
        )
        
        self.test_model()

if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser()
    parser.required=True
    parser.add_argument("-classify", help="Set the classifiction to be 'Emotion' or 'Sentiment'", required=True)
    parser.add_argument("-modality", help="Set the modality to be 'text' or 'audio' or 'bimodal'", required=True)
    parser.add_argument("-train", default=False, action="store_true", help="Flag to intiate training")
    parser.add_argument("-test", default=False, action="store_true", help="Flag to initiate testing")
    parser.add_argument("-augment", default=False, action="store_true", help="Flag to use data augmentation")
    parser.add_argument("-mixup", default=False, action="store_true", help="Flag to use mixup augmentation")
    parser.add_argument("-focal", default=False, action="store_true", help="Flag to use focal loss")
    parser.add_argument("-cv", default=False, action="store_true", help="Flag to perform cross-validation")
    parser.add_argument("-resume", default=False, action="store_true", help="Flag to resume training from checkpoint")
    args = parser.parse_args()

    if args.classify.lower() not in ["emotion", "sentiment"]:
        print("Classification mode hasn't been set properly. Please set the classifiction flag to be: -classify Emotion/Sentiment")
        exit()
    if args.modality.lower() not in ["text", "audio", "bimodal"]:
        print("Modality hasn't been set properly. Please set the modality flag to be: -modality text/audio/bimodal")
        exit()

    args.classify = args.classify.title()
    args.modality = args.modality.lower()
    
    # Check directory existence
    for directory in ["./data/pickles", "./data/models", "./data/results", "./data/models/ensemble"]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    model = bc_LSTM(args)
    model.load_data()

    # Apply data augmentation if specified
    if args.augment and args.modality == "audio" and not args.test:
        model.augment_audio_data()

    if args.resume:
        # Resume training from checkpoint
        model.resume_training()
    elif args.cv:
        # Perform cross-validation
        model.cross_validate(k=5)
    elif args.test:
        model.test_model()
    else:
        model.train_model()