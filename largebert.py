#!/usr/bin/env python
# coding: utf-8

# # FineTuning BERT for Multi-Class Classification with custom datasets
# 
# 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch, os, re, string
from torch.utils.data import Dataset
from sklearn.metrics import precision_recall_fscore_support,  accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
import matplotlib.pyplot as plt
print(torch.cuda.is_available())  # should print True


# In[ ]:





# In[2]:


#df = pd.read_csv('../data/atdt_data.csv')
df = pd.read_csv('all_data_combined_fixed.csv', encoding='utf-8')


# Drop na
df = df.dropna()

df


# In[3]:


def preprocess_text(text: str) -> str:
    # Remove URLs
    # text = re.sub(r"http\S+", "", text)
    # Remove punctuation (optional; note that BERT is pretrained on punctuation, so use carefully)
    try:
        text = text.translate(str.maketrans("", "", string.punctuation))
    except:
        print(text)
    # Lowercase the text (if using a cased model, skip this)
    text = text.lower()
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

df['text'] = df['text'].apply(preprocess_text)


# In[4]:


from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
device 


# In[5]:


labels = df['label'].unique().tolist()
len(labels)


# In[6]:


id_to_label = {
    0: 'Non‑CUI',
    1: 'Ammonium Nitrate',
    2: 'Chemical‑terrorism Vulnerability Information',
    3: 'Critical Energy Infrastructure Information',
    4: 'Emergency Management',
    5: 'General Critical Infrastructure Information',
    6: 'Information Systems Vulnerability Information',
    7: 'Physical Security',
    8: 'Protected Critical Infrastructure Information',
    9: 'SAFETY Act Information',
    10: 'Toxic Substances',
    11: 'Water Assessments',
    12: 'Controlled Technical Information',
    13: 'DoD Critical Infrastructure Security Information',
    14: 'Naval Nuclear Propulsion Information',
    15: 'Privileged Safety Information',
    16: 'Unclassified Controlled Nuclear Information - Defense',
    17: 'Export Controlled',
    18: 'Export Controlled Research',
    19: 'Bank Secrecy',
    20: 'Budget',
    21: 'Comptroller General',
    22: 'Consumer Complaint Information',
    23: 'Electronic Funds Transfer',
    24: 'Federal Housing Finance Non‑Public Information',
    25: 'Financial Supervision Information',
    26: 'General Financial Information',
    27: 'International Financial Institutions',
    28: 'Mergers',
    29: 'Net Worth',
    30: 'Retirement',
    31: 'Asylee',
    32: 'Battered Spouse or Child',
    33: 'Permanent Resident Status',
    34: 'Status Adjustment',
    35: 'Temporary Protected Status',
    36: 'Victims of Human Trafficking',
    37: 'Visas',
    38: 'Agriculture',
    39: 'Foreign Intelligence Surveillance Act',
    40: 'Foreign Intelligence Surveillance Act Business Records',
    41: 'General Intelligence',
    42: 'Geodetic Product Information',
    43: 'Intelligence Financial Records',
    44: 'Internal Data',
    45: 'Operations Security',
    46: 'International Agreement Information',
    47: 'Accident Investigation',
    48: 'Campaign Funds',
    49: 'Committed Person',
    50: 'Communications',
    51: 'Controlled Substances',
    52: 'Criminal History Records Information',
    53: 'DNA',
    54: 'General Law Enforcement',
    55: 'Informant',
    56: 'Investigation',
    57: 'Juvenile',
    58: 'Law Enforcement Financial Records',
    59: 'National Security Letter',
    60: 'Pen Register/Trap & Trace',
    61: 'Reward',
    62: 'Sex Crime Victim',
    63: 'Terrorist Screening',
    64: 'Whistleblower Identity',
    65: 'Administrative Proceedings',
    66: 'Child Victim/Witness',  # Originally 67
    67: 'Collective Bargaining',
    68: 'Federal Grand Jury',
    69: 'Legal Privilege',
    70: 'Legislative Materials',
    71: 'Presentence Report',
    72: 'Prior Arrest',
    73: 'Protective Order',
    74: 'Victim',
    75: 'Witness Protection',
    76: 'Archaeological Resources',
    77: 'Historic Properties',
    78: 'National Park System Resources',
    79: 'NATO Restricted',
    80: 'NATO Unclassified',
    81: 'General Nuclear',
    82: 'Nuclear Recommendation Material',
    83: 'Nuclear Security‑Related Information',
    84: 'Safeguards Information',
    85: 'Unclassified Controlled Nuclear Information - Energy',
    86: 'Patent Applications',
    87: 'Inventions',
    88: 'Secrecy Orders',
    89: 'Contract Use',
    90: 'Death Records',
    91: 'General Privacy',
    92: 'Genetic Information',
    93: 'Health Information',
    94: 'Inspector General Protected',
    95: 'Military Personnel Records',
    96: 'Personnel Records',
    97: 'Student Records',
    98: 'General Procurement and Acquisition',
    99: 'Small Business Research and Technology',
    100: 'Source Selection',
    101: 'Entity Registration Information',
    102: 'General Proprietary Business Information',
    103: 'Ocean Common Carrier and Marine Terminal Operator Agreements',
    104: 'Ocean Common Carrier Service Contracts',
    105: 'Proprietary Manufacturer',
    106: 'Proprietary Postal',
    107: 'Homeland Security Agreement Information',
    108: 'Homeland Security Enforcement Information',
    109: 'Information Systems Vulnerability Information - Homeland',
    110: 'International Agreement Information - Homeland',
    111: 'Operations Security Information',
    112: 'Personnel Security Information',
    113: 'Physical Security - Homeland',
    114: 'Privacy Information',
    115: 'Sensitive Personally Identifiable Information',
    116: 'Investment Survey',
    117: 'Pesticide Producer Survey',
    118: 'Statistical Information',
    119: 'US Census',
    120: 'Federal Taxpayer Information',
    121: 'Tax Convention',
    122: 'Taxpayer Advocate Information',
    123: 'Written Determinations',
    124: 'Railroad Safety Analysis Records',
    125: 'Sensitive Security Information'
}

label_to_id = {v: k for k, v in id_to_label.items()}
NUM_LABELS = len(labels)  # 126


# In[7]:


#NUM_LABELS= len(labels)

#id_to_label= {0 : 'Non-CUI' , 1: 'Ammonium Nitrate', 2: 'Chemical Terrorism', 3 : 'Critical Energy', 4 : 'Emergency Management', 5 : 'General Critical Infrastructure', 6: 'Information System Vulnerability', 7: 'Physical Security'}

#label_to_id= { 'Non-CUI':0, 'Ammonium Nitrate' : 1 , 'Chemical Terrorism' : 2, 'Critical Energy': 3, 'Emergency Management' :4, 'General Critical Infrastructure' : 5, 'Information System Vulnerability' : 6, 'Physical Security' : 7}


# In[8]:


plt.hist(df['label'])
print(f"Label Skew: {df['label'].skew()}")


# In[9]:


from sklearn.model_selection import train_test_split
label_counts = df['label'].value_counts()
df = df[df['label'].isin(label_counts[label_counts > 1].index)]

# Assuming df is your DataFrame with 'text' and 'label' columns
print("Total samples:", len(df))

# First, split off the training set (e.g., 50% of data) with stratification
train_df, temp_df = train_test_split(
    df, 
    test_size=0.2, 
    stratify=df['label'], 
    random_state=42
)

# Then, split the remaining data into validation and test sets (each 25% of total)
val_df, test_df = train_test_split(
    temp_df, 
    test_size=0.5, 
    stratify=temp_df['label'], 
    random_state=42
)

print("Train samples:", len(train_df))
print("Validation samples:", len(val_df))
print("Test samples:", len(test_df))

# Convert the splits to lists as needed:
train_texts = train_df['text'].tolist()
train_labels = train_df['label'].tolist()

val_texts = val_df['text'].tolist()
val_labels = val_df['label'].tolist()

test_texts = test_df['text'].tolist()
test_labels = test_df['label'].tolist()


# In[10]:


print(f'Number of Labels in Validation Set: {len(val_labels)}')


# In[11]:


def get_class_weights(labels):
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=labels)
    return torch.tensor(weights, dtype=torch.float)

# Assuming train_labels is a list or numpy array of label IDs
class_weights = get_class_weights(train_labels)

class_weights # Display label weights


# In[12]:


from transformers import AutoTokenizer


# In[13]:


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased") # , max_length=512


# In[14]:


train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)


# In[15]:


from transformers import pipeline, AutoModelForSequenceClassification


# In[16]:


model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(labels), id2label=id_to_label, label2id=label_to_id) #bert-large-uncased


##model.to('cuda')


# In[17]:


from transformers import TrainingArguments


# In[18]:


from transformers import TrainingArguments
import inspect
print(inspect.signature(TrainingArguments.__init__))


# In[ ]:


training_args = TrainingArguments(
    output_dir='./ModernBERT-AT-DT-finetuned',
    do_train=True,
    do_eval=True,
    load_best_model_at_end=True,
    metric_for_best_model="eval_F1",
    greater_is_better=True,

    num_train_epochs=8,
    per_device_train_batch_size=4,   # ← sweet spot for A100 16
    per_device_eval_batch_size=4,    # 32
    gradient_accumulation_steps=2,    # effective batch size = 32
    fp16=True,                        # speeds up on A100 (Tensor Cores)

    warmup_steps=100,
    weight_decay=0.01,

    optim='adamw_torch_fused',        # great choice for fused kernel
    learning_rate=3e-5,

    logging_strategy='steps',
    logging_dir='./multi-class-logs',
    logging_steps=50,

    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=8,
)


# In[20]:


training_args = TrainingArguments(
    # The output directory where the model predictions and checkpoints will be written
    output_dir='./ModernBERT-AT-DT-finetuned',
    do_train=True,
    do_eval=True,

    # Load the best model at the end
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    #  The number of epochs, defaults to 3.0
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,  # simulates a larger batch size without increasing memory usage
    fp16=True,                      # enable mixed precision training if supported
    # Number of steps used for a linear warmup
    warmup_steps=100,
    weight_decay=0.01,
    logging_strategy='steps',
    # The initial learning rate
    optim='adamw_torch_fused',
    learning_rate=2e-5,
   # TensorBoard log directory
    logging_dir='./multi-class-logs',
    logging_steps=25,
    eval_strategy="steps",
    eval_steps=25,
    save_strategy="steps",
    save_total_limit=10,             # limits the number of checkpoints saved 
)





# In[21]:


from transformers import TrainingArguments
import inspect

print(inspect.signature(TrainingArguments.__init__))


# In[22]:


class DataLoader(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        """
          This construct a dict that is (index position) to encoding pairs.
          Where the Encoding becomes tensor(Encoding), which is an requirements
          for training the model
        """
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        """
        Returns the number of data items in the dataset.

        """
        return len(self.labels)


# In[23]:


train_dataloader = DataLoader(train_encodings, train_labels)
val_dataloader = DataLoader(val_encodings,val_labels)
test_dataloader = DataLoader(test_encodings,test_labels)


# In[24]:


def compute_metrics(pred):

    # Extract true labels from the input object
    labels = pred.label_ids

    # Obtain predicted class labels by finding the column index with the maximum probability
    preds = pred.predictions.argmax(-1)

    # Compute macro precision, recall, and F1 score using sklearn's precision_recall_fscore_support function
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro',zero_division=1)

    # Calculate the accuracy score using sklearn's accuracy_score function
    acc = accuracy_score(labels, preds)

    # Return the computed metrics as a dictionary
    return {
        'Accuracy': acc,
        'F1': f1,
        'Precision': precision,
        'Recall': recall
    }


# In[25]:


import torch
import torch.nn as nn
from transformers import Trainer

class WeightedLossTrainer(Trainer):
    def __init__(self, class_weights, num_labels, *args, **kwargs):
        """
        Initializes the trainer with class weights and number of labels.
        :param class_weights: A torch.Tensor of shape (num_labels,) with the weight for each class.
        :param num_labels: The total number of classes.
        """
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.num_labels = num_labels

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        # Ensure the class weights are on the same device as logits
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


# In[26]:


trainer = WeightedLossTrainer(
    #the pre-trained bert model that will be fine-tuned
    model=model,
    #training arguments that we defined above
    args=training_args,
    train_dataset= train_dataloader,
    eval_dataset = val_dataloader,
    compute_metrics= compute_metrics,
    class_weights=class_weights,
    num_labels=NUM_LABELS
)


# In[27]:


import torch._dynamo
torch._dynamo.config.suppress_errors = True


# In[28]:


print(set['label'])  # Or however your labels are stored




# In[29]:


trainer.train()


# In[26]:


print("Labels in training dataset:", set(train_df['label']))
print("Labels in validation dataset:", set(val_df['label']))


# In[27]:


model_path = "bert-large-AT-DT-8epoch-balancedloss-fp16"

trainer.save_model(model_path)
tokenizer.save_pretrained(model_path)


# In[28]:


model_path = "bert-large-AT-DT-8epoch-balancedloss-fp16" #"ModernBERT-AT-DT-1/checkpoint-309"

model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer= AutoTokenizer.from_pretrained(model_path)
nlp= pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)


# In[29]:


def prediction(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    # Move inputs to the same device as the model
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_probabilities = torch.softmax(logits, dim=1)
    predicted_label_id = torch.argmax(predicted_probabilities, dim=1).item()
    confidence = predicted_probabilities[0][predicted_label_id].item() * 100  # Convert to percentage
    predicted_label = id_to_label[predicted_label_id]
    return predicted_label_id, predicted_label, confidence


# In[30]:


# Test with a small sample first
predictions = []
confidences = []
labels = []
for text in test_texts[:30]:
    pred_id, pred_label, confidence = prediction(text)
    predictions.append(pred_id)
    confidences.append(confidence)
    labels.append(pred_label)

# Create a DataFrame to display results
pd.DataFrame({
    'text': test_texts[:30],
    'predicted_label': labels,
    'confidence': [f"{conf:.2f}%" for conf in confidences],
    'actual': [id_to_label[label] for label in test_labels[:30]]
})


# # Measurement

# In[31]:


# Run predictions on the full test set
all_predictions = []
all_confidences = []
all_labels = []

for text in test_texts:
    pred_id, pred_label, confidence = prediction(text)
    all_predictions.append(pred_id)
    all_confidences.append(confidence)
    all_labels.append(pred_label)

# Calculate accuracy with the prediction IDs (for metrics)
accuracy = accuracy_score(test_labels, all_predictions)
print('Accuracy on Test Set: {:.2%}'.format(accuracy))

# Calculate average confidence
avg_confidence = sum(all_confidences) / len(all_confidences)
print('Average Confidence: {:.2f}%'.format(avg_confidence))

# Show confidence distribution
plt.figure(figsize=(10, 6))
plt.hist(all_confidences, bins=20)
plt.title('Confidence Distribution')
plt.xlabel('Confidence (%)')
plt.ylabel('Count')
plt.show()


# In[32]:


from sklearn.metrics import confusion_matrix
import seaborn as sns

# Compute the confusion matrix
cm = confusion_matrix(test_labels, all_predictions)

# Create a heatmap of the confusion matrix
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=[id_to_label[i] for i in range(NUM_LABELS)],
            yticklabels=[id_to_label[i] for i in range(NUM_LABELS)])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')

# Save the confusion matrix as an image
plt.savefig('confusion_matrix.png')
plt.show()

# Calculate per-class metrics
print("\nPer-class Metrics:")
for i in range(NUM_LABELS):
    class_indices = [idx for idx, label in enumerate(test_labels) if label == i]
    if class_indices:
        class_accuracy = sum(1 for idx in class_indices if all_predictions[idx] == i) / len(class_indices)
        class_avg_confidence = sum(all_confidences[idx] for idx in class_indices if all_predictions[idx] == i) / max(1, sum(1 for idx in class_indices if all_predictions[idx] == i))
        print(f"Class {id_to_label[i]}: Accuracy = {class_accuracy:.2%}, Avg Confidence = {class_avg_confidence:.2f}%")


# In[33]:


# Threshold analysis to find optimal confidence cutoff
import numpy as np
from sklearn.metrics import precision_recall_curve

# Create arrays for thresholds analysis
thresholds = np.arange(0, 100, 1)  # Test thresholds from 0% to 100% in 1% increments
accuracies = []
coverage = []  # Percentage of data points retained at each threshold

for threshold in thresholds:
    # Filter predictions with confidence >= threshold
    filtered_indices = [i for i, conf in enumerate(all_confidences) if conf >= threshold]

    if filtered_indices:  # Avoid division by zero
        # Calculate accuracy on filtered predictions
        filtered_accuracy = sum(1 for i in filtered_indices if all_predictions[i] == test_labels[i]) / len(filtered_indices)
        accuracies.append(filtered_accuracy)

        # Calculate coverage (percentage of data retained)
        coverage_pct = len(filtered_indices) / len(all_predictions)
        coverage.append(coverage_pct)
    else:
        accuracies.append(0)
        coverage.append(0)

# Plot accuracy vs threshold
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(thresholds, accuracies, 'b-', linewidth=2)
plt.xlabel('Confidence Threshold (%)')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Confidence Threshold')
plt.grid(True)

# Plot coverage vs threshold
plt.subplot(1, 2, 2)
plt.plot(thresholds, coverage, 'r-', linewidth=2)
plt.xlabel('Confidence Threshold (%)')
plt.ylabel('Coverage (% of data retained)')
plt.title('Coverage vs Confidence Threshold')
plt.grid(True)
plt.tight_layout()
plt.show()

# Find optimal threshold (balancing accuracy and coverage)
# You can adjust the weights to prioritize accuracy or coverage
accuracy_weight = 0.7
coverage_weight = 0.3
combined_score = [(a * accuracy_weight + c * coverage_weight) for a, c in zip(accuracies, coverage)]
optimal_idx = np.argmax(combined_score)
optimal_threshold = thresholds[optimal_idx]

print(f"Optimal confidence threshold: {optimal_threshold:.1f}%")
print(f"At this threshold:")
print(f"  - Accuracy: {accuracies[optimal_idx]:.2%}")
print(f"  - Coverage: {coverage[optimal_idx]:.2%} of data")
print(f"  - Discarded: {(1-coverage[optimal_idx]):.2%} of predictions")

# Create a table showing accuracy at different thresholds
threshold_table = pd.DataFrame({
    'Threshold (%)': thresholds[::5],  # Show every 5th threshold for brevity
    'Accuracy': [f"{a:.2%}" for a in accuracies[::5]],
    'Coverage': [f"{c:.2%}" for c in coverage[::5]],
    'Discarded': [f"{(1-c):.2%}" for c in coverage[::5]]
})
threshold_table


# In[ ]: