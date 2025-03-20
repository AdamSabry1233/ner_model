import json
import spacy
from spacy.training import Example
from spacy.util import minibatch
import random


# Load dataset
DATASET_PATH = "ner_dataset.json"  

with open(DATASET_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# Initialize a fresh spaCy model
nlp = spacy.blank("en")  # Start from scratch (no pre-trained labels)
ner = nlp.add_pipe("ner", last=True)  # Add a fresh NER pipeline

# Add custom entity labels
LABELS = ["AMOUNT", "DATE", "VENDOR", "PAYMENT_METHOD"]
for label in LABELS:
    ner.add_label(label)

print("Custom entity labels added:", ner.labels)  # Verify labels

# Convert data to spaCy Example format
examples = []
for item in data["annotations"]:
    doc = nlp.make_doc(item["text"])
    entities = [(ent["start"], ent["end"], ent["label"]) for ent in item["entities"]]
    example = Example.from_dict(doc, {"entities": entities})
    examples.append(example)

# Initialize training
optimizer = nlp.initialize()

# Training loop
n_iter = 35  # Increased training iterations for stability

for epoch in range(n_iter):
    random.shuffle(examples)
    losses = {}
    batches = minibatch(examples, size=20)
    
    for batch in batches:
        nlp.update(batch, drop=0.3, losses=losses)
    
    print(f"Losses at iteration {epoch+1}: {losses}")

# **TEST MODEL BEFORE SAVING**
test_text = "I spent $150.75 at Amazon on 2023-08-12 using my PayPal."
doc = nlp(test_text)

print("\nEntities extracted from test text:")
print([(ent.text, ent.label_) for ent in doc.ents])  # Verify entities

# Save trained model
MODEL_DIR = "ner_expense_model"
nlp.to_disk(MODEL_DIR)
print(f"\nModel saved to {MODEL_DIR}")
