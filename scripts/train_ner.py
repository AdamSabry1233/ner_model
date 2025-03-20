import json
import spacy


# Load trained NER model from Google Drive
MODEL_PATH = "/data//ner_expense_model"
nlp = spacy.load(MODEL_PATH)

# Load test sentences
TEST_DATA_PATH = "/content/drive/My Drive/test_expense_sentences.json"  # Update path if needed
with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
    test_data = json.load(f)

test_sentences = test_data["test_sentences"]

# Run model on test data
for i, text in enumerate(test_sentences):
    doc = nlp(text)
    print(f"Test Case {i+1}: {text}")
    print("Entities:", [(ent.text, ent.label_) for ent in doc.ents])
    print("-" * 50)
