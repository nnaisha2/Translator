import torch
from transformers import MarianMTModel, MarianTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import corpus_bleu

# Load pre-trained model and tokenizer
model_name = "Helsinki-NLP/opus-mt-de-en" 
model = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)

# Load data
with open('news-commentary-v9.de-en.de', 'r', encoding='utf-8') as file:
    german_corpus = file.readlines()
with open('news-commentary-v9.de-en.en', 'r', encoding='utf-8') as file:
    english_corpus = file.readlines()

#  both corpora have the same length
data = list(zip(german_corpus, english_corpus))

# Split into training and validation sets
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

# Define a data preparation function
def prepare_data(data, tokenizer, max_length=128):
    inputs, targets = [], []
    for src, tgt in data:
        inputs.append(src.strip())
        targets.append(tgt.strip())

    # Tokenize inputs and targets
    input_encodings = tokenizer(inputs, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
    target_encodings = tokenizer(targets, truncation=True, padding=True, max_length=max_length, return_tensors="pt")

    return input_encodings, target_encodings

# Prepare training and validation data
train_inputs, train_targets = prepare_data(train_data, tokenizer)
val_inputs, val_targets = prepare_data(val_data, tokenizer)

# Create DataLoader
def create_dataloader(inputs, targets, batch_size=32):
    dataset = list(zip(inputs["input_ids"], inputs["attention_mask"], targets["input_ids"], targets["attention_mask"]))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

train_loader = create_dataloader(train_inputs, train_targets)
val_loader = create_dataloader(val_inputs, val_targets)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define optimizer and criterion
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

# Training function
def train_epoch(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    # Wrap data_loader in tqdm for a progress bar
    for batch in tqdm(data_loader, desc="Training", leave=False):
        input_ids, attention_mask, target_ids, target_mask = [b.to(device) for b in batch]

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=target_ids)
        loss = outputs.loss
        total_loss += loss.item()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return total_loss / len(data_loader)

def evaluate_epoch(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    references, hypotheses = [], []

    # Wrap data_loader in tqdm for a progress bar
    for batch in tqdm(data_loader, desc="Evaluating", leave=False):
        input_ids, attention_mask, target_ids, target_mask = [b.to(device) for b in batch]

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=target_ids)
        loss = outputs.loss
        total_loss += loss.item()

        # Generate translations for BLEU score
        generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=128)
        hypotheses.extend([tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids])
        references.extend([[tokenizer.decode(t, skip_special_tokens=True)] for t in target_ids])

    bleu_score = corpus_bleu(references, hypotheses)
    return total_loss / len(data_loader), bleu_score

# Training loop
num_epochs = 3
for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, bleu = evaluate_epoch(model, val_loader, criterion, device)

    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, BLEU: {bleu * 100:.2f}")

# Save the model
model.save_pretrained("trained_translation_model")
tokenizer.save_pretrained("trained_translation_model")
