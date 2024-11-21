import os
import logging
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import classification_report, roc_auc_score

# Set up logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def load_evaluation_data(file_path, tokenizer, max_seq_length, batch_size):
    """Load and preprocess the evaluation data."""
    data = pd.read_csv(file_path)
    input_ids, attention_masks, token_type_ids, labels = [], [], [], []

    for _, row in data.iterrows():
        if pd.isna(row['TEXT']):
            continue  # Skip rows with missing text

        # Tokenize the input text
        encoded = tokenizer.encode_plus(
            text=row['TEXT'],
            add_special_tokens=True,
            max_length=max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        input_ids.append(encoded['input_ids'].squeeze(0))
        attention_masks.append(encoded['attention_mask'].squeeze(0))
        # Ensure token_type_ids has the correct shape
        token_type_ids.append(encoded.get('token_type_ids', torch.zeros(max_seq_length, dtype=torch.long)))

        labels.append(int(row['Label']))

    # Debugging: Print the shapes of the tensors
    print(f"Sample input_ids shape: {input_ids[0].shape}")
    print(f"Sample attention_masks shape: {attention_masks[0].shape}")
    print(f"Sample token_type_ids shape: {token_type_ids[0].shape}")

    # Stack tensors to create a TensorDataset
    dataset = TensorDataset(
        torch.stack(input_ids),
        torch.stack(attention_masks),
        torch.stack(token_type_ids).squeeze(1),  # Remove the extra dimension
        torch.tensor(labels)
    )

    return DataLoader(dataset, batch_size=batch_size)

def evaluate_model(model, dataloader, device):
    """Evaluate the model on the provided data."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_masks, token_type_ids, labels = [b.to(device) for b in batch]

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_masks,
                token_type_ids=token_type_ids
            )
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)[:, 1]  # Probability of the positive class
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Calculate metrics
    report = classification_report(all_labels, all_preds, target_names=["Not Readmitted", "Readmitted"])
    roc_auc = roc_auc_score(all_labels, all_probs)
    accuracy = (torch.tensor(all_preds) == torch.tensor(all_labels)).float().mean().item()

    return accuracy, roc_auc, report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, type=str, help="Directory containing the saved model.")
    parser.add_argument("--data_file", required=True, type=str, help="CSV file containing the evaluation data.")
    parser.add_argument("--max_seq_length", default=128, type=int, help="Maximum sequence length for BERT inputs.")
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size for evaluation.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load the model and tokenizer
    model = BertForSequenceClassification.from_pretrained(args.model_dir)
    tokenizer = BertTokenizer.from_pretrained(args.model_dir)
    model.to(device)

    # Load the evaluation data
    dataloader = load_evaluation_data(args.data_file, tokenizer, args.max_seq_length, args.batch_size)

    # Evaluate the model
    logger.info("Starting evaluation...")
    accuracy, roc_auc, report = evaluate_model(model, dataloader, device)

    # Print evaluation results to the console
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print("Classification Report:")
    print(report)

if __name__ == "__main__":
    main()
