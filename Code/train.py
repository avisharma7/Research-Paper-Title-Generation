from torch import nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torch.nn.utils.rnn import pad_sequence


batch_size = 32
num_workers = 4
shuffle = True  # Set to True to shuffle the dataset at each epoch
pin_memory = True


# Define the dataset class
class AbstractTitleDataset(Dataset):
    def __init__(self, abstracts, titles, tokenizer, max_length):
        self.abstracts = abstracts
        self.titles = titles
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def __len__(self):
        return len(self.abstracts)

    def __getitem__(self, idx):
        abstract = self.abstracts[idx]
        title = self.titles[idx]
        inputs = self.tokenizer.encode_plus(
            abstract,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding='longest',
            return_tensors='pt'
        )
        labels = self.tokenizer.encode(
            title,
            add_special_tokens=False
        )
        labels = torch.tensor(labels)
        inputs['labels'] = labels
        return inputs


# Define the collate function to handle padding of data
def collate_fn(batch):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    input_ids = pad_sequence([sample['input_ids'].squeeze() for sample in batch], batch_first=True, padding_value=tokenizer.pad_token_id)
    padded_input_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    padded_input_ids = padded_input_ids.float()
    attention_mask = pad_sequence([sample['attention_mask'].squeeze() for sample in batch], batch_first=True, padding_value=0)
    labels = pad_sequence([sample['labels'].squeeze() for sample in batch], batch_first=True, padding_value=-100)
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}


if __name__ == '__main__':
    # Load the dataset
    df=pd.read_csv("nlp_data.csv")
    abstracts=df['abstract']
    titles=df['title']
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    dataset = AbstractTitleDataset(abstracts, titles, tokenizer, max_length=512)
    shuffle = True
    pin_memory = True
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn,
                        num_workers=num_workers, shuffle=shuffle, pin_memory=pin_memory,
                        drop_last=True)

    # Load the model and tokenizer
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Set up the optimizer and training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    num_epochs = 5

    for epoch in range(num_epochs):
        print("Epoch:", epoch + 1)

        # Set the model to training mode
        model.train()

        # Iterate over the data loader
        for batch_idx, batch in enumerate(loader):
            # Send the batch to the GPU if available
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            labels = batch['labels'].cuda()

            # Zero the gradients
            optimizer.zero_grad()

            # Calculate the loss
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            # Backpropagate the loss and update the weights
            loss.backward()
            optimizer.step()

            # Print the loss every 50 batches
            if batch_idx % 50 == 0:
                print("Batch:", batch_idx, "Loss:", loss.item())

    # Save the model
    model.save_pretrained("my_finetuned_gpt2_model")
    tokenizer.save_pretrained("my_finetuned_gpt2_tokenizer")

    # Load the model and tokenizer
    model = GPT2LMHeadModel.from_pretrained("my_finetuned_gpt2_model")
    tokenizer = GPT2Tokenizer.from_pretrained("my_finetuned_gpt2_tokenizer")


    # Function to generate a title from an abstract
    def generate_title(abstract, tokenizer, model):
        input_ids = tokenizer.encode(abstract, add_special_tokens=True, return_tensors='pt')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=device)

        output = model.generate(input_ids, attention_mask=attention_mask, pad_token_id=model.config.pad_token_id,
                                max_new_tokens=50)

        title = tokenizer.decode(output[0], skip_special_tokens=True)
        return title


    # Example usage
    new_abstract="the paper proposed a newton third law of gravitation and ddeploy a new kind of app with 0.80 accuracy."
    title = generate_title(new_abstract, tokenizer, model)
    print(title)




