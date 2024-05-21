# -*- coding: utf-8 -*-
"""
Created on Sun May 19 21:11:23 2024

@author: 21443
"""

import os
import random

import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
import torch
from collections import Counter
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, \
    roc_auc_score
import tqdm
from matplotlib import pyplot as plt

# Set CUDA device visibility
os.environ['CUDA_VISIBLE_DEVICES'] = ''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Custom dataset class for text data
class TextDataset(Dataset):
    def __init__(self, datas):
        self.datas = datas

    def __getitem__(self, idx):
        return self.datas[idx][0], self.datas[idx][1]

    def __len__(self):
        return len(self.datas)

# Classifier model based on BERT
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.part1 = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.part2 = nn.Linear(768, 2)
        for x in self.part1.parameters():
            x.requires_grad = False

    def forward(self, x):
        x = self.part1(x)
        x = self.part2(x['pooler_output'])
        x = F.softmax(x, dim=1)
        return x

# Main class for text mining
class TextMinner(object):
    def __init__(self):
        self.accuracies = []
        self.precessions = []
        self.recalls = []
        self.aucs = []

    def seperate_data(self):
        try:
            # Load dataset from CSV file
            self.df = pd.read_csv("C:/Users/21443/Desktop/大三下 春季/cps 3320/W02_5/code/dataset.csv", encoding='unicode_escape')
            print(self.df.head)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return

        # self.df = self.df.iloc[:100, :]
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # Tokenize text and create input tensors
        text = [self.df.iloc[i]["Article"] for i in range(len(self.df))]
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids = inputs["input_ids"]
        labels = []
        for i in range(len(self.df)):
            if self.df.iloc[i]["NewsType"] == 'business':
                labels.append(0)
            else:
                labels.append(1)

        print(Counter(labels))
        data = [[input_ids[i], labels[i]] for i in range(len(labels))]
        random.seed(12)
        random.shuffle(data)
        self.train_dataset = TextDataset(data[:int(len(data) * 0.9)])
        self.test_dataset = TextDataset(data[int(len(data) * 0.9):])
        self.test_loader = DataLoader(self.test_dataset, batch_size=64, shuffle=False, num_workers=0)

    def build_model(self, load_from=''):
        # Initialize the model and load pre-trained weights if provided
        self.model = Classifier()
        if load_from:
            try:
                self.model.part2.load_state_dict(torch.load(load_from))
            except Exception as e:
                print(f"Error loading model weights: {e}")
        self.model.to(device)

    def train_model(self):
        # Train the model
        train_loader = DataLoader(self.train_dataset, batch_size=16, shuffle=True, num_workers=0)
        optimizer = torch.optim.Adam(self.model.part2.parameters(), lr=0.00002)
        step_all = 0
        for epoch in range(4):
            for step, batch in tqdm.tqdm(enumerate(train_loader)):
                out = self.model(batch[0].to(device))
                label = batch[1].to(device)
                loss = F.cross_entropy(out, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if step_all % 10 == 0:
                    self.evaluate_model()
                step_all = step_all + 1

        try:
            # Save the trained model weights
            self.model.cpu()
            torch.save(self.model.part2.state_dict(), "C:/Users/21443/Desktop/大三下 春季/cps 3320/W02_5/code/model.pt")
        except Exception as e:
            print(f"Error saving model weights: {e}")

        # Plot evaluation metrics
        fig, axs = plt.subplots(4, 1, figsize=(10, 15))
        axs[0].plot(self.accuracies, label=f'accuracies')
        axs[1].plot(self.precessions, label=f'precessions')
        axs[2].plot(self.recalls, label=f'recalls')
        axs[3].plot(self.aucs, label=f'aucs')
        axs[0].legend()
        axs[1].legend()
        axs[2].legend()
        axs[3].legend()
        plt.tight_layout()
        plt.savefig('curve.jpg')
        plt.show()

    def evaluate_model(self, plot=False):
        # Evaluate the model on the test dataset
        y_test, y_pred = [], []
        with torch.no_grad():
            for step, batch in enumerate(self.test_loader):
                print('eval batch', step)
                out = self.model(batch[0].to(device))
                out = out.cpu()
                label = batch[1]
                y_test.extend(label.numpy().tolist())
                y_pred.extend(out.numpy().tolist())

        y_pred_label = []
        for x in y_pred:
            y_pred_label.append(0 if x[0] > x[1] else 1)

        cm = confusion_matrix(y_test, y_pred_label)
        print("Confusion Matrix:\n", cm)
        cr = classification_report(y_test, y_pred_label)
        print("\nClassification Report:\n", cr)

        if not plot:
            y_pred = [x[1] for x in y_pred]
            self.accuracies.append(accuracy_score(y_test, y_pred_label))
            self.precessions.append(precision_score(y_test, y_pred_label))
            self.recalls.append(recall_score(y_test, y_pred_label))
            self.aucs.append(roc_auc_score(y_test, y_pred))

        if plot:
            points = [[], []]
            for x, label in zip(y_pred, y_test):
                points[label].append(x)
            plt.scatter([x[0] for x in points[0]], [x[1] for x in points[0]], label='business')
            plt.scatter([x[0] for x in points[1]], [x[1] for x in points[1]], label='sports')
            plt.legend()
            plt.savefig('points.jpg')
            plt.show()

    def predict_text(self, text):
        # Predict the label for a given text
        try:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            input_ids = inputs["input_ids"]
            self.model.cpu()
            with torch.no_grad():
                out = self.model(input_ids)
            digit = out.detach().cpu().numpy()
            out = out.detach().cpu().numpy()
            print('input:', text)
            if out[0][0] > out[0][1]:
                ret = 'business'
            else:
                ret = 'sports'
            print(f'predict result: {digit}, {ret}')
        except Exception as e:
            print(f"Error predicting text: {e}")

if __name__ == '__main__':
    
    # Train the model
    # text_minner = TextMinner()
    # text_minner.build_model()
    # text_minner.seperate_data()
    # text_minner.train_model()

    # Evaluate the model
    #text_minner = TextMinner()
    #text_minner.build_model(load_from="C:/Users/21443/Desktop/大三下 春季/cps 3320/3320 final project/code/model.pt")
    #text_minner.seperate_data()
    #text_minner.evaluate_model(plot=True)    

      
    # Predict text labels
    text_minner = TextMinner()
    try:
        text_minner.seperate_data()
    except Exception as e:
        print(f"Error separating data: {e}")
        exit()

    try:
        text_minner.build_model(load_from="C:/Users/21443/Desktop/大三下 春季/cps 3320/W02_5/code/model.pt")
    except Exception as e:
        print(f"Error building model: {e}")
        exit()

    while True:
        try:
            input_text = input("Enter a news content (or 'q' to quit): ")
            if input_text.lower() == 'q':
                break
            text_minner.predict_text(input_text)
            texts = input_text.split('.')
            for text in texts:
                text_minner.predict_text(text)
        except Exception as e:
            print(f"Error: {e}")
            print("Please enter a valid news content.")