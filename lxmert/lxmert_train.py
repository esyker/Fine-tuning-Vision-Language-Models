import torch
import pandas as pd
import numpy as np
from transformers import LxmertTokenizer, LxmertConfig, LxmertModel, LxmertForPreTraining, get_scheduler
from modeling_frcnn import GeneralizedRCNN
import utils
from processing_image import Preprocess
from torch.optim import AdamW
from torch.utils.data import DataLoader
import lmdb
import pickle
import time
import io
from tqdm import tqdm
import math

class ASLSingleLabel(torch.nn.Module):
    '''
    This loss is intended for single-label classification problems
    '''
    def __init__(self, gamma_pos=0, gamma_neg=4, eps: float = 0.1, reduction='mean'):
        super(ASLSingleLabel, self).__init__()

        self.eps = eps
        self.logsoftmax = torch.nn.LogSoftmax(dim=-1)
        self.targets_classes = []
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.reduction = reduction

    def forward(self, inputs, target):
        '''
        "input" dimensions: - (batch_size,number_classes)
        "target" dimensions: - (batch_size)
        '''
        num_classes = inputs.size()[-1]
        log_preds = self.logsoftmax(inputs)
        self.targets_classes = torch.zeros_like(inputs).scatter_(1, target.long().unsqueeze(1), 1)

        # ASL weights
        targets = self.targets_classes
        anti_targets = 1 - targets
        xs_pos = torch.exp(log_preds)
        xs_neg = 1 - xs_pos
        xs_pos = xs_pos * targets
        xs_neg = xs_neg * anti_targets
        asymmetric_w = torch.pow(1 - xs_pos - xs_neg,
                                 self.gamma_pos * targets + self.gamma_neg * anti_targets)
        log_preds = log_preds * asymmetric_w

        if self.eps > 0:  # label smoothing
            self.targets_classes = self.targets_classes.mul(1 - self.eps).add(self.eps / num_classes)

        # loss calculation
        loss = - self.targets_classes.mul(log_preds)

        loss = loss.sum(dim=-1)
        if self.reduction == 'mean':
            loss = loss.mean()

        return loss
    
class MyDataset(torch.utils.data.Dataset):
    def __init__(self,dataset_path,image_path, vsr = None, vsr_image_path = './data/vsr-images', max_length = 77):
        self.lxmert_tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
        self.max_length = max_length
        if(vsr):#assess only for the vsr dataset
            self.dataset = self.read_vsr_dataset(vsr)
            img_env = lmdb.open(
                vsr_image_path, readonly=True, create=False, readahead=not False
            )
            self.img_txn = img_env.begin(buffers=True)
        else:#assess for the SNLI-VE dataset
            self.dataset = self.read_dataset(dataset_path)
            img_env = lmdb.open(
                image_path, readonly=True, create=False, readahead=not False
            )
            self.img_txn = img_env.begin(buffers=True)
    
    def read_vsr_dataset(self,dataset_name, dataset_path = '../visual-spatial-reasoning/',splits_path='splits/', 
                         image_path = 'images/',sort = False, encode_labels = False):
        dataset = pd.read_json(dataset_path+splits_path+dataset_name, lines =True)
        dataset = dataset[['caption','image','label']]
        dataset.rename(columns = {'caption':'hypothesis', 'image':'Flickr30kID', 'label' : 'gold_label'}, inplace = True)
        if encode_labels:
            labels_encoding = {0:0,1:2}#leave the label 0 the same and convert 1 to 2 to mean entailment
            dataset['gold_label']=dataset['gold_label'].apply(lambda label: labels_encoding[label])
        if(dataset_name=='train.json'):
            dataset.drop(labels=[1786,3569,4553,4912], axis=0, inplace = True)
        elif(dataset_name=='test.json'):
            dataset.drop(labels=[135,614,1071,1621,1850], axis=0, inplace = True)
        elif(dataset_name=='dev.json'):
            dataset.drop(labels=[807], axis=0, inplace = True)
        dataset.reset_index(drop=True, inplace=True)
        if sort:
            dataset.sort_values(by="hypothesis", key=lambda x: x.str.len(), inplace = True)
        return dataset
    
    def read_dataset(self, url,sort = False):
        dataset = pd.read_csv(url)
        labels_encoding = {'contradiction':0,'neutral': 1,
                           'entailment':2}
        dataset = dataset[['hypothesis','Flickr30kID','gold_label']]
        dataset['gold_label']=dataset['gold_label'].apply(lambda label: labels_encoding[label])
        if sort:
            dataset.sort_values(by="hypothesis", key=lambda x: x.str.len(), inplace = True)
        return dataset
    
    def get_text_features(self,text): 
        #preprocess text
        inputs = self.lxmert_tokenizer(
            text,
            truncation=True,
            padding = True,
            return_token_type_ids=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )
        return inputs
    
    def __getitem__(self, idx):
        sample = self.dataset.loc[idx]
        img_name = sample['Flickr30kID']
        text = sample['hypothesis']
        label = sample['gold_label']
        item_img = pickle.loads(self.img_txn.get(img_name.encode()))
        
        item = {'text': text,
                'normalized_boxes': torch.tensor(item_img['normalized_boxes'][0], dtype = torch.float32),
                'features': torch.tensor(item_img['features'][0], dtype = torch.float32),
                'label': torch.tensor(label,dtype = torch.long)}
        return item
    
    def collate_fn(self,batch):
        #print(batch)
        text = [item['text'] for item in batch]
        inputs = self.get_text_features(text)
        item = {'input_ids': inputs['input_ids'].to(torch.int32),
                'attention_mask': inputs['attention_mask'].to(torch.int32),
                'token_type_ids': inputs['token_type_ids'].to(torch.int32),
                'label':torch.tensor([item['label'] for item in batch],dtype = torch.long),
                 'normalized_boxes': torch.stack(([item['normalized_boxes'] for item in batch])),
                'features': torch.stack(([item['features'] for item in batch]))
               }
        return item
        
        

    def __len__(self):
        return len(self.dataset.index)
    
    def __exit__(self):
        self.img_env.close()
        self.env.close()
        
class MyTrainer():
    def __init__(self,model,train,eval_test, device = None, num_labels = 3):
        self.device = device
        self.model = model
        self.train = train
        self.eval_test = eval_test
        self.test_acc_list = []#init
        self.model_path = "./models/new_my_model_epoch_"
        self.num_labels = num_labels
        self.config_problem_type = "single_label_classification"
        if self.config_problem_type == "single_label_classification":
          self.loss_fct = torch.nn.CrossEntropyLoss()
          #self.loss_fct = ASLSingleLabel()
          self.output_loss = lambda output,labels : self.loss_fct(output.logits.view(-1, self.num_labels), labels.view(-1)) 
        elif self.config_problem_type == "regression":
          self.loss_fct = torch.nn.MSELoss()
          if self.num_labels == 1: self.output_loss = lambda output,labels : self.loss_fct(output.logits.squeeze(), labels.squeeze())
          else: self.output_loss =  lambda output,labels : self.loss_fct(output.logits, labels)
        elif self.config_problem_type == "multi_label_classification":
          self.loss_fct = torch.nn.BCEWithLogitsLoss()
          self.output_loss = lambda output,labels : self.loss_fct(output.logits, labels)
        self.train_loader = DataLoader(self.train, batch_size=batch_size, shuffle=True, num_workers = 4,
                                 collate_fn = self.train.collate_fn)
        
    def train_model(self,batch_size = None, lr= None, epochs=None):
        optimizer = AdamW(self.model.parameters(), lr=lr)
        lr_scheduler = get_scheduler(
            name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps= epochs * len(self.train_loader)
        )
        for epoch in range(epochs):
            progress_bar = tqdm(range(math.ceil(len(self.train)/batch_size)))
            train_losses = []
            for item in self.train_loader:
                """
                print(item.keys())
                for key, value in item.items() :
                    print(value.shape)
                    print(key,'\n',value)
                """
                item['input_ids']=item['input_ids'].to(self.device)
                item['attention_mask']= item['attention_mask'].to(self.device)
                item['token_type_ids']= item['token_type_ids'].to(self.device)
                item['normalized_boxes'] = item['normalized_boxes'].to(self.device)
                item['features']= item['features'].to(self.device)
                item['label'] = item['label'].to(self.device)
                optimizer.zero_grad()
                outputs = self.model.forward(**item)
                label = item['label']
                loss = self.output_loss(outputs, label)
                train_losses.append(loss)
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                progress_bar.update(1)
            #print("Saving model ....")
            #model.save_model(self.model_path+str(epoch))
            #print("Model Saved!")
            test_acc = self.eval_test.evaluate(batch_size = batch_size)
            self.test_acc_list.append(test_acc)
            print('--- Epoch ',epoch,' Acc: ',test_acc)
            mean_loss = torch.tensor(train_losses).mean().item()
            print('Training loss: %.4f' % (mean_loss))
        return
    
class MyEvaluator():
  def __init__(self,model,test, device = None):
    self.test_dataset = test
    self.model = model
    self.device = device
  
  def evaluate(self, batch_size = 8):
      self.model.eval()
      loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle = False, num_workers = 4,
                         collate_fn = self.test_dataset.collate_fn)
      n_correct = 0
      n_possible = 0
      for item in loader:
        item['input_ids']=item['input_ids'].to(self.device)
        item['attention_mask']= item['attention_mask'].to(self.device)
        item['token_type_ids']= item['token_type_ids'].to(self.device)
        item['normalized_boxes'] = item['normalized_boxes'].to(self.device)
        item['features']= item['features'].to(self.device)
        item['label'] = item['label'].to(self.device)
        y_hat = self.model.predict(item)
        y = item['label']
        n_correct += (y == y_hat).sum().item()
        n_possible += float(y.shape[0])
      self.model.train()
      return n_correct / n_possible
    
class LxmertForBinaryClassification(torch.nn.Module):
    def __init__(self, lxmert):
        super(LxmertForBinaryClassification, self).__init__()
        self.lxmert = lxmert
        #self.classification = torch.nn.Linear(2, 2)

    def forward(self, input_ids, attention_mask, features, normalized_boxes, token_type_ids, label=None):

        outputs = self.lxmert(
                input_ids = input_ids,
                attention_mask = attention_mask,
                visual_feats = features,
                visual_pos = normalized_boxes,
                token_type_ids = token_type_ids)
        #outputs.logits = self.classification(outputs["cross_relationship_score"])
        outputs.logits = outputs["cross_relationship_score"]
        """
        reshaped_logits = outputs["cross_relationship_score"]

        if label is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, label.view(-1))
        else:
            loss = None

        return SequenceClassifierOutput(
                loss=loss,
                logits=reshaped_logits,
        )
        """
        return outputs
    
    def predict(self,item):
      """
      item (n_examples x n_features)
      """
      scores = model(**item)  # (n_examples x n_classes)
      predicted_labels = scores.logits.argmax(dim=-1)  # (n_examples)
      return predicted_labels
    
    def save_model(self,path):
        torch.save(self.state_dict(), path)
        
    def load_model(self,path):
        self.load_state_dict(torch.load(path))
        
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = 'vsr'
task = 'train'
batch_size = 32
epochs = 100
lr = 1e-5
if dataset =='vsr':
    num_labels = 2
elif dataset =='snli-ve':
    num_labels =3
    
model = LxmertForBinaryClassification(LxmertForPreTraining.from_pretrained('unc-nlp/lxmert-base-uncased'))

max_length = 32
train = MyDataset("../e-ViL/data/esnlive_train.csv",
                      "./data/my_image_db",
                      max_length = max_length, vsr= 'train.json')
test = MyDataset("../e-ViL/data/esnlive_test.csv",
                      "./data/my_image_db",
                      max_length = max_length,
                        vsr= 'test.json')
dev = MyDataset("../e-ViL/data/esnlive_dev.csv",
                      "./data/my_image_db",
                      max_length = max_length, vsr= 'dev.json')

print(len(train))
print(len(test))
print(len(dev))

if task =='train':
    test_evaluator = MyEvaluator(model,test, device = device)
    dev_evaluator = MyEvaluator(model,dev, device = device)
    trainer = MyTrainer(model,train,test_evaluator, device = device, num_labels = num_labels)
    model = model.to(device)
    print("-----Training Model-----")
    trainer.train_model(epochs=epochs ,batch_size = batch_size, lr = lr)
    print('----Training finished-----')
    dev_acc = dev_evaluator.evaluate(batch_size = batch_size)
    print("---- Dev Acc: ",dev_acc)
    train_acc = MyEvaluator(model,train,device=device).evaluate(batch_size = batch_size)
    print("--- Train Acc: ", train_acc)
    model.save_model(dataset+'_len'+str(max_length)+'_batch'+str(batch_size)+'_lr'+str(lr))
elif task =='test':
    model = model.to(device)
    model.load_model("my_model_epoch_9")
    evaluator = MyEvaluator(model,dev, device = device)
    acc = evaluator.evaluate(batch_size = batch_size)
    print(acc)
    #output = run_example(model,train)
    
    