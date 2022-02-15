import json
import argparse
from os.path import isfile, join
import re
import numpy as np
import pprint
import pickle
import os

os.chdir("D:/bk")

t_q_json_file = join('./data', 'train2014_questions.json')
t_a_json_file = join('./data', 'train2014_annotations.json')

v_q_json_file = join('./data', 'val2014_questions.json')
v_a_json_file = join('./data', 'val2014_annotations.json')
qa_data_file = join('./data', 'qa_data_file1.pkl')
vocab_file = join('./data', 'vocab_file1.pkl')

# with open(qa_data_file) as f:
#     data = pickle.load(f)
#
# with open('D:\\김보경\\data\\qa_data_file1.pkl', 'r',encoding='utf-16') as t:
#     data = pickle.load(t)

print ("Loading Training questions")
with open(t_q_json_file) as f:
	t_questions = json.loads(f.read())

print ("Loading Training anwers")
with open(t_a_json_file) as f:
	t_answers = json.loads(f.read())


print ("Loading Val questions")
with open(v_q_json_file) as f:
	v_questions = json.loads(f.read())

print ("Loading Val answers")
with open(v_a_json_file) as f:
	v_answers = json.loads(f.read())

print ("Answer", 'train_annotations:'+str(len(t_answers['annotations'])), 'val_annotations:'+str(len(v_answers['annotations'])))
print ("Question", 'train_annotations:'+str(len(t_questions['questions'])), 'val_annotations:'+str(len(v_questions['questions'])))

answers = t_answers['annotations'] + v_answers['annotations']
questions = t_questions['questions'] + v_questions['questions']

answer_vocab = make_answer_vocab(answers)
question_vocab, max_question_length = make_questions_vocab(questions, answers, answer_vocab)
print ("Max Question Length", max_question_length)

word_regex = re.compile(r'\w+')
training_data = []

for i, question in enumerate(t_questions['questions']):
    ans = t_answers['annotations'][i]['multiple_choice_answer']
    if ans in answer_vocab:
        training_data.append({
            'image_id': t_answers['annotations'][i]['image_id'],
            'question': np.zeros(max_question_length),
            'answer': answer_vocab[ans]
        })
        question_words = re.findall(word_regex, question['question'])

        base = max_question_length - len(question_words)
        for i in range(0, len(question_words)):
            training_data[-1]['question'][base + i] = question_vocab[question_words[i]]

print("Training Data", len(training_data))

val_data = []
for i,question in enumerate( v_questions['questions']):
	ans = v_answers['annotations'][i]['multiple_choice_answer']
	if ans in answer_vocab:
		val_data.append({
			'image_id' : v_answers['annotations'][i]['image_id'],
			'question' : np.zeros(max_question_length),
			'answer' : answer_vocab[ans]
			})
		question_words = re.findall(word_regex, question['question'])

		base = max_question_length - len(question_words)
		for i in range(0, len(question_words)):
			val_data[-1]['question'][base + i] = question_vocab[ question_words[i] ]

print ("Validation Data", len(val_data))




def load_fc7_features(data_dir, split):
	import h5py
	fc7_features = None
	image_id_list = None
	with h5py.File( join( data_dir, (split + '_fc7.h5')),'r') as hf:
		fc7_features = np.array(hf.get('fc7_features'))
	with h5py.File( join( data_dir, (split + '_image_id_list.h5')),'r') as hf:
		image_id_list = np.array(hf.get('image_id_list'))
	return fc7_features, image_id_list

fc7_features, image_id_list = load_fc7_features('./data', 'train')

fc7_features.shape
image_id_list.shape
import h5py
fc7_features = None
image_id_list = None

with h5py.File('D:/김보경/data/train_fc7.h5', 'r') as hf:
	fc7_features = np.array(hf.get('fc7_features'))

with h5py.File('D:/김보경/data/train_image_id_list.h5', 'r') as hf:
	image_id_list = np.array(hf.get('image_id_list'))

len(fc7_features)
len(image_id_list)

(fc7_features[0]).shape

image_id_map = {}
for i in range(len(image_id_list)):
	image_id_map[ image_id_list[i] ] = i

ans_map = { qa_data['answer_vocab'][ans] : ans for ans in qa_data['answer_vocab']}

import pickle
vocab_data = data_loader.get_question_answer_vocab('./data')
data_dir = 'D:/bk/data'
vocab_file = join(data_dir, 'vocab_file1.pkl')
vocab_data = pickle.load(open(vocab_file,'rb'))

import scipy.misc
scipy.misc.imread

question_vocab = vocab_data['question_vocab']
import re
word_regex = re.compile(r'\w+')
question_ids = np.zeros((1, vocab_data['max_question_length']), dtype = 'int32')
question_words = re.findall(word_regex, '사진속에있는것은무엇인가요?')

base = vocab_data['max_question_length'] - len(question_words)

for i in range(0, len(question_words)):
	if question_words[i] in question_vocab:
		question_ids[0][base + i] = question_vocab[question_words[i]]
	else:
		question_ids[0][base + i] = question_vocab['UNK']