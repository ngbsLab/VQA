import json
import argparse
from os.path import isfile, join
import re
import numpy as np
import pprint
import pickle
import os
from abc import ABC, abstractmethod
from dataset.data_generator import Database

# os.chdir("C:/Users/woohy/Desktop/bokyung/data")
import sys
sys.path.append('D:/bk/vis_lstm')
sys.path.append('D:/bk')



class VQADataset(Database):
    def __init__(self, data_dir, input_shape, num_train_classes, num_val_classes): #수정필요
        self.data_dir = os.chdir("C:/Users/woohy/Desktop/bokyung/data")
        self.train_folders = os.chdir("C:/Users/woohy/Desktop/bokyung/data/train")
        self.val_folders = os.chdir("C:/Users/woohy/Desktop/bokyung/data/val")
        
        self.prepare_database()
        self.input_shape = self.get_input_shape()
        self.num_train_classes = num_train_classes
        self.num_val_classes = num_val_classes
        
        
    def get_class(self):
    
        train_dict = defaultdict(list)
        val_dict = defaultdict(list)

        for train_class in self.train_folders:
            for train_image_path in glob(train_class + '/*.*'):
                train_dict[train_class.split('\\')[-1]].append(train_image_path)

        for val_class in self.train_folders:
            for val_image_path in glob(val_class + '/*.*'):
                val_dict[val_class.split('\\')[-1]].append(val_image_path)

      

        return train_dict, val_dict
        
        
    def preview_image(self, image_path): #numpy 파일을 image 로 읽을 수 있는지 알아보기
        image = Image.open(image_path)
        
        return image
        
        
    def get_input_shape(self):
        return # 1, 4096        
    
    
    def get_train_val_test_folders(self):
        return False
                
           
    def prepare_database(self, version = 1, data_dir = 'data'):     
	if version == 1:
		t_q_json_file = join(data_dir, 'train2014_questions.json')
		t_a_json_file = join(data_dir, 'train2014_annotations.json')

		v_q_json_file = join(data_dir, 'val2014_questions.json')
		v_a_json_file = join(data_dir, 'val2014_annotations.json')
		qa_data_file = join(data_dir, 'qa_data_file1.pkl')
		vocab_file = join(data_dir, 'vocab_file1.pkl')
	else:
		t_q_json_file = join(data_dir, 'v2_OpenEnded_mscoco_train2014_questions.json')
		t_a_json_file = join(data_dir, 'v2_mscoco_train2014_annotations.json')

		v_q_json_file = join(data_dir, 'v2_OpenEnded_mscoco_val2014_questions.json')
		v_a_json_file = join(data_dir, 'v2_mscoco_val2014_annotations.json')
		qa_data_file = join(data_dir, 'qa_data_file2.pkl')
		vocab_file = join(data_dir, 'vocab_file2.pkl')

	# IF ALREADY EXTRACTED
	# qa_data_file = join(data_dir, 'qa_data_file{}.pkl'.format(version))
	if isfile(qa_data_file):
		with open(qa_data_file) as f:
			data = pickle.load(f)
			return data

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

	
	print ("Ans", len(t_answers['annotations']), len(v_answers['annotations']))
	print ("Qu", len(t_questions['questions']), len(v_questions['questions']))

	answers = t_answers['annotations'] + v_answers['annotations']
	questions = t_questions['questions'] + v_questions['questions']
	
	answer_vocab = make_answer_vocab(answers)
	question_vocab, max_question_length = make_questions_vocab(questions, answers, answer_vocab)
	print ("Max Question Length", max_question_length)
	word_regex = re.compile(r'\w+')
	training_data = []
	for i,question in enumerate( t_questions['questions']):
		ans = t_answers['annotations'][i]['multiple_choice_answer']
		if ans in answer_vocab:
			training_data.append({
				'image_id' : t_answers['annotations'][i]['image_id'],
				'question' : np.zeros(max_question_length),
				'answer' : answer_vocab[ans]
				})
			question_words = re.findall(word_regex, question['question'])

			base = max_question_length - len(question_words)
			for i in range(0, len(question_words)):
				training_data[-1]['question'][base + i] = question_vocab[ question_words[i] ]

	print ("Training Data", len(training_data))
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

	data = {
		'training' : training_data,
		'validation' : val_data,
		'answer_vocab' : answer_vocab,
		'question_vocab' : question_vocab,
		'max_question_length' : max_question_length
	}

	print ("Saving qa_data")
	with open(qa_data_file, 'wb') as f:
		pickle.dump(data, f)

	with open(vocab_file, 'wb') as f:
		vocab_data = {
			'answer_vocab' : data['answer_vocab'],
			'question_vocab' : data['question_vocab'],
			'max_question_length' : data['max_question_length']
		}
		pickle.dump(vocab_data, f)

	return data    
	
	
    def load_quesions_answers(self, version = 1, data_dir = 'Data'):
	qa_data_file = join(data_dir, 'qa_data_file{}.pkl'.format(version))
	
	if isfile(qa_data_file):
		with open(qa_data_file,'rb') as f:
			data = pickle.load(f)
			return data
			
			
    def get_question_answer_vocab(self, version =1, data_dir = 'data'):
	vocab_file = join(data_dir, 'vocab_file{}.pkl'.format(version))
	vocab_data = pickle.load(open(vocab_file,'rb'))
	return vocab_data
	
	
    def make_answer_vocab(self, answers):
	top_n = 1000
	answer_frequency = {} 
	for annotation in answers:
		answer = annotation['multiple_choice_answer']
		if answer in answer_frequency:
			answer_frequency[answer] += 1
		else:
			answer_frequency[answer] = 1

	answer_frequency_tuples = [ (-frequency, answer) for answer, frequency in answer_frequency.items()]
	answer_frequency_tuples.sort()
	answer_frequency_tuples = answer_frequency_tuples[0:top_n-1]

	answer_vocab = {}
	for i, ans_freq in enumerate(answer_frequency_tuples):
		# print i, ans_freq
		ans = ans_freq[1]
		answer_vocab[ans] = i

	answer_vocab['UNK'] = top_n - 1
	return answer_vocab


    def make_questions_vocab(questions, answers, answer_vocab):
	word_regex = re.compile(r'\w+')
	question_frequency = {}

	max_question_length = 0
	for i,question in enumerate(questions):
		ans = answers[i]['multiple_choice_answer']
		count = 0
		if ans in answer_vocab:
			question_words = re.findall(word_regex, question['question'])
			for qw in question_words:
				if qw in question_frequency:
					question_frequency[qw] += 1
				else:
					question_frequency[qw] = 1
				count += 1
		if count > max_question_length:
			max_question_length = count


	qw_freq_threhold = 0
	qw_tuples = [ (-frequency, qw) for qw, frequency in question_frequency.items()]
	# qw_tuples.sort()

	qw_vocab = {}
	for i, qw_freq in enumerate(qw_tuples):
		frequency = -qw_freq[0]
		qw = qw_freq[1]
		# print frequency, qw
		if frequency > qw_freq_threhold:
			# +1 for accounting the zero padding for batc training
			qw_vocab[qw] = i + 1
		else:
			break

	qw_vocab['UNK'] = len(qw_vocab) + 1

	return qw_vocab, max_question_length
	     
    def load_fc7_features(self, data_dir, split):
	import h5py
	fc7_features = None
	image_id_list = None
	with h5py.File( join( data_dir, (split + '_fc7.h5')),'r') as hf:
		fc7_features = np.array(hf.get('fc7_features'))
	with h5py.File( join( data_dir, (split + '_image_id_list.h5')),'r') as hf:
		image_id_list = np.array(hf.get('image_id_list'))
	return fc7_features, image_id_list                 	
	
	

	
	
	
	
	
	
