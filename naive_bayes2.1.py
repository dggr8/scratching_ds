#!/usr/bin/env python

import math
import random
import time
def tokenize(message):
	message = message.lower()
	all_words = re.findall("[a-z0-9']+",message)
	return set(all_words)

def count_words(training_set):
	"""training set consists of pairs (message,is_spam)"""
	counts = defaultdict(lambda:[0,0])
	for message,is_spam in training_set:
		for word in tokenize(message):
			counts[word][0 if is_spam else 1] += 1

	print "Number of words - ",len(counts.keys())
	return counts

def word_probabilities(counts,total_spams,total_non_spams,k=0.5):
	"""turn the word_counts into a dict where key is the word
	and values are p(w|spam) and p(w| -spam)"""
	g = {}
	for w,(spam,non_spam) in counts.iteritems():
		g[w] = [(spam+k)/(total_spams+2*k),(non_spam+k)/(total_non_spams+2*k)]
	return g

def spam_probability(word_probs,message):
	message_words = tokenize(message)
	log_prob_if_spam = log_prob_if_not_spam = 0.0

	#keep the keys in a local bucket
	vocabulary = word_probs.keys()

	#iterate through each word in our message
	for word in message_words:
		try:
			#print word," ",word_probs[word][0]," ",word_probs[word][1]
			log_prob_if_spam += math.log(word_probs[word][0])
			log_prob_if_not_spam += math.log(word_probs[word][1])
		except KeyError:
			#print "Ha!"
			pass
	
	log_prob_if_spam /= len(message_words)
	log_prob_if_not_spam /= len(message_words)
	prob_if_spam  = math.exp(log_prob_if_spam)
	prob_if_not_spam = math.exp(log_prob_if_not_spam)
	#print "log_prob_if_spam is ",log_prob_if_spam
	#print "log_prob_if_not_spam is ",log_prob_if_not_spam

	try:
		spamness = prob_if_spam / (prob_if_spam + prob_if_not_spam)
		return spamness
	except ZeroDivisionError:
		print "This mail is weird"
		return 1.0
	
class NaiveBayesClassifier:

	def __init__(self,k=0.5):
		self.k = k
		self.word_probs = {}

	def train(self,training_set):

		num_spams = len([is_spam
						for message,is_spam in training_set
						if is_spam])
		num_non_spams = len(training_set) - num_spams

		# run training data through our "pipeline"
		word_counts = count_words(training_set)
		self.word_probs = word_probabilities(word_counts,
											num_spams,
											num_non_spams,
											self.k)

	def classify(self,message):
		return spam_probability(self.word_probs,message)

import glob,re
from collections import Counter
from collections import defaultdict

def split_data(data,prob):
	"""split data into fractions [prob, 1-prob]"""
	results = [],[]
	for row in data:
		results[0 if random.random()< prob else 1].append(row)
	return results

#give path here
path = r"/home/qwerty/coding/scratching_ds/spamassassin/*/*"
data = []

file_count = 0
for fn in glob.glob(path):
	is_spam = "ham" not in fn

	file_count += 1
	with open(fn,'r') as file:
		words = file.read()
		data.append((words,is_spam))

random.seed(0)
train_data,test_data = split_data(data,0.75)

classifier = NaiveBayesClassifier()
classifier.train(train_data)

words_to_be_tested = sum([len(line) for line,_ in test_data])
classified = [(line,is_spam,classifier.classify(line))
			for line,is_spam in test_data]


counts = Counter((is_spam,spam_probability>0.5)
				for _,is_spam,spam_probability in classified)

print counts
